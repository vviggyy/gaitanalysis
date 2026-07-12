"""
ICC(3,1) bar charts from the NEW long-format composite table.

Replaces plot_icc.py, which expected the OLD wide per-system ICC CSVs
(one row per Metric; columns Ataxia_NG / PD_RG / ..._CI95). This reads the
single long "Composite Scores" table instead:

    feature, system, week, test, metric, cohort, ICC, n,
    spearman_r, spearman_p, comp_score, all_conditions_met

Only the ICC column is used here (Spearman / composite are ignored for now).
The new table has NO CI95 column, so bars are drawn without error whiskers.

The old 4 "categories" map onto (cohort, test) pairs in the long table:
    Ataxia_NG -> (ATX_SARA, NG)   PD_NG -> (PD_UPDRS, NG)
    Ataxia_RG -> (ATX_SARA, RG)   PD_RG -> (PD_UPDRS, RG)

Three public functions (mirroring plot_icc.py):
  plot_individual   – one bar chart per category -> 4 PNGs
  plot_combined     – 2x2 grid of all 4 categories -> 1 PNG
  plot_multi_system – 2x2 grid, systems stacked per subplot -> 1 PNG
"""

import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

# Save vector SVG so charts stay editable in Illustrator; 'none' keeps text as
# real <text> elements (editable/selectable) rather than outlined paths.
FMT = "svg"
plt.rcParams["svg.fonttype"] = "none"

HERE = pathlib.Path(__file__).resolve().parent
COMPOSITE_CSV = (HERE.parent.parent /
                 "Composite Scores - Averaged [JUNE 26 2026] - "
                 "SPEARMAN CORR + ICC MERGED, AVG, T1.csv")
OUT = HERE / "out" / "071126"
OUT.mkdir(parents=True, exist_ok=True)

# (cohort, test) in the long table  ->  display title
CATEGORIES = [
    ("ATX_SARA", "NG", "Ataxia — Normal Gait"),
    ("ATX_SARA", "RG", "Ataxia — Backward Gait"),
    ("PD_UPDRS", "NG", "PD — Normal Gait"),
    ("PD_UPDRS", "RG", "PD — Backward Gait"),
]

BANDS = [
    (-1.0, 0.5,  "#fee0d2", "#d9534f", "Poor (<0.5)"),
    (0.5,  0.75, "#ffffcc", "#f0ad4e", "Moderate (0.5–0.75)"),
    (0.75, 0.9,  "#c7e9c0", "#5cb85c", "Good (0.75–0.9)"),
    (0.9,  1.01, "#9ecae1", "#337ab7", "Excellent (>0.9)"),
]

# system code (as it appears in the long table's `system` column) → color / legend
SYSTEM_COLORS = {"Xsens": "#7570b3", "JT": "#1b9e77", "Myo": "#d95f02",
                 "Arm": "#e7298a"}
SYSTEM_LABELS = {"Xsens": "Xsens Leg Features", "Arm": "Xsens Arm Features",
                 "JT": "JTrack Smartphone", "Myo": "Force plate"}
# short codes for titles (system column value → code)
SYSTEM_CODES = {"Xsens": "Leg", "Arm": "Arm", "JT": "JT", "Myo": "FP"}

# Stacking order in the multi-system charts (list-last renders at the top), kept
# so the two Xsens groups (Arm + Leg) sit next to each other rather than split
# across the top and bottom of each panel.
SYSTEM_ORDER = ("Myo", "JT", "Xsens", "Arm")

# Lower x-limit for the bar axes. New-table ICCs can dip slightly negative
# (min ≈ -0.18), so leave headroom below 0 rather than clipping at 0.2.
X_LO, X_HI = -0.25, 1.05


# ── helpers ──────────────────────────────────────────────────────────

def load(csv_path=COMPOSITE_CSV):
    """Load the long composite table (ICC column coerced to float)."""
    df = pd.read_csv(csv_path)
    df["ICC"] = pd.to_numeric(df["ICC"], errors="coerce")
    return df


def band_color(val):
    """ICC value → bar hex color based on reliability band."""
    for lo, hi, _, bar_c, _ in BANDS:
        if lo <= val < hi:
            return bar_c
    return BANDS[-1][3]


def _icc_arrays(df, cohort, test, system=None):
    """(icc_vals, metric_names) for one (cohort, test)[, system], NaN dropped."""
    sub = df[(df["cohort"] == cohort) & (df["test"] == test)]
    if system is not None:
        sub = sub[sub["system"] == system]
    sub = sub.dropna(subset=["ICC"])
    return sub["ICC"].to_numpy(), sub["metric"].to_numpy()


def _sort_ascending(icc, names, top_n=None):
    """Sort by ICC ascending (highest ends at top of horizontal bars).
    Optionally keep only the top_n highest first."""
    if top_n is not None:
        idx = np.argsort(icc)[::-1][:top_n]
        icc, names = icc[idx], names[idx]
    order = np.argsort(icc)
    return icc[order], names[order]


def _bar_colors(icc_sorted, color, system_name=None):
    if color == "reliability":
        return [band_color(v) for v in icc_sorted]
    if color == "system" and system_name in SYSTEM_COLORS:
        return [SYSTEM_COLORS[system_name]] * len(icc_sorted)
    return ["#4878a8"] * len(icc_sorted)


# ── plot_individual ──────────────────────────────────────────────────

def plot_individual(df, out_dir, tag, system=None, system_name=None,
                    color="reliability", bg_bands=True, top_n=None):
    """One bar chart per category → 4 PNGs."""
    out_dir = pathlib.Path(out_dir)
    label = system_name or system or "All systems"

    for cohort, test, title in CATEGORIES:
        icc, names = _icc_arrays(df, cohort, test, system)
        if len(icc) == 0:
            continue
        icc, names = _sort_ascending(icc, names, top_n)
        colors = _bar_colors(icc, color, system)

        n = len(names)
        fig, ax = plt.subplots(figsize=(8, max(4, n * 0.45 + 1.5)))

        if bg_bands:
            for x0, x1, bg_c, _, _ in BANDS:
                ax.axvspan(x0, x1, color=bg_c, alpha=0.4, zorder=0)

        y_pos = np.arange(n)
        ax.barh(y_pos, icc, height=0.6, color=colors, edgecolor="white", zorder=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=max(7, 9 - n // 20))
        ax.set_xlabel("ICC(3,1)")
        suffix = f" ({label})" if system else ""
        ax.set_title(f"ICC(3,1) — {title}{suffix}", fontsize=12, fontweight="bold")
        ax.set_xlim(X_LO, X_HI)
        ax.axvline(0, color="black", lw=0.5, ls="--", zorder=1)

        handles = [mpatches.Patch(facecolor=bc, label=l)
                   for _, _, _, bc, l in BANDS]
        ax.legend(handles=handles, loc="lower right", fontsize=7,
                  title="Reliability", title_fontsize=8)

        plt.tight_layout()
        sep = f"_{tag}" if tag else ""
        fname = out_dir / f"icc_bar{sep}_{cohort.lower()}_{test.lower()}.{FMT}"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved {fname}")


# ── plot_combined ────────────────────────────────────────────────────

def plot_combined(df, out_dir, tag, system=None, system_name=None,
                  top_n=None, color="reliability"):
    """2×2 grid of all 4 categories → 1 PNG."""
    out_dir = pathlib.Path(out_dir)
    label = system_name or system or "All systems"

    max_n = 0
    for cohort, test, _ in CATEGORIES:
        icc, _ = _icc_arrays(df, cohort, test, system)
        cnt = min(len(icc), top_n) if top_n else len(icc)
        max_n = max(max_n, cnt)

    fig, axes = plt.subplots(2, 2, figsize=(14, max(6, max_n * 0.4 + 2)))

    for ax, (cohort, test, title) in zip(axes.flat, CATEGORIES):
        icc, names = _icc_arrays(df, cohort, test, system)
        if len(icc) == 0:
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        icc, names = _sort_ascending(icc, names, top_n)
        colors = _bar_colors(icc, color, system)

        y_pos = np.arange(len(names))
        ax.barh(y_pos, icc, height=0.6, color=colors, edgecolor="white", zorder=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("ICC(3,1)", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(X_LO, X_HI)
        for thresh in (0.5, 0.75, 0.9):
            ax.axvline(thresh, color="gray", lw=0.5, ls=":", zorder=1)
        ax.axvline(0, color="black", lw=0.5, ls="--", zorder=1)

    handles = [mpatches.Patch(facecolor=c, label=l) for _, _, _, c, l in BANDS]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=True, title="Reliability", title_fontsize=10)
    top_str = f"Top {top_n} " if top_n else ""
    fig.suptitle(f"ICC(3,1) — {top_str}Metrics ({label}, All Subjects)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fname = out_dir / f"icc_combined_{tag}.{FMT}"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


# ── plot_multi_system ────────────────────────────────────────────────

def plot_multi_system(df, out_dir, systems=SYSTEM_ORDER,
                      top_n=None, top_n_per_system=None):
    """2×2 grid with multiple systems stacked per subplot → 1 PNG.

    systems: iteration/plot order of the `system` values to include.
    top_n_per_system: dict like {"Xsens": 10, "JT": None, ...} — None keeps all
        features for that system. Takes precedence over top_n.
    """
    out_dir = pathlib.Path(out_dir)
    per_system_filter = top_n_per_system is not None
    use_sys_colors = top_n is None
    bar_color_plain = "#4d4d4d"

    figsize = (14, 14) if (top_n is not None or per_system_filter) else (16, 22)
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for ax, (cohort, test, title) in zip(axes.flat, CATEGORIES):
        all_rows = []
        for sys_name in systems:
            icc, names = _icc_arrays(df, cohort, test, sys_name)
            entries = list(zip(names, icc, [sys_name] * len(icc)))
            n = top_n_per_system.get(sys_name) if per_system_filter else top_n
            if n is not None:
                entries.sort(key=lambda x: x[1], reverse=True)
                entries = entries[:n]
            entries.sort(key=lambda x: x[1])
            all_rows.extend(entries)

        if not all_rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title)
            continue

        labels = [r[0] for r in all_rows]
        iccs = np.array([r[1] for r in all_rows])
        sys_names = [r[2] for r in all_rows]
        colors = ([SYSTEM_COLORS.get(s, bar_color_plain) for s in sys_names]
                  if use_sys_colors else bar_color_plain)

        for x0, x1, bg_c, _, _ in BANDS:
            ax.axvspan(x0, x1, color=bg_c, alpha=0.35, zorder=0)

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, iccs, height=0.7, color=colors, edgecolor="white", zorder=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel("ICC(3,1)", fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(X_LO, X_HI)
        ax.axvline(0, color="black", lw=0.5, ls="--", zorder=1)

        # separator lines between systems
        prev_sys = sys_names[0]
        for i, s in enumerate(sys_names):
            if s != prev_sys:
                ax.axhline(i - 0.5, color="black", lw=0.8, ls="-", zorder=3)
                prev_sys = s

    band_handles = [mpatches.Patch(facecolor=bg, alpha=0.35, label=l)
                    for _, _, bg, _, l in BANDS]

    if use_sys_colors:
        sys_handles = [mpatches.Patch(facecolor=SYSTEM_COLORS[s],
                                      label=SYSTEM_LABELS.get(s, s))
                       for s in systems if s in SYSTEM_COLORS]
        sep = mlines.Line2D([], [], color="none", label="")
        fig.legend(handles=sys_handles + [sep] + band_handles,
                   loc="lower center", ncol=4, fontsize=9, frameon=True,
                   title="System                                              Reliability",
                   title_fontsize=10, columnspacing=1.5)
        if per_system_filter:
            parts = [f"{SYSTEM_CODES.get(s, s)}: top {n}" if n is not None
                     else f"{SYSTEM_CODES.get(s, s)}: all"
                     for s, n in top_n_per_system.items()]
            suptitle = "ICC(3,1) — " + ", ".join(parts)
        else:
            suptitle = ("ICC(3,1) — All Systems ("
                        + ", ".join(SYSTEM_CODES.get(s, s) for s in systems) + ")")
        fname = out_dir / f"icc_all_systems_combined.{FMT}"
        rect = [0, 0.035, 1, 0.99]
    else:
        fig.legend(handles=band_handles, loc="lower center", ncol=4, fontsize=9,
                   frameon=True, title="Reliability", title_fontsize=10,
                   columnspacing=1.5)
        suptitle = (f"ICC(3,1) — Top {top_n} Features per System ("
                    + ", ".join(SYSTEM_CODES.get(s, s) for s in systems) + ")")
        fname = out_dir / f"icc_top{top_n}_all_systems.{FMT}"
        rect = [0, 0.04, 1, 0.99]

    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=rect)
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


# ── __main__ ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load()

    # Per-system individual + combined charts
    for sys_name in ("Xsens", "JT", "Myo", "Arm"):
        plot_individual(df, OUT, sys_name.lower(), system=sys_name)
        plot_combined(df, OUT, sys_name.lower(), system=sys_name,
                      top_n=10 if sys_name in ("Xsens", "Arm") else None)

    # All-systems comparison (single file already holds every system).
    # Dict order follows SYSTEM_ORDER so the Arm + Leg Xsens groups stay adjacent.
    plot_multi_system(df, OUT,
                      top_n_per_system={"Myo": 10, "JT": None,
                                        "Xsens": 10, "Arm": 10})
    plot_multi_system(df, OUT, top_n=5)

    print("\nDone.")
