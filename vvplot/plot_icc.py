"""
Consolidated ICC(1,1) bar-chart plotting.

Three public functions:
  plot_individual  – one bar chart per category → 4 PNGs
  plot_combined    – 2×2 grid of all 4 categories → 1 PNG
  plot_multi_system – 2×2 grid with multiple systems stacked → 1 PNG
"""

import pathlib
import ast
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"

CATEGORIES = [
    ("Ataxia_NG", "Ataxia — Normal Gait"),
    ("Ataxia_RG", "Ataxia — Backward Gait"),
    ("PD_NG", "PD — Normal Gait"),
    ("PD_RG", "PD — Backward Gait"),
]

BANDS = [
    (-1.0, 0.5,  "#fee0d2", "#d9534f", "Poor (<0.5)"),
    (0.5,  0.75, "#ffffcc", "#f0ad4e", "Moderate (0.5–0.75)"),
    (0.75, 0.9,  "#c7e9c0", "#5cb85c", "Good (0.75–0.9)"),
    (0.9,  1.01, "#9ecae1", "#337ab7", "Excellent (>0.9)"),
]

SYSTEM_COLORS = {"Xsens": "#7570b3", "JT": "#1b9e77", "Myo": "#d95f02"}


# ── helpers ──────────────────────────────────────────────────────────

def parse_ci(df, ci_col):
    """Parse CI95 string column → (ci_lo, ci_hi) numpy arrays, NaN-safe."""
    ci_lo = np.empty(len(df))
    ci_hi = np.empty(len(df))
    for i, raw in enumerate(df[ci_col]):
        if pd.isna(raw) or raw == "":
            ci_lo[i] = np.nan
            ci_hi[i] = np.nan
        else:
            lo, hi = ast.literal_eval(raw)
            ci_lo[i] = lo
            ci_hi[i] = hi
    return ci_lo, ci_hi


def band_color(val):
    """ICC value → bar hex color based on reliability band."""
    for lo, hi, _, bar_c, _ in BANDS:
        if lo <= val < hi:
            return bar_c
    return BANDS[-1][3]


def _icc_arrays(df, col_prefix):
    """Return (icc_vals, ci_lo, ci_hi, metric_names) with NaN rows masked out."""
    icc_raw = df[col_prefix].values
    icc_vals = np.array([float(v) if v != "" and not pd.isna(v) else np.nan
                         for v in icc_raw])
    ci_lo, ci_hi = parse_ci(df, col_prefix + "_CI95")
    mask = ~np.isnan(icc_vals)
    return icc_vals[mask], ci_lo[mask], ci_hi[mask], df["Metric"].values[mask]


def _sort_ascending(icc, ci_lo, ci_hi, names, top_n=None):
    """Sort by ICC ascending (highest at top of horizontal bar chart).
    Optionally keep only top_n highest."""
    if top_n is not None:
        idx = np.argsort(icc)[::-1][:top_n]
        icc, ci_lo, ci_hi, names = icc[idx], ci_lo[idx], ci_hi[idx], names[idx]
    order = np.argsort(icc)
    return icc[order], ci_lo[order], ci_hi[order], names[order]


def _bar_colors(icc_sorted, color, system_name=None):
    """Return list of colors for bars."""
    if color == "reliability":
        return [band_color(v) for v in icc_sorted]
    if color == "system" and system_name and system_name in SYSTEM_COLORS:
        return [SYSTEM_COLORS[system_name]] * len(icc_sorted)
    # uniform
    return ["#4878a8"] * len(icc_sorted)


# ── plot_individual ──────────────────────────────────────────────────

def plot_individual(csv_path, out_dir, tag, system_name=None,
                    color="reliability", bg_bands=True, top_n=None):
    """One bar chart per category → 4 PNGs."""
    df = pd.read_csv(csv_path)
    out_dir = pathlib.Path(out_dir)
    label = system_name or "Xsens"

    for col_prefix, title in CATEGORIES:
        icc, ci_lo, ci_hi, names = _icc_arrays(df, col_prefix)
        if len(icc) == 0:
            continue
        icc, ci_lo, ci_hi, names = _sort_ascending(icc, ci_lo, ci_hi, names, top_n)
        err_lo = icc - ci_lo
        err_hi = ci_hi - icc
        colors = _bar_colors(icc, color, system_name)

        n = len(names)
        fig_h = max(4, n * 0.45 + 1.5)
        fig, ax = plt.subplots(figsize=(8, fig_h))

        if bg_bands:
            for x0, x1, bg_c, _, _ in BANDS:
                ax.axvspan(x0, x1, color=bg_c, alpha=0.4, zorder=0)

        y_pos = np.arange(n)
        ax.barh(y_pos, icc, height=0.6, color=colors, edgecolor="white",
                xerr=[err_lo, err_hi], capsize=2,
                error_kw=dict(ecolor="gray", lw=0.8), zorder=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=max(7, 9 - n // 20))
        ax.set_xlabel("ICC(1,1)")
        suffix = f" ({label})" if system_name else ""
        ax.set_title(f"ICC(1,1) — {title}{suffix}",
                     fontsize=12, fontweight="bold")
        ax.set_xlim(-0.6, 1.05)
        ax.axvline(0, color="black", lw=0.5, ls="--", zorder=1)

        handles = [mpatches.Patch(facecolor=bc, label=l)
                   for _, _, _, bc, l in BANDS]
        ax.legend(handles=handles, loc="lower right", fontsize=7,
                  title="Reliability", title_fontsize=8)

        plt.tight_layout()
        sep = f"_{tag}" if tag else ""
        fname = out_dir / f"icc_bar{sep}_{col_prefix.lower()}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved {fname}")


# ── plot_combined ────────────────────────────────────────────────────

def plot_combined(csv_path, out_dir, tag, system_name=None,
                  top_n=None, color="reliability"):
    """2×2 grid of all 4 categories → 1 PNG."""
    df = pd.read_csv(csv_path)
    out_dir = pathlib.Path(out_dir)
    label = system_name or "Xsens"

    max_n = 0
    for col_prefix, _ in CATEGORIES:
        icc, *_ = _icc_arrays(df, col_prefix)
        cnt = min(len(icc), top_n) if top_n else len(icc)
        max_n = max(max_n, cnt)

    fig_h = max(6, max_n * 0.4 + 2)
    fig, axes = plt.subplots(2, 2, figsize=(14, fig_h))

    for ax, (col_prefix, title) in zip(axes.flat, CATEGORIES):
        icc, ci_lo, ci_hi, names = _icc_arrays(df, col_prefix)
        if len(icc) == 0:
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        icc, ci_lo, ci_hi, names = _sort_ascending(icc, ci_lo, ci_hi, names, top_n)
        err_lo = icc - ci_lo
        err_hi = ci_hi - icc
        colors = _bar_colors(icc, color, system_name)

        y_pos = np.arange(len(names))
        ax.barh(y_pos, icc, height=0.6, color=colors, edgecolor="white",
                xerr=[err_lo, err_hi], capsize=2,
                error_kw=dict(ecolor="gray", lw=0.8), zorder=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("ICC(1,1)", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(-0.6 if top_n is None else 0, 1.05)
        for thresh in (0.5, 0.75, 0.9):
            ax.axvline(thresh, color="gray", lw=0.5, ls=":", zorder=1)
        ax.axvline(0, color="black", lw=0.5, ls="--", zorder=1)

    handles = [mpatches.Patch(facecolor=c, label=l) for _, _, _, c, l in BANDS]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=True, title="Reliability", title_fontsize=10)
    top_str = f"Top {top_n} " if top_n else ""
    fig.suptitle(f"ICC(1,1) — {top_str}Metrics ({label}, All Subjects)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fname = out_dir / f"icc_combined_{tag}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


# ── plot_multi_system ────────────────────────────────────────────────

def plot_multi_system(system_csvs, out_dir, top_n=None):
    """2×2 grid with multiple systems stacked per subplot → 1 PNG.

    system_csvs: dict like {"Xsens": path, "JT": path, "Myo": path}
    """
    out_dir = pathlib.Path(out_dir)
    systems_order = list(system_csvs.items())  # preserve insertion order
    dfs = {name: pd.read_csv(path) for name, path in systems_order}

    use_sys_colors = top_n is None
    bar_color_plain = "#4d4d4d"

    figsize = (14, 14) if top_n is not None else (16, 22)
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for ax, (col_prefix, title) in zip(axes.flat, CATEGORIES):
        all_rows = []
        for sys_name, _ in systems_order:
            df = dfs[sys_name]
            icc, ci_lo, ci_hi, names = _icc_arrays(df, col_prefix)
            entries = list(zip(names, icc, ci_lo, ci_hi,
                               [sys_name] * len(icc)))
            if top_n is not None:
                entries.sort(key=lambda x: x[1], reverse=True)
                entries = entries[:top_n]
            entries.sort(key=lambda x: x[1])
            all_rows.extend(entries)

        if not all_rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title)
            continue

        labels = [r[0] for r in all_rows]
        iccs = np.array([r[1] for r in all_rows])
        ci_los = np.array([r[2] for r in all_rows])
        ci_his = np.array([r[3] for r in all_rows])
        sys_names = [r[4] for r in all_rows]

        err_lo = iccs - ci_los
        err_hi = ci_his - iccs
        colors = ([SYSTEM_COLORS.get(s, bar_color_plain) for s in sys_names]
                  if use_sys_colors else bar_color_plain)

        for x0, x1, bg_c, _, _ in BANDS:
            ax.axvspan(x0, x1, color=bg_c, alpha=0.35, zorder=0)

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, iccs, height=0.7, color=colors, edgecolor="white",
                xerr=[err_lo, err_hi], capsize=1.5,
                error_kw=dict(ecolor="gray", lw=0.6), zorder=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel("ICC(1,1)", fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(-0.6, 1.05)
        ax.axvline(0, color="black", lw=0.5, ls="--", zorder=1)

        # separator lines between systems
        prev_sys = sys_names[0]
        for i, s in enumerate(sys_names):
            if s != prev_sys:
                ax.axhline(i - 0.5, color="black", lw=0.8, ls="-", zorder=3)
                prev_sys = s

    # legend
    band_handles = [mpatches.Patch(facecolor=bg, alpha=0.35, label=l)
                    for _, _, bg, _, l in BANDS]

    if use_sys_colors:
        sys_handles = [mpatches.Patch(facecolor=SYSTEM_COLORS[s], label=s)
                       for s in system_csvs]
        sep = mlines.Line2D([], [], color="none", label="")
        fig.legend(handles=sys_handles + [sep] + band_handles,
                   loc="lower center", ncol=4, fontsize=9, frameon=True,
                   title="System                                              Reliability",
                   title_fontsize=10, columnspacing=1.5)
        suptitle = "ICC(1,1) — All Systems (" + ", ".join(system_csvs) + ")"
        fname = out_dir / "icc_all_systems_combined.png"
        rect = [0, 0.035, 1, 0.99]
    else:
        fig.legend(handles=band_handles, loc="lower center", ncol=4, fontsize=9,
                   frameon=True, title="Reliability", title_fontsize=10,
                   columnspacing=1.5)
        suptitle = (f"ICC(1,1) — Top {top_n} Features per System ("
                    + ", ".join(system_csvs) + ")")
        fname = out_dir / f"icc_top{top_n}_all_systems.png"
        rect = [0, 0.04, 1, 0.99]

    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=rect)
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


# ── __main__ ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    lr_csv = OUT / "icc11_lr_averaged_all_subjects.csv"
    jt_csv = OUT / "icc11_jt_all_subjects.csv"
    myo_csv = OUT / "icc11_myo_all_subjects.csv"

    # Xsens (original split L/R)
    plot_individual(OUT / "icc11_all_subjects.csv", OUT, "",
                    color="uniform", bg_bands=True)
    plot_combined(OUT / "icc11_all_subjects.csv", OUT, "xsens", top_n=10)

    # LR Averaged
    plot_individual(lr_csv, OUT, "lr_avg", system_name="LR Averaged")
    plot_combined(lr_csv, OUT, "lr_averaged", top_n=10)

    # JT & Myo
    plot_individual(jt_csv, OUT, "jt", system_name="JT")
    plot_combined(jt_csv, OUT, "jt", system_name="JT")
    plot_individual(myo_csv, OUT, "myo", system_name="Myo")
    plot_combined(myo_csv, OUT, "myo", system_name="Myo")

    # All systems comparison
    plot_multi_system({"Xsens": lr_csv, "JT": jt_csv, "Myo": myo_csv}, OUT)
    plot_multi_system({"Xsens": lr_csv, "JT": jt_csv, "Myo": myo_csv}, OUT,
                      top_n=5)

    print("\nDone.")
