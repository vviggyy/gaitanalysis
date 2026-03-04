"""
Bar charts of ICC(1,1) for JT and Myo systems:
  - Individual bar charts per system × category
  - Combined 2x2 chart per system
"""

import pathlib
import ast
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"

categories = [
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


def band_color(val):
    for lo, hi, _, bar_c, _ in BANDS:
        if lo <= val < hi:
            return bar_c
    return BANDS[-1][3]


def parse_ci(df, ci_col):
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


def plot_individual_bars(df, system_name, system_tag):
    """One bar chart per category with all metrics."""
    for col_prefix, title in categories:
        icc_raw = df[col_prefix].values
        # Handle empty strings
        icc_vals = np.array([float(v) if v != "" else np.nan for v in icc_raw])
        ci_lo, ci_hi = parse_ci(df, col_prefix + "_CI95")

        # Drop metrics with no ICC
        mask = ~np.isnan(icc_vals)
        if mask.sum() == 0:
            continue

        icc_vals = icc_vals[mask]
        ci_lo = ci_lo[mask]
        ci_hi = ci_hi[mask]
        metric_names = df["Metric"].values[mask]

        err_lo = icc_vals - ci_lo
        err_hi = ci_hi - icc_vals

        order = np.argsort(icc_vals)
        metrics = metric_names[order]
        icc_sorted = icc_vals[order]
        err_lo_sorted = err_lo[order]
        err_hi_sorted = err_hi[order]
        colors = [band_color(v) for v in icc_sorted]

        n = len(metrics)
        fig_h = max(4, n * 0.45 + 1.5)
        fig, ax = plt.subplots(figsize=(8, fig_h))
        for x0, x1, bg_c, _, label in BANDS:
            ax.axvspan(x0, x1, color=bg_c, alpha=0.4, zorder=0)

        y_pos = np.arange(n)
        ax.barh(y_pos, icc_sorted, height=0.6, color=colors, edgecolor="white",
                xerr=[err_lo_sorted, err_hi_sorted], capsize=2,
                error_kw=dict(ecolor="gray", lw=0.8), zorder=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics, fontsize=9)
        ax.set_xlabel("ICC(1,1)")
        ax.set_title(f"ICC(1,1) — {system_name} — {title}",
                      fontsize=12, fontweight="bold")
        ax.set_xlim(-0.6, 1.05)
        ax.axvline(0, color="black", lw=0.5, ls="--", zorder=1)

        handles = [mpatches.Patch(facecolor=bc, label=l) for _, _, _, bc, l in BANDS]
        ax.legend(handles=handles, loc="lower right", fontsize=7,
                  title="Reliability", title_fontsize=8)

        plt.tight_layout()
        fname = OUT / f"icc_bar_{system_tag}_{col_prefix.lower()}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved {fname}")


def plot_combined_2x2(df, system_name, system_tag):
    """Combined 2x2 subplot with all metrics per category."""
    fig, axes = plt.subplots(2, 2, figsize=(14, max(6, len(df) * 0.4 + 2)))

    for ax, (col_prefix, title) in zip(axes.flat, categories):
        icc_raw = df[col_prefix].values
        icc_vals = np.array([float(v) if v != "" else np.nan for v in icc_raw])
        ci_lo, ci_hi = parse_ci(df, col_prefix + "_CI95")

        mask = ~np.isnan(icc_vals)
        if mask.sum() == 0:
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        icc_vals = icc_vals[mask]
        ci_lo = ci_lo[mask]
        ci_hi = ci_hi[mask]
        metric_names = df["Metric"].values[mask]

        err_lo = icc_vals - ci_lo
        err_hi = ci_hi - icc_vals

        order = np.argsort(icc_vals)[::-1][::-1]  # ascending for bottom-to-top
        order = np.argsort(icc_vals)
        metrics = metric_names[order]
        icc_sorted = icc_vals[order]
        err_lo_sorted = err_lo[order]
        err_hi_sorted = err_hi[order]
        colors = [band_color(v) for v in icc_sorted]

        y_pos = np.arange(len(metrics))
        ax.barh(y_pos, icc_sorted, height=0.6, color=colors, edgecolor="white",
                xerr=[err_lo_sorted, err_hi_sorted], capsize=2,
                error_kw=dict(ecolor="gray", lw=0.8), zorder=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics, fontsize=8)
        ax.set_xlabel("ICC(1,1)", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(-0.6, 1.05)
        for thresh in (0.5, 0.75, 0.9):
            ax.axvline(thresh, color="gray", lw=0.5, ls=":", zorder=1)
        ax.axvline(0, color="black", lw=0.5, ls="--", zorder=1)

    handles = [mpatches.Patch(facecolor=c, label=l) for _, _, _, c, l in BANDS]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=True, title="Reliability", title_fontsize=10)
    fig.suptitle(f"ICC(1,1) — {system_name} (All Subjects)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fname = OUT / f"icc_combined_{system_tag}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


# ── JT ───────────────────────────────────────────────────────────────
print("=== JT plots ===")
df_jt = pd.read_csv(OUT / "icc11_jt_all_subjects.csv")
plot_individual_bars(df_jt, "JT", "jt")
plot_combined_2x2(df_jt, "JT", "jt")

# ── Myo ──────────────────────────────────────────────────────────────
print("\n=== Myo plots ===")
df_myo = pd.read_csv(OUT / "icc11_myo_all_subjects.csv")
plot_individual_bars(df_myo, "Myo", "myo")
plot_combined_2x2(df_myo, "Myo", "myo")

print("\nDone.")
