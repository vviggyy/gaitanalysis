"""
Plots for LR-averaged ICC(1,1) results:
  1. Four individual horizontal bar charts (all metrics)
  2. One combined 2x2 top-10 chart
"""

import pathlib
import ast
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
CSV = OUT / "icc11_lr_averaged_all_subjects.csv"

df = pd.read_csv(CSV)

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
        lo, hi = ast.literal_eval(raw)
        ci_lo[i] = lo
        ci_hi[i] = hi
    return ci_lo, ci_hi

# ── 1. Individual bar charts (all metrics) ───────────────────────────
for col_prefix, title in categories:
    icc_vals = df[col_prefix].values.astype(float)
    ci_lo, ci_hi = parse_ci(df, col_prefix + "_CI95")
    err_lo = icc_vals - ci_lo
    err_hi = ci_hi - icc_vals

    order = np.argsort(icc_vals)
    metrics = df["Metric"].values[order]
    icc_sorted = icc_vals[order]
    err_lo_sorted = err_lo[order]
    err_hi_sorted = err_hi[order]
    colors = [band_color(v) for v in icc_sorted]

    fig, ax = plt.subplots(figsize=(8, 9))
    for x0, x1, bg_c, _, label in BANDS:
        ax.axvspan(x0, x1, color=bg_c, alpha=0.4, zorder=0)

    y_pos = np.arange(len(metrics))
    ax.barh(y_pos, icc_sorted, height=0.6, color=colors, edgecolor="white",
            xerr=[err_lo_sorted, err_hi_sorted], capsize=2,
            error_kw=dict(ecolor="gray", lw=0.8), zorder=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=8)
    ax.set_xlabel("ICC(1,1)")
    ax.set_title(f"ICC(1,1) — {title} (LR Averaged)", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.6, 1.05)
    ax.axvline(0, color="black", lw=0.5, ls="--", zorder=1)

    handles = [mpatches.Patch(facecolor=bc, label=l) for _, _, _, bc, l in BANDS]
    ax.legend(handles=handles, loc="lower right", fontsize=7,
              title="Reliability", title_fontsize=8)

    plt.tight_layout()
    fname = OUT / f"icc_bar_lr_avg_{col_prefix.lower()}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")

# ── 2. Combined 2x2 top-10 chart ────────────────────────────────────
TOP_N = 10
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, (col_prefix, title) in zip(axes.flat, categories):
    icc_vals = df[col_prefix].values.astype(float)
    ci_lo, ci_hi = parse_ci(df, col_prefix + "_CI95")
    err_lo = icc_vals - ci_lo
    err_hi = ci_hi - icc_vals

    top_idx = np.argsort(icc_vals)[::-1][:TOP_N][::-1]
    metrics = df["Metric"].values[top_idx]
    icc_top = icc_vals[top_idx]
    err_lo_top = err_lo[top_idx]
    err_hi_top = err_hi[top_idx]
    colors = [band_color(v) for v in icc_top]

    y_pos = np.arange(TOP_N)
    ax.barh(y_pos, icc_top, height=0.6, color=colors, edgecolor="white",
            xerr=[err_lo_top, err_hi_top], capsize=2,
            error_kw=dict(ecolor="gray", lw=0.8), zorder=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=8)
    ax.set_xlabel("ICC(1,1)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.05)
    for thresh in (0.5, 0.75, 0.9):
        ax.axvline(thresh, color="gray", lw=0.5, ls=":", zorder=1)

handles = [mpatches.Patch(facecolor=c, label=l) for _, _, _, c, l in BANDS]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
           frameon=True, title="Reliability", title_fontsize=10)
fig.suptitle("ICC(1,1) — Top 10 Metrics (LR Averaged, All Subjects)",
             fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
fname = OUT / "icc_top10_lr_averaged.png"
fig.savefig(fname, dpi=150)
plt.close(fig)
print(f"Saved {fname}")

print("Done.")
