"""
Combined 2x2 subplot of top-10 ICC(1,1) metrics per category,
with bars color-coded by reliability band and 95% CIs.
"""

import pathlib
import ast
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
CSV = OUT / "icc11_all_subjects.csv"

df = pd.read_csv(CSV)

categories = [
    ("Ataxia_NG", "Ataxia — Normal Gait"),
    ("Ataxia_RG", "Ataxia — Backward Gait"),
    ("PD_NG", "PD — Normal Gait"),
    ("PD_RG", "PD — Backward Gait"),
]

# ICC quality bands: (lo, hi, color, label)
BANDS = [
    (-1.0, 0.5,  "#d9534f", "Poor (<0.5)"),
    (0.5,  0.75, "#f0ad4e", "Moderate (0.5–0.75)"),
    (0.75, 0.9,  "#5cb85c", "Good (0.75–0.9)"),
    (0.9,  1.01, "#337ab7", "Excellent (>0.9)"),
]

TOP_N = 10

def band_color(val):
    for lo, hi, color, _ in BANDS:
        if lo <= val < hi:
            return color
    return BANDS[-1][2]


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, (col_prefix, title) in zip(axes.flat, categories):
    icc_col = col_prefix
    ci_col = col_prefix + "_CI95"

    icc_vals = df[icc_col].values.astype(float)
    ci_lo = np.empty(len(df))
    ci_hi = np.empty(len(df))
    for i, raw in enumerate(df[ci_col]):
        lo, hi = ast.literal_eval(raw)
        ci_lo[i] = lo
        ci_hi[i] = hi

    err_lo = icc_vals - ci_lo
    err_hi = ci_hi - icc_vals

    # Top N by ICC (descending)
    top_idx = np.argsort(icc_vals)[::-1][:TOP_N][::-1]  # reverse for bottom-to-top plot
    metrics = df["Metric"].values[top_idx]
    icc_top = icc_vals[top_idx]
    err_lo_top = err_lo[top_idx]
    err_hi_top = err_hi[top_idx]
    colors = [band_color(v) for v in icc_top]

    y_pos = np.arange(TOP_N)
    ax.barh(
        y_pos, icc_top, height=0.6, color=colors, edgecolor="white",
        xerr=[err_lo_top, err_hi_top], capsize=2,
        error_kw=dict(ecolor="gray", lw=0.8),
        zorder=2,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=8)
    ax.set_xlabel("ICC(1,1)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.axvline(0.75, color="gray", lw=0.5, ls=":", zorder=1)
    ax.axvline(0.5, color="gray", lw=0.5, ls=":", zorder=1)
    ax.axvline(0.9, color="gray", lw=0.5, ls=":", zorder=1)

# Shared legend
handles = [mpatches.Patch(facecolor=c, label=l) for _, _, c, l in BANDS]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
           frameon=True, title="Reliability", title_fontsize=10)

fig.suptitle("ICC(1,1) — Top 10 Metrics by Category (All Subjects)",
             fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
fname = OUT / "icc_top10_combined.png"
fig.savefig(fname, dpi=150)
plt.close(fig)
print(f"Saved {fname}")
