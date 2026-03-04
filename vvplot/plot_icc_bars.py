"""
Horizontal bar charts of ICC(1,1) with 95% CIs for each of the 4 categories:
Ataxia NG, Ataxia RG, PD NG, PD RG.

Reads icc11_all_subjects.csv and outputs 4 PNGs into out/.
"""

import pathlib
import ast
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

# ICC quality bands
BANDS = [
    (-1.0, 0.5,  "#fee0d2", "Poor"),
    (0.5,  0.75, "#ffffcc", "Moderate"),
    (0.75, 0.9,  "#c7e9c0", "Good"),
    (0.9,  1.0,  "#9ecae1", "Excellent"),
]

for col_prefix, title in categories:
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

    # Sort by ICC descending (top = highest)
    order = np.argsort(icc_vals)  # ascending; plotted bottom-to-top
    metrics = df["Metric"].values[order]
    icc_sorted = icc_vals[order]
    err_lo_sorted = err_lo[order]
    err_hi_sorted = err_hi[order]

    fig, ax = plt.subplots(figsize=(8, 10))

    # Background bands
    for x0, x1, color, label in BANDS:
        ax.axvspan(x0, x1, color=color, alpha=0.4, zorder=0)

    y_pos = np.arange(len(metrics))
    ax.barh(
        y_pos, icc_sorted, height=0.6, color="#4878a8", edgecolor="white",
        xerr=[err_lo_sorted, err_hi_sorted], capsize=2,
        error_kw=dict(ecolor="gray", lw=0.8),
        zorder=2,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=7)
    ax.set_xlabel("ICC(1,1)")
    ax.set_title(f"ICC(1,1) — {title}", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.6, 1.05)
    ax.axvline(0, color="black", lw=0.5, ls="--", zorder=1)

    # Legend for bands
    for x0, x1, color, label in BANDS:
        ax.barh([], [], color=color, alpha=0.4, label=label)
    ax.legend(loc="lower right", fontsize=7, title="Reliability", title_fontsize=8)

    plt.tight_layout()
    fname = OUT / f"icc_bar_{col_prefix.lower()}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")

print("Done.")
