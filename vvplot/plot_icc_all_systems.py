"""
Combined 2x2 plot with all 3 systems (Xsens LR-avg, JT, Myo) side by side,
grouped by system within each category subplot.

Usage:
  python plot_icc_all_systems.py              # all features, colored by system
  python plot_icc_all_systems.py --top 5      # top-5 per system, no color
  python plot_icc_all_systems.py --top 10     # top-10 per system, no color
"""

import argparse
import pathlib
import ast
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"

parser = argparse.ArgumentParser()
parser.add_argument("--top", type=int, default=None,
                    help="Show only the top N features per system (omit for all)")
args = parser.parse_args()
TOP_N = args.top

# Load all 3 ICC CSVs
df_xsens = pd.read_csv(OUT / "icc11_lr_averaged_all_subjects.csv")
df_jt = pd.read_csv(OUT / "icc11_jt_all_subjects.csv")
df_myo = pd.read_csv(OUT / "icc11_myo_all_subjects.csv")

# Tag each with system name
df_xsens["System"] = "Xsens"
df_jt["System"] = "JT"
df_myo["System"] = "Myo"

# Order: JT (smallest) on top, then Myo, then Xsens at bottom
# Within each system, sort by ICC descending (highest at top)
systems_order = [("Xsens", df_xsens), ("Myo", df_myo), ("JT", df_jt)]

SYSTEM_COLORS = {"JT": "#1b9e77", "Myo": "#d95f02", "Xsens": "#7570b3"}
BAR_COLOR_PLAIN = "#4d4d4d"  # used when --top is set (no system color)

categories = [
    ("Ataxia_NG", "Ataxia — Normal Gait"),
    ("Ataxia_RG", "Ataxia — Backward Gait"),
    ("PD_NG", "PD — Normal Gait"),
    ("PD_RG", "PD — Backward Gait"),
]

BANDS = [
    (-1.0, 0.5,  "#fee0d2", "Poor (<0.5)"),
    (0.5,  0.75, "#ffffcc", "Moderate"),
    (0.75, 0.9,  "#c7e9c0", "Good"),
    (0.9,  1.01, "#9ecae1", "Excellent"),
]


def parse_ci_row(raw):
    if pd.isna(raw) or raw == "":
        return np.nan, np.nan
    lo, hi = ast.literal_eval(raw)
    return lo, hi


figsize = (14, 14) if TOP_N is not None else (16, 22)
fig, axes = plt.subplots(2, 2, figsize=figsize)

for ax, (col_prefix, title) in zip(axes.flat, categories):
    icc_col = col_prefix
    ci_col = col_prefix + "_CI95"

    # Build combined data: list of (label, icc, ci_lo, ci_hi, system)
    all_rows = []
    for sys_name, sys_df in systems_order:
        entries = []
        for _, r in sys_df.iterrows():
            val = r[icc_col]
            icc = float(val) if val != "" and not pd.isna(val) else np.nan
            if np.isnan(icc):
                continue
            lo, hi = parse_ci_row(r[ci_col])
            entries.append((r["Metric"], icc, lo, hi, sys_name))
        # Optionally keep only top N within this system
        if TOP_N is not None:
            entries.sort(key=lambda x: x[1], reverse=True)
            entries = entries[:TOP_N]
        # Sort ascending so highest ends up at top of section when plotted
        entries.sort(key=lambda x: x[1])
        all_rows.extend(entries)

    if not all_rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        continue

    labels = [f"{r[0]}" for r in all_rows]
    iccs = np.array([r[1] for r in all_rows])
    ci_los = np.array([r[2] for r in all_rows])
    ci_his = np.array([r[3] for r in all_rows])
    sys_names = [r[4] for r in all_rows]

    err_lo = iccs - ci_los
    err_hi = ci_his - iccs
    colors = (BAR_COLOR_PLAIN if TOP_N is not None
              else [SYSTEM_COLORS[s] for s in sys_names])

    # Background reliability bands
    for x0, x1, bg_c, _ in BANDS:
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

    # Draw horizontal separators between systems
    prev_sys = sys_names[0]
    for i, s in enumerate(sys_names):
        if s != prev_sys:
            ax.axhline(i - 0.5, color="black", lw=0.8, ls="-", zorder=3)
            prev_sys = s

# Shared legend
band_handles = [mpatches.Patch(facecolor=c, alpha=0.35, label=l)
                for _, _, c, l in BANDS]

if TOP_N is None:
    # Full plot: show system colors + reliability bands
    sys_handles = [mpatches.Patch(facecolor=SYSTEM_COLORS[s], label=s)
                   for s in ["Xsens", "Myo", "JT"]]
    sep = mlines.Line2D([], [], color="none", label="")
    fig.legend(handles=sys_handles + [sep] + band_handles,
               loc="lower center", ncol=4, fontsize=9, frameon=True,
               title="System                                              Reliability",
               title_fontsize=10, columnspacing=1.5)
    suptitle = "ICC(1,1) — All Systems (Xsens LR-Averaged, JT, Myo)"
    fname = OUT / "icc_all_systems_combined.png"
    rect = [0, 0.035, 1, 0.99]
else:
    # Top-N plot: reliability bands only (no system colors)
    fig.legend(handles=band_handles, loc="lower center", ncol=4, fontsize=9,
               frameon=True, title="Reliability", title_fontsize=10,
               columnspacing=1.5)
    suptitle = f"ICC(1,1) — Top {TOP_N} Features per System (Xsens LR-Avg, JT, Myo)"
    fname = OUT / f"icc_top{TOP_N}_all_systems.png"
    rect = [0, 0.04, 1, 0.99]

fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=0.995)
plt.tight_layout(rect=rect)
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {fname}")
