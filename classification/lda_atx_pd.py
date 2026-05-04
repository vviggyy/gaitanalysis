"""
LDA: Ataxia vs PD using top-10 ICC features per group.
Produces a 2x2 figure: NG T1 | NG T2 | RG T1 | RG T2.

ICC source : claude/out/icc11_patients_only.csv
Data source: All_Results_T0T1_Gait_VV_CR_20260303.csv
Subjects   : Control == 0 (patients only), labelled by 1ATX2PD (1=Ataxia, 2=PD)
"""

import pathlib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE     = pathlib.Path(__file__).resolve().parent
ROOT     = HERE.parent
ICC_CSV  = HERE.parent / "claude" / "out" / "icc11_patients_only.csv"
DATA_CSV = ROOT / "All_Results_T0T1_Gait_VV_CR_20260303.csv"
OUT_PNG  = HERE / "lda_atx_pd.png"

# ── 1. Load data ─────────────────────────────────────────────────────
icc = pd.read_csv(ICC_CSV)
df  = pd.read_csv(DATA_CSV)

# Patients only
df = df[df["Control"] == 0].copy()

# ── 2. Column mapping (same pattern as calc_icc_1_1.py) ─────────────
# Canonical metric names come from T1_NG columns (no corruption there)
metrics = [c.split(".", 1)[1] for c in df.columns if c.startswith("Xsens_T1_NG.")]

col_map = {}
for trial in ("T1", "T2"):
    for gait in ("NG", "RG"):
        prefix = f"Xsens_{trial}_{gait}."
        for col in [c for c in df.columns if c.startswith(prefix)]:
            suffix    = col.split(".", 1)[1]
            canonical = suffix if gait == "NG" else suffix.replace("RG", "ng")
            col_map[(trial, gait, canonical)] = col

# ── 3. Feature selection helpers ─────────────────────────────────────
def top_features(icc_df, col, n=10):
    """Top-n metrics by ICC (positive values only)."""
    sub = icc_df[["Metric", col]].copy()
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna().query(f"`{col}` > 0")
    return sub.nlargest(n, col)["Metric"].tolist()

# ── 4. LDA + plot helpers ─────────────────────────────────────────────
COLORS = {1: "#2196F3", 2: "#FF5722"}   # Ataxia=blue, PD=orange
LABELS = {1: "Ataxia", 2: "PD"}

def plot_lda_panel(ax, trial, gait, icc_atx_col, icc_pd_col):
    # Select features
    top_atx = top_features(icc, icc_atx_col)
    top_pd  = top_features(icc, icc_pd_col)
    # Union preserving order (Ataxia first, then any PD-only additions)
    seen = {}
    for f in top_atx + top_pd:
        seen[f] = True
    feature_names = list(seen.keys())

    # Resolve to actual column names, drop any missing
    cols = []
    kept = []
    for f in feature_names:
        key = (trial, gait, f)
        if key in col_map and col_map[key] in df.columns:
            cols.append(col_map[key])
            kept.append(f)

    if not cols:
        ax.text(0.5, 0.5, "No features available", ha="center", va="center",
                transform=ax.transAxes)
        return

    # Build subject-level matrix
    sub = df[["1ATX2PD"] + cols].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()

    X = sub[cols].values
    y = sub["1ATX2PD"].values          # 1=Ataxia, 2=PD

    n_atx = (y == 1).sum()
    n_pd  = (y == 2).sum()

    if n_atx < 3 or n_pd < 3:
        ax.text(0.5, 0.5, f"Too few subjects\n(Ataxia n={n_atx}, PD n={n_pd})",
                ha="center", va="center", transform=ax.transAxes)
        return

    # Scale → LDA
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lda      = LinearDiscriminantAnalysis()
    lda.fit(X_scaled, y)
    ld1      = lda.transform(X_scaled)[:, 0]

    # LOO cross-validated accuracy
    cv_acc = cross_val_score(lda, X_scaled, y, cv=LeaveOneOut(), scoring="accuracy").mean()

    # ── Plot KDE density per group ───────────────────────────────────
    x_min, x_max = ld1.min() - 0.5, ld1.max() + 0.5
    x_grid = np.linspace(x_min, x_max, 300)

    for grp in [1, 2]:
        vals = ld1[y == grp]
        if len(vals) < 2:
            continue
        kde  = gaussian_kde(vals, bw_method="scott")
        dens = kde(x_grid)
        ax.fill_between(x_grid, dens, alpha=0.25, color=COLORS[grp])
        ax.plot(x_grid, dens, color=COLORS[grp], lw=2, label=f"{LABELS[grp]} (n={len(vals)})")
        # Rug plot
        ax.plot(vals, np.full_like(vals, -0.02 * dens.max()),
                "|", color=COLORS[grp], markersize=8, alpha=0.7)

    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("LD1", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle=":")

    # Title: condition + accuracy + n features used
    ax.set_title(
        f"{gait} {trial}   |   LOO acc = {cv_acc:.1%}   |   {len(kept)} features",
        fontsize=11, fontweight="bold"
    )

    # Annotate top features used (small text)
    feat_str = "Features: " + ", ".join(kept)
    ax.annotate(feat_str, xy=(0.01, 0.01), xycoords="axes fraction",
                fontsize=6, color="gray", va="bottom",
                wrap=True)

# ── 5. Build 2×2 figure ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

conditions = [
    ("T1", "NG", "Ataxia_NG", "PD_NG"),
    ("T2", "NG", "Ataxia_NG", "PD_NG"),
    ("T1", "RG", "Ataxia_RG", "PD_RG"),
    ("T2", "RG", "Ataxia_RG", "PD_RG"),
]
panel_labels = ["a)", "b)", "c)", "d)"]

for ax, (trial, gait, atx_col, pd_col), lbl in zip(axes, conditions, panel_labels):
    plot_lda_panel(ax, trial, gait, atx_col, pd_col)
    # Add panel letter
    ax.text(-0.08, 1.02, lbl, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")

fig.suptitle("LDA: Ataxia vs PD\n(Top 10 ICC features per group, patients only)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print(f"Saved → {OUT_PNG}")
plt.show()
