"""
LDA scatter plot: Ataxia vs PD in PCA space with decision boundary.

Strategy:
  1. Train LDA on full scaled feature space (same features as lda_atx_pd.py).
  2. PCA(2) for visualisation only.
  3. Draw decision boundary by inverse-transforming a PC-space meshgrid back
     to feature space and classifying with the original LDA — so the boundary
     is faithful to the full-dimensional LDA, just shown in 2D.

ICC source : claude/out/icc11_patients_only.csv
Data source: All_Results_T0T1_Gait_VV_CR_20260303.csv
Subjects   : Control == 0, labelled by 1ATX2PD (1=Ataxia, 2=PD)
"""

import pathlib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE     = pathlib.Path(__file__).resolve().parent
ROOT     = HERE.parent
ICC_CSV  = HERE.parent / "claude" / "out" / "icc11_patients_only.csv"
DATA_CSV = ROOT / "All_Results_T0T1_Gait_VV_CR_20260303.csv"
OUT_PNG  = HERE / "lda_scatter_atx_pd.png"

# ── 1. Load ───────────────────────────────────────────────────────────
icc = pd.read_csv(ICC_CSV)
df  = pd.read_csv(DATA_CSV)
df  = df[df["Control"] == 0].copy()

# ── 2. Column map (identical to lda_atx_pd.py) ───────────────────────
col_map = {}
for trial in ("T1", "T2"):
    for gait in ("NG", "RG"):
        prefix = f"Xsens_{trial}_{gait}."
        for col in [c for c in df.columns if c.startswith(prefix)]:
            suffix    = col.split(".", 1)[1]
            canonical = suffix if gait == "NG" else suffix.replace("RG", "ng")
            col_map[(trial, gait, canonical)] = col

# ── 3. Feature selection ──────────────────────────────────────────────
def top_features(icc_df, col, n=10):
    sub = icc_df[["Metric", col]].copy()
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna().query(f"`{col}` > 0")
    return sub.nlargest(n, col)["Metric"].tolist()

# ── 4. Colours / style ────────────────────────────────────────────────
POINT_COLORS = {1: "#1565C0", 2: "#BF360C"}   # dark blue / dark orange
BG_COLORS    = ["#BBDEFB", "#FFCCBC"]          # light blue / light orange
LABELS       = {1: "Ataxia", 2: "PD"}
MARKERS      = {1: "o", 2: "s"}

# ── 5. Panel function ─────────────────────────────────────────────────
def plot_scatter_panel(ax, trial, gait, icc_atx_col, icc_pd_col):
    # --- feature union ---
    top_atx = top_features(icc, icc_atx_col)
    top_pd  = top_features(icc, icc_pd_col)
    seen = {}
    for f in top_atx + top_pd:
        seen[f] = True
    feature_names = list(seen.keys())

    cols = []
    for f in feature_names:
        key = (trial, gait, f)
        if key in col_map and col_map[key] in df.columns:
            cols.append(col_map[key])

    if not cols:
        ax.text(0.5, 0.5, "No features available", ha="center", va="center",
                transform=ax.transAxes)
        return

    sub = df[["1ATX2PD"] + cols].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()

    X = sub[cols].values
    y = sub["1ATX2PD"].values          # 1=Ataxia, 2=PD
    n_atx, n_pd = (y == 1).sum(), (y == 2).sum()

    if n_atx < 3 or n_pd < 3:
        ax.text(0.5, 0.5, f"Too few subjects (Ataxia n={n_atx}, PD n={n_pd})",
                ha="center", va="center", transform=ax.transAxes)
        return

    # --- scale → LDA (full feature space) ---
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_scaled, y)
    cv_acc = cross_val_score(lda, X_scaled, y, cv=LeaveOneOut(),
                             scoring="accuracy").mean()

    # --- PCA to 2D for visualisation ---
    pca    = PCA(n_components=2, random_state=42)
    X_pca  = pca.fit_transform(X_scaled)
    var_ex = pca.explained_variance_ratio_ * 100

    # --- decision-boundary meshgrid in PCA space ---
    pad = 0.5
    x_min, x_max = X_pca[:, 0].min() - pad, X_pca[:, 0].max() + pad
    y_min, y_max = X_pca[:, 1].min() - pad, X_pca[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid_pca     = np.c_[xx.ravel(), yy.ravel()]
    grid_feature = pca.inverse_transform(grid_pca)   # back to feature space
    Z = lda.predict(grid_feature).reshape(xx.shape)

    # --- background shading ---
    bg_cmap = ListedColormap(BG_COLORS)
    ax.contourf(xx, yy, Z, levels=1, cmap=bg_cmap, alpha=0.35, zorder=0)

    # --- decision boundary line ---
    ax.contour(xx, yy, Z, levels=1, colors="black", linewidths=1.5,
               linestyles="--", zorder=1)

    # --- scatter points ---
    for grp in [1, 2]:
        mask = y == grp
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=POINT_COLORS[grp], marker=MARKERS[grp],
                   s=60, edgecolors="white", linewidths=0.5,
                   label=f"{LABELS[grp]} (n={mask.sum()})",
                   zorder=2, alpha=0.9)

    ax.set_xlabel(f"PC1 ({var_ex[0]:.1f}% var)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_ex[1]:.1f}% var)", fontsize=10)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
    ax.grid(True, alpha=0.2, linestyle=":")
    ax.set_title(
        f"{gait} {trial}   |   LOO acc = {cv_acc:.1%}   |   {len(cols)} features",
        fontsize=11, fontweight="bold"
    )

# ── 6. Build 2×2 figure ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

conditions = [
    ("T1", "NG", "Ataxia_NG", "PD_NG"),
    ("T2", "NG", "Ataxia_NG", "PD_NG"),
    ("T1", "RG", "Ataxia_RG", "PD_RG"),
    ("T2", "RG", "Ataxia_RG", "PD_RG"),
]
panel_labels = ["a)", "b)", "c)", "d)"]

for ax, (trial, gait, atx_col, pd_col), lbl in zip(axes, conditions, panel_labels):
    plot_scatter_panel(ax, trial, gait, atx_col, pd_col)
    ax.text(-0.08, 1.02, lbl, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")

fig.suptitle(
    "LDA Decision Boundary: Ataxia vs PD\n"
    "(PCA 2D projection — boundary from full-dimensional LDA)",
    fontsize=13, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print(f"Saved → {OUT_PNG}")
plt.show()
