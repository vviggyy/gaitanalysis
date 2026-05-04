"""
L2-regularised Logistic Regression: Ataxia vs PD.
C is chosen by inner LOO-CV (LogisticRegressionCV).
Outer LOO-CV gives the reported accuracy.

Produces two figures:
  logistic_scatter_atx_pd.png  — 2×2 PCA scatter + decision boundary
  logistic_coeff_atx_pd.png    — 2×2 feature coefficient bar charts

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
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE     = pathlib.Path(__file__).resolve().parent
ROOT     = HERE.parent
ICC_CSV  = HERE.parent / "claude" / "out" / "icc11_patients_only.csv"
DATA_CSV = ROOT / "All_Results_T0T1_Gait_VV_CR_20260303.csv"

# ── 1. Load ───────────────────────────────────────────────────────────
icc = pd.read_csv(ICC_CSV)
df  = pd.read_csv(DATA_CSV)
df  = df[df["Control"] == 0].copy()

# ── 2. Column map ─────────────────────────────────────────────────────
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

# ── 4. Style ──────────────────────────────────────────────────────────
POINT_COLORS = {1: "#1565C0", 2: "#BF360C"}
BG_COLORS    = ["#BBDEFB", "#FFCCBC"]
LABELS       = {1: "Ataxia", 2: "PD"}
MARKERS      = {1: "o", 2: "s"}
CONDITIONS   = [
    ("T1", "NG", "Ataxia_NG", "PD_NG"),
    ("T2", "NG", "Ataxia_NG", "PD_NG"),
    ("T1", "RG", "Ataxia_RG", "PD_RG"),
    ("T2", "RG", "Ataxia_RG", "PD_RG"),
]
PANEL_LABELS = ["a)", "b)", "c)", "d)"]

# ── 5. Build X, y, feature names per condition ────────────────────────
def build_dataset(trial, gait, icc_atx_col, icc_pd_col):
    top_atx = top_features(icc, icc_atx_col)
    top_pd  = top_features(icc, icc_pd_col)
    seen = {}
    for f in top_atx + top_pd:
        seen[f] = True

    cols, feat_names = [], []
    for f in seen:
        key = (trial, gait, f)
        if key in col_map and col_map[key] in df.columns:
            cols.append(col_map[key])
            feat_names.append(f)

    sub = df[["1ATX2PD"] + cols].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()

    X = sub[cols].values
    y = sub["1ATX2PD"].values
    return X, y, feat_names

# ── 6. Scatter figure ─────────────────────────────────────────────────
fig_scatter, axes_s = plt.subplots(2, 2, figsize=(14, 11))
axes_s = axes_s.flatten()

fig_coeff, axes_c = plt.subplots(2, 2, figsize=(14, 11))
axes_c = axes_c.flatten()

for idx, ((trial, gait, atx_col, pd_col), lbl) in enumerate(
        zip(CONDITIONS, PANEL_LABELS)):

    ax_s = axes_s[idx]
    ax_c = axes_c[idx]

    X, y, feat_names = build_dataset(trial, gait, atx_col, pd_col)
    n_atx, n_pd = (y == 1).sum(), (y == 2).sum()

    if n_atx < 3 or n_pd < 3:
        for ax in (ax_s, ax_c):
            ax.text(0.5, 0.5, f"Too few subjects\n(Ataxia n={n_atx}, PD n={n_pd})",
                    ha="center", va="center", transform=ax.transAxes)
        continue

    # ── Fit: StandardScaler + LogisticRegressionCV (inner LOO selects C) ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 20),
        cv=LeaveOneOut(),
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_scaled, y)
    best_C = model.C_[0]

    # ── Outer LOO accuracy (avoids optimistic bias from inner C selection) ──
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegressionCV(
            Cs=[best_C], cv=5, penalty="l2",
            solver="lbfgs", max_iter=2000, random_state=42,
        )),
    ])
    loo_acc = cross_val_score(pipe, X, y, cv=LeaveOneOut(),
                              scoring="accuracy").mean()

    # ── PCA 2D for scatter ────────────────────────────────────────────
    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_ex = pca.explained_variance_ratio_ * 100

    pad = 0.5
    x_min, x_max = X_pca[:, 0].min() - pad, X_pca[:, 0].max() + pad
    y_min, y_max = X_pca[:, 1].min() - pad, X_pca[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid_feat = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
    Z = model.predict(grid_feat).reshape(xx.shape)

    bg_cmap = ListedColormap(BG_COLORS)
    ax_s.contourf(xx, yy, Z, levels=1, cmap=bg_cmap, alpha=0.35, zorder=0)
    ax_s.contour(xx, yy, Z, levels=1, colors="black",
                 linewidths=1.5, linestyles="--", zorder=1)

    for grp in [1, 2]:
        mask = y == grp
        ax_s.scatter(X_pca[mask, 0], X_pca[mask, 1],
                     c=POINT_COLORS[grp], marker=MARKERS[grp],
                     s=65, edgecolors="white", linewidths=0.5,
                     label=f"{LABELS[grp]} (n={mask.sum()})",
                     zorder=2, alpha=0.9)

    ax_s.set_xlabel(f"PC1 ({var_ex[0]:.1f}% var)", fontsize=10)
    ax_s.set_ylabel(f"PC2 ({var_ex[1]:.1f}% var)", fontsize=10)
    ax_s.legend(fontsize=9, framealpha=0.8)
    ax_s.grid(True, alpha=0.2, linestyle=":")
    ax_s.set_title(
        f"{gait} {trial}   |   LOO acc = {loo_acc:.1%}   |   C = {best_C:.3f}",
        fontsize=11, fontweight="bold"
    )
    ax_s.text(-0.08, 1.02, lbl, transform=ax_s.transAxes,
              fontsize=13, fontweight="bold", va="bottom")

    # ── Coefficient bar chart ─────────────────────────────────────────
    # Coefficients are already on standardised scale → directly comparable
    coefs = model.coef_[0]
    order = np.argsort(np.abs(coefs))          # sort by magnitude
    sorted_names = [feat_names[i] for i in order]
    sorted_coefs = coefs[order]
    bar_colors   = [POINT_COLORS[2] if c > 0 else POINT_COLORS[1]
                    for c in sorted_coefs]      # orange = PD, blue = Ataxia

    ax_c.barh(sorted_names, sorted_coefs, color=bar_colors,
              edgecolor="white", height=0.7)
    ax_c.axvline(0, color="black", lw=1)
    ax_c.set_xlabel("Standardised coefficient\n(+ve → PD, −ve → Ataxia)", fontsize=9)
    ax_c.grid(True, axis="x", alpha=0.3, linestyle=":")
    ax_c.set_title(
        f"{gait} {trial}   |   C = {best_C:.3f}   |   LOO acc = {loo_acc:.1%}",
        fontsize=11, fontweight="bold"
    )
    ax_c.text(-0.22, 1.02, lbl, transform=ax_c.transAxes,
              fontsize=13, fontweight="bold", va="bottom")

# ── 7. Save ───────────────────────────────────────────────────────────
for fig, name in [
    (fig_scatter, "logistic_scatter_atx_pd.png"),
    (fig_coeff,   "logistic_coeff_atx_pd.png"),
]:
    fig.suptitle(
        "L2 Logistic Regression: Ataxia vs PD\n"
        "(Top 10 ICC features per group, C by inner LOO-CV, patients only)",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    out = HERE / name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved → {out}")

plt.show()
