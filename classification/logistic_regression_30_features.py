"""
L2 Logistic Regression: Ataxia vs PD using top-5 ICC(3,1) features per
(system, task) cell.

Cells: 3 systems (Xsens [LR-averaged], JT, Myo) × 2 tasks (NG, RG) = 6.
Per cell, top features ranked by mean(Ataxia_ICC, PD_ICC) from
patients-only ICC tables. JT only has 4 metrics → 4 features in those
cells, so total = 5+4+5+5+4+5 = 28 features.

Features: T1 and T2 averaged per subject. Patients only (Control==0).
C selected via inner LOO-CV; outer LOO-CV reports accuracy.

Inputs (all in icc_py/claude/out/042926/):
  - All_Results_LR_averaged.csv
  - icc31_lr_averaged_patients_only.csv
  - icc31_jt_patients_only.csv
  - icc31_myo_patients_only.csv

Outputs (icc_py/claude/out/042926/):
  - logistic_30feat_scatter.png
  - logistic_30feat_coeff.png
  - logistic_30feat_selected.csv
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

HERE = pathlib.Path(__file__).resolve().parent
ICC_DIR = HERE.parent / "claude" / "out" / "042926"
DATA_CSV = ICC_DIR / "All_Results_LR_averaged.csv"

ICC_PATHS = {
    "Xsens": ICC_DIR / "icc31_lr_averaged_patients_only.csv",
    "JT":    ICC_DIR / "icc31_jt_patients_only.csv",
    "Myo":   ICC_DIR / "icc31_myo_patients_only.csv",
}

TOP_N = 5
SYSTEMS = ("Xsens", "JT", "Myo")
TASKS = ("NG", "RG")


# ── 1. Load ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_CSV)
df = df[df["Control"] == 0].copy()  # patients only

icc_tables = {sys: pd.read_csv(p) for sys, p in ICC_PATHS.items()}


# ── 2. Pick top-N features per (system, task) ────────────────────────
def top_features(icc_df, task, n):
    atx = pd.to_numeric(icc_df[f"Ataxia_{task}"], errors="coerce")
    pdv = pd.to_numeric(icc_df[f"PD_{task}"], errors="coerce")
    mean_icc = (atx + pdv) / 2.0
    sub = pd.DataFrame({"Metric": icc_df["Metric"],
                        "Ataxia": atx, "PD": pdv, "mean_icc": mean_icc})
    sub = sub.dropna(subset=["mean_icc"])
    sub = sub.nlargest(min(n, len(sub)), "mean_icc")
    return sub.reset_index(drop=True)


selected_rows = []
for system in SYSTEMS:
    for task in TASKS:
        top = top_features(icc_tables[system], task, TOP_N)
        for _, r in top.iterrows():
            selected_rows.append({
                "system": system, "task": task, "metric": r["Metric"],
                "icc_ataxia": r["Ataxia"], "icc_pd": r["PD"],
                "icc_mean": r["mean_icc"],
            })

selected = pd.DataFrame(selected_rows)
print(f"Selected {len(selected)} features:")
print(selected.to_string(index=False))


# ── 3. Build feature matrix (mean of T1, T2) ─────────────────────────
def col_name(system, trial, task, metric):
    """Account for the 'ng'→'RG' corruption in Xsens RG suffixes."""
    if system == "Xsens" and task == "RG":
        metric = metric.replace("ng", "RG")
    return f"{system}_{trial}_{task}.{metric}"


feat_labels, feat_cols = [], []
for _, r in selected.iterrows():
    sys, task, metric = r["system"], r["task"], r["metric"]
    c1 = col_name(sys, "T1", task, metric)
    c2 = col_name(sys, "T2", task, metric)
    if c1 not in df.columns or c2 not in df.columns:
        print(f"WARN: missing column for {sys} {task} {metric}")
        continue
    v1 = pd.to_numeric(df[c1], errors="coerce")
    v2 = pd.to_numeric(df[c2], errors="coerce")
    feat_cols.append(((v1 + v2) / 2.0).values)
    feat_labels.append(f"{sys[:1]}|{task}|{metric}")  # short label

X = np.column_stack(feat_cols)
y = df["1ATX2PD"].values
ids = df["ID"].values

mask = ~np.isnan(X).any(axis=1)
X_full, y_full, ids_full = X[mask], y[mask], ids[mask]
print(f"\nN subjects (complete cases): {len(y_full)}  "
      f"(Ataxia={int((y_full == 1).sum())}, PD={int((y_full == 2).sum())})")
print(f"N features: {X_full.shape[1]}")


# ── 4. Fit ───────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

model = LogisticRegressionCV(
    Cs=np.logspace(-3, 2, 20),
    cv=LeaveOneOut(),
    penalty="l2",
    solver="lbfgs",
    max_iter=4000,
    random_state=42,
)
model.fit(X_scaled, y_full)
best_C = model.C_[0]

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegressionCV(
        Cs=[best_C], cv=5, penalty="l2",
        solver="lbfgs", max_iter=4000, random_state=42,
    )),
])
loo_acc = cross_val_score(pipe, X_full, y_full, cv=LeaveOneOut(),
                          scoring="accuracy").mean()
print(f"\nBest C: {best_C:.4f}")
print(f"LOO-CV accuracy: {loo_acc:.1%}")


# ── 5. PCA scatter + decision boundary ───────────────────────────────
POINT_COLORS = {1: "#1565C0", 2: "#BF360C"}
BG_COLORS = ["#BBDEFB", "#FFCCBC"]
LABELS = {1: "Ataxia", 2: "PD"}
MARKERS = {1: "o", 2: "s"}

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_ex = pca.explained_variance_ratio_ * 100

pad = 0.5
xmn, xmx = X_pca[:, 0].min() - pad, X_pca[:, 0].max() + pad
ymn, ymx = X_pca[:, 1].min() - pad, X_pca[:, 1].max() + pad
xx, yy = np.meshgrid(np.linspace(xmn, xmx, 300),
                     np.linspace(ymn, ymx, 300))
grid_feat = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
Z = model.predict(grid_feat).reshape(xx.shape)

fig_s, ax_s = plt.subplots(figsize=(9, 7))
ax_s.contourf(xx, yy, Z, levels=1, cmap=ListedColormap(BG_COLORS),
              alpha=0.35, zorder=0)
ax_s.contour(xx, yy, Z, levels=1, colors="black",
             linewidths=1.5, linestyles="--", zorder=1)
for grp in (1, 2):
    m = y_full == grp
    ax_s.scatter(X_pca[m, 0], X_pca[m, 1],
                 c=POINT_COLORS[grp], marker=MARKERS[grp],
                 s=70, edgecolors="white", linewidths=0.5,
                 label=f"{LABELS[grp]} (n={int(m.sum())})",
                 zorder=2, alpha=0.9)
ax_s.set_xlabel(f"PC1 ({var_ex[0]:.1f}% var)", fontsize=11)
ax_s.set_ylabel(f"PC2 ({var_ex[1]:.1f}% var)", fontsize=11)
ax_s.set_title(
    f"Ataxia vs PD — {X_full.shape[1]} features (top-{TOP_N} ICC per system×task)\n"
    f"LOO acc = {loo_acc:.1%}   |   C = {best_C:.3f}",
    fontsize=12, fontweight="bold",
)
ax_s.legend(fontsize=10, framealpha=0.85)
ax_s.grid(True, alpha=0.2, linestyle=":")
fig_s.tight_layout()


# ── 6. Coefficient bar chart ─────────────────────────────────────────
coefs = model.coef_[0]
order = np.argsort(np.abs(coefs))
sorted_names = [feat_labels[i] for i in order]
sorted_coefs = coefs[order]
bar_colors = [POINT_COLORS[2] if c > 0 else POINT_COLORS[1]
              for c in sorted_coefs]

fig_c, ax_c = plt.subplots(figsize=(10, max(6, len(feat_labels) * 0.28)))
ax_c.barh(sorted_names, sorted_coefs, color=bar_colors,
          edgecolor="white", height=0.75)
ax_c.axvline(0, color="black", lw=1)
ax_c.set_xlabel("Standardised coefficient   (+ve → PD,  −ve → Ataxia)",
                fontsize=10)
ax_c.set_title(
    f"Logistic regression coefficients — {X_full.shape[1]} features\n"
    f"LOO acc = {loo_acc:.1%}   |   C = {best_C:.3f}",
    fontsize=12, fontweight="bold",
)
ax_c.grid(True, axis="x", alpha=0.3, linestyle=":")
fig_c.tight_layout()


# ── 7. Save ──────────────────────────────────────────────────────────
selected.to_csv(ICC_DIR / "logistic_30feat_selected.csv", index=False)
print(f"\nSaved → {ICC_DIR / 'logistic_30feat_selected.csv'}")

fig_s.savefig(ICC_DIR / "logistic_30feat_scatter.png", dpi=200,
              bbox_inches="tight")
print(f"Saved → {ICC_DIR / 'logistic_30feat_scatter.png'}")

fig_c.savefig(ICC_DIR / "logistic_30feat_coeff.png", dpi=200,
              bbox_inches="tight")
print(f"Saved → {ICC_DIR / 'logistic_30feat_coeff.png'}")

plt.close("all")
