"""
Compare logistic-regression variants for Ataxia vs PD using top-K ICC(3,1)
features per (system, task) cell. JT is missing for ~half the patients,
so we explore mitigations.

Variants:
  A. baseline      — all 3 systems × 2 tasks, top-5, complete cases
  B. drop_JT       — Xsens + Myo only, top-5, complete cases
  C. impute_median — all 3 systems × 2 tasks, top-5, group-wise median impute
  D. top3          — all 3 systems × 2 tasks, top-3, complete cases
  E. drop_JT_top3  — Xsens + Myo, top-3, complete cases

Outputs (in icc_py/claude/out/042926/):
  - logistic_variants_summary.csv
  - logistic_variants_<name>_scatter.png  (one per variant)
  - logistic_variants_<name>_coeff.png
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
TASKS = ("NG", "RG")


def col_name(system, trial, task, metric):
    if system == "Xsens" and task == "RG":
        metric = metric.replace("ng", "RG")
    return f"{system}_{trial}_{task}.{metric}"


def top_features(icc_df, task, n):
    atx = pd.to_numeric(icc_df[f"Ataxia_{task}"], errors="coerce")
    pdv = pd.to_numeric(icc_df[f"PD_{task}"], errors="coerce")
    mean_icc = (atx + pdv) / 2.0
    sub = pd.DataFrame({"Metric": icc_df["Metric"], "icc": mean_icc,
                        "Ataxia": atx, "PD": pdv}).dropna(subset=["icc"])
    return sub.nlargest(min(n, len(sub)), "icc").reset_index(drop=True)


def build_feature_matrix(df, systems, top_n):
    icc_tables = {s: pd.read_csv(ICC_PATHS[s]) for s in systems}
    selected = []
    for s in systems:
        for t in TASKS:
            top = top_features(icc_tables[s], t, top_n)
            for _, r in top.iterrows():
                selected.append({
                    "system": s, "task": t, "metric": r["Metric"],
                    "icc_ataxia": r["Ataxia"], "icc_pd": r["PD"],
                    "icc_mean": r["icc"],
                })
    selected = pd.DataFrame(selected)

    feat_labels, feat_cols = [], []
    for _, r in selected.iterrows():
        s, t, m = r["system"], r["task"], r["metric"]
        c1 = col_name(s, "T1", t, m)
        c2 = col_name(s, "T2", t, m)
        if c1 not in df.columns or c2 not in df.columns:
            continue
        v1 = pd.to_numeric(df[c1], errors="coerce")
        v2 = pd.to_numeric(df[c2], errors="coerce")
        feat_cols.append(((v1 + v2) / 2.0).values)
        feat_labels.append(f"{s[:1]}|{t}|{m}")
    X = np.column_stack(feat_cols)
    return X, feat_labels, selected


def fit_eval(X, y, labels, name):
    """StandardScaler + LogisticRegressionCV with inner LOO C selection;
    LOO-CV outer accuracy. Returns dict of metrics."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 20), cv=LeaveOneOut(), penalty="l2",
        solver="lbfgs", max_iter=4000, random_state=42,
    )
    model.fit(Xs, y)
    best_C = model.C_[0]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegressionCV(
            Cs=[best_C], cv=5, penalty="l2",
            solver="lbfgs", max_iter=4000, random_state=42,
        )),
    ])
    loo_acc = cross_val_score(pipe, X, y, cv=LeaveOneOut(),
                              scoring="accuracy").mean()
    return {"name": name, "n": len(y), "n_atx": int((y == 1).sum()),
            "n_pd": int((y == 2).sum()), "n_features": X.shape[1],
            "best_C": best_C, "loo_acc": loo_acc,
            "model": model, "labels": labels, "Xs": Xs, "y": y}


def plot_variant(res, out_dir):
    POINT_COLORS = {1: "#1565C0", 2: "#BF360C"}
    BG = ["#BBDEFB", "#FFCCBC"]
    LBL = {1: "Ataxia", 2: "PD"}
    MK = {1: "o", 2: "s"}

    Xs, y, model = res["Xs"], res["y"], res["model"]
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xs)
    var_ex = pca.explained_variance_ratio_ * 100

    pad = 0.5
    xmn, xmx = Xp[:, 0].min() - pad, Xp[:, 0].max() + pad
    ymn, ymx = Xp[:, 1].min() - pad, Xp[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(xmn, xmx, 250),
                         np.linspace(ymn, ymx, 250))
    grid = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
    Z = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.contourf(xx, yy, Z, levels=1, cmap=ListedColormap(BG),
                alpha=0.35, zorder=0)
    ax.contour(xx, yy, Z, levels=1, colors="black",
               linewidths=1.5, linestyles="--", zorder=1)
    for grp in (1, 2):
        m = y == grp
        ax.scatter(Xp[m, 0], Xp[m, 1], c=POINT_COLORS[grp],
                   marker=MK[grp], s=70, edgecolors="white",
                   linewidths=0.5, label=f"{LBL[grp]} (n={int(m.sum())})",
                   zorder=2, alpha=0.9)
    ax.set_xlabel(f"PC1 ({var_ex[0]:.1f}% var)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_ex[1]:.1f}% var)", fontsize=10)
    ax.set_title(
        f"{res['name']}   |   N={res['n']}   |   "
        f"{res['n_features']} feats   |   "
        f"LOO acc = {res['loo_acc']:.1%}   |   C = {res['best_C']:.3f}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.2, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_dir / f"logistic_variants_{res['name']}_scatter.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    coefs = model.coef_[0]
    order = np.argsort(np.abs(coefs))
    sorted_names = [res["labels"][i] for i in order]
    sorted_coefs = coefs[order]
    bar_colors = [POINT_COLORS[2] if c > 0 else POINT_COLORS[1]
                  for c in sorted_coefs]

    fig_c, ax_c = plt.subplots(figsize=(9, max(5, len(res["labels"]) * 0.28)))
    ax_c.barh(sorted_names, sorted_coefs, color=bar_colors,
              edgecolor="white", height=0.75)
    ax_c.axvline(0, color="black", lw=1)
    ax_c.set_xlabel("Standardised coefficient   (+ → PD,  − → Ataxia)",
                    fontsize=10)
    ax_c.set_title(
        f"{res['name']}   |   N={res['n']}, {res['n_features']} feats   |   "
        f"LOO acc = {res['loo_acc']:.1%}",
        fontsize=11, fontweight="bold",
    )
    ax_c.grid(True, axis="x", alpha=0.3, linestyle=":")
    fig_c.tight_layout()
    fig_c.savefig(out_dir / f"logistic_variants_{res['name']}_coeff.png",
                  dpi=200, bbox_inches="tight")
    plt.close(fig_c)


# ── Load ─────────────────────────────────────────────────────────────
df_full = pd.read_csv(DATA_CSV)
df_full = df_full[df_full["Control"] == 0].copy()


def prep(systems, top_n, impute=False):
    """Build X,y dropping NaN rows (or imputing within group)."""
    X, labels, selected = build_feature_matrix(df_full, systems, top_n)
    y_all = df_full["1ATX2PD"].values

    if impute:
        # group-wise median imputation
        Xi = X.copy()
        for grp in (1, 2):
            mask = y_all == grp
            med = np.nanmedian(Xi[mask], axis=0)
            for j in range(Xi.shape[1]):
                col = Xi[mask, j]
                col[np.isnan(col)] = med[j]
                Xi[mask, j] = col
        # drop any remaining NaNs (shouldn't happen if each group has data)
        keep = ~np.isnan(Xi).any(axis=1)
        return Xi[keep], y_all[keep], labels, selected

    keep = ~np.isnan(X).any(axis=1)
    return X[keep], y_all[keep], labels, selected


print("=" * 70)
results = []

variants = [
    ("baseline",      ("Xsens", "JT", "Myo"), 5, False),
    ("drop_JT",       ("Xsens", "Myo"),       5, False),
    ("impute_median", ("Xsens", "JT", "Myo"), 5, True),
    ("top3",          ("Xsens", "JT", "Myo"), 3, False),
    ("drop_JT_top3",  ("Xsens", "Myo"),       3, False),
]

for name, systems, k, impute in variants:
    X, y, labels, _ = prep(systems, k, impute=impute)
    res = fit_eval(X, y, labels, name)
    results.append(res)
    plot_variant(res, ICC_DIR)
    print(f"{name:>16}   N={res['n']:>2}   feats={res['n_features']:>2}   "
          f"acc={res['loo_acc']:.1%}   C={res['best_C']:.3f}")

# Summary CSV
summary = pd.DataFrame([{k: r[k] for k in
                         ("name", "n", "n_atx", "n_pd",
                          "n_features", "best_C", "loo_acc")}
                        for r in results])
summary.to_csv(ICC_DIR / "logistic_variants_summary.csv", index=False)
print(f"\nSaved → {ICC_DIR / 'logistic_variants_summary.csv'}")
