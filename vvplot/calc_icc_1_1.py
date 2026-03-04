"""
Compute ICC(1,1) + 95% CI for each Xsens gait metric, W1 vs W2,
split by disease group (Ataxia / PD), with and without controls.

Outputs two CSVs:
  - icc11_all_subjects.csv       (all subjects)
  - icc11_patients_only.csv      (controls excluded, Control==0)
"""

import pathlib
import pandas as pd
import pingouin as pg
import warnings

warnings.filterwarnings("ignore")

HERE = pathlib.Path(__file__).resolve().parent
INPUT_CSV = HERE.parent / "All_Results_T0T1_Gait_VV_012526.csv"

# ── 1. Load data ────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)

# ── 2. Identify Xsens columns and canonical metric names ───────────
t1_ng_cols = [c for c in df.columns if c.startswith("Xsens_T1_NG.")]
# canonical metric names from NG columns (no corruption there)
metrics = [c.split(".", 1)[1] for c in t1_ng_cols]  # 33 metrics

# Build column map: for each (trial, test_type) → metric → column name
# trial: T1, T2   test_type: NG, RG
def _rg_to_canonical(rg_suffix: str) -> str:
    """Reverse the RG corruption: replace 'RG' with 'ng' in metric suffix."""
    return rg_suffix.replace("RG", "ng")

col_map = {}  # (trial, test_type, metric) → column name
for trial in ("T1", "T2"):
    for tt in ("NG", "RG"):
        prefix = f"Xsens_{trial}_{tt}."
        cols = [c for c in df.columns if c.startswith(prefix)]
        for col in cols:
            suffix = col.split(".", 1)[1]
            canonical = suffix if tt == "NG" else _rg_to_canonical(suffix)
            col_map[(trial, tt, canonical)] = col

# Verify all 33 metrics map for every (trial, test_type)
for trial in ("T1", "T2"):
    for tt in ("NG", "RG"):
        for m in metrics:
            assert (trial, tt, m) in col_map, f"Missing column for {trial}_{tt}.{m}"


# ── 3. Reshape to long format and compute ICC(1,1) ─────────────────
def compute_icc_table(data: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with one row per metric, ICC columns per group×test."""
    group_map = {1: "Ataxia", 2: "PD"}
    results = []

    for m in metrics:
        row = {"Metric": m}
        for grp_code, grp_name in group_map.items():
            sub = data[data["1ATX2PD"] == grp_code].copy()
            for tt in ("NG", "RG"):
                col_t1 = col_map[("T1", tt, m)]
                col_t2 = col_map[("T2", tt, m)]
                # Build wide frame per subject, keep only complete pairs
                pair = sub[["ID"]].copy()
                pair["T1"] = pd.to_numeric(sub[col_t1], errors="coerce").values
                pair["T2"] = pd.to_numeric(sub[col_t2], errors="coerce").values
                pair = pair.dropna(subset=["T1", "T2"])

                icc_val = ""
                ci_str = ""
                if len(pair) >= 2:
                    long = pair.melt(id_vars="ID", var_name="raters",
                                     value_name="ratings")
                    try:
                        icc_df = pg.intraclass_corr(
                            data=long, targets="ID",
                            raters="raters", ratings="ratings",
                        )
                        icc1 = icc_df[icc_df["Type"] == "ICC1"].iloc[0]
                        icc_val = round(icc1["ICC"], 3)
                        ci_str = f"[{icc1['CI95%'][0]:.3f}, {icc1['CI95%'][1]:.3f}]"
                    except Exception:
                        pass

                col_icc = f"{grp_name}_{tt}"
                col_ci = f"{grp_name}_{tt}_CI95"
                row[col_icc] = icc_val
                row[col_ci] = ci_str
        results.append(row)

    out_cols = ["Metric"]
    for g in ("Ataxia", "PD"):
        for tt in ("NG", "RG"):
            out_cols += [f"{g}_{tt}", f"{g}_{tt}_CI95"]
    return pd.DataFrame(results)[out_cols]


# ── 4. Run for all subjects and patients-only ───────────────────────
print("Computing ICC(1,1) for all subjects...")
icc_all = compute_icc_table(df)
out_all = HERE / "out" / "icc11_all_subjects.csv"
icc_all.to_csv(out_all, index=False)
print(f"  → {out_all}  ({len(icc_all)} rows)")

print("Computing ICC(1,1) for patients only (Control==0)...")
icc_pat = compute_icc_table(df[df["Control"] == 0])
out_pat = HERE / "out" / "icc11_patients_only.csv"
icc_pat.to_csv(out_pat, index=False)
print(f"  → {out_pat}  ({len(icc_pat)} rows)")

print("Done.")
