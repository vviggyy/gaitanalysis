"""
1. Average left-foot / right-foot Xsens features into single bilateral features.
2. Save a new CSV with averaged columns.
3. Compute ICC(1,1) + 95% CI on the averaged data (all subjects & patients-only).

Outputs (in out/):
  - All_Results_LR_averaged.csv          (input CSV with LF/RF averaged)
  - icc11_lr_averaged_all_subjects.csv   (ICC results, all subjects)
  - icc11_lr_averaged_patients_only.csv  (ICC results, patients only)
"""

import pathlib
import pandas as pd
import pingouin as pg
import warnings

warnings.filterwarnings("ignore")

HERE = pathlib.Path(__file__).resolve().parent
INPUT_CSV = HERE.parent / "All_Results_T0T1_Gait_VV_012526.csv"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)

# ── 2. Define LF/RF pairs to average ────────────────────────────────
# Each tuple: (LF suffix, RF suffix, new averaged suffix)
LR_PAIRS = [
    ("AvgLFstancetime",        "AvgRFstancetime",        "Avgstancetime"),
    ("AvgLFswingtime",         "AvgRFswingtime",          "Avgswingtime"),
    ("VariabilityLFstancetime","VariabilityRFstancetime", "Variabilitystancetime"),
    ("VariabilityLFswingtime", "VariabilityRFswingtime",  "Variabilityswingtime"),
    ("AvgLFmaxheight",         "AvgRFmaxheight",          "Avgmaxheight"),
    ("VariabilityLFmaxheight", "VariabilityRFmaxheight",  "Variabilitymaxheight"),
]

# RG columns have the "ng" → "RG" corruption in metric names
def _corrupt_for_rg(metric: str) -> str:
    """Apply the same RG corruption the original data has."""
    return metric.replace("ng", "RG")

# For each (trial, test_type) prefix, average the LF/RF columns
for trial in ("T1", "T2"):
    for tt in ("NG", "RG"):
        for lf_suf, rf_suf, avg_suf in LR_PAIRS:
            if tt == "RG":
                lf_col = f"Xsens_{trial}_{tt}.{_corrupt_for_rg(lf_suf)}"
                rf_col = f"Xsens_{trial}_{tt}.{_corrupt_for_rg(rf_suf)}"
                avg_col = f"Xsens_{trial}_{tt}.{_corrupt_for_rg(avg_suf)}"
            else:
                lf_col = f"Xsens_{trial}_{tt}.{lf_suf}"
                rf_col = f"Xsens_{trial}_{tt}.{rf_suf}"
                avg_col = f"Xsens_{trial}_{tt}.{avg_suf}"

            lf_vals = pd.to_numeric(df[lf_col], errors="coerce")
            rf_vals = pd.to_numeric(df[rf_col], errors="coerce")
            df[avg_col] = (lf_vals + rf_vals) / 2.0

            # Drop the original LF/RF columns
            df.drop(columns=[lf_col, rf_col], inplace=True)

avg_csv = OUT / "All_Results_LR_averaged.csv"
df.to_csv(avg_csv, index=False)
print(f"Saved averaged CSV → {avg_csv}")

# ── 3. Build metric list and column map for the new data ────────────
t1_ng_cols = [c for c in df.columns if c.startswith("Xsens_T1_NG.")]
metrics = [c.split(".", 1)[1] for c in t1_ng_cols]

def _rg_to_canonical(rg_suffix: str) -> str:
    return rg_suffix.replace("RG", "ng")

col_map = {}
for trial in ("T1", "T2"):
    for tt in ("NG", "RG"):
        prefix = f"Xsens_{trial}_{tt}."
        cols = [c for c in df.columns if c.startswith(prefix)]
        for col in cols:
            suffix = col.split(".", 1)[1]
            canonical = suffix if tt == "NG" else _rg_to_canonical(suffix)
            col_map[(trial, tt, canonical)] = col

# Verify mapping
for trial in ("T1", "T2"):
    for tt in ("NG", "RG"):
        for m in metrics:
            assert (trial, tt, m) in col_map, f"Missing column for {trial}_{tt}.{m}"

print(f"Metrics ({len(metrics)}): {metrics}")

# ── 4. Compute ICC(1,1) ─────────────────────────────────────────────
def compute_icc_table(data: pd.DataFrame) -> pd.DataFrame:
    group_map = {1: "Ataxia", 2: "PD"}
    results = []

    for m in metrics:
        row = {"Metric": m}
        for grp_code, grp_name in group_map.items():
            sub = data[data["1ATX2PD"] == grp_code].copy()
            for tt in ("NG", "RG"):
                col_t1 = col_map[("T1", tt, m)]
                col_t2 = col_map[("T2", tt, m)]
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

                row[f"{grp_name}_{tt}"] = icc_val
                row[f"{grp_name}_{tt}_CI95"] = ci_str
        results.append(row)

    out_cols = ["Metric"]
    for g in ("Ataxia", "PD"):
        for tt in ("NG", "RG"):
            out_cols += [f"{g}_{tt}", f"{g}_{tt}_CI95"]
    return pd.DataFrame(results)[out_cols]


# ── 5. Run for all subjects and patients-only ────────────────────────
print("Computing ICC(1,1) for all subjects (LR averaged)...")
icc_all = compute_icc_table(df)
out_all = OUT / "icc11_lr_averaged_all_subjects.csv"
icc_all.to_csv(out_all, index=False)
print(f"  → {out_all}  ({len(icc_all)} rows)")

print("Computing ICC(1,1) for patients only (LR averaged, Control==0)...")
icc_pat = compute_icc_table(df[df["Control"] == 0])
out_pat = OUT / "icc11_lr_averaged_patients_only.csv"
icc_pat.to_csv(out_pat, index=False)
print(f"  → {out_pat}  ({len(icc_pat)} rows)")

print("Done.")
