"""
Compute ICC(1,1) + 95% CI for JT and Myo gait features, W1 vs W2,
split by disease group (Ataxia / PD), with and without controls.

Outputs (in out/):
  - icc11_jt_all_subjects.csv / icc11_jt_patients_only.csv
  - icc11_myo_all_subjects.csv / icc11_myo_patients_only.csv
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

df = pd.read_csv(INPUT_CSV)


def compute_icc_for_system(data, system, metrics):
    """Compute ICC(1,1) table for a given system prefix and metric list."""
    group_map = {1: "Ataxia", 2: "PD"}
    results = []

    for m in metrics:
        row = {"Metric": m}
        for grp_code, grp_name in group_map.items():
            sub = data[data["1ATX2PD"] == grp_code].copy()
            for tt in ("NG", "RG"):
                col_t1 = f"{system}_T1_{tt}.{m}"
                col_t2 = f"{system}_T2_{tt}.{m}"

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


# ── JT features ──────────────────────────────────────────────────────
jt_metrics = ["cadence", "stridetime", "stridetimeSD", "v"]

print("=== JT ===")
print("Computing ICC(1,1) for all subjects...")
icc = compute_icc_for_system(df, "JT", jt_metrics)
out_path = OUT / "icc11_jt_all_subjects.csv"
icc.to_csv(out_path, index=False)
print(f"  → {out_path}  ({len(icc)} rows)")

print("Computing ICC(1,1) for patients only (Control==0)...")
icc = compute_icc_for_system(df[df["Control"] == 0], "JT", jt_metrics)
out_path = OUT / "icc11_jt_patients_only.csv"
icc.to_csv(out_path, index=False)
print(f"  → {out_path}  ({len(icc)} rows)")

# ── Myo features ─────────────────────────────────────────────────────
myo_metrics = [
    "Schrittzeit", "SchrittzeitSD",
    "DoubleSupport%", "DoubleSupport%SD",
    "Kadenz", "KadenzSD",
    "Geschwindigkeit", "GeschwindigkeitSD",
    "FußrotationR", "FußrotationR_SD",
    "FußrotationL", "FußrotationL_SD",
    "Schrittbreite", "SchrittbreiteSD",
]

print("\n=== Myo ===")
print("Computing ICC(1,1) for all subjects...")
icc = compute_icc_for_system(df, "Myo", myo_metrics)
out_path = OUT / "icc11_myo_all_subjects.csv"
icc.to_csv(out_path, index=False)
print(f"  → {out_path}  ({len(icc)} rows)")

print("Computing ICC(1,1) for patients only (Control==0)...")
icc = compute_icc_for_system(df[df["Control"] == 0], "Myo", myo_metrics)
out_path = OUT / "icc11_myo_patients_only.csv"
icc.to_csv(out_path, index=False)
print(f"  → {out_path}  ({len(icc)} rows)")

print("\nDone.")
