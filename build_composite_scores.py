"""
build_composite_scores.py
==========================
Reproduce the long-format "Composite Scores" table (ICC + Spearman merged) from a
reconciled WIDE csv, for AVERAGED features, T1.

This consolidates the previously-separate steps:
  - icc_py/claude/avg_lr_*_and_icc.py  (L/R averaging + ICC(3,1), per system, WIDE out)
  - plot_feature_vs_clinical_lr_avg.py (L/R averaging + Spearman, long out)
into one script that emits the long composite directly, matching:
  "Composite Scores - Averaged [JUNE 22 2026] - SPEARMAN CORR + ICC MERGED, AVG, T1.csv"

Per T1 feature x cohort:
  - ICC   : ICC(3,1) test-retest (T1<->T2) within that disease group (pingouin ICC3)
  - spearman_r/p : T1 feature vs clinical score within that cohort
  - n     : Spearman sample size (pairs with both feature & score present)
  - comp_score        = ICC + |spearman_r|
  - all_conditions_met = (ICC >= ICC_THR) and (|spearman_r| >= R_THR)
Rows where ICC or spearman_r can't be computed are dropped (matches the source file).

Cohorts:  PD_UPDRS -> group 2 vs UPDRS_T1 ;  ATX_SARA -> group 1 vs SARA_T1
"""

from functools import partial
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent   # the dir this script lives in (data CSVs sit beside it)
INPUT_CSV = ROOT / "All_Results_T0T1_Arm_Gait_RECONCILED_062626_FINAL.csv"
OUTPUT_CSV = ROOT / "Composite Scores - Averaged [JUNE 26 2026] - SPEARMAN CORR + ICC MERGED, AVG, T1.csv"

GROUP_COL = "1ATX2PD0HEALTHY"   # 0 healthy, 1 ataxia, 2 PD
UPDRS_COL = "UPDRS_T1"
SARA_COL = "SARA_T1"

ICC_THR = 0.6      # all_conditions_met thresholds (reverse-engineered from June-22 file)
R_THR = 0.3

LR_SYSTEMS = ("Arm", "JT", "Myo", "Xsens")
SYSTEMS = ("Arm", "JT", "Myo", "Xsens")

COHORTS = [
    # label,      group code, clinical score column
    ("PD_UPDRS",  2, UPDRS_COL),
    ("ATX_SARA",  1, SARA_COL),
]

# ── L/R averaging (mirrors plot_feature_vs_clinical_lr_avg.py) ───────────────
def _unmojibake(s: str) -> str:
    try:
        return s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s

def _find_token_pairs(suffixes, *, r_token, l_token, avg_token):
    pairs, seen = [], set()
    for suf in suffixes:
        if r_token not in suf:
            continue
        l_suf = suf.replace(r_token, l_token)
        if l_suf not in suffixes:
            continue
        avg_suf = suf.replace(r_token, avg_token)
        if avg_suf in seen:
            continue
        seen.add(avg_suf)
        pairs.append((suf, l_suf, avg_suf))
    return pairs

def _find_trailing_rl_pairs(suffixes):
    pairs, seen = [], set()
    for suf in suffixes:
        m = re.match(r"^(.+?)R(_SD)?$", suf)
        if not m:
            continue
        base, tail = m.group(1), m.group(2) or ""
        l_suf = f"{base}L{tail}"
        avg_suf = f"{base}{tail}"
        if l_suf not in suffixes or avg_suf in seen:
            continue
        seen.add(avg_suf)
        pairs.append((suf, l_suf, avg_suf))
    return pairs

PAIR_FINDERS = {
    "Arm":   partial(_find_token_pairs, r_token="RFA", l_token="LFA", avg_token="FA"),
    "JT":    _find_trailing_rl_pairs,
    "Myo":   _find_trailing_rl_pairs,
    "Xsens": partial(_find_token_pairs, r_token="RF", l_token="LF", avg_token=""),
}

def apply_lr_averaging(df):
    df.columns = [_unmojibake(c) for c in df.columns]
    for system in LR_SYSTEMS:
        finder = PAIR_FINDERS[system]
        t1_ng_suffixes = {c.split(".", 1)[1] for c in df.columns
                          if c.startswith(f"{system}_T1_NG.")}
        pairs = finder(t1_ng_suffixes)
        print(f"  {system}: L/R pairs averaged = {len(pairs)}")
        for trial in ("T1", "T2"):
            for tt in ("NG", "RG"):
                for r_suf, l_suf, avg_suf in pairs:
                    r_col = f"{system}_{trial}_{tt}.{r_suf}"
                    l_col = f"{system}_{trial}_{tt}.{l_suf}"
                    avg_col = f"{system}_{trial}_{tt}.{avg_suf}"
                    if r_col not in df.columns or l_col not in df.columns:
                        continue
                    r_vals = pd.to_numeric(df[r_col], errors="coerce")
                    l_vals = pd.to_numeric(df[l_col], errors="coerce")
                    df[avg_col] = (r_vals + l_vals) / 2.0
                    df.drop(columns=[r_col, l_col], inplace=True)
    return df

FEATURE_RE = re.compile(r"^(Arm|JT|Myo|Xsens)_T(\d+)_(NG|RG)\.(.+)$")

def icc31(df_group, t1_col, t2_col):
    """ICC(3,1) test-retest for one metric within one disease group."""
    pair = pd.DataFrame({
        "ID": df_group["ID"].values,
        "T1": pd.to_numeric(df_group[t1_col], errors="coerce").values,
        "T2": pd.to_numeric(df_group[t2_col], errors="coerce").values,
    }).dropna(subset=["T1", "T2"])
    if len(pair) < 2:
        return np.nan
    long = pair.melt(id_vars="ID", var_name="raters", value_name="ratings")
    try:
        res = pg.intraclass_corr(data=long, targets="ID", raters="raters", ratings="ratings")
        return round(float(res.loc[res["Type"] == "ICC3", "ICC"].iloc[0]), 3)
    except Exception:
        return np.nan

def main():
    print(f"Input:  {INPUT_CSV.name}")
    df = pd.read_csv(INPUT_CSV)
    df["ID"] = df["ID"].astype(int)
    print("L/R averaging:")
    apply_lr_averaging(df)

    for col in (UPDRS_COL, SARA_COL, GROUP_COL):
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}' in {INPUT_CSV.name}")

    # Clinical scores are intentionally blank in the shared file; they are filled in
    # locally before running. Fail loudly (rather than emit an empty table) if they
    # are still empty -- the Spearman/composite step needs them.
    pd_n = pd.to_numeric(df.loc[df[GROUP_COL] == 2, UPDRS_COL], errors="coerce").notna().sum()
    atx_n = pd.to_numeric(df.loc[df[GROUP_COL] == 1, SARA_COL], errors="coerce").notna().sum()
    if pd_n == 0 and atx_n == 0:
        raise SystemExit(
            f"'{UPDRS_COL}' and '{SARA_COL}' are both empty in {INPUT_CSV.name}. "
            "Fill in the clinical scores and re-run -- the Spearman/composite step needs them."
        )

    # T1 features of the four systems (post-averaging)
    t1_feats = [c for c in df.columns
                if (m := FEATURE_RE.match(c)) and m.group(2) == "1"]
    print(f"T1 features: {len(t1_feats)}")

    rows = []
    for feat in t1_feats:
        m = FEATURE_RE.match(feat)
        system, _, test, metric = m.group(1), m.group(2), m.group(3), m.group(4)
        t2_feat = f"{system}_T2_{test}.{metric}"
        if t2_feat not in df.columns:
            continue
        for label, gcode, score_col in COHORTS:
            grp = df[df[GROUP_COL] == gcode]
            # ICC(3,1) within group from T1<->T2
            icc = icc31(grp, feat, t2_feat)
            # Spearman: T1 feature vs clinical score within cohort
            x = pd.to_numeric(grp[feat], errors="coerce")
            y = pd.to_numeric(grp[score_col], errors="coerce")
            mask = x.notna() & y.notna()
            n = int(mask.sum())
            if n >= 3 and x[mask].std() > 0 and y[mask].std() > 0:
                rho, p = stats.spearmanr(x[mask], y[mask])
            else:
                rho, p = np.nan, np.nan
            if pd.isna(icc) or pd.isna(rho):
                continue  # drop uncomputable rows (matches source)
            comp = icc + abs(rho)
            rows.append({
                "feature": feat, "system": system, "week": "T1", "test": test,
                "metric": metric, "cohort": label, "ICC": icc, "n": n,
                "spearman_r": rho, "spearman_p": p, "comp_score": round(comp, 3),
                "all_conditions_met": bool(icc >= ICC_THR and abs(rho) >= R_THR),
            })

    out = pd.DataFrame(rows).sort_values("comp_score", ascending=False).reset_index(drop=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nOutput: {OUTPUT_CSV.name}")
    print(f"rows: {len(out)} | systems: {out.system.value_counts().to_dict()} "
          f"| cohorts: {out.cohort.value_counts().to_dict()}")
    print(f"all_conditions_met: {out.all_conditions_met.value_counts().to_dict()}")

if __name__ == "__main__":
    main()
