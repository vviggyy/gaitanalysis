# ICC Analysis Pipeline

Computes ICC(3,1) test-retest reliability for gait metrics across three sensor systems (Xsens, JT, Myo) and generates bar-chart visualizations.

Two-way mixed-effects, single-measure consistency (Koo & Li 2015), via `pingouin.intraclass_corr` (the `ICC3` row).

## Input

All scripts read from:

```
../All_Results_T0T1_Gait_VV_CR_20260303.csv
```

This CSV contains gait metrics from two trials (T1, T2) for Ataxia and PD patient groups, with a `Control` column (0 = patient, 1 = control) and `1ATX2PD` (1 = Ataxia, 2 = PD).

## Pipeline

Outputs are written to `out/042926/`. Run in order:

### 1. `calc_icc_3_1.py` — Xsens ICC

Computes ICC(3,1) with 95% CI for 33 Xsens metrics (including separate Left/Right foot metrics), split by disease group (Ataxia/PD) and gait type (NG/RG).

**Outputs:**
- `out/042926/icc31_all_subjects.csv`
- `out/042926/icc31_patients_only.csv`

### 2. `avg_lr_and_icc.py` — LR-averaged Xsens ICC

Averages 6 left/right foot metric pairs (e.g. stance time, swing time, max height + their variabilities) into bilateral metrics, then computes ICC(3,1).

**Outputs:**
- `out/042926/All_Results_LR_averaged.csv`
- `out/042926/icc31_lr_averaged_all_subjects.csv`
- `out/042926/icc31_lr_averaged_patients_only.csv`

### 3. `calc_icc_jt_myo.py` — JT & Myo ICC

Computes ICC(3,1) for JT (4 metrics: cadence, stride time, stride time SD, velocity) and Myo (15 metrics).

**Outputs:**
- `out/042926/icc31_jt_all_subjects.csv`, `out/042926/icc31_jt_patients_only.csv`
- `out/042926/icc31_myo_all_subjects.csv`, `out/042926/icc31_myo_patients_only.csv`

### 4. `plot_icc.py` — All plots

Reads the ICC CSVs from steps 1-3 and generates horizontal bar charts. Three functions:

| Function | Output | Description |
|----------|--------|-------------|
| `plot_individual()` | 4 PNGs per call | One bar chart per category (Ataxia NG/RG, PD NG/RG) |
| `plot_combined()` | 1 PNG per call | 2x2 grid of all categories |
| `plot_multi_system()` | 1 PNG per call | 2x2 grid with multiple systems stacked |

Running `python plot_icc.py` produces all plots (22 PNGs total).

**Parameters:**
- `color` — `"reliability"` (colored by ICC band), `"uniform"` (single color), or `"system"` (system-specific color)
- `bg_bands` — show/hide background reliability shading
- `top_n` — filter to top N metrics by ICC value
- `system_name` — label for titles

## Quick start

```bash
cd icc_py/claude
python calc_icc_3_1.py
python avg_lr_and_icc.py
python calc_icc_jt_myo.py
python plot_icc.py
```
