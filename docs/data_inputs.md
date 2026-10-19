# Data inputs

## UK Biobank

`enrollment_age` denotes age at the accelerometer assessment. The label table
contains `eid`, `accelerometer_start_date`, `first_dementia_date`,
`days_accel_to_dementia`, `incident_analysis_eligible`, `enrollment_age`,
`gender`, `Followingyears_birthday` and the four activity-duration fields below.
`Followingyears_birthday` is the age at the end of available outcome follow-up.
The shared endpoint definition selects dementia after day 365 and through seven
years, with sufficient observation for non-events.

`data/protocol/ukb_fixed_80_20_split.csv` contains `eid,split`, where split is
`development` or `test`. `development_internal_epoch_selection.csv` contains
`eid,role`, where role is `inner_training` or `epoch_selection`. These are the
study participant assignments. Participant identifiers remain in the local
authorized data environment.

`data/cache/` contains `activity_7d.npy`, `observed_7d.npy` and `eids.npy` in
matching participant order. The first two arrays have shape `(N,7,1440)`;
the mask is 1 for observed and 0 for missing. `build_activity_cache.py` can
prepare these arrays from a CSV with one `eid` column and 10,080 chronologically
ordered minute columns, already aligned to seven local-calendar days.

`APOE_phenotype.csv` contains `eid,e4`; dosage is 0, 1 or 2. Subtype labels are
produced by `build_dementia_labels.py`. Additional clinical fields can be joined
with `--clinical-outcomes-csv` and explicit `--comorbidity-cols`.

The default survival analysis uses the study's fixed window. Optional
`--followup-days-col`, `--death-days-col` and `--mortality-time-days-col` supply
recorded times since accelerometry for separate observed-follow-up analyses.
Mortality Cox times represent death for events and censoring for non-events;
the age-difference fallback assumes `Followingyears_birthday` has that meaning.

The IV descriptor uses the full 10,080-minute sequence at one-minute resolution:
mean squared differences of observed adjacent minutes divided by the variance
of observed activity. Gaps are not joined. Concept distribution diagnostics and
common-scale/volume-adjusted sensitivities are written separately from the
primary concept association table.

## External validation

Set `PAMED_DATA_ROOT` and `PAMED_RESULTS_ROOT`, or pass explicit paths to each
entry point. The default NHATS input directory is
`PAMED_DATA_ROOT/external_validation`; its default result directory is
`PAMED_RESULTS_ROOT/04_nhats_external`.

## NHATS activity and outcomes

Activity CSVs contain `eid` and minute columns `1` through `1440`. Each
participant has exactly seven rows in chronological local-calendar day order,
with minutes ordered from local midnight. A measured zero denotes observed
inactivity. Blank/NaN values denote missing observations. Activity values are
in the measurement domain before the frozen UKB `log1p` transformation and
location/scale normalization; do not supply values already normalized by the
model preprocessor.

The default filenames are:

| Mapping | Filename | Role |
| --- | --- | --- |
| `raw_count` | `nhats_r11_daily_7x1440_raw_count.csv` | Required raw-transfer comparison |
| `positive_robust` | `nhats_r11_daily_7x1440_positive_robust.csv` | Required primary mapping |
| `positive_quantile` | `nhats_r11_daily_7x1440_positive_quantile.csv` | Optional sensitivity mapping |
| `learned_adapter` | `nhats_r11_adapter_daily_7x1440_raw.csv` | Optional genuinely prealigned adapter input |

Use `--activity-manifest PATH` to override discovery. Its JSON object maps
these mode names to CSV paths; relative paths resolve beside the manifest.
Every supplied mapping must cover the primary validation participants.

`NHATS_Round_11_SP_File_V2.csv` supplies `spid` (or `eid`), `r11d2intvrage`
and `r5dgender`. Age groups 70–74, 75–79, 80–84, 85–89 and 90+ are represented
by 72, 77, 82, 87 and 92.5 years; inapplicable age is excluded. Sex codes are
1 for male and 2 for female. `enrollment_age` is the model's field name for
age at the accelerometer assessment.

`NHATS_R11_BEAT384_revised_longitudinal_labels_R11_R14.csv` contains one row
per `eid` and the numeric binary fields `dementia_binary_r11`,
`incident_dementia_binary` and `incident_probable_dementia`. Missing follow-up
endpoints remain missing. The primary analysis excludes R11 dementia and
uses the observed possible-or-probable incident endpoint. Outcome fields
are consumed by evaluation, never by measurement alignment.

## Traditional-PA comparator

Supply `--pa-summary-csv PATH`, or place `nhats_pa_summaries.csv` in the NHATS
input directory. It has one row per participant and these exact columns:

| Column | Meaning | Unit |
| --- | --- | --- |
| `eid` | Participant identifier used by activity and outcome inputs | Identifier |
| `Light_Total_Hour` | Light-intensity activity duration | Hours/day |
| `MVPA_Total_Hour` | Moderate-to-vigorous activity duration | Hours/day |
| `Sedentary_Total_Hour` | Sedentary duration | Hours/day |
| `Sleep_Total_Hour` | Sleep duration | Hours/day |

Prepare these participant-level summaries using documented NHATS-specific
activity/sleep definitions, valid-day rules and aggregation procedures.
Minute counts alone do not identify sleep or validated intensity categories.
The validation script requires finite
values between 0 and 24 hours/day for every primary participant and fails
with an `invalid_pa_summaries.csv` diagnostic when required values are absent
or invalid. It does not assume the four durations necessarily sum to 24.

The principal comparator is the frozen UKB
`age_sex_traditional_pa_80pct.joblib`, evaluated on `enrollment_age`, `gender`
and the four PA fields above. Its output key is
`age_sex_traditional_pa`. If `age_sex_80pct.joblib` is also available, the
separate `age_sex` score is evaluated under its own name.

## Frozen UKB checkpoints

Pass `--core-dir` containing:

- `grufcn_80pct_refit.pt`
- `convcontext_mamba_80pct_refit.pt`
- `dual_pa_feature_80pct_refit.pt`
- `final_hybrid_80pct_refit.pt`
- `age_sex_preprocessor_80pct.joblib`
- `age_sex_traditional_pa_80pct.joblib`
- Optional `age_sex_80pct.joblib`

The checkpoints must come from the same completed UKB development fit.
Neural inference retains the stored UKB normalization, feature parameters,
fusion parameters and risk head. No NHATS outcome is used for refitting,
checkpoint selection or probability calibration. Transported scores assess
ranking rather than calibrated three-year NHATS absolute risk.

Example invocation, with paths replaced by the corresponding local inputs:

```text
python analysis/nhats_validation.py --input-dir PATH_TO_NHATS --core-dir PATH_TO_FROZEN_UKB_MODELS --pa-summary-csv PATH_TO_NHATS_PA_SUMMARIES --activity-manifest PATH_TO_ACTIVITY_MANIFEST --output-dir PATH_TO_RESULTS
```

## Outcome-blind mapping generation

`analysis/align_measurements.py` reads only `eid` and minute activity columns.
Supply `--nhats-raw-csv`, `--ukb-reference-csv` and
`--ukb-development-ids-csv`. The ID file has an `eid` column and must identify
development participants only; the script filters the UKB reference to that
list. The reference and target inputs must use the same day/minute layout.
The source reference is read in chunks, although all selected positive
values are retained for exact empirical quantiles.

The main-text positive-robust formula is applied in the original activity
domain: `mU + (sU / sE) * (A - mE)` for observed `A > 0`. The generator's
available estimator is positive-value median and IQR. It requires the
explicit option `--robust-estimator median-iqr`; select it only when that
estimator matches the intended analysis protocol. `--robust-nonpositive`
must explicitly be `error` or `clip-zero`. The former stops if the affine
mapping produces any nonpositive value. The latter truncates such values
at zero and records the count. Original zeros and missing observations
are preserved.

Positive-quantile mapping uses an evenly spaced probability grid, NumPy's
linear empirical quantiles and piecewise-linear interpolation. Tied source
knots map to the mean of their corresponding target quantiles. Values beyond
the fitted knot range are mapped to the endpoint target quantile. Specify
`--quantile-points` and `--quantile-tail clip` explicitly. The estimator,
boundary policy and quantile settings must be matched to the intended
protocol; a named alignment method alone does not specify those choices.

The generator writes two activity CSVs, `alignment_parameters.json` and
`activity_manifest.json`. The parameter JSON records locations/scales,
quantile knots, zero/missingness policy, development-sample size and input
file hashes. Retain it with the analysis outputs. Pass the manifest to
`nhats_validation.py`. Reusing an existing output directory requires
`--overwrite`.

The constrained monotone adapter requires an independently specified and
fitted alignment procedure. Supply an actual prealigned CSV using
`--learned-adapter-csv`; the generator records that input in its manifest
and does not create or train an adapter.

## Output interpretation

`nhats_external_predictions.csv`, `nhats_external_metrics.csv` and
`risk_enrichment.csv` include the frozen age–sex–traditional-PA comparator.
The metric table also includes strict-endpoint and age-overlap sensitivity
sets. `paired_vs_age_sex_traditional_pa_MODE.csv` reports each model minus
the principal comparator; `paired_vs_final_MODE.csv` reports each comparator
minus the final model. Bootstrap rows are participant-paired within each
comparison and use identical resampling draws across compared models.

Top-risk groups contain `ceil(N * requested_fraction)` participants.
`selected_fraction` records the actual fraction, and `enrichment` is the
selected-group event proportion divided by overall prevalence.
