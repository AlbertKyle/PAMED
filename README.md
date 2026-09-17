# PAMED NMI manuscript reproducibility code

This folder contains the minimal code set retained for the manuscript
“Learning a transportable prediagnostic behavioural phenotype of dementia from wearable activity”
and its Supplementary Information, based on the versions available on 2026-09-17.

## Included scope

### 01 cohort and descriptive

- `build_ukb_dementia_labels.py`: constructs all-cause and subtype dementia labels, landmark timing, and audit tables from UK Biobank source fields.
- `dementia_descriptive_nature.py`: derives seven-day profiles and directly computable activity and rhythm descriptors used by the descriptive figures and downstream publication atlas.

### 02 v13 analysis

- `v13_final.py` and `run_v13_final.sh`: stage orchestration and immutable split handling.
- `v13_cohort_engine.py`: cohort loading, minute-level activity/mask preprocessing, tabular preprocessing, calibration, and metric utilities.
- `v13_classic_engine.py`, `v13_rhythm_base.py`, `v13_rhythm_engine.py`, and `v13_mamba_engine.py`: model components imported by the final GRU-FCN and ConvContext-Mamba workflow and algorithm pool.
- `v13_fusion_engine.py` and `v13_core_80_20.py`: dual-pathway fusion, age-sex context, fixed 80:20 training, calibration, thresholding, and internal evaluation.
- `v13_algorithm_pool.py`: expanded PA-only architecture benchmark.
- `v13_apoe_secondary.py`: genotype-complete secondary comparison.
- `v13_nhats_external.py`: frozen-model NHATS transport, alignment, and sensitivity analyses.
- `v13_supplementary.py`: repeated-fit stability, paired comparisons, subgroup analyses, ablation, occlusion, and decision summaries.
- `v13_clinical_extension.py`: representations, clustering, subtypes, temporal Shapley analysis, rhythm concepts, survival, interactions, mortality, and comorbidity analyses.

### 03 figure generation

- `publication_atlas.py`: common publication-figure and table-generation utilities.
- `publication_atlas_expanded.py`: maps the v13 result modules into main and supplementary figure candidates.
- `publication_atlas_v3.py`: final visual-semantics and figure-output revision used by the manuscript figure workflow.
- `run_publication_atlas_v3.sh`: figure-generation entry point.
- `install_publication_fonts_v3.sh`: font availability and rendering helper.

## Manuscript mapping

- Cohort definition and dementia outcomes: `01_cohort_and_descriptive/build_ukb_dementia_labels.py`, `02_analysis/v13_cohort_engine.py`
- Seven-day preprocessing and conventional activity characterization: `01_cohort_and_descriptive/dementia_descriptive_nature.py`, `02_analysis/v13_cohort_engine.py`
- PAMED architecture, training, calibration, and internal metrics: `02_analysis/v13_mamba_engine.py`, `v13_fusion_engine.py`, `v13_core_80_20.py`
- Algorithm benchmarks and ablations: `02_analysis/v13_algorithm_pool.py`, `v13_supplementary.py`
- APOE analyses: `02_analysis/v13_apoe_secondary.py`
- NHATS external transport: `02_analysis/v13_nhats_external.py`
- Explanations, rhythm concepts, clusters, survival, subtypes, mortality, and comorbidity: `02_analysis/v13_supplementary.py`, `v13_clinical_extension.py`
- Main and supplementary publication panels: `03_figure_generation/publication_atlas.py`, `publication_atlas_expanded.py`, `publication_atlas_v3.py`

## Deliberately excluded

- Superseded exploratory and development versions v2 through v12.
- Duplicate older descriptive, external-validation, and publication-atlas implementations.
- Smoke/full wrapper scripts belonging only to superseded versions.
- Participant-level UK Biobank or NHATS data, derived participant-level tables, model checkpoints, caches, predictions, logs, and rendered figures.
- Point-and-click statistical-software operations. The retained Python code already covers the manuscript's programmatic statistical analyses.
- The Mi Watch acquisition/interface illustration, because no corresponding application source code was found under the searched project path; it is described in the manuscript as a research workflow rather than a released analytical module.

## Before public release

1. Replace machine-specific absolute paths with command-line arguments or a project configuration file. Several scripts preserve the original local and AutoDL paths.
2. Record and pin the exact package and CUDA versions used for the final run. `requirements.txt` lists the imported top-level packages but does not invent historical version numbers.
3. Add the immutable outer split and configuration/calibration artifacts only if data-use rules permit. Do not publish participant identifiers or participant-level predictions.
4. Confirm that the repository license, third-party package licenses, font licenses, and UK Biobank/NHATS data-use terms are satisfied.
5. Run the complete workflow in a clean environment and compare aggregate outputs with the manuscript and Supplementary Information before assigning a release tag and archival identifier.

## Integrity files

- `CODE_MANIFEST.csv` records each retained file's original path, copied path, size, modification time, and SHA-256 hash.
- `requirements.txt` records the direct third-party Python packages imported by this code set.

