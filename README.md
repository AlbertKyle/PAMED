# PAMED

Code accompanying **Multiscale wearable learning reveals a transportable
prediagnostic behavioural phenotype of dementia**.

PAMED integrates minute-level activity, weekly context and demographic
information to study prediagnostic dementia risk in UK Biobank and NHATS.

## Repository

| Location | Purpose |
| --- | --- |
| `data_preparation/build_dementia_labels.py` | Hospital-record dementia and subtype labels |
| `data_preparation/build_activity_cache.py` | Seven-day activity and observation arrays |
| `data_preparation/describe_activity.py` | Cohort and activity summaries |
| `analysis/train_evaluate.py` | Model development and internal evaluation |
| `analysis/pamed_models.py` | Motion and weekly-context encoders |
| `analysis/fusion.py` | Representation fusion and demographic interaction |
| `analysis/temporal_models.py` | Activity-sequence comparators |
| `analysis/benchmark_models.py` | Comparator evaluation |
| `analysis/align_measurements.py` | Cross-device measurement alignment |
| `analysis/nhats_validation.py` | Frozen-model external evaluation |
| `analysis/apoe_analysis.py` | Genotype-complete comparisons |
| `analysis/sensitivity_analysis.py` | Stability, ablation and perturbation analyses |
| `analysis/clinical_analysis.py` | Temporal attribution, behaviour concepts and clinical outcomes |
| `figures/generate_figures.py` | Publication panel assets |

Shared cohort definitions and paths are in `study_design.py` and
`project_paths.py`. Detailed file formats are in [Data inputs](docs/data_inputs.md).

## Environment and data

Use Python 3.11 with the packages in `requirements.txt`. Model training requires
a compatible PyTorch, CUDA and `mamba-ssm` environment.

```bash
pip install -r requirements.txt
export PAMED_DATA_ROOT=/path/to/data
export PAMED_RESULTS_ROOT=/path/to/results
```

UK Biobank and NHATS participant-level data require access through the respective
cohort providers: [UK Biobank](https://www.ukbiobank.ac.uk/use-our-data/) and
[NHATS](https://www.nhats.org/data-access).

## Analysis

Place the study participant partitions in `data/protocol/` and the prepared
activity arrays in `data/cache/`, or provide explicit paths.

```bash
python analysis/run_pipeline.py protocol
python analysis/run_pipeline.py core
python analysis/run_pipeline.py pool
python analysis/run_pipeline.py apoe
python analysis/run_pipeline.py external
```

Supplementary stages are `stability`, `paired`, `subgroups`, `ablation`,
`interpretability` and `decision`. Clinical stages are `representations`,
`subtypes`, `temporal_shapley`, `rhythm_concepts`, `survival`, `interactions`,
`clinical_associations` and `final_paired`; `clinical_all` runs these in order.
Run `stability` and `ablation` before analyses that use their saved outputs.

```bash
python analysis/run_pipeline.py stability
python analysis/run_pipeline.py ablation
python analysis/run_pipeline.py clinical_all
python figures/generate_figures.py --module all
```

Figures are generated from saved analysis outputs. Source diagrams and final
composite layouts are separate artwork.
