#!/usr/bin/env python3
"""Run PAMED analyses with a shared participant partition and frozen model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA_ROOT, RESULTS_ROOT
import pandas as pd

HERE = Path(__file__).resolve().parent
CLINICAL_FOLDERS = {
    "representations": "11_representations",
    "subtypes": "12_subtypes",
    "temporal_shapley": "13_hierarchical_shapley",
    "rhythm_concepts": "14_rhythm_knowledge",
    "survival": "15_survival",
    "interactions": "16_interactions",
    "clinical_associations": "17_clinical_associations",
    "final_paired": "18_final_paired",
}
SUPPLEMENT_FOLDERS = {
    "stability": "05_stability", "paired": "06_paired_comparisons",
    "subgroups": "07_subgroups", "ablation": "08_ablation",
    "interpretability": "09_interpretability", "decision": "10_decision_analysis",
}


def protocol(results, split_dir):
    """Import the study participant partition without redrawing it."""
    out = results / "00_protocol"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("ukb_fixed_80_20_split.csv", "development_internal_epoch_selection.csv"):
        source, target = split_dir / name, out / name
        if source.exists():
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError(f"Participant partition differs: {target}")
            if not target.exists():
                shutil.copy2(source, target)
        elif not target.exists():
            raise FileNotFoundError(f"Supply the study participant partition: {source}")
    outer = pd.read_csv(out / "ukb_fixed_80_20_split.csv", dtype={"eid": "string"})
    inner = pd.read_csv(out / "development_internal_epoch_selection.csv", dtype={"eid": "string"})
    for frame in (outer, inner):
        frame["eid"] = frame["eid"].str.strip()
        if frame["eid"].isna().any() or frame["eid"].duplicated().any():
            raise ValueError("Participant partition contains missing or duplicate IDs")
    if set(outer["split"]) != {"development", "test"}:
        raise ValueError("Outer partition requires development and test roles")
    development = outer.loc[outer["split"].eq("development"), "eid"]
    if set(inner["eid"]) != set(development):
        raise ValueError("Inner partition must cover exactly the development participants")
    if set(inner["role"]) != {"inner_training", "epoch_selection"}:
        raise ValueError("Inner partition requires inner_training and epoch_selection roles")
    test = outer.loc[outer["split"].eq("test"), "eid"]
    lock = {
        "outer_design": "participant-disjoint stratified 80:20",
        "development_n": len(development), "test_n": len(test),
        "test_eid_sha256": hashlib.sha256("\n".join(test).encode()).hexdigest(),
        "split_seed": 20260820,
        "endpoint": "incident dementia after day 365 and through year 7",
        "predictors_primary": ["seven-day activity", "observation mask", "age at accelerometry", "sex"],
    }
    (out / "PROTOCOL_LOCK.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return out


def ensure_core_split(protocol_dir, core_dir):
    core_dir.mkdir(parents=True, exist_ok=True)
    for name in ("ukb_fixed_80_20_split.csv", "development_internal_epoch_selection.csv"):
        source, target = protocol_dir / name, core_dir / name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"Core participant partition differs: {target}")
        if not target.exists():
            shutil.copy2(source, target)


def dispatch(script, arguments):
    subprocess.run([sys.executable, str(HERE / script), *map(str, arguments)], check=True, cwd=HERE)


def main(cfg):
    results = Path(cfg.results_root)
    core = results / "02_core_model"
    common = ["--labels-csv", cfg.labels_csv, "--apoe-csv", cfg.apoe_csv, "--cache-dir", cfg.cache_dir]
    if cfg.stage == "external":
        arguments = ["--input-dir", cfg.nhats_dir, "--core-dir", core,
                     "--output-dir", results / "04_nhats_external"]
        if cfg.pa_summary_csv:
            arguments += ["--pa-summary-csv", cfg.pa_summary_csv]
        if cfg.activity_manifest:
            arguments += ["--activity-manifest", cfg.activity_manifest]
        dispatch("nhats_validation.py", arguments)
        return
    pdir = protocol(results, Path(cfg.split_dir))
    if cfg.stage == "protocol":
        return
    if cfg.stage == "core":
        ensure_core_split(pdir, core)
        dispatch("train_evaluate.py", ["--output-dir", core, *common])
    elif cfg.stage == "pool":
        dispatch("benchmark_models.py", ["--master-dir", pdir,
                 "--output-dir", results / "01_algorithm_pool", "--models", cfg.models, *common])
    elif cfg.stage == "apoe":
        dispatch("apoe_analysis.py", ["--master-dir", core,
                 "--output-dir", results / "03_apoe_secondary", *common])
    elif cfg.stage in SUPPLEMENT_FOLDERS:
        dispatch("sensitivity_analysis.py", [cfg.stage, "--core-dir", core,
                 "--protocol-dir", pdir, "--stability-dir", results / "05_stability",
                 "--output-dir", results / SUPPLEMENT_FOLDERS[cfg.stage], *common])
    elif cfg.stage in CLINICAL_FOLDERS or cfg.stage == "clinical_all":
        stages = list(CLINICAL_FOLDERS) if cfg.stage == "clinical_all" else [cfg.stage]
        for stage in stages:
            arguments = [stage, *common, "--core-dir", core, "--protocol-dir", pdir,
                         "--subtypes-csv", cfg.subtypes_csv,
                         "--clinical-outcomes-csv", cfg.clinical_outcomes_csv,
                         "--comorbidity-cols", cfg.comorbidity_cols,
                         "--representation-dir", results / "11_representations",
                         "--ablation-dir", results / "08_ablation",
                         "--apoe-dir", results / "03_apoe_secondary",
                         "--output-dir", results / CLINICAL_FOLDERS[stage]]
            for option, value in (("--followup-days-col", cfg.followup_days_col),
                                  ("--death-days-col", cfg.death_days_col),
                                  ("--mortality-time-days-col", cfg.mortality_time_days_col)):
                if value:
                    arguments += [option, value]
            dispatch("clinical_analysis.py", arguments)
    else:
        raise ValueError(cfg.stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["protocol", "pool", "core", "apoe", "external",
                        *SUPPLEMENT_FOLDERS, *CLINICAL_FOLDERS, "clinical_all"])
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--split-dir", type=Path, default=DATA_ROOT / "protocol")
    parser.add_argument("--labels-csv", default=str(DATA_ROOT / "ukb_dementia_labels_covariates_N80990.csv"))
    parser.add_argument("--apoe-csv", default=str(DATA_ROOT / "APOE_phenotype.csv"))
    parser.add_argument("--cache-dir", default=str(DATA_ROOT / "cache"))
    parser.add_argument("--nhats-dir", default=str(DATA_ROOT / "external_validation"))
    parser.add_argument("--pa-summary-csv")
    parser.add_argument("--activity-manifest")
    parser.add_argument("--subtypes-csv", default=str(DATA_ROOT / "ukb_dementia_subtypes_N80990.csv"))
    parser.add_argument("--clinical-outcomes-csv", default=os.environ.get(
        "PAMED_CLINICAL_OUTCOMES_CSV", str(DATA_ROOT / "UKB_Demographic_clinic_PA_baseline.csv")))
    parser.add_argument("--comorbidity-cols", default=os.environ.get("PAMED_COMORBIDITY_COLS", ""))
    parser.add_argument("--followup-days-col", default="")
    parser.add_argument("--death-days-col", default="")
    parser.add_argument("--mortality-time-days-col", default="")
    parser.add_argument("--models", default="fcn,resnet1d,tcn,xceptiontime,inception,patchtst,sameclock,mamba_fcn,bimambasl_fcn,chronoview_mamba_fcn,gru_mamba_fcn")
    main(parser.parse_args())
