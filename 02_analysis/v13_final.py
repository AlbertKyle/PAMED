#!/usr/bin/env python3
"""Single command centre for the frozen Dementia v13 UKB/NHATS workflow.

This file owns the study protocol and dispatches the independently testable
stages.  All stages share one EID-level 80:20 split stored in 00_protocol.
"""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path("/root/autodl-tmp/PA_CR_CNN/Project_new/PA_Foundation_Model/dementia")
RESULTS = ROOT / "Results" / "v13_final"
LEGACY_SPLIT = ROOT / "supervised_v12_ukb_80_20_retrained"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def run_command(args: list[str]) -> None:
    print("RUN:", " ".join(args), flush=True)
    subprocess.run(args, cwd=HERE, check=True)


def protocol(results: Path) -> Path:
    """Create one immutable protocol directory, reusing the established EIDs."""
    out = results / "00_protocol"; out.mkdir(parents=True, exist_ok=True)
    split = out / "ukb_fixed_80_20_split.csv"
    inner = out / "development_internal_epoch_selection.csv"
    if not split.exists():
        source = LEGACY_SPLIT / split.name
        if not source.exists():
            raise FileNotFoundError(
                f"Established 80:20 split not found: {source}. "
                "Do not silently generate a new split; restore this CSV first.")
        shutil.copy2(source, split)
    if not inner.exists():
        source = LEGACY_SPLIT / inner.name
        if not source.exists(): raise FileNotFoundError(source)
        shutil.copy2(source, inner)
    z = pd.read_csv(split, dtype={"eid":"string"})
    z["eid"] = z.eid.str.strip()
    if set(z.split) != {"development", "test"}: raise ValueError("Invalid outer split roles")
    if z.eid.duplicated().any(): raise ValueError("Duplicate EIDs in outer split")
    dev = z.split.eq("development"); test = z.split.eq("test")
    q = pd.read_csv(inner, dtype={"eid":"string"}); q["eid"] = q.eid.str.strip()
    if q.eid.duplicated().any() or not set(q.eid).issubset(set(z.loc[dev,"eid"])):
        raise ValueError("Inner split is not a unique subset of development EIDs")
    digest = hashlib.sha256("\n".join(z.loc[test,"eid"]).encode()).hexdigest()
    write_json(out / "V13_PROTOCOL_LOCK.json", {
        "created": datetime.now().isoformat(), "protocol_version":"v13_final_1.0",
        "outer_design":"single fixed stratified UKB 80:20 split",
        "development_n":int(dev.sum()), "test_n":int(test.sum()),
        "test_eid_sha256":digest, "split_seed":20260820,
        "primary_horizon":"incident dementia >1 year and within 7 years",
        "predictors_primary":["7-day minute PA", "baseline age", "sex"],
        "traditional_PA":"separate comparator", "APOE":"separate complete-case secondary analysis",
        "prohibited_predictors":["Followingyears_birthday", "Townsend_index", "comorbidity_index",
                                  "post-baseline diagnoses", "outcome-derived variables"],
        "note":"This imports the already established v12 EID list; v13 never redraws it."
    })
    print(f"Protocol locked: development={dev.sum():,}; test={test.sum():,}; sha256={digest}")
    return out


def ensure_core_split(protocol_dir: Path, core_dir: Path) -> None:
    core_dir.mkdir(parents=True, exist_ok=True)
    for name in ("ukb_fixed_80_20_split.csv",):
        dst=core_dir/name; src=protocol_dir/name
        if dst.exists() and dst.read_bytes()!=src.read_bytes():
            raise RuntimeError(f"Refusing mismatched split in {dst}")
        if not dst.exists(): shutil.copy2(src,dst)


def main(cfg) -> None:
    results=Path(cfg.results_root); pdir=protocol(results)
    core=results/"02_core_model"
    common=["--labels-csv",cfg.labels_csv,"--apoe-csv",cfg.apoe_csv,"--cache-dir",cfg.cache_dir]
    if cfg.stage=="protocol": return
    if cfg.stage in ("pool_smoke","pool"):
        mode="smoke" if cfg.stage.endswith("smoke") else "full"
        run_command([sys.executable,str(HERE/"v13_algorithm_pool.py"),"--mode",mode,
                     "--master-dir",str(pdir),"--output-dir",str(results/"01_algorithm_pool"),
                     "--models",cfg.models,*common])
    elif cfg.stage in ("core_smoke","core"):
        ensure_core_split(pdir,core); mode="smoke" if cfg.stage.endswith("smoke") else "full"
        run_command([sys.executable,str(HERE/"v13_core_80_20.py"),"--mode",mode,
                     "--output-dir",str(core),*common])
    elif cfg.stage in ("apoe_smoke","apoe"):
        if not (core/"COMPLETED.json").exists(): raise RuntimeError("Run core first")
        mode="smoke" if cfg.stage.endswith("smoke") else "full"
        run_command([sys.executable,str(HERE/"v13_apoe_secondary.py"),"--mode",mode,
                     "--master-dir",str(core),"--output-dir",str(results/"03_apoe_secondary"),*common])
    elif cfg.stage in ("external_smoke","external"):
        if not (core/"COMPLETED.json").exists(): raise RuntimeError("Run core first")
        mode="smoke" if cfg.stage.endswith("smoke") else "full"
        run_command([sys.executable,str(HERE/"v13_nhats_external.py"),"--mode",mode,
                     "--input-dir",cfg.nhats_dir,"--core-dir",str(core),
                     "--output-dir",str(results/"04_nhats_external")])
    elif cfg.stage in ("supplement_smoke","stability","paired","subgroups","ablation","interpretability","decision"):
        if not (core/"COMPLETED.json").exists(): raise RuntimeError("Run core first")
        folders={"supplement_smoke":"05_supplement_smoke","stability":"05_stability",
                 "paired":"06_paired_comparisons","subgroups":"07_subgroups",
                 "ablation":"08_ablation","interpretability":"09_interpretability",
                 "decision":"10_decision_analysis"}
        run_command([sys.executable,str(HERE/"v13_supplementary.py"),cfg.stage,
                     "--labels-csv",cfg.labels_csv,"--apoe-csv",cfg.apoe_csv,
                     "--cache-dir",cfg.cache_dir,"--core-dir",str(core),
                     "--protocol-dir",str(pdir),"--output-dir",str(results/folders[cfg.stage])])
    elif cfg.stage in ("clinical_smoke","representations","subtypes","temporal_shapley",
                       "rhythm_concepts","survival","interactions",
                       "clinical_associations","final_paired","clinical_all"):
        if not (core/"COMPLETED.json").exists(): raise RuntimeError("Run core first")
        folders={"clinical_smoke":"11_clinical_smoke","representations":"11_representations",
                 "subtypes":"12_subtypes","temporal_shapley":"13_hierarchical_shapley",
                 "rhythm_concepts":"14_rhythm_knowledge","survival":"15_survival",
                 "interactions":"16_interactions","clinical_associations":"17_clinical_associations",
                 "final_paired":"18_final_paired"}
        stages=list(folders)[1:] if cfg.stage=="clinical_all" else [cfg.stage]
        for stage in stages:
            if cfg.stage=="clinical_all" and (results/folders[stage]/"COMPLETED.json").exists():
                print(f"Skipping completed clinical stage: {stage}",flush=True);continue
            if stage not in ("clinical_smoke","representations","final_paired") and not (results/"11_representations"/"COMPLETED.json").exists():
                raise RuntimeError("Run representations first")
            run_command([sys.executable,str(HERE/"v13_clinical_extension.py"),stage,
                         "--labels-csv",cfg.labels_csv,"--apoe-csv",cfg.apoe_csv,
                         "--subtypes-csv",cfg.subtypes_csv,
                         "--clinical-outcomes-csv",cfg.clinical_outcomes_csv,
                         "--comorbidity-cols",cfg.comorbidity_cols,
                         "--cache-dir",cfg.cache_dir,"--core-dir",str(core),
                         "--protocol-dir",str(pdir),"--representation-dir",str(results/"11_representations"),
                         "--ablation-dir",str(results/"08_ablation"),
                         "--apoe-dir",str(results/"03_apoe_secondary"),
                         "--output-dir",str(results/folders[stage])])
    else: raise ValueError(cfg.stage)


if __name__ == "__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("stage",choices=["protocol","pool_smoke","pool","core_smoke","core",
                                    "apoe_smoke","apoe","external_smoke","external",
                                    "supplement_smoke","stability","paired","subgroups",
                                    "ablation","interpretability","decision",
                                    "clinical_smoke","representations","subtypes",
                                    "temporal_shapley","rhythm_concepts","survival",
                                    "interactions","clinical_associations","final_paired",
                                    "clinical_all"])
    a.add_argument("--results-root",default=str(RESULTS))
    a.add_argument("--labels-csv",default=str(ROOT/"ukb_dementia_labels_covariates_N80990.csv"))
    a.add_argument("--apoe-csv",default=str(ROOT/"APOE_phenotype.csv"))
    a.add_argument("--cache-dir",default=str(ROOT/"supervised_results"/"cache"))
    a.add_argument("--nhats-dir",default=str(ROOT/"external_validation"))
    a.add_argument("--subtypes-csv",default=str(ROOT/"ukb_dementia_subtypes_N80990.csv"))
    a.add_argument("--clinical-outcomes-csv",default=os.environ.get("V13_CLINICAL_OUTCOMES_CSV",str(ROOT/"UKB_Demographic_clinic_PA_baseline.csv")))
    a.add_argument("--comorbidity-cols",default=os.environ.get("V13_COMORBIDITY_COLS",""))
    a.add_argument("--models",default="fcn,resnet1d,tcn,xceptiontime,inception,patchtst,sameclock,mamba_fcn,bimambasl_fcn,chronoview_mamba_fcn,gru_mamba_fcn")
    main(a.parse_args())
