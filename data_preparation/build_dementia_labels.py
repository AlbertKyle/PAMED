#!/usr/bin/env python3
"""Build hospital-record dementia labels for the 7-day UKB accelerometer cohort.

The ICD-10 code list follows the UK Biobank algorithmically-defined dementia
outcome documentation (version 2.0). This local dataset contains hospital
inpatient ICD-10 arrays (Fields 41270/41280) only; consequently the output is
an official-code-list, hospital-only ascertainment rather than the complete
UKB ADO, which additionally uses baseline self-report and death registration.

Raw source files are read-only. All outputs are written below this script's
directory. ICD codes and dates are paired strictly by matching array index.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA_ROOT, RESULTS_ROOT

import numpy as np
import pandas as pd


ROOT = DATA_ROOT
OUT_DIR = DATA_ROOT
MAP_DIR = DATA_ROOT / "hospital_records"
ICD_PATH = MAP_DIR / "41270.csv"
DATE_PATH = MAP_DIR / "41280.csv"
ACC_PATH = ROOT / "representative_mean_acc_wide_matched_7days_filtered_QC_pass_N80990.csv"
BASELINE_PATH = ROOT / "UKB_Demographic_clinic_PA_baseline.csv"
WEAR_PATH = ROOT / "Wear_strat_time.csv"

OFFICIAL_SOURCE_URL = "https://biobank.ndph.ox.ac.uk/ukb/ukb/docs/alg_outcome_main.pdf"
DEFINITION_VERSION = "UKB_ADO_v2.0_2022_ICD10_hospital_only"
WASHOUT_DAYS = 365

# UKB ADO v2.0 dementia code list. Parent codes are retained in the codebook;
# matching is exact after normalisation, with descendants accepted for parent
# codes so that locally present national extensions are not inadvertently lost.
CODEBOOK = [
    ("A81.0", "Sporadic Creutzfeldt-Jakob disease", 0, 0, 0, 1, "other"),
    ("F00", "Dementia in Alzheimer's disease", 1, 0, 0, 1, "alzheimer"),
    ("F00.0", "Dementia in Alzheimer's disease with early onset", 1, 0, 0, 1, "alzheimer"),
    ("F00.1", "Dementia in Alzheimer's disease with late onset", 1, 0, 0, 1, "alzheimer"),
    ("F00.2", "Dementia in Alzheimer's disease, atypical or mixed type", 1, 0, 0, 1, "alzheimer"),
    ("F00.9", "Dementia in Alzheimer's disease, unspecified", 1, 0, 0, 1, "alzheimer"),
    ("F01", "Vascular dementia", 0, 1, 0, 1, "vascular"),
    ("F01.0", "Vascular dementia of acute onset", 0, 1, 0, 1, "vascular"),
    ("F01.1", "Multi-infarct dementia", 0, 1, 0, 1, "vascular"),
    ("F01.2", "Subcortical vascular dementia", 0, 1, 0, 1, "vascular"),
    ("F01.3", "Mixed cortical and sub-cortical vascular dementia", 0, 1, 0, 1, "vascular"),
    ("F01.8", "Other vascular dementia", 0, 1, 0, 1, "vascular"),
    ("F01.9", "Vascular dementia, unspecified", 0, 1, 0, 1, "vascular"),
    ("F02", "Dementia in other diseases classified elsewhere", 0, 0, 0, 1, "other"),
    ("F02.0", "Dementia in Pick's disease", 0, 0, 1, 1, "frontotemporal"),
    ("F02.1", "Dementia in Creutzfeldt-Jakob disease", 0, 0, 0, 1, "other"),
    ("F02.2", "Dementia in Huntington's disease", 0, 0, 0, 1, "other"),
    ("F02.3", "Dementia in Parkinson's disease", 0, 0, 0, 1, "other"),
    ("F02.4", "Dementia in HIV disease", 0, 0, 0, 1, "other"),
    ("F02.8", "Dementia in other specified diseases classified elsewhere", 0, 0, 0, 1, "other"),
    ("F03", "Unspecified dementia", 0, 0, 0, 1, "unspecified"),
    ("F05.1", "Delirium superimposed on dementia", 0, 0, 0, 1, "other"),
    ("F10.6", "Alcohol-related amnesic syndrome", 0, 0, 0, 1, "other"),
    ("G30", "Alzheimer's disease", 1, 0, 0, 1, "alzheimer"),
    ("G30.0", "Alzheimer's disease with early onset", 1, 0, 0, 1, "alzheimer"),
    ("G30.1", "Alzheimer's disease with late onset", 1, 0, 0, 1, "alzheimer"),
    ("G30.8", "Other Alzheimer's disease", 1, 0, 0, 1, "alzheimer"),
    ("G30.9", "Alzheimer's disease unspecified", 1, 0, 0, 1, "alzheimer"),
    ("G31.0", "Circumscribed brain atrophy", 0, 0, 1, 1, "frontotemporal"),
    ("G31.1", "Senile degeneration of brain", 0, 0, 0, 1, "other"),
    ("G31.8", "Other specified degenerative diseases of nervous system", 0, 0, 0, 1, "other"),
    ("I67.3", "Binswanger's disease", 0, 1, 0, 1, "vascular"),
]

CODE_RE = re.compile(r"\b([A-TV-Z][0-9]{2}(?:\.[0-9A-Z]+)?)\b", re.I)


def normalise_eid(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def extract_icd10(value: object) -> str | None:
    if pd.isna(value):
        return None
    match = CODE_RE.search(str(value).upper())
    return match.group(1).upper() if match else None


def code_matches(observed: str, official: str) -> bool:
    return observed == official or (
        "." not in official and observed.startswith(official + ".")
    )


def classify_code(code: str) -> dict | None:
    if not isinstance(code, str) or not code:
        return None
    matches = [row for row in CODEBOOK if code_matches(code, row[0])]
    if not matches:
        return None
    # Prefer the most specific official code when both a parent and child match.
    row = max(matches, key=lambda item: len(item[0]))
    return {
        "official_code": row[0], "description": row[1],
        "alzheimer": row[2], "vascular": row[3], "frontotemporal": row[4],
        "all_cause": row[5], "etiology_group": row[6],
    }


def file_fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
    stat = path.stat()
    return {
        "path": str(path), "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "sha256_first_1MiB": digest.hexdigest(),
    }


def load_accelerometer_eids() -> list[str]:
    acc = pd.read_csv(ACC_PATH, usecols=["eid"], dtype={"eid": "string"})
    acc["eid"] = acc["eid"].map(normalise_eid)
    counts = acc["eid"].value_counts(sort=False)
    if not counts.eq(7).all():
        raise ValueError(f"Accelerometer file contains non-7-row eids: {(counts != 7).sum()}")
    return acc["eid"].drop_duplicates(keep="first").tolist()


def parse_hospital_records(accel_eids: set[str], chunksize: int) -> tuple[pd.DataFrame, dict]:
    icd_header = pd.read_csv(ICD_PATH, nrows=0).columns.tolist()
    date_header = pd.read_csv(DATE_PATH, nrows=0).columns.tolist()
    icd_by_index = {int(c.rsplit("a", 1)[1]): c for c in icd_header if re.fullmatch(r"p41270_a\d+", c)}
    date_by_index = {int(c.rsplit("a", 1)[1]): c for c in date_header if re.fullmatch(r"p41280_a\d+", c)}
    common_indices = sorted(set(icd_by_index) & set(date_by_index))
    if set(icd_by_index) != set(date_by_index):
        raise ValueError("Fields 41270 and 41280 do not expose identical array indices")

    audit = Counter()
    records = []
    icd_reader = pd.read_csv(ICD_PATH, dtype="string", chunksize=chunksize, low_memory=False)
    date_reader = pd.read_csv(DATE_PATH, dtype="string", chunksize=chunksize, low_memory=False)
    for chunk_number, (codes, dates) in enumerate(zip(icd_reader, date_reader), start=1):
        codes["eid"] = codes["eid"].map(normalise_eid)
        dates["eid"] = dates["eid"].map(normalise_eid)
        if not codes["eid"].equals(dates["eid"]):
            raise ValueError(f"41270/41280 eid or row order mismatch in chunk {chunk_number}")
        keep = codes["eid"].isin(accel_eids)
        codes, dates = codes.loc[keep], dates.loc[keep]
        audit["rows_in_accelerometer_cohort"] += len(codes)
        for idx in common_indices:
            code_col, date_col = icd_by_index[idx], date_by_index[idx]
            code_raw, date_raw = codes[code_col], dates[date_col]
            has_code, has_date = code_raw.notna() & code_raw.str.strip().ne(""), date_raw.notna() & date_raw.str.strip().ne("")
            audit["code_without_date"] += int((has_code & ~has_date).sum())
            audit["date_without_code"] += int((~has_code & has_date).sum())
            candidate = has_code
            if not candidate.any():
                continue
            extracted = code_raw.loc[candidate].map(extract_icd10)
            audit["nonempty_code_cells"] += int(candidate.sum())
            audit["unparseable_code_cells"] += int(extracted.isna().sum())
            relevant = extracted.map(classify_code)
            relevant_mask = relevant.notna()
            if not relevant_mask.any():
                continue
            positions = relevant.index[relevant_mask]
            parsed_dates = pd.to_datetime(date_raw.loc[positions], errors="coerce")
            audit["official_dementia_code_cells"] += len(positions)
            audit["official_dementia_code_missing_or_invalid_date"] += int(parsed_dates.isna().sum())
            for row_index, diagnosis_date in parsed_dates.items():
                info = relevant.loc[row_index]
                records.append({
                    "eid": codes.at[row_index, "eid"],
                    "array_index": idx,
                    "icd10_observed": extracted.loc[row_index],
                    "icd10_official_match": info["official_code"],
                    "icd10_description": info["description"],
                    "diagnosis_date": diagnosis_date,
                    "alzheimer": info["alzheimer"],
                    "vascular": info["vascular"],
                    "frontotemporal": info["frontotemporal"],
                    "etiology_group": info["etiology_group"],
                })
        if chunk_number == 1 or chunk_number % 10 == 0:
            logging.info("Parsed chunk %d; dementia records accumulated=%d", chunk_number, len(records))

    events = pd.DataFrame.from_records(records)
    if events.empty:
        raise RuntimeError("No dementia records found")
    audit["dementia_event_rows_before_deduplication"] = len(events)
    duplicate_cols = ["eid", "icd10_observed", "diagnosis_date"]
    audit["duplicate_eid_code_date_rows"] = int(events.duplicated(duplicate_cols).sum())
    events = events.sort_values(["eid", "diagnosis_date", "icd10_observed"], na_position="last")
    events = events.drop_duplicates(duplicate_cols, keep="first").reset_index(drop=True)
    audit["dementia_event_rows_after_deduplication"] = len(events)
    return events, {**audit, "array_indices": common_indices}


def earliest_event(events: pd.DataFrame, flag: str | None = None) -> pd.DataFrame:
    subset = events if flag is None else events[events[flag].eq(1)]
    subset = subset.dropna(subset=["diagnosis_date"]).sort_values(
        ["eid", "diagnosis_date", "icd10_observed"]
    )
    return subset.drop_duplicates("eid", keep="first").set_index("eid")


def build_outputs(accel_eids: list[str], events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_csv(BASELINE_PATH, dtype={"eid": "string"}, low_memory=False)
    wear = pd.read_csv(WEAR_PATH, dtype={"eid": "string"}, low_memory=False)
    for frame in (baseline, wear):
        frame["eid"] = frame["eid"].map(normalise_eid)
        if frame["eid"].duplicated().any():
            raise ValueError("Duplicate eid in baseline/wear data")
    wear = wear[["eid", "p90010"]].rename(columns={"p90010": "accelerometer_start_date"})
    wear["accelerometer_start_date"] = pd.to_datetime(wear["accelerometer_start_date"], errors="coerce")
    cohort = pd.DataFrame({"eid": accel_eids}).merge(wear, on="eid", how="left", validate="one_to_one")
    cohort = cohort.merge(baseline, on="eid", how="left", validate="one_to_one")
    if cohort["accelerometer_start_date"].isna().any():
        raise ValueError("Missing accelerometer start date in 7-day cohort")

    first = earliest_event(events)
    cohort["first_dementia_date"] = cohort["eid"].map(first["diagnosis_date"])
    cohort["first_dementia_icd10"] = cohort["eid"].map(first["icd10_observed"])
    cohort["first_dementia_description"] = cohort["eid"].map(first["icd10_description"])
    cohort["days_accel_to_dementia"] = (
        cohort["first_dementia_date"] - cohort["accelerometer_start_date"]
    ).dt.days.astype("Int64")
    days = cohort["days_accel_to_dementia"]
    cohort["dementia_state"] = np.select(
        [
            (days.notna() & days.le(0)).fillna(False).to_numpy(dtype=bool),
            days.between(1, WASHOUT_DAYS).fillna(False).to_numpy(dtype=bool),
            days.gt(WASHOUT_DAYS).fillna(False).to_numpy(dtype=bool),
        ],
        ["prevalent_at_accelerometer", "washout_1_365_days", "incident_after_365_days"],
        default="no_hospital_dementia_observed",
    )
    cohort["prevalent_dementia"] = cohort["dementia_state"].eq("prevalent_at_accelerometer").astype("int8")
    cohort["incident_dementia_after_1y"] = pd.Series(pd.NA, index=cohort.index, dtype="Int8")
    eligible = cohort["dementia_state"].isin(["incident_after_365_days", "no_hospital_dementia_observed"])
    cohort.loc[eligible, "incident_dementia_after_1y"] = cohort.loc[eligible, "dementia_state"].eq("incident_after_365_days").astype("int8")
    cohort["incident_analysis_eligible"] = eligible.astype("int8")
    cohort["dementia_definition_version"] = DEFINITION_VERSION
    cohort["dementia_ascertainment_source"] = "hospital_inpatient_41270_41280"

    subtype = cohort[["eid", "accelerometer_start_date"]].copy()
    for flag, stem in [("alzheimer", "alzheimers_disease"), ("vascular", "vascular_dementia"), ("frontotemporal", "frontotemporal_dementia")]:
        subfirst = earliest_event(events, flag)
        subtype[f"first_{stem}_date"] = subtype["eid"].map(subfirst["diagnosis_date"])
        subtype[f"first_{stem}_icd10"] = subtype["eid"].map(subfirst["icd10_observed"])
        subtype[f"days_accel_to_{stem}"] = (subtype[f"first_{stem}_date"] - subtype["accelerometer_start_date"]).dt.days.astype("Int64")
        d = subtype[f"days_accel_to_{stem}"]
        subtype[f"prevalent_{stem}"] = (d.notna() & d.le(0)).fillna(False).astype("int8")
        subtype[f"incident_{stem}_after_1y"] = pd.Series(pd.NA, index=subtype.index, dtype="Int8")
        subeligible = d.isna() | d.gt(WASHOUT_DAYS)
        subtype.loc[subeligible, f"incident_{stem}_after_1y"] = d.loc[subeligible].gt(WASHOUT_DAYS).fillna(False).astype("int8")
    subtype["any_multiple_recorded_subtypes"] = (
        subtype[["first_alzheimers_disease_date", "first_vascular_dementia_date", "first_frontotemporal_dementia_date"]].notna().sum(axis=1) > 1
    ).astype("int8")
    subtype["dementia_definition_version"] = DEFINITION_VERSION

    event_details = events.merge(
        cohort[["eid", "accelerometer_start_date"]], on="eid", how="left", validate="many_to_one"
    )
    event_details["days_accel_to_event"] = (event_details["diagnosis_date"] - event_details["accelerometer_start_date"]).dt.days.astype("Int64")
    return cohort, subtype, event_details


def save_codebook() -> pd.DataFrame:
    columns = ["icd10_code", "description", "alzheimer_subtype", "vascular_subtype", "frontotemporal_subtype", "all_cause_dementia", "etiology_group"]
    codebook = pd.DataFrame(CODEBOOK, columns=columns)
    codebook["matching_rule"] = np.where(codebook["icd10_code"].str.contains(r"\.", regex=True), "exact", "exact_or_descendant")
    codebook["source_url"] = OFFICIAL_SOURCE_URL
    codebook.to_csv(OUT_DIR / "dementia_icd10_codebook_ukb_official.csv", index=False, encoding="utf-8-sig")
    return codebook


def main() -> None:
    global ICD_PATH, DATE_PATH, ACC_PATH, BASELINE_PATH, WEAR_PATH, OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunksize", type=int, default=4000)
    parser.add_argument("--icd-csv", type=Path, default=ICD_PATH)
    parser.add_argument("--diagnosis-dates-csv", type=Path, default=DATE_PATH)
    parser.add_argument("--activity-csv", type=Path, default=ACC_PATH)
    parser.add_argument("--baseline-csv", type=Path, default=BASELINE_PATH)
    parser.add_argument("--wear-dates-csv", type=Path, default=WEAR_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    ICD_PATH, DATE_PATH = args.icd_csv, args.diagnosis_dates_csv
    ACC_PATH, BASELINE_PATH, WEAR_PATH = args.activity_csv, args.baseline_csv, args.wear_dates_csv
    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(OUT_DIR / "dementia_label_processing.log", mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Starting dementia label build")
    inputs = [ICD_PATH, DATE_PATH, ACC_PATH, BASELINE_PATH, WEAR_PATH]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    codebook = save_codebook()
    accel_eids = load_accelerometer_eids()
    logging.info("7-day accelerometer participants=%d", len(accel_eids))
    events, parsing_audit = parse_hospital_records(set(accel_eids), args.chunksize)
    cohort, subtype, event_details = build_outputs(accel_eids, events)

    cohort.to_csv(OUT_DIR / "ukb_dementia_labels_covariates_N80990.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    subtype.to_csv(OUT_DIR / "ukb_dementia_subtypes_N80990.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    event_details.to_csv(OUT_DIR / "ukb_dementia_hospital_events_long.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    state_counts = cohort["dementia_state"].value_counts(dropna=False).rename_axis("dementia_state").reset_index(name="n")
    state_counts["percent_of_N80990"] = state_counts["n"] / len(cohort) * 100
    state_counts.to_csv(OUT_DIR / "dementia_state_summary.csv", index=False, encoding="utf-8-sig")
    subtype_rows = []
    for stem in ["alzheimers_disease", "vascular_dementia", "frontotemporal_dementia"]:
        subtype_rows.append({
            "subtype": stem,
            "ever_recorded": int(subtype[f"first_{stem}_date"].notna().sum()),
            "prevalent_at_accelerometer": int(subtype[f"prevalent_{stem}"].sum()),
            "incident_after_365_days": int(subtype[f"incident_{stem}_after_1y"].eq(1).sum()),
            "washout_1_365_days": int(subtype[f"days_accel_to_{stem}"].between(1, WASHOUT_DAYS).sum()),
        })
    pd.DataFrame(subtype_rows).to_csv(OUT_DIR / "dementia_subtype_summary.csv", index=False, encoding="utf-8-sig")

    audit = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "definition_version": DEFINITION_VERSION,
        "official_source_url": OFFICIAL_SOURCE_URL,
        "scope_limitation": "Hospital inpatient ICD-10 only; self-report and death registry are not included in 41270/41280.",
        "washout_days": WASHOUT_DAYS,
        "input_fingerprints": [file_fingerprint(p) for p in inputs],
        "n_accelerometer_participants": len(accel_eids),
        "n_covariate_rows": len(cohort),
        "n_subtype_rows": len(subtype),
        "n_official_codebook_rows": len(codebook),
        "parsing_audit": parsing_audit,
        "dementia_state_counts": dict(cohort["dementia_state"].value_counts()),
        "baseline_missing_counts": {c: int(cohort[c].isna().sum()) for c in baseline_columns()},
    }
    with (OUT_DIR / "dementia_processing_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2, default=str)
    logging.info("State counts: %s", audit["dementia_state_counts"])
    logging.info("Finished; outputs written to %s", OUT_DIR)


def baseline_columns() -> list[str]:
    return [
        "BMI", "Townsend_index", "enrollment_age", "gender", "Disable_long",
        "education_level", "smoking", "alcohol_use", "ethic", "overall_health",
        "is_dead", "Followingyears_birthday", "Light_Total_Hour", "MVPA_Total_Hour",
        "Sedentary_Total_Hour", "Sleep_Total_Hour",
    ]


if __name__ == "__main__":
    main()
