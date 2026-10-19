#!/usr/bin/env python3
"""Prepare outcome-blind positive-value NHATS-to-UKB activity mappings.

Only participant IDs and minute activity values are read. The UKB reference
is restricted to the supplied development IDs. This script does not load a
disease model, outcomes, traditional-PA summaries, or a learned adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MINUTES = [str(i) for i in range(1, 1441)]


def normalize_eid(values):
    text = pd.Series(values, dtype="string").str.strip()
    digits = text.str.extract(r"(\d{6,})", expand=False)
    return digits.fillna(text.str.replace(r"\.0+$", "", regex=True)).astype("string")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_reference(path, development_ids_path, chunksize):
    ids = pd.read_csv(development_ids_path, usecols=["eid"], dtype={"eid": "string"})
    ids["eid"] = normalize_eid(ids.eid)
    if ids.empty or ids.eid.isna().any() or ids.eid.duplicated().any():
        raise ValueError("UKB development IDs must be nonempty, unique participant IDs")
    selected = set(ids.eid)
    observed = set()
    positives = []
    for frame in pd.read_csv(path, usecols=["eid", *MINUTES], dtype={"eid": "string"}, chunksize=chunksize):
        frame["eid"] = normalize_eid(frame.eid)
        frame = frame.loc[frame.eid.isin(selected)]
        observed.update(frame.eid)
        values = frame[MINUTES].to_numpy(float)
        if np.isinf(values).any() or np.any(values[np.isfinite(values)] < 0):
            raise ValueError("UKB activity must be nonnegative; missing observations must be blank/NaN")
        values = values[np.isfinite(values) & (values > 0)]
        if values.size:
            positives.append(values)
    if observed != selected:
        raise ValueError(f"UKB activity reference lacks {len(selected - observed)} supplied development IDs")
    if not positives:
        raise ValueError("UKB development reference has no observed positive activity")
    return np.concatenate(positives), len(selected)


def positive_values(values, label):
    if np.isinf(values).any() or np.any(values[np.isfinite(values)] < 0):
        raise ValueError(f"{label}: activity must be nonnegative; encode missing observations as blank/NaN")
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size < 2:
        raise ValueError(f"{label}: at least two observed positive values are required")
    return positive


def median_iqr(values):
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Positive-value IQR must be finite and strictly positive")
    return {"location": float(median), "scale": scale}


def run(cfg):
    nhats_path = Path(cfg.nhats_raw_csv).expanduser().resolve()
    reference_path = Path(cfg.ukb_reference_csv).expanduser().resolve()
    ids_path = Path(cfg.ukb_development_ids_csv).expanduser().resolve()
    out = Path(cfg.output_dir).expanduser().resolve()
    if cfg.quantile_points < 2 or cfg.chunksize < 1:
        raise ValueError("Quantile points must be at least two and chunksize must be positive")
    outputs = {"positive_robust": out / "nhats_r11_daily_7x1440_positive_robust.csv",
               "positive_quantile": out / "nhats_r11_daily_7x1440_positive_quantile.csv"}
    sidecars = [out / "alignment_parameters.json", out / "activity_manifest.json"]
    adapter = Path(cfg.learned_adapter_csv).expanduser().resolve() if cfg.learned_adapter_csv else None
    inputs = [nhats_path, reference_path, ids_path] + ([adapter] if adapter else [])
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if set(inputs) & set([*outputs.values(), *sidecars]):
        raise ValueError("Output paths must not replace input files")
    for path in [*outputs.values(), *sidecars]:
        if path.exists() and not cfg.overwrite:
            raise FileExistsError(f"{path} exists; choose a separate output directory or explicitly use --overwrite")

    frame = pd.read_csv(nhats_path, usecols=["eid", *MINUTES], dtype={"eid": "string"})
    frame = frame[["eid", *MINUTES]]
    frame["eid"] = normalize_eid(frame.eid)
    if frame.empty or frame.eid.isna().any() or frame.eid.eq("").any():
        raise ValueError("NHATS activity requires nonempty participant IDs")
    if not frame.eid.value_counts().eq(7).all():
        raise ValueError("NHATS activity requires exactly seven chronological daily rows per participant")
    values = frame[MINUTES].to_numpy(float)
    external = positive_values(values, "NHATS")
    reference, development_n = read_reference(reference_path, ids_path, cfg.chunksize)
    ukb = median_iqr(reference)
    nhats = median_iqr(external)
    mask = np.isfinite(values) & (values > 0)

    # Main manuscript equation (6) acts in the original positive-value domain.
    robust = values.copy()
    mapped = ukb["location"] + (ukb["scale"] / nhats["scale"]) * (values[mask] - nhats["location"])
    nonpositive_n = int(np.count_nonzero(mapped <= 0))
    if nonpositive_n and cfg.robust_nonpositive == "error":
        raise ValueError(f"Affine positive-robust mapping produces {nonpositive_n} nonpositive values. Specify a measurement-domain policy explicitly; --robust-nonpositive clip-zero is available only when justified by the analysis protocol.")
    robust[mask] = np.maximum(mapped, 0) if cfg.robust_nonpositive == "clip-zero" else mapped

    probabilities = np.linspace(0.0, 1.0, cfg.quantile_points)
    external_knots = np.quantile(external, probabilities, method="linear")
    ukb_knots = np.quantile(reference, probabilities, method="linear")
    unique_knots, first, counts = np.unique(external_knots, return_index=True, return_counts=True)
    if unique_knots.size < 2:
        raise ValueError("NHATS positive quantiles have fewer than two distinct values")
    # Tied source knots use the mean of corresponding target quantiles.
    target_knots = np.add.reduceat(ukb_knots, first) / counts
    quantile = values.copy()
    quantile[mask] = np.interp(values[mask], unique_knots, target_knots,
                              left=target_knots[0], right=target_knots[-1])
    out.mkdir(parents=True, exist_ok=True)
    for name, mapped_values in [("positive_robust", robust), ("positive_quantile", quantile)]:
        mapped_frame = frame.copy()
        mapped_frame[MINUTES] = mapped_values
        mapped_frame.to_csv(outputs[name], index=False)
    manifest = {"raw_count": str(nhats_path), **{name: str(path) for name, path in outputs.items()}}
    if adapter:
        manifest["learned_adapter"] = str(adapter)
    parameters = {
        "domain": "original_activity_values_before_log1p", "outcomes_used": False,
        "ukb_reference_scope": "supplied_development_participant_ids", "ukb_development_n": development_n,
        "nhats_n": int(frame.eid.nunique()), "reference_positive_count": int(reference.size),
        "nhats_positive_count": int(external.size),
        "robust": {"estimator": cfg.robust_estimator, "UKB": ukb, "NHATS": nhats,
                   "formula": "mU + (sU / sE) * (A - mE), for observed A > 0",
                   "nonpositive_policy": cfg.robust_nonpositive, "nonpositive_mapped_count": nonpositive_n},
        "quantile": {"points": cfg.quantile_points, "probabilities": probabilities.tolist(),
                     "quantile_method": "linear", "interpolation": "piecewise_linear",
                     "tied_source_knots": "mean_target_quantile", "tail_policy": cfg.quantile_tail,
                     "source_knots": unique_knots.tolist(), "target_knots": target_knots.tolist()},
        "original_zeros": "preserved", "missing_observations": "preserved_as_NaN",
        "subsequent_model_preprocessing": "frozen_UKB_log1p_location_and_scale",
        "input_files": {str(path): {"sha256": file_sha256(path)} for path in inputs},
        "learned_adapter": "prealigned_input_only" if adapter else "not_supplied",
    }
    (out / "activity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "alignment_parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")


def args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nhats-raw-csv", required=True)
    parser.add_argument("--ukb-reference-csv", required=True, help="UKB minute-activity CSV; selection uses development IDs")
    parser.add_argument("--ukb-development-ids-csv", required=True, help="CSV with an eid column containing development participants only")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--robust-estimator", required=True, choices=["median-iqr"], help="Explicit estimator choice for robust location and scale")
    parser.add_argument("--robust-nonpositive", required=True, choices=["error", "clip-zero"])
    parser.add_argument("--quantile-points", required=True, type=int)
    parser.add_argument("--quantile-tail", required=True, choices=["clip"])
    parser.add_argument("--learned-adapter-csv", help="Optional genuinely prealigned adapter CSV; no adapter is fitted here")
    parser.add_argument("--chunksize", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(args())
