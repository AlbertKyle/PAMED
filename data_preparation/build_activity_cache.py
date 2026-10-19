#!/usr/bin/env python3
"""Cache participant-aligned seven-day minute activity and observation masks."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

MINUTES_PER_WEEK = 7 * 1440


def build_cache(activity_csv, output_dir, id_column="eid", observed_csv=None, chunksize=256):
    """Read a wide CSV already aligned to seven local-calendar days.

    Minute columns must be chronologically ordered. A separate observation-mask
    CSV, when supplied, must have identical IDs and minute columns. Measured zero
    remains observed; absent, negative or non-finite activity is missing.
    """
    header = pd.read_csv(activity_csv, nrows=0).columns.tolist()
    if id_column not in header:
        raise ValueError(f"Participant ID column is missing: {id_column}")
    minutes = [name for name in header if name != id_column]
    if len(minutes) != MINUTES_PER_WEEK:
        raise ValueError("Activity input requires exactly 10,080 ordered minute columns")
    ids = pd.read_csv(activity_csv, usecols=[id_column], dtype={id_column: "string"})[id_column].str.strip()
    if ids.isna().any() or ids.duplicated().any() or len(ids) == 0:
        raise ValueError("Participant IDs must be unique and non-missing")
    if observed_csv:
        mask_header = pd.read_csv(observed_csv, nrows=0).columns.tolist()
        mask_ids = pd.read_csv(observed_csv, usecols=[id_column], dtype={id_column: "string"})[id_column].str.strip()
        if mask_header != header or mask_ids.tolist() != ids.tolist():
            raise ValueError("Observation-mask input must match activity column and participant order")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / name for name in ("activity_7d.npy", "observed_7d.npy", "eids.npy")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Use an empty cache directory to avoid replacing study input arrays")
    activity = np.lib.format.open_memmap(outputs[0], mode="w+", dtype=np.float32,
                                         shape=(len(ids), 7, 1440))
    observed = np.lib.format.open_memmap(outputs[1], mode="w+", dtype=np.uint8,
                                         shape=(len(ids), 7, 1440))
    mask_reader = pd.read_csv(observed_csv, chunksize=chunksize, dtype={id_column: "string"}) if observed_csv else None
    offset = 0
    for frame in pd.read_csv(activity_csv, chunksize=chunksize, dtype={id_column: "string"}):
        values = frame[minutes].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        valid = np.isfinite(values) & (values >= 0)
        if mask_reader is not None:
            mask = next(mask_reader)[minutes].apply(pd.to_numeric, errors="coerce").to_numpy()
            if not np.isin(mask, [0, 1]).all():
                raise ValueError("Observation mask must contain only 0 and 1")
            valid &= mask.astype(bool)
        end = offset + len(frame)
        activity[offset:end] = np.where(valid, values, 0).reshape(-1, 7, 1440)
        observed[offset:end] = valid.reshape(-1, 7, 1440)
        offset = end
    activity.flush()
    observed.flush()
    np.save(outputs[2], ids.to_numpy(dtype=str), allow_pickle=False)
    metadata = {"participants": len(ids), "shape": [len(ids), 7, 1440],
                "time_alignment": "seven input local-calendar days; chronological minute columns",
                "observation_channel": "1 observed; 0 missing", "source_units": "unchanged",
                "normalization": "applied later using development participants"}
    (output_dir / "activity_cache.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity-csv", type=Path, required=True)
    parser.add_argument("--observed-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-column", default="eid")
    parser.add_argument("--chunksize", type=int, default=256)
    args = parser.parse_args()
    build_cache(args.activity_csv, args.output_dir, args.id_column, args.observed_csv, args.chunksize)
