"""Fixed-window outcome definition shared by descriptive and predictive analyses."""
from __future__ import annotations

import numpy as np
import pandas as pd


def fixed_window_cohort(frame, horizon_years=7.0, landmark_days=365):
    """Select events after the landmark and observable horizon non-events.

    enrollment_age is age at the wearable assessment. Followingyears_birthday
    is age at the end of recorded follow-up and is used only for eligibility.
    A later recorded dementia event also establishes observation through the
    horizon. Outcome ascertainment must use the same record coverage as the
    follow-up variable.
    """
    frame = frame.copy()
    frame["eid"] = frame["eid"].astype("string").str.strip()
    if frame["eid"].isna().any() or frame["eid"].duplicated().any():
        raise ValueError("Cohort participant IDs must be unique and non-missing")
    baseline = pd.to_datetime(frame["accelerometer_start_date"], errors="coerce")
    diagnosis = pd.to_datetime(frame["first_dementia_date"], errors="coerce")
    days = (diagnosis - baseline).dt.days.astype(float)
    if "days_accel_to_dementia" in frame:
        days = days.fillna(pd.to_numeric(frame["days_accel_to_dementia"], errors="coerce"))
    followup = (
        pd.to_numeric(frame["Followingyears_birthday"], errors="coerce")
        - pd.to_numeric(frame["enrollment_age"], errors="coerce")
    )
    horizon = int(round(365.25 * horizon_years))
    eligible = frame["incident_analysis_eligible"].eq(1) & baseline.notna()
    event = eligible & days.gt(landmark_days) & days.le(horizon)
    non_event = eligible & ~event & (
        days.gt(horizon) | (days.isna() & followup.ge(horizon_years))
    )
    selected = frame.loc[event | non_event].copy()
    selected["target"] = event.loc[selected.index].astype(np.int8)
    selected["days_accel_to_dementia"] = days.loc[selected.index]
    selected["available_followup_years"] = followup.loc[selected.index]
    return selected.reset_index(drop=True)
