"""Measurement summaries and diagnostics for frozen clinical analyses."""
from __future__ import annotations

import warnings

import numpy as np


def full_sequence_iv(activity, observed):
    """One-minute intradaily variability over the ordered seven-day record.

    The numerator is the mean squared difference between adjacent observed
    minutes; the denominator is the population variance of observed minutes.
    Day boundaries are retained. Missing gaps are never joined or imputed.
    For a complete N-minute record this is
    N * sum(diff(x)**2) / ((N - 1) * sum((x - mean(x))**2)).
    """
    values = np.asarray(activity, dtype=np.float64).reshape(len(activity), -1)
    if values.shape[1] != 10080:
        raise ValueError("IV expects seven chronological days at one-minute resolution (10,080 values)")
    mask = np.asarray(observed, dtype=bool).reshape(values.shape) & np.isfinite(values)
    adjacent = mask[:, :-1] & mask[:, 1:]
    count = mask.sum(axis=1)
    pairs = adjacent.sum(axis=1)
    mean = np.divide(np.where(mask, values, 0.).sum(axis=1), count,
                     out=np.full(len(values), np.nan), where=count > 0)
    variance = np.divide(np.where(mask, (values - mean[:, None]) ** 2, 0.).sum(axis=1),
                         count, out=np.full(len(values), np.nan), where=count > 0)
    differences = np.where(adjacent, np.diff(values, axis=1) ** 2, 0.).sum(axis=1)
    numerator = np.divide(differences, pairs, out=np.full(len(values), np.nan), where=pairs > 0)
    iv = np.divide(numerator, variance, out=np.full(len(values), np.nan),
                   where=(count >= 2) & (pairs > 0) & (variance > 0))
    return iv, {"observed_minutes": count, "observed_adjacent_minute_pairs": pairs,
                "observed_fraction": count / values.shape[1],
                "raw_activity_variance": variance}


def distribution_summary(values):
    """Report the untransformed scale, using ddof=0 as in concept models."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"finite_n": 0, "mean": np.nan, "sd": np.nan, "min": np.nan,
                "p01": np.nan, "p25": np.nan, "median": np.nan,
                "p75": np.nan, "p99": np.nan, "max": np.nan,
                "iqr": np.nan, "outside_three_iqr_n": 0}
    p01, q1, median, q3, p99 = np.quantile(values, [.01, .25, .5, .75, .99])
    iqr = q3 - q1
    return {"finite_n": len(values), "mean": values.mean(), "sd": values.std(ddof=0),
            "min": values.min(), "p01": p01, "p25": q1, "median": median,
            "p75": q3, "p99": p99, "max": values.max(), "iqr": iqr,
            "outside_three_iqr_n": int(((values < q1 - 3 * iqr) |
                                        (values > q3 + 3 * iqr)).sum())}


def binomial_term(y, features, names, target):
    """Unpenalized binomial GLM, with convergence and influence diagnostics.

    Influence measures are descriptive diagnostics and do not remove records
    or change the fitted coefficients. A failed fit remains an explicit row.
    """
    import statsmodels.api as sm

    features = np.asarray(features, dtype=float)
    y = np.asarray(y, dtype=float)
    result = {"coefficient": np.nan, "se": np.nan, "odds_ratio": np.nan,
              "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan,
              "converged": False, "fit_n": len(y), "fit_events": int(y.sum()),
              "design_columns": features.shape[1] + 1,
              "design_rank": np.nan, "design_condition_number": np.nan,
              "max_leverage": np.nan, "max_cooks_distance": np.nan,
              "max_abs_pearson_residual": np.nan, "fit_warnings": "", "error": ""}
    try:
        design = sm.add_constant(features, has_constant="add")
        result["design_rank"] = int(np.linalg.matrix_rank(design))
        result["design_condition_number"] = float(np.linalg.cond(design))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = sm.GLM(y, design, family=sm.families.Binomial()).fit()
        j = names.index(target) + 1
        with np.errstate(over="ignore", invalid="ignore"):
            result.update(coefficient=fit.params[j], se=fit.bse[j],
                          odds_ratio=np.exp(fit.params[j]),
                          ci_low=np.exp(fit.params[j] - 1.96 * fit.bse[j]),
                          ci_high=np.exp(fit.params[j] + 1.96 * fit.bse[j]),
                          p_value=fit.pvalues[j])
        result["converged"] = bool(fit.converged)
        result["fit_warnings"] = " | ".join(str(item.message) for item in caught)
        result["max_abs_pearson_residual"] = float(np.nanmax(np.abs(fit.resid_pearson)))
        try:
            influence = fit.get_influence(observed=True)
            result["max_leverage"] = float(np.nanmax(influence.hat_matrix_diag))
            result["max_cooks_distance"] = float(np.nanmax(influence.cooks_distance[0]))
        except Exception as exc:
            result["influence_error"] = repr(exc)
    except Exception as exc:
        result["error"] = repr(exc)
    return result
