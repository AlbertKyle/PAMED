from __future__ import annotations

import argparse, json, logging, math, re, shutil
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA_ROOT, RESULTS_ROOT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from statsmodels.stats.proportion import proportion_confint

import plotting_utils as core


def safe(s): return re.sub(r"[^A-Za-z0-9_.-]+","_",str(s)).strip("_")


def save(fig, module, name, title, source, use="supplement"):
    core.save_figure(fig,module,name,title,source,"main" if use=="main" else "supplement")


def label_column(d):
    for c in ["model","comparison","subtype","concept","modifier","outcome","group","analysis","source","term","window","level"]:
        if c in d:return c
    return None


def read_csv(path, max_rows=200000):
    try:return pd.read_csv(path,nrows=max_rows,low_memory=False)
    except Exception as e:
        logging.warning("Skipping %s: %r",path,e);return None


def metric_forest(d, metric, module, stem, source):
    lab=label_column(d)
    if not lab or metric not in d:return
    q=d.dropna(subset=[metric]).copy()
    if not len(q):return
    q["_label"]=q[lab].astype(str)
    for extra in ["harmonization","endpoint","split","analysis","level","term","subtype","seed"]:
        if extra in q and extra!=lab:q["_label"]+= " | "+q[extra].astype(str)
    q=q.drop_duplicates("_label").sort_values(metric).tail(35)
    lo=q.get(f"{metric}_ci_low", q.get("ci_low", q[metric]))
    hi=q.get(f"{metric}_ci_high", q.get("ci_high", q[metric]))
    x=q[metric].to_numpy(float);y=np.arange(len(q));fig,ax=core.new_fig(8.4,max(5,.34*len(q)+1.8))
    ax.errorbar(x,y,xerr=[np.maximum(0,x-lo.to_numpy(float)),np.maximum(0,hi.to_numpy(float)-x)],fmt="o",ms=5.5,capsize=2.5,color=core.PALETTE["hybrid"],ecolor="#88959D")
    core.apply_effect_reference(ax, metric)
    ax.set_yticks(y, [core.readable_label(x) for x in q._label])
    core.finish(ax, stem.replace("_", " ").title(), core.metric_axis_label(metric), "", False)
    save(fig, module, stem, stem, source, "main")


def performance_suite(d, module, stem, source):
    for metric in ["auroc", "auprc", "calibration_intercept", "calibration_slope", "sensitivity", "specificity"]:
        if metric in d:
            metric_forest(d, metric, module, f"{stem}_{metric}_forest", source)


def repertoire_contrasts(cohort, repeats=2000, seed=20260820):
    """Figure 2f: evaluation-set standardized mean differences and intervals."""
    module = "00B_pa_pattern_mining"
    path = core.RESULTS / "14_rhythm_knowledge" / "posthoc_rhythm_concepts.csv"
    concepts = pd.read_csv(path, dtype={"eid": "string"})
    evaluation = cohort.loc[cohort.split.eq("test"), ["eid", "target"]].merge(concepts, on="eid", validate="one_to_one")
    descriptors = ["day_mean_sd", "sameclock_between_day_sd", "adjacent_day_absolute_change", "mean_pairwise_day_correlation", "interdaily_stability"]
    rng = np.random.default_rng(seed)
    rows = []
    for descriptor in descriptors:
        control = pd.to_numeric(evaluation.loc[evaluation.target.eq(0), descriptor], errors="coerce").dropna().to_numpy()
        incident = pd.to_numeric(evaluation.loc[evaluation.target.eq(1), descriptor], errors="coerce").dropna().to_numpy()
        if min(len(control), len(incident)) < 2:
            raise ValueError(f"Insufficient observed descriptor values: {descriptor}")
        effect = core.standardized_difference(control, incident)
        bootstrap = np.asarray([core.standardized_difference(rng.choice(control, len(control), replace=True), rng.choice(incident, len(incident), replace=True)) for _ in range(repeats)])
        low, high = np.nanquantile(bootstrap, [.025, .975])
        rows.append({"descriptor": descriptor, "standardized_mean_difference": effect, "ci_low": low, "ci_high": high, "controls": len(control), "cases": len(incident)})
    results = pd.DataFrame(rows)
    fig, ax = core.new_fig(8, 5)
    y = np.arange(len(results))
    ax.errorbar(results.standardized_mean_difference, y, xerr=[np.maximum(0, results.standardized_mean_difference - results.ci_low), np.maximum(0, results.ci_high - results.standardized_mean_difference)], fmt="o", capsize=3, color=core.PALETTE["hybrid"])
    ax.axvline(0, linestyle="--", color="#879198")
    ax.set_yticks(y, [x.replace("_", " ") for x in results.descriptor])
    core.finish(ax, "Cross-day behavioural descriptors", "Incident − control standardized mean difference (95% CI)", "", False)
    save(fig, module, "cross_day_repertoire_contrasts", "Cross-day descriptor mean contrasts", str(path), "main")
    destination = core.OUT / module
    destination.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination / "cross_day_repertoire_contrasts.csv", index=False)
    metadata = {"sample": "UKB fixed internal evaluation set; actual non-missing counts are recorded per descriptor", "numerator": "mean(incident) − mean(control)", "denominator": "sqrt((sample_variance(control) + sample_variance(incident))/2)", "interval": "2.5th and 97.5th percentiles from within-outcome participant bootstrap; both numerator and denominator recomputed in each replicate", "bootstrap_repeats": repeats, "seed": seed, "adjustment": "Unadjusted descriptive contrast; distinct from age–sex-adjusted logistic ORs", "other_panels": "The standardized median contrasts are separate descriptive results without confidence intervals."}
    (destination / "cross_day_repertoire_contrasts_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def interpretation_panels(cohort):
    module = "13_hierarchical_shapley"; root = core.RESULTS / module
    temporal = pd.read_csv(root / "temporal_shapley_7day_x_4period.csv")
    signed = temporal.pivot_table(index=["day", "clock_start"], columns="group", values="mean_shapley_probability")
    control_column = "matched_control" if "matched_control" in signed else "control"
    if control_column not in signed or "incident" not in signed:
        raise ValueError("Temporal summaries must identify incident and matched-control groups")
    contrast = (signed["incident"] - signed[control_column]).unstack("clock_start")
    fig, ax = core.new_fig(8, 5)
    sns.heatmap(contrast, cmap=core.DIV_CMAP, center=0, annot=True, fmt=".3f", cbar_kws={"label": "Incident − control Shapley contribution"}, ax=ax)
    ax.set_xlabel("6-h clock-period start (hour)"); ax.set_ylabel("Monitoring day")
    save(fig, module, "temporal_signed_outcome_contrast", "Signed temporal attribution contrast", str(root), "main")
    individual = pd.read_csv(root / "modality_shapley_individual.csv")
    fig, ax = core.new_fig(7, 6)
    scatter = ax.scatter(individual.shapley_gru, individual.shapley_mamba, c=individual.full_probability, cmap=core.SEQ_CMAP, s=14, alpha=.45)
    fig.colorbar(scatter, ax=ax, label="PAMED predicted probability")
    core.finish(ax, "Activity-pathway evidence", "GRU–FCN Shapley contribution", "ConvContext–Mamba Shapley contribution", False)
    save(fig, module, "source_complementarity", "Activity-pathway complementarity", str(root), "supplement")
    occlusion_path = core.RESULTS / "09_interpretability" / "temporal_occlusion_summary.csv"
    occlusion = pd.read_csv(occlusion_path)
    days = occlusion[occlusion.window.str.fullmatch(r"day_[1-7]")].copy()
    days["day"] = days.window.str.extract(r"day_(\d)").astype(int)
    fig, ax = core.new_fig()
    for i, (group, frame) in enumerate(days.groupby("group")):
        frame = frame.sort_values("day")
        ax.plot(frame.day, frame.mean_delta_logit, "o-", label=group, color=core.SERIES[i % len(core.SERIES)])
    ax.axhline(0, color="#879198", linewidth=.8)
    core.finish(ax, "Whole-day occlusion", "Occluded monitoring day", "Original − occluded risk logit")
    save(fig, module, "occlusion_whole_day", "Whole-day occlusion", str(occlusion_path), "main")
    windows = occlusion[occlusion.group.eq("incident") & occlusion.window.str.fullmatch(r"day_[1-7]_\d{2}_\d{2}")].copy()
    extracted = windows.window.str.extract(r"day_(\d)_(\d{2})_(\d{2})")
    windows["day"] = extracted[0].astype(int); windows["period"] = extracted[1] + "–" + extracted[2]
    matrix = windows.pivot(index="day", columns="period", values="mean_delta_logit")
    fig, ax = core.new_fig(8, 5)
    sns.heatmap(matrix, cmap=core.DIV_CMAP, center=0, annot=True, fmt=".3f", cbar_kws={"label": "Original − occluded risk logit"}, ax=ax)
    ax.set_xlabel("Occluded 6-h clock period"); ax.set_ylabel("Monitoring day")
    save(fig, module, "occlusion_incident_day_period", "Incident-case day-period occlusion", str(occlusion_path), "main")
    pcs = cohort[cohort.split.eq("test")].dropna(subset=["latent_pc01", "latent_pc02", "latent_pc03"])
    sample = pcs.sample(min(5000, len(pcs)), random_state=20260820)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sample.latent_pc01, sample.latent_pc02, sample.latent_pc03, c=sample.target, cmap=core.DIV_CMAP, s=6, alpha=.35)
    ax.set_xlabel("PC1", labelpad=10); ax.set_ylabel("PC2", labelpad=10); ax.set_zlabel("PC3", labelpad=10)
    ax.tick_params(labelsize=8, pad=1)
    ax.set_box_aspect((1.25, 1.25, 1)); fig.subplots_adjust(left=.05, right=.84, bottom=.10, top=.95)
    save(fig, module, "latent_pc_3d", "Evaluation representation in development-fitted PCs", str(core.RESULTS / "11_representations"), "main")
    concept_path = core.RESULTS / "14_rhythm_knowledge" / "posthoc_rhythm_concepts.csv"
    concepts = pd.read_csv(concept_path, dtype={"eid": "string"})
    evaluation = cohort.loc[cohort.split.eq("test"), ["eid", "target"]].merge(concepts, on="eid", validate="one_to_one")
    shown = ["overall_activity", "sameclock_between_day_sd", "mean_pairwise_day_correlation", "intradaily_variability", "interdaily_stability", "relative_amplitude"]
    normalized = evaluation[shown].apply(pd.to_numeric, errors="coerce")
    median = normalized.median(); scale = (normalized.quantile(.75) - normalized.quantile(.25)).replace(0, np.nan)
    normalized = ((normalized - median) / scale).clip(-4, 4)
    normalized["Outcome"] = np.where(evaluation.target.eq(1), "Incident", "Control")
    long = normalized.melt(id_vars="Outcome", var_name="Descriptor", value_name="Robust-scaled value")
    fig, ax = core.new_fig(9, 6)
    sns.violinplot(data=long, x="Robust-scaled value", y="Descriptor", hue="Outcome", inner=None, cut=0, split=True, palette=[core.PALETTE["control"], core.PALETTE["incident"]], ax=ax)
    save(fig, module, "rhythm_descriptor_distributions", "Evaluation descriptor distributions", str(concept_path), "supplement")
    numeric = evaluation.drop(columns=["eid", "target"]).apply(pd.to_numeric, errors="coerce")
    contrast = (numeric[evaluation.target.eq(1)].median() - numeric[evaluation.target.eq(0)].median()) / numeric.std(ddof=0).replace(0, np.nan)
    contrast = contrast.dropna().sort_values()
    fig, ax = core.new_fig(8, 8)
    ax.barh([x.replace("_", " ") for x in contrast.index], contrast, color=core.PALETTE["incident"])
    ax.axvline(0, linestyle="--", color="#879198")
    core.finish(ax, "Standardized median contrasts", "(Incident median − control median) / evaluation SD", "", False)
    save(fig, module, "rhythm_standardized_median_differences", "Standardized median contrasts; no CI", str(concept_path), "supplement")


def seed_summary():
    path = core.RESULTS / "05_stability" / "per_seed_test_metrics.csv"
    data = pd.read_csv(path)
    for metric in ["auroc", "auprc"]:
        fig, ax = core.new_fig(8, 5)
        for i, (model, frame) in enumerate(data.groupby("model")):
            x = np.full(len(frame), i) + np.linspace(-.12, .12, len(frame))
            ax.scatter(x, frame[metric], s=30, color=core.SERIES[i % len(core.SERIES)])
            ax.errorbar(i, frame[metric].mean(), yerr=[[frame[metric].mean() - frame[metric].min()], [frame[metric].max() - frame[metric].mean()]], fmt="D", color=core.PALETTE["hybrid"], capsize=4)
        models = list(data.groupby("model").groups)
        ax.set_xticks(range(len(models)), [core.metric_label(model) for model in models], rotation=25, ha="right")
        core.finish(ax, "Repeated-fit discrimination", "", core.metric_axis_label(metric), False)
        save(fig, "05_stability", f"three_seed_mean_sd_{metric}", "Seed fits with mean and observed range (not confidence intervals)", str(path), "supplement")


if __name__ == "__main__":
    from generate_figures import main
    main()
