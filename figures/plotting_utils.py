from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import warnings
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA_ROOT, RESULTS_ROOT

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import font_manager
import seaborn as sns
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    auc,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from statsmodels.stats.proportion import proportion_confint


warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = DATA_ROOT
RESULTS = RESULTS_ROOT
DESC = RESULTS_ROOT / "descriptive"
EXT_BASE = DATA_ROOT / "external_validation"
OUT = RESULTS_ROOT / "figures"

PALETTE = {
    "control": "#496C83",
    "incident": "#8A6668",
    "alzheimers_disease": "#657E91",
    "vascular_dementia": "#66857F",
    "frontotemporal_dementia": "#7D8290",
    "unclassified_or_other_dementia": "#9AA3A8",
    "gru": "#385D75",
    "mamba": "#587681",
    "clinical": "#777F86",
    "hybrid": "#294D63",
    "external": "#527D78",
    "neutral": "#747D84",
}
SERIES = ["#294D63", "#496C83", "#66857F", "#7D8290", "#8A6668", "#82958D", "#68747C", "#A5ADB1"]
DIV_CMAP = mpl.colors.LinearSegmentedColormap.from_list("navy_gray_teal", ["#294D63", "#EEF1F2", "#527D78"])
SEQ_CMAP = mpl.colors.LinearSegmentedColormap.from_list("steel_scale", ["#E8EEF0", "#78909C", "#294D63"])

MANIFEST: list[dict] = []


def configure_style() -> None:
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    for font_dir in [BASE/"fonts",RESULTS/"fonts"]:
        if font_dir.exists():
            for fp in font_dir.glob("*.ttf"):
                try:font_manager.fontManager.addfont(str(fp))
                except Exception:pass
    available={f.name for f in font_manager.fontManager.ttflist}
    if "Arial" in available:journal_font="Arial"
    elif "Liberation Sans" in available:
        journal_font="Liberation Sans"
        logging.warning("Arial was not found; using metrically compatible Liberation Sans. Add Arial TTF files to %s/fonts for final rendering.",BASE)
    else:
        journal_font="DejaVu Sans"
        logging.warning("Arial and Liberation Sans were not found; using DejaVu Sans. Add Arial TTF files before final rendering.")
    mpl.rcParams.update(
        {
            "font.family": journal_font,
            "font.sans-serif": [journal_font],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 450,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    sns.set_theme(style="ticks", context="paper", font=journal_font, rc={"axes.spines.top": False, "axes.spines.right": False})


def clean_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def save_figure(fig: plt.Figure, module: str, name: str, title: str, source: str, use: str = "supplement") -> None:
    section = "main_figure_panels" if "main" in str(use).lower() else "supplementary_figures"
    png_dir = OUT / module / section / "png"
    pdf_dir = OUT / module / section / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    stem = clean_name(name)
    fig.savefig(png_dir / f"{stem}.png", dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    MANIFEST.append(
        {
            "module": module,
            "figure": stem,
            "title": title,
            "source": source,
            "recommended_use": use,
            "png": str((png_dir / f"{stem}.png").relative_to(OUT)),
            "pdf": str((pdf_dir / f"{stem}.pdf").relative_to(OUT)),
        }
    )


def new_fig(width: float = 6.8, height: float = 5.2) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    return fig, ax


def finish(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "", legend: bool = True) -> None:
    ax.set_title(title, loc="left", fontweight="bold", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7, alpha=0.8)
    ax.grid(axis="x", visible=False)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    sns.despine(ax=ax)


def ci_mean(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan, np.nan)
    m = x.mean()
    se = stats.sem(x)
    q = stats.t.ppf(1 - alpha / 2, len(x) - 1)
    return m, m - q * se, m + q * se


def bootstrap_curve(y: np.ndarray, p: np.ndarray, kind: str, grid: np.ndarray, n_boot: int = 250, seed: int = 20260820):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    curves = []
    for _ in range(n_boot):
        ii = np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]
        yy, pp = y[ii], p[ii]
        if kind == "roc":
            x, z, _ = roc_curve(yy, pp)
            curves.append(np.interp(grid, x, z))
        else:
            precision, recall, _ = precision_recall_curve(yy, pp)
            order = np.argsort(recall)
            curves.append(np.interp(grid, recall[order], precision[order]))
    a = np.asarray(curves)
    return np.quantile(a, 0.025, axis=0), np.quantile(a, 0.975, axis=0)


def metric_label(s: str) -> str:
    return {
        "age_sex": "Age + sex",
        "traditional_pa": "Traditional PA",
        "age_sex_traditional_pa": "Age + sex + traditional PA",
        "grufcn_pa": "GRU–FCN PA",
        "convcontext_pa": "ConvContext–Mamba PA",
        "dual_pa_feature": "Dual PA representation",
        "final_hybrid": "PAMED",
        "gru": "GRU–FCN", "mamba": "ConvContext–Mamba", "clinical": "Age + sex",
        "frozen_allcause_score": "Frozen all-cause PAMED score",
        "latent_pa_age_sex_probe": "Latent PA + age + sex probe",
        "age_sex_apoe": "Age + sex + APOE", "apoe_only": "APOE only",
        "final_risk_age_sex": "PAMED score + age + sex",
        "risk_only": "PAMED score", "risk_age_sex": "PAMED score + age + sex",
        "final_risk_per_SD": "PAMED logit per SD", "score_age_sex": "PAMED score + age + sex",
        "latent_pa_age_sex": "Latent PA + age + sex",
        "apoe_hybrid": "PAMED + APOE", "pamed_apoe": "PAMED + APOE",
        "ensemble_final_hybrid": "Three-seed ensemble",
    }.get(s, s.replace("_", " ").title())


def load_fixed_cohort() -> pd.DataFrame:
    lab = pd.read_csv(BASE / "ukb_dementia_labels_covariates_N80990.csv", dtype={"eid": "string"}, low_memory=False)
    rep = pd.read_csv(RESULTS / "11_representations" / "frozen_latent_pcs.csv", dtype={"eid": "string"}, low_memory=False)
    keep = ["eid", "split", "target", "final_probability", "grufcn_logit", "convcontext_logit"] + [f"latent_pc{i:02d}" for i in range(1, 17)]
    lab = lab.drop(columns=[column for column in keep if column != "eid" and column in lab])
    d = rep[keep].merge(lab, on="eid", how="left", validate="one_to_one")
    apoe = pd.read_csv(BASE / "APOE_phenotype.csv", dtype={"eid": "string"}, low_memory=False)
    apoe = apoe.rename(columns={"e4": "apoe_e4"})[["eid", "apoe_e4"]]
    d = d.merge(apoe, on="eid", how="left", validate="one_to_one")
    com = BASE / "UKB_Demographic_combordity.csv"
    if com.exists():
        c = pd.read_csv(com, dtype={"eid": "string"}, low_memory=False)
        cols = [x for x in ["eid", "charlson_score", "elixhauser_score"] if x in c]
        d = d.merge(c[cols].drop_duplicates("eid"), on="eid", how="left", validate="one_to_one")
    return d


def standardized_difference(x0: np.ndarray, x1: np.ndarray) -> float:
    x0 = pd.to_numeric(pd.Series(x0), errors="coerce").dropna().to_numpy()
    x1 = pd.to_numeric(pd.Series(x1), errors="coerce").dropna().to_numpy()
    return (x1.mean() - x0.mean()) / np.sqrt((x0.var(ddof=1) + x1.var(ddof=1)) / 2)


def table1_numeric(d: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    continuous = [
        ("enrollment_age", "Age, years"),
        ("BMI", "Body mass index, kg m^-2"),
        ("Light_Total_Hour", "Light activity, h day^-1"),
        ("MVPA_Total_Hour", "MVPA, h day^-1"),
        ("Sedentary_Total_Hour", "Sedentary time, h day^-1"),
        ("Sleep_Total_Hour", "Sleep duration, h day^-1"),
        ("charlson_score", "Charlson comorbidity score"),
        ("elixhauser_score", "Elixhauser comorbidity score"),
    ]
    y = d.target.to_numpy(int)
    rows.append({"dataset": dataset, "variable": "Participants", "type": "count", "overall": f"{len(d):,}", "control": f"{(y==0).sum():,}", "incident": f"{(y==1).sum():,}", "smd": np.nan, "p_value": np.nan})
    for c, label in continuous:
        if c not in d:
            continue
        z = pd.to_numeric(d[c], errors="coerce")
        a, b = z[y == 0].dropna(), z[y == 1].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        rows.append(
            {
                "dataset": dataset,
                "variable": label,
                "type": "continuous",
                "overall": f"{z.mean():.1f} ({z.std():.1f})",
                "control": f"{a.mean():.1f} ({a.std():.1f})",
                "incident": f"{b.mean():.1f} ({b.std():.1f})",
                "smd": standardized_difference(a, b),
                "p_value": p,
            }
        )
    categorical = [("gender", "Female sex", 2), ("apoe_e4", "APOE e4 carrier", "carrier"), ("is_dead", "Death during follow-up", 1)]
    for c, label, level in categorical:
        if c not in d:
            continue
        z = pd.to_numeric(d[c], errors="coerce")
        if level == "carrier":
            zz = (z > 0).astype(float).where(z.notna())
        else:
            zz = (z == level).astype(float).where(z.notna())
        tab = pd.crosstab(y[zz.notna()], zz.dropna().astype(int))
        p = stats.fisher_exact(tab.to_numpy())[1] if tab.shape == (2, 2) else np.nan
        vals = []
        for mask in [np.ones(len(d), bool), y == 0, y == 1]:
            q = zz[mask].dropna()
            vals.append(f"{int(q.sum()):,} ({100*q.mean():.1f}%)")
        p0, p1 = zz[y == 0].mean(), zz[y == 1].mean()
        smd = (p1 - p0) / np.sqrt((p0 * (1-p0) + p1 * (1-p1)) / 2)
        rows.append({"dataset": dataset, "variable": label, "type": "categorical", "overall": vals[0], "control": vals[1], "incident": vals[2], "smd": smd, "p_value": p})
    return pd.DataFrame(rows)


def pvalue_text(x) -> str:
    if not np.isfinite(pd.to_numeric(x,errors="coerce")):return ""
    x=float(x)
    if x<.001:return f"{x:.2e}"
    return f"{x:.3f}"


def demographic_module(d: pd.DataFrame) -> None:
    module = "00A_demographics_mining"
    for column, label in [("Sedentary_Total_Hour", "Sedentary time"), ("Light_Total_Hour", "Light PA"), ("MVPA_Total_Hour", "MVPA"), ("Sleep_Total_Hour", "Sleep duration")]:
        values = pd.to_numeric(d[column], errors="coerce")
        plot = pd.DataFrame({"value": values, "group": np.where(d.target.eq(1), "Incident dementia", "Control")}).dropna()
        fig, ax = new_fig()
        sns.violinplot(data=plot, x="group", y="value", inner="quart", cut=0, palette=[PALETTE["control"], PALETTE["incident"]], ax=ax)
        finish(ax, label, "", "Hours per day", False)
        save_figure(fig, module, f"distribution_{column}", label, "UKB fixed-horizon cohort", "main")
    columns = [c for c in ["Sedentary_Total_Hour", "Light_Total_Hour", "MVPA_Total_Hour", "Sleep_Total_Hour", "enrollment_age", "BMI", "Townsend_index", "gender", "target"] if c in d]
    correlation = d[columns].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    labels = {"enrollment_age": "Age", "Townsend_index": "Deprivation", "gender": "Sex", "target": "Incident dementia", "BMI": "BMI"}
    correlation = correlation.rename(index=lambda x: labels.get(x, x.replace("_Total_Hour", "").replace("_", " ")), columns=lambda x: labels.get(x, x.replace("_Total_Hour", "").replace("_", " ")))
    fig, ax = new_fig(8.5, 7)
    sns.heatmap(correlation, cmap=DIV_CMAP, center=0, vmin=-1, vmax=1, square=True, annot=True, fmt=".2f", cbar_kws={"label": "Spearman correlation"}, ax=ax)
    save_figure(fig, module, "clinical_pa_risk_correlation_heatmap", "Activity and demographic correlations", "UKB fixed-horizon cohort", "main")

def load_profiles():
    hourly = pd.read_csv(DESC / "participant_hourly_profiles.csv", dtype={"eid":"string"})
    daily = pd.read_csv(DESC / "participant_daily_means.csv", dtype={"eid":"string"})
    return hourly, daily


def line_ci_profile(ax, frame: pd.DataFrame, cols: list[str], target_col: str, groups: list[tuple], xlabel: str, ylabel: str):
    x=np.arange(len(cols))
    for value,label,color in groups:
        a=frame.loc[frame[target_col].eq(value),cols].apply(pd.to_numeric,errors="coerce").to_numpy(float)
        mean=np.nanmean(a,axis=0);se=stats.sem(a,axis=0,nan_policy="omit");ci=1.96*se
        ax.plot(x,mean,color=color,lw=2.2,label=f"{label} (n={len(a):,})")
        ax.fill_between(x,mean-ci,mean+ci,color=color,alpha=.18,lw=0)
    ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)


def pa_pattern_module(d: pd.DataFrame) -> None:
    module = "00B_pa_pattern_mining"
    hourly, _ = load_profiles()
    columns = [f"hour_{i:02d}" for i in range(24)]
    fig, ax = new_fig(8, 5.2)
    line_ci_profile(ax, hourly, columns, "target", [(0, "Control", PALETTE["control"]), (1, "Incident dementia", PALETTE["incident"])], "Clock hour", "Mean activity")
    ax.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    finish(ax, "Activity by clock time", "Clock hour", "Mean activity")
    save_figure(fig, module, "hourly_profile_incident_control_ci", "Activity by clock time", str(DESC), "main")
    subtype_labels = pd.read_csv(BASE / "ukb_dementia_subtypes_N80990.csv", dtype={"eid": "string"})
    profiles = hourly.merge(subtype_labels, on="eid", how="left", validate="one_to_one")
    for subtype, label in [("alzheimers_disease", "Alzheimer’s disease"), ("vascular_dementia", "Vascular dementia"), ("frontotemporal_dementia", "Frontotemporal dementia")]:
        column = f"incident_{subtype}_after_1y"
        if column not in profiles:
            continue
        # The comparator is all-cause non-incident controls, never other subtype cases.
        selected = profiles[profiles.target.eq(0) | profiles[column].eq(1)].copy()
        selected["display_group"] = selected[column].eq(1).astype(int)
        fig, ax = new_fig(8, 5.2)
        line_ci_profile(ax, selected, columns, "display_group", [(0, "Non-incident controls", PALETTE["control"]), (1, label, PALETTE.get(subtype, PALETTE["incident"]))], "Clock hour", "Mean activity")
        finish(ax, label, "Clock hour", "Mean activity")
        save_figure(fig, module, f"hourly_profile_{column}", label, str(DESC), "supplement")
    # Only measured day-by-hour aggregates are plotted. No additive reconstruction.
    aggregate_path = DESC / "day_hour_aggregates.npz"
    if aggregate_path.exists():
        with np.load(aggregate_path, allow_pickle=False) as arrays:
            for key in arrays.files:
                values = arrays[key]
                if values.shape != (7, 24):
                    continue
                is_difference = any(token in key.lower() for token in ["difference", "contrast", "delta"])
                color_label = key.replace("_", " ") if is_difference else "Mean activity"
                fig, ax = new_fig(9, 4.8)
                sns.heatmap(values, cmap=DIV_CMAP if is_difference else SEQ_CMAP, center=0 if is_difference else None, cbar_kws={"label": color_label}, ax=ax)
                ax.set_xlabel("Clock hour"); ax.set_ylabel("Monitoring day")
                save_figure(fig, module, f"day_hour_heatmap_{key}", key.replace("_", " "), str(aggregate_path), "main")
                if not is_difference and any(token in key.lower() for token in ["mean", "control", "incident"]):
                    fig = plt.figure(figsize=(9, 6))
                    ax = fig.add_subplot(111, projection="3d")
                    activity_surface(ax, values, label=key.replace("_", " "))
                    fig.subplots_adjust(left=.06, right=.85, bottom=.12, top=.90)
                    save_figure(fig, module, f"day_hour_surface_{key}", key.replace("_", " "), str(aggregate_path), "main")

def probability_columns(pred: pd.DataFrame) -> list[str]:
    return [c for c in pred if c.startswith("probability_")]


def model_from_probability(c: str) -> str:
    return c.removeprefix("probability_")


def curve_panel(pred: pd.DataFrame, col: str, module: str, stem: str, kind: str, source: str, color: str = "#4C78A8") -> None:
    q=pred[["target",col]].dropna();y=q.target.to_numpy(int);p=q[col].to_numpy(float)
    fig,ax=new_fig(6.2,5.2)
    if kind=="roc":
        fpr,tpr,_=roc_curve(y,p);score=roc_auc_score(y,p);grid=np.linspace(0,1,101);lo,hi=bootstrap_curve(y,p,"roc",grid,n_boot=120)
        ax.fill_between(grid,lo,hi,color=color,alpha=.14,label="95% bootstrap CI");ax.plot(fpr,tpr,color=color,lw=2.4,label=f"AUROC = {score:.3f}");ax.plot([0,1],[0,1],"--",color="#999",lw=1);finish(ax,f"ROC curve: {metric_label(model_from_probability(col))}","1 − specificity","Sensitivity")
    else:
        pr,rc,_=precision_recall_curve(y,p);score=average_precision_score(y,p);grid=np.linspace(0,1,101);lo,hi=bootstrap_curve(y,p,"pr",grid,n_boot=120)
        ax.fill_between(grid,lo,hi,color=color,alpha=.14,label="95% bootstrap CI");ax.plot(rc,pr,color=color,lw=2.4,label=f"AUPRC = {score:.3f}");ax.axhline(y.mean(),ls="--",color="#999",lw=1,label=f"Prevalence = {y.mean():.3f}");finish(ax,f"Precision–recall curve: {metric_label(model_from_probability(col))}","Recall","Precision")
    save_figure(fig,module,f"{stem}_{kind}",f"{kind.upper()} curve for {model_from_probability(col)}",source,"supplement")


def metric_forest(frame: pd.DataFrame, metric: str, module: str, stem: str, title: str, source: str, label_col: str="model", use: str="supplement") -> None:
    q=frame.dropna(subset=[metric]).copy().sort_values(metric);labels=q[label_col].map(metric_label) if label_col=="model" else q[label_col].astype(str)
    lo=q.get(f"{metric}_ci_low", q.get("ci_low", q[metric]));hi=q.get(f"{metric}_ci_high", q.get("ci_high", q[metric]));x=q[metric].to_numpy();y=np.arange(len(q))
    fig,ax=new_fig(8,max(4.8,.43*len(q)+1.7));ax.errorbar(x,y,xerr=[np.maximum(0,x-np.asarray(lo)),np.maximum(0,np.asarray(hi)-x)],fmt="o",ms=6,color=PALETTE["hybrid"],ecolor="#6B7280",capsize=3);ax.set_yticks(y,labels);apply_effect_reference(ax, metric);finish(ax,title,metric_axis_label(metric),"",False);save_figure(fig,module,stem,title,source,use)


def calibration_panel(pred: pd.DataFrame, col: str, module: str, stem: str, source: str) -> None:
    q=pred[["target",col]].dropna().sort_values(col);q["bin"]=pd.qcut(q[col],10,duplicates="drop");g=q.groupby("bin",observed=True).agg(predicted=(col,"mean"),events=("target","sum"),n=("target","size")).reset_index();g["observed"]=g.events/g.n
    cis=np.array([proportion_confint(e,n,method="wilson") for e,n in zip(g.events,g.n)])
    lower=np.maximum(0,g.observed.to_numpy(float)-cis[:,0]);upper=np.maximum(0,cis[:,1]-g.observed.to_numpy(float));fig,ax=new_fig();ax.errorbar(g.predicted,g.observed,yerr=np.vstack([lower,upper]),fmt="o-",color=PALETTE["hybrid"],capsize=3,label="Deciles (95% CI)");mx=max(g.predicted.max(),g.observed.max(),1e-6)*1.08;ax.plot([0,mx],[0,mx],"--",color="#999",label="Ideal");finish(ax,f"Calibration: {metric_label(model_from_probability(col))}","Mean predicted risk","Observed incidence");save_figure(fig,module,stem,"Calibration with Wilson confidence intervals",source,"main")


def internal_module(d=None) -> None:
    module = "02_core_model"; root = RESULTS / module
    predictions = pd.read_csv(root / "ukb_20pct_final_test_predictions.csv")
    calibration_panel(predictions, "probability_final_hybrid", module, "final_hybrid_calibration", str(root))
    fig, ax = new_fig()
    for target, label, color in [(0, "Control", PALETTE["control"]), (1, "Incident dementia", PALETTE["incident"])]:
        sns.kdeplot(data=predictions[predictions.target.eq(target)], x="probability_final_hybrid", cut=0, fill=True, alpha=.2, color=color, label=label, ax=ax)
    finish(ax, "PAMED score by outcome", "Predicted probability", "Density")
    save_figure(fig, module, "risk_density_final_hybrid", "PAMED score by outcome", str(root), "main")
    concentration = pd.read_csv(root / "ukb_20pct_top_risk_enrichment.csv")
    fig, ax = new_fig()
    for i, (model, frame) in enumerate(concentration.groupby("model")):
        ax.plot(frame.top_fraction * 100, frame.event_capture_fraction * 100, "o-", label=metric_label(model), color=SERIES[i % len(SERIES)])
    ax.plot([0, 20], [0, 20], "--", color="#A7AFB3", label="Random selection")
    finish(ax, "Case concentration", "Population selected (%)", "Incident cases captured (%)")
    save_figure(fig, module, "top_risk_event_capture_fraction", "Case concentration", str(root), "main")

def ablation_module() -> None:
    module = "08_ablation"
    path = RESULTS / "18_final_paired" / "final_paired_increment_tests.csv"
    paired = pd.read_csv(path)
    for metric in ["delta_auroc", "delta_auprc"]:
        metric_forest(paired, metric, module, f"paired_increment_{metric}", "Paired comparison with PAMED", str(path), label_col="comparison", use="main")
    path = RESULTS / "13_hierarchical_shapley" / "modality_shapley_individual.csv"
    individual = pd.read_csv(path)
    for source in ["gru", "mamba", "clinical"]:
        fig, ax = new_fig()
        sns.violinplot(data=individual, x="target", y=f"shapley_{source}", inner="quart", cut=0, palette=[PALETTE["control"], PALETTE["incident"]], ax=ax)
        ax.set_xticks([0, 1], ["Control", "Incident dementia"])
        finish(ax, metric_label(source), "", "Shapley probability contribution", False)
        save_figure(fig, module, f"individual_shapley_{source}", metric_label(source), str(path), "main")

def external_module() -> None:
    module = "04_nhats_external"; root = RESULTS / module
    metrics = pd.read_csv(root / "nhats_external_metrics.csv")
    predictions = pd.read_csv(root / "nhats_external_predictions.csv")
    primary = metrics[metrics.analysis.eq("primary_complete_demographics")].copy()
    if "endpoint" in primary:
        primary = primary[primary.endpoint.eq("incident_possible_or_probable")]
    robust = primary[primary.harmonization.eq("positive_robust") | primary.model.eq("age_sex_traditional_pa")].copy()
    metric_forest(robust, "auroc", module, "external_auroc_forest", "NHATS discrimination", str(root), use="main")
    aligned = primary[primary.harmonization.isin(["positive_robust", "learned_adapter", "positive_quantile", "raw_count"])].copy()
    if "auprc_over_prevalence" not in aligned:
        prevalence = predictions.target.mean()
        aligned["auprc_over_prevalence"] = aligned.auprc / prevalence
    for metric in ["auroc", "auprc_over_prevalence"]:
        pivot = aligned.pivot(index="model", columns="harmonization", values=metric)
        pivot.index = pivot.index.map(metric_label)
        pivot.columns = [str(x).replace("_", " ").capitalize() for x in pivot.columns]
        fig, ax = new_fig(8, 5.5)
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap=SEQ_CMAP, cbar_kws={"label": metric_axis_label(metric)}, ax=ax)
        ax.set_ylabel(""); ax.set_xlabel("Measurement mapping")
        save_figure(fig, module, f"harmonization_{metric}_heatmap", metric_axis_label(metric), str(root), "main")
    fig, ax = new_fig()
    for target, label, color in [(0, "No incident dementia", PALETTE["control"]), (1, "Incident dementia", PALETTE["incident"])]:
        sns.kdeplot(data=predictions[predictions.target.eq(target)], x="positive_robust__final_hybrid", cut=0, fill=True, alpha=.2, color=color, label=label, ax=ax)
    finish(ax, "NHATS transported score", "Transported PAMED score", "Density")
    save_figure(fig, module, "nhats_risk_density", "NHATS transported score", str(root), "main")
    enrichment = pd.read_csv(root / "risk_enrichment.csv")
    for endpoint, subset in enrichment.groupby("endpoint"):
        if endpoint != "incident_possible_or_probable":
            continue
        fig, ax = new_fig()
        for i, (score, frame) in enumerate(subset.groupby("score")):
            ax.plot(frame.top_fraction * 100, frame.event_capture_fraction * 100, "o-", label=readable_label(score), color=SERIES[i % len(SERIES)])
        finish(ax, "NHATS case concentration", "Population selected (%)", "Incident cases captured (%)")
        save_figure(fig, module, f"enrichment_{endpoint}_event_capture_fraction", "NHATS case concentration", str(root), "main")
    sensitivity = metrics[~metrics.analysis.eq("primary_complete_demographics") & metrics.model.eq("final_hybrid")].copy()
    for metric in ["auroc", "auprc"]:
        sensitivity["label"] = sensitivity.analysis.str.replace("_", " ") + " | " + sensitivity.harmonization.str.replace("_", " ")
        metric_forest(sensitivity, metric, module, f"external_sensitivity_{metric}", "NHATS sensitivity", str(root), label_col="label")

def interpretability_module(d: pd.DataFrame) -> None:
    module = "13_hierarchical_shapley"; root = RESULTS / module
    for filename, xcolumn, prefix in [("temporal_shapley_by_day.csv", "day", "day"), ("temporal_shapley_by_clock.csv", "clock_start", "clock")]:
        frame = pd.read_csv(root / filename)
        fig, ax = new_fig()
        for i, (group, subset) in enumerate(frame.groupby("group")):
            ax.plot(subset[xcolumn], subset.mean_absolute_shapley, "o-", label=str(group).replace("_", " "), color=SERIES[i % len(SERIES)])
        finish(ax, "Absolute temporal Shapley", "Monitoring day" if prefix == "day" else "Clock-period start (hour)", "Mean absolute probability contribution")
        save_figure(fig, module, f"shapley_by_{prefix}_mean_absolute_shapley", "Absolute temporal Shapley", str(root), "main")
    path = RESULTS / "14_rhythm_knowledge" / "rhythm_concept_latent_correlations.csv"
    correlations = pd.read_csv(path)
    pivot = correlations[correlations.split.eq("test")].pivot(index="concept", columns="latent_component", values="spearman_r")
    pivot.index = [str(x).replace("_", " ") for x in pivot.index]
    fig, ax = new_fig(10, 8)
    sns.heatmap(pivot, cmap=DIV_CMAP, center=0, vmin=-1, vmax=1, cbar_kws={"label": "Spearman correlation"}, ax=ax)
    save_figure(fig, module, "rhythm_latent_correlation_heatmap", "Rhythm and latent components", str(path), "supplement")

def clinical_module(d=None) -> None:
    module = "08_clinical_extensions"
    rhythm = RESULTS / "14_rhythm_knowledge" / "rhythm_concept_dementia_associations.csv"
    associations = pd.read_csv(rhythm)
    metric_forest(associations[associations.split.eq("test")], "odds_ratio", module, "rhythm_association_forest_test", "Age- and sex-adjusted rhythm associations", str(rhythm), label_col="concept", use="supplement")
    root = RESULTS / "15_survival"
    curves = pd.read_csv(root / "landmark_km_by_risk_quartile.csv")
    fig, ax = new_fig(7.5, 5.5)
    for i, (group, frame) in enumerate(curves.groupby("group")):
        ax.step(frame.time_years + 1, (1 - frame.survival) * 100, where="post", label=str(group), color=SERIES[i % len(SERIES)])
    finish(ax, "Cumulative dementia incidence", "Years after accelerometry", "Cumulative incidence (%)")
    save_figure(fig, module, "survival_cumulative_incidence_quartiles", "Cumulative incidence", str(root), "main")
    last = curves.sort_values("time_years").groupby("group", as_index=False).tail(1)
    fig, ax = new_fig()
    ax.bar(last.group, (1 - last.survival) * 100, color=SERIES[:len(last)])
    finish(ax, "Seven-year dementia incidence", "PAMED risk quartile", "Cumulative incidence (%)", False)
    save_figure(fig, module, "survival_quartile_seven_year_incidence", "Seven-year incidence", str(root), "main")
    horizons = pd.read_csv(root / "time_dependent_auc.csv")
    fig, ax = new_fig()
    ax.plot(horizons.horizon_years_from_accelerometer, horizons.dynamic_auroc, "o-", color=PALETTE["hybrid"])
    for _, row in horizons.iterrows():
        ax.annotate(f"{int(row.events_by_horizon)} events", (row.horizon_years_from_accelerometer, row.dynamic_auroc), xytext=(3, 6), textcoords="offset points")
    ax.axhline(.5, color="#999", linestyle="--")
    finish(ax, "Horizon-specific discrimination", "Years after accelerometry", "Horizon-specific AUROC", False)
    save_figure(fig, module, "time_dependent_auroc", "Horizon-specific AUROC (not IPCW)", str(root), "main")
    cox = pd.read_csv(root / "landmark_cox_models.csv")
    metric_forest(cox[cox.term.eq("final_risk_per_SD")], "hazard_ratio", module, "landmark_cox_forest", "Dementia Cox associations", str(root), use="main")
    subtypes = pd.read_csv(root / "cause_specific_subtype_cox.csv")
    subtypes = subtypes[subtypes.term.eq("final_risk_per_SD")]
    metric_forest(subtypes, "hazard_ratio", module, "subtype_cause_specific_cox_forest", "Subtype Cox associations", str(root), label_col="subtype", use="main")
    subtype_root = RESULTS / "12_subtypes"
    metrics = pd.read_csv(subtype_root / "subtype_test_metrics.csv")
    for metric in ["auroc", "auprc"]:
        frame = metrics.copy()
        frame["label"] = frame.subtype.str.replace("_", " ") + " | " + frame.model.map(metric_label)
        metric_forest(frame, metric, module, f"subtype_{metric}_forest", f"Subtype {metric.upper()}", str(subtype_root), label_col="label", use="main")
    counts = pd.read_csv(subtype_root / "subtype_cohort_counts.csv")
    fig, ax = new_fig()
    ax.bar(counts.subtype.str.replace("_", " "), counts.test_events, color=SERIES[:len(counts)])
    finish(ax, "Subtype evaluation events", "", "Events", False)
    save_figure(fig, module, "subtype_event_counts", "Subtype evaluation events", str(subtype_root), "supplement")
    clinical_root = RESULTS / "17_clinical_associations"
    probes = pd.read_csv(clinical_root / "clinical_linear_probe_test_metrics.csv")
    for metric in ["auroc", "r2"]:
        frame = probes.dropna(subset=[metric]).copy()
        frame["label"] = frame.outcome.str.replace("_", " ") + " | " + frame.analysis.map(metric_label)
        metric_forest(frame, metric, module, f"clinical_probe_{metric}", f"Clinical probe {metric_axis_label(metric)}", str(clinical_root), label_col="label", use="main")
    mortality = pd.read_csv(clinical_root / "mortality_cox_associations.csv")
    if len(mortality):
        mortality = mortality[mortality.term.eq("final_risk_per_SD")].copy()
        mortality["label"] = mortality.split.str.replace("_", " ")
        metric_forest(mortality, "hazard_ratio", module, "mortality_cox_forest", "Age- and sex-adjusted mortality association", str(clinical_root), label_col="label", use="main")

def metric_axis_label(metric: str) -> str:
    return {
        "auroc": "AUROC", "auprc": "AUPRC", "auprc_over_prevalence": "AUPRC / prevalence",
        "hazard_ratio": "Hazard ratio (95% CI)", "odds_ratio": "Odds ratio (95% CI)",
        "delta_auroc": "AUROC difference (95% CI)", "delta_auprc": "AUPRC difference (95% CI)",
        "r2": "R²", "mae": "Mean absolute error", "brier": "Brier score",
        "spearman_r": "Spearman correlation", "coefficient": "Coefficient (95% CI)",
        "sensitivity": "Sensitivity", "specificity": "Specificity",
    }.get(metric, metric.replace("_", " ").capitalize())


def apply_effect_reference(ax, metric: str) -> None:
    """Use 1 for ratio measures and 0 for signed additive effects."""
    if metric in {"hazard_ratio", "odds_ratio"}:
        ax.axvline(1, color="#879198", linestyle="--", linewidth=.9)
        ax.set_xscale("log")
    elif metric.startswith("delta_") or metric in {"coefficient", "spearman_r"}:
        ax.axvline(0, color="#879198", linestyle="--", linewidth=.9)


def readable_label(value: str) -> str:
    parts = str(value).removeprefix("probability_").split("__")
    return " | ".join(metric_label(part.removeprefix("score_")) for part in parts)


def activity_surface(ax, values, *, label="", color="#4F7085") -> None:
    """Plot a 7 × 24 activity landscape with spaced labels and fixed ticks."""
    values = np.asarray(values, dtype=float)
    if values.shape != (7, 24):
        raise ValueError("Activity surface requires a 7 × 24 day-by-hour array")
    hours = np.arange(24)
    for day, profile in enumerate(values, 1):
        ax.plot(hours, np.full(24, day), profile, color=color, linewidth=1.1)
    ax.set_xlabel("Clock hour", labelpad=12)
    ax.set_ylabel("Monitoring day", labelpad=15)
    ax.set_zlabel("Mean activity", labelpad=12)
    ax.set_xticks([0, 6, 12, 18, 23], ["00", "06", "12", "18", "23"])
    ax.set_yticks([1, 4, 7])
    ax.tick_params(axis="both", labelsize=8, pad=1)
    ax.set_box_aspect((1.7, 1.2, .85))
    ax.view_init(elev=25, azim=-58)
    if label:
        ax.set_title(label, loc="left", pad=16)


def write_manifest() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    columns = ["module", "figure", "title", "source", "recommended_use", "png", "pdf"]
    current = pd.DataFrame(MANIFEST, columns=columns)
    current.to_csv(OUT / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    current.groupby("module").size().rename("n_panels").reset_index().to_csv(
        OUT / "figure_counts_by_module.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    from generate_figures import main
    main()
