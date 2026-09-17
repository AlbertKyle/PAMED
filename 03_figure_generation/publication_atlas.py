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

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import font_manager
import seaborn as sns
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    auc,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint


warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = Path(r"E:\PA_foundation_model\dementia")
V13 = BASE / "Results" / "v13_final" / "v13_final"
DESC = BASE / "descriptive_nature_results"
EXT_BASE = BASE / "external_validation"
OUT = V13 / "publication_outputs"

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
    for font_dir in [BASE/"fonts",V13/"fonts"]:
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


def save_figure(fig: plt.Figure, module: str, name: str, title: str, source: str, use: str = "candidate") -> None:
    section = "main_candidates" if "main" in str(use).lower() else "supplementary_figures"
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
        "grufcn_pa": "GRU-FCN",
        "convcontext_pa": "ConvContext-Mamba",
        "dual_pa_feature": "Dual PA feature fusion",
        "final_hybrid": "Final hybrid",
        "ensemble_final_hybrid": "Three-seed ensemble",
    }.get(s, s.replace("_", " ").title())


def load_fixed_cohort() -> pd.DataFrame:
    lab = pd.read_csv(BASE / "ukb_dementia_labels_covariates_N80990.csv", dtype={"eid": "string"}, low_memory=False)
    rep = pd.read_csv(V13 / "11_representations" / "frozen_latent_pcs.csv", dtype={"eid": "string"}, low_memory=False)
    keep = ["eid", "split", "target", "final_probability", "grufcn_logit", "convcontext_logit"] + [f"latent_pc{i:02d}" for i in range(1, 17)]
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


def export_publication_tables(d: pd.DataFrame) -> None:
    td=OUT/"09_tables";td.mkdir(parents=True,exist_ok=True)
    uk=table1_numeric(d,"UKB");uk.to_csv(td/"Table_1A_UKB_baseline_numeric.csv",index=False,encoding="utf-8-sig");fmt=uk.copy();fmt["smd"]=fmt.smd.map(lambda x:"" if not np.isfinite(pd.to_numeric(x,errors="coerce")) else f"{float(x):.2f}");fmt["p_value"]=fmt.p_value.map(pvalue_text);fmt.to_csv(td/"Table_1A_UKB_manuscript_formatted.csv",index=False,encoding="utf-8-sig")
    ep=V13/"04_nhats_external"/"nhats_external_predictions.csv"
    if ep.exists():
        nh=pd.read_csv(ep);rows=[];y=nh.target.to_numpy(int);rows.append({"variable":"Participants","overall":f"{len(nh):,}","no_incident":f"{(y==0).sum():,}","incident":f"{(y==1).sum():,}"})
        age=pd.to_numeric(nh.age_midpoint,errors="coerce")
        rows.append({"variable":"Age midpoint, years","overall":f"{age.mean():.1f} ({age.std():.1f})","no_incident":f"{age[y==0].mean():.1f} ({age[y==0].std():.1f})","incident":f"{age[y==1].mean():.1f} ({age[y==1].std():.1f})"})
        sex=pd.to_numeric(nh.gender,errors="coerce");female=(sex==2)
        rows.append({"variable":"Female sex","overall":f"{female.sum():,} ({100*female.mean():.1f}%)","no_incident":f"{female[y==0].sum():,} ({100*female[y==0].mean():.1f}%)","incident":f"{female[y==1].sum():,} ({100*female[y==1].mean():.1f}%)"})
        nhfmt=pd.DataFrame(rows);nhfmt.to_csv(td/"Table_1B_NHATS_manuscript_formatted.csv",index=False,encoding="utf-8-sig")
        ukmain=fmt.rename(columns={"control":"no_incident"})[["variable","overall","no_incident","incident","smd","p_value"]].copy();ukmain.insert(0,"cohort","UK Biobank")
        nhmain=nhfmt.copy();nhmain.insert(0,"cohort","NHATS");nhmain["smd"]="";nhmain["p_value"]="";nhmain=nhmain[ukmain.columns]
        pd.concat([ukmain,nhmain],ignore_index=True).to_csv(td/"MAIN_TABLE_1_UKB_NHATS.csv",index=False,encoding="utf-8-sig")
    selected=[
        "01_algorithm_pool/fixed_20pct_algorithm_metrics.csv","02_core_model/ukb_20pct_final_test_metrics.csv","02_core_model/ukb_20pct_top_risk_enrichment.csv","03_apoe_secondary/apoe_complete_test_metrics.csv","04_nhats_external/nhats_external_metrics.csv","04_nhats_external/risk_enrichment.csv","05_stability/per_seed_test_metrics.csv","05_stability/three_seed_ensemble_metrics.csv","06_paired_comparisons/paired_all_models_vs_final_hybrid.csv","07_subgroups/subgroup_performance.csv","07_subgroups/intersectional_subgroups_min10events.csv","08_ablation/ablation_test_metrics.csv","12_subtypes/subtype_test_metrics.csv","13_hierarchical_shapley/modality_shapley_summary.csv","13_hierarchical_shapley/temporal_shapley_7day_x_4period.csv","14_rhythm_knowledge/rhythm_concept_dementia_associations.csv","15_survival/landmark_cox_models.csv","15_survival/time_dependent_auc.csv","16_interactions/risk_score_interaction_tests.csv","17_clinical_associations/clinical_linear_probe_test_metrics.csv","17_clinical_associations/mortality_cox_associations.csv","18_final_paired/final_paired_increment_tests.csv"]
    index=[]
    for rel in selected:
        src=V13/rel
        if src.exists():
            name="Supplementary_"+rel.replace("/","__");shutil.copy2(src,td/name);index.append({"table_file":name,"source":rel})
    pd.DataFrame(index).to_csv(td/"Supplementary_table_index.csv",index=False,encoding="utf-8-sig")


def demographic_module(d: pd.DataFrame) -> None:
    module = "00A_demographics_mining"
    vars_ = [
        ("enrollment_age", "Age (years)"), ("BMI", "BMI (kg m$^{-2}$)"),
        ("Light_Total_Hour", "Light activity (h day$^{-1}$)"),
        ("MVPA_Total_Hour", "MVPA (h day$^{-1}$)"),
        ("Sedentary_Total_Hour", "Sedentary time (h day$^{-1}$)"),
        ("Sleep_Total_Hour", "Sleep duration (h day$^{-1}$)"),
        ("charlson_score", "Charlson score"), ("elixhauser_score", "Elixhauser score"),
    ]
    for c, label in vars_:
        if c not in d:
            continue
        z = pd.to_numeric(d[c], errors="coerce")
        if z.notna().sum() < 50:
            continue
        qlo, qhi = z.quantile([0.005, 0.995])
        plot = pd.DataFrame({"value": z.clip(qlo, qhi), "group": np.where(d.target.eq(1), "Incident dementia", "Control")}).dropna()
        fig, ax = new_fig()
        sns.violinplot(data=plot, x="group", y="value", palette=[PALETTE["control"], PALETTE["incident"]], inner=None, cut=0, linewidth=0.8, ax=ax)
        sns.boxplot(data=plot, x="group", y="value", width=.18, showfliers=False, color="white", boxprops={"zorder": 3}, ax=ax)
        finish(ax, f"Distribution of {label.lower()}", "", label, False)
        save_figure(fig, module, f"distribution_{c}", f"Distribution of {label}", "UKB fixed-horizon cohort", "supplement")

        fig, ax = new_fig()
        for target, group, color in [(0, "Control", PALETTE["control"]), (1, "Incident dementia", PALETTE["incident"])]:
            x = z[d.target.eq(target)].dropna().clip(qlo, qhi)
            sns.kdeplot(x=x, fill=True, alpha=.18, linewidth=2, color=color, label=f"{group} (n={len(x):,})", ax=ax)
        finish(ax, f"Density of {label.lower()}", label, "Density")
        save_figure(fig, module, f"density_{c}", f"Density of {label}", "UKB fixed-horizon cohort", "candidate")

    pa_cols = [c for c in ["Light_Total_Hour", "MVPA_Total_Hour", "Sedentary_Total_Hour", "Sleep_Total_Hour"] if c in d]
    labels = ["Light", "MVPA", "Sedentary", "Sleep"][:len(pa_cols)]
    means = []
    for c, lab in zip(pa_cols, labels):
        for target, group in [(0, "Control"), (1, "Incident dementia")]:
            m, lo, hi = ci_mean(pd.to_numeric(d.loc[d.target.eq(target), c], errors="coerce"))
            means.append({"activity": lab, "group": group, "mean": m, "low": lo, "high": hi})
    mm = pd.DataFrame(means)
    fig, ax = new_fig(7.4, 5.2)
    x = np.arange(len(labels)); width=.34
    for j, (g, color) in enumerate([("Control", PALETTE["control"]), ("Incident dementia", PALETTE["incident"])]):
        z = mm[mm.group.eq(g)]
        ax.bar(x+(j-.5)*width, z["mean"], width, color=color, alpha=.9, label=g)
        ax.errorbar(x+(j-.5)*width, z["mean"], yerr=[z["mean"]-z["low"], z["high"]-z["mean"]], fmt="none", ecolor="#303030", capsize=3)
    ax.set_xticks(x, labels)
    finish(ax, "Traditional activity phenotypes", "", "Mean hours per day")
    save_figure(fig, module, "traditional_pa_mean_ci", "Traditional PA means with 95% CIs", "UKB fixed-horizon cohort", "main candidate")

    corr_cols = [c for c, _ in vars_ if c in d] + ["final_probability"]
    cr = d[corr_cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    fig, ax = new_fig(9.0, 7.2)
    sns.heatmap(cr, cmap=DIV_CMAP, center=0, vmin=-.6, vmax=.6, square=True, linewidths=.5, cbar_kws={"label": "Spearman r", "shrink": .8}, ax=ax)
    ax.set_title("Clinical, traditional PA and model-risk correlations", loc="left", fontweight="bold", pad=10)
    save_figure(fig, module, "clinical_pa_risk_correlation_heatmap", "Clinical and traditional PA correlation heatmap", "UKB fixed-horizon cohort", "main candidate")

    smd_rows=[]
    for c,label in vars_:
        if c in d:
            a=pd.to_numeric(d.loc[d.target.eq(0),c],errors="coerce");b=pd.to_numeric(d.loc[d.target.eq(1),c],errors="coerce")
            if a.notna().sum()>2 and b.notna().sum()>2:smd_rows.append((label,standardized_difference(a,b)))
    smd=pd.DataFrame(smd_rows,columns=["variable","smd"]).sort_values("smd")
    fig,ax=new_fig(7.5,6.0);colors=np.where(smd.smd.abs()>=.2,PALETTE["incident"],"#AAB2BD");ax.barh(smd.variable,smd.smd,color=colors);ax.axvline(0,color="#333",lw=.8);ax.axvline(.2,color="#999",ls="--",lw=.8);ax.axvline(-.2,color="#999",ls="--",lw=.8);finish(ax,"Standardized differences: incident dementia vs controls","Standardized mean difference","");save_figure(fig,module,"baseline_smd_love_plot","Baseline and PA standardized differences","UKB fixed-horizon cohort","main candidate")

    if "apoe_e4" in d:
        q=d.dropna(subset=["apoe_e4"]).copy();q["apoe_group"]=pd.to_numeric(q.apoe_e4).astype(int).astype(str)+" e4 allele(s)"
        fig,ax=new_fig();order=[x for x in ["0 e4 allele(s)","1 e4 allele(s)","2 e4 allele(s)"] if x in q.apoe_group.unique()];sns.violinplot(data=q,x="apoe_group",y="final_probability",order=order,palette=SERIES[:len(order)],inner=None,cut=0,ax=ax);ax.set_yscale("log");finish(ax,"Model-risk distribution by APOE e4 dosage","APOE e4 dosage","Predicted risk (log scale)",False);save_figure(fig,module,"apoe_risk_distribution","Model risk by APOE dosage","UKB APOE complete cases","supplement")
        rates=q.groupby("apoe_group").target.agg(["sum","count"]);rates["rate"]=rates["sum"]/rates["count"]
        lo=[];hi=[]
        for _,r in rates.iterrows():a,b=proportion_confint(r["sum"],r["count"],method="wilson");lo.append(a);hi.append(b)
        rates["lo"],rates["hi"]=lo,hi;rates=rates.reindex(order)
        fig,ax=new_fig();x=np.arange(len(rates));ax.errorbar(x,rates.rate*100,yerr=[(rates.rate-rates.lo)*100,(rates.hi-rates.rate)*100],fmt="o",ms=8,color=PALETTE["incident"],capsize=4);ax.set_xticks(x,rates.index);finish(ax,"Incident dementia by APOE e4 dosage","APOE e4 dosage","Incident dementia (%)",False);save_figure(fig,module,"apoe_event_rate_ci","Incident dementia by APOE dosage","UKB APOE complete cases","supplement")

    # Age-risk and PA-risk smooth relationships.
    for c,label in [("enrollment_age","Age (years)"),("Light_Total_Hour","Light activity (h day$^{-1}$)"),("MVPA_Total_Hour","MVPA (h day$^{-1}$)"),("Sedentary_Total_Hour","Sedentary time (h day$^{-1}$)"),("Sleep_Total_Hour","Sleep duration (h day$^{-1}$)")]:
        if c not in d:continue
        q=d[[c,"final_probability","target"]].apply(pd.to_numeric,errors="coerce").dropna();q["bin"]=pd.qcut(q[c],10,duplicates="drop");g=q.groupby("bin",observed=True).agg(x=(c,"median"),risk=("final_probability","mean"),events=("target","sum"),n=("target","size")).reset_index();g["rate"]=g.events/g.n
        fig,ax=new_fig();ax.plot(g.x,g.risk,"o-",color=PALETTE["hybrid"],lw=2,label="Mean predicted risk");ax.plot(g.x,g.rate,"s--",color=PALETTE["incident"],lw=1.8,label="Observed incidence");finish(ax,f"Model risk across {label.lower()} deciles",label,"Probability");save_figure(fig,module,f"risk_gradient_{c}",f"Risk gradient across {label}","UKB fixed-horizon cohort","supplement")


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
    module="00B_pa_pattern_mining"
    hourly,daily=load_profiles();sub=pd.read_csv(BASE/"ukb_dementia_subtypes_N80990.csv",dtype={"eid":"string"});hourly=hourly.merge(sub,on="eid",how="left");daily=daily.merge(sub,on="eid",how="left")
    hcols=[f"hour_{i:02d}" for i in range(24)];dcols=[f"day_{i}" for i in range(1,8)]
    fig,ax=new_fig(8,5.2);line_ci_profile(ax,hourly,hcols,"target",[(0,"Control",PALETTE["control"]),(1,"Incident dementia",PALETTE["incident"])],"Clock hour","Mean acceleration");ax.set_xticks(np.arange(0,24,3),[f"{i:02d}:00" for i in range(0,24,3)]);ax.axvspan(0,6,color="#DCE6F1",alpha=.35);ax.axvspan(18,23,color="#DCE6F1",alpha=.25);finish(ax,"Twenty-four-hour activity profile with 95% confidence bands","Clock hour","Mean acceleration");save_figure(fig,module,"hourly_profile_incident_control_ci","24-hour PA profile by dementia outcome","Participant hourly profiles","main candidate")
    fig,ax=new_fig(7.2,5);line_ci_profile(ax,daily,dcols,"target",[(0,"Control",PALETTE["control"]),(1,"Incident dementia",PALETTE["incident"])],"Monitoring day","Mean acceleration");ax.set_xticks(np.arange(7),[f"Day {i}" for i in range(1,8)]);finish(ax,"Seven-day activity trajectory with 95% confidence bands","Monitoring day","Mean acceleration");save_figure(fig,module,"daily_profile_incident_control_ci","Seven-day PA trajectory","Participant daily profiles","main candidate")

    # Hour-by-hour difference with confidence interval.
    a=hourly.loc[hourly.target.eq(0),hcols].to_numpy(float);b=hourly.loc[hourly.target.eq(1),hcols].to_numpy(float);delta=np.nanmean(b,0)-np.nanmean(a,0);se=np.sqrt(np.nanvar(a,0,ddof=1)/len(a)+np.nanvar(b,0,ddof=1)/len(b));ci=1.96*se
    fig,ax=new_fig(8,5);x=np.arange(24);ax.plot(x,delta,color=PALETTE["incident"],lw=2);ax.fill_between(x,delta-ci,delta+ci,color=PALETTE["incident"],alpha=.2);ax.axhline(0,color="#333",lw=.8);ax.set_xticks(np.arange(0,24,3),[f"{i:02d}:00" for i in range(0,24,3)]);finish(ax,"Hourly activity difference: incident dementia minus controls","Clock hour","Mean difference (95% CI)",False);save_figure(fig,module,"hourly_difference_ci","Hourly PA difference with 95% CI","Participant hourly profiles","main candidate")

    subtype_specs=[("incident_alzheimers_disease_after_1y","Alzheimer's disease",PALETTE["alzheimers_disease"]),("incident_vascular_dementia_after_1y","Vascular dementia",PALETTE["vascular_dementia"]),("incident_frontotemporal_dementia_after_1y","Frontotemporal dementia",PALETTE["frontotemporal_dementia"])]
    for col,label,color in subtype_specs:
        if col not in hourly:continue
        fig,ax=new_fig(8,5.2);line_ci_profile(ax,hourly,hcols,col,[(0,"No recorded subtype",PALETTE["control"]),(1,label,color)],"Clock hour","Mean acceleration");ax.set_xticks(np.arange(0,24,3),[f"{i:02d}:00" for i in range(0,24,3)]);finish(ax,f"Twenty-four-hour activity profile: {label}","Clock hour","Mean acceleration");save_figure(fig,module,f"hourly_profile_{col}",f"24-hour profile for {label}","UKB subtype labels and hourly profiles","supplement")
        fig,ax=new_fig(7.2,5);line_ci_profile(ax,daily,dcols,col,[(0,"No recorded subtype",PALETTE["control"]),(1,label,color)],"Monitoring day","Mean acceleration");ax.set_xticks(np.arange(7),[f"Day {i}" for i in range(1,8)]);finish(ax,f"Seven-day trajectory: {label}","Monitoring day","Mean acceleration");save_figure(fig,module,f"daily_profile_{col}",f"Seven-day profile for {label}","UKB subtype labels and daily profiles","supplement")

    # Day-by-hour mean and effect-size heatmaps reconstructed from hourly x daily additive components.
    dh=DESC/"day_hour_aggregates.npz"
    if dh.exists():
        z=np.load(dh,allow_pickle=True)
        for key in z.files:
            arr=z[key]
            if arr.ndim==2 and arr.shape==(7,24):
                fig,ax=new_fig(10,4.8);sns.heatmap(arr,cmap=SEQ_CMAP,ax=ax,cbar_kws={"label":"Mean acceleration"});ax.set_xlabel("Clock hour");ax.set_ylabel("Monitoring day");ax.set_yticklabels([f"Day {i}" for i in range(1,8)],rotation=0);ax.set_title(key.replace("_"," ").title(),loc="left",fontweight="bold");save_figure(fig,module,f"day_hour_heatmap_{key}",f"Day-hour heatmap {key}","day_hour_aggregates.npz","candidate")

    # Individual profile heatmaps sampled deterministically.
    rng=np.random.default_rng(20260820)
    for target,label,color in [(0,"controls",PALETTE["control"]),(1,"incident_dementia",PALETTE["incident"])]:
        q=hourly[hourly.target.eq(target)].copy();n=min(250,len(q));q=q.iloc[rng.choice(len(q),n,False)];a=q[hcols].to_numpy(float);order=np.argsort(np.nanmean(a,1));a=a[order]
        fig,ax=new_fig(9,6);sns.heatmap(a,cmap=SEQ_CMAP,xticklabels=[str(i) if i%3==0 else "" for i in range(24)],yticklabels=False,cbar_kws={"label":"Mean acceleration"},ax=ax);ax.set_xlabel("Clock hour");ax.set_ylabel(f"Participants (n={n})");ax.set_title(f"Participant-level circadian profiles: {label.replace('_',' ')}",loc="left",fontweight="bold");save_figure(fig,module,f"individual_hourly_heatmap_{label}",f"Individual PA heatmap: {label}","Participant hourly profiles","supplement")

    # Polar clocks for outcome and subtypes.
    for target,label,color in [(0,"Control",PALETTE["control"]),(1,"Incident dementia",PALETTE["incident"])]:
        mean=hourly.loc[hourly.target.eq(target),hcols].mean().to_numpy();theta=np.linspace(0,2*np.pi,24,endpoint=False);theta=np.r_[theta,theta[0]];v=np.r_[mean,mean[0]]
        fig=plt.figure(figsize=(6.2,6.2),constrained_layout=True);ax=fig.add_subplot(111,projection="polar");ax.plot(theta,v,color=color,lw=2.5);ax.fill(theta,v,color=color,alpha=.18);ax.set_theta_zero_location("N");ax.set_theta_direction(-1);ax.set_xticks(np.linspace(0,2*np.pi,8,endpoint=False),["00","03","06","09","12","15","18","21"]);ax.set_title(f"Activity clock: {label}",fontweight="bold",pad=18);save_figure(fig,module,f"activity_clock_{label}",f"PA clock for {label}","Participant hourly profiles","candidate")

    # Distribution of hourly peak timing and amplitude.
    hh=hourly[hcols].to_numpy(float);hourly["peak_hour"]=np.nanargmax(np.where(np.isfinite(hh),hh,-np.inf),axis=1);hourly["profile_amplitude"]=np.nanmax(hh,axis=1)-np.nanmin(hh,axis=1)
    for c,label in [("peak_hour","Peak activity clock hour"),("profile_amplitude","24-hour profile amplitude")]:
        fig,ax=new_fig();
        for target,g,color in [(0,"Control",PALETTE["control"]),(1,"Incident dementia",PALETTE["incident"])]:
            if c=="peak_hour":
                counts=hourly.loc[hourly.target.eq(target),c].value_counts(normalize=True).reindex(range(24),fill_value=0);ax.plot(range(24),counts*100,"o-",color=color,label=g)
            else:sns.kdeplot(data=hourly[hourly.target.eq(target)],x=c,fill=True,alpha=.15,color=color,label=g,ax=ax)
        finish(ax,label,"Clock hour" if c=="peak_hour" else "Acceleration difference","Participants (%)" if c=="peak_hour" else "Density");save_figure(fig,module,f"distribution_{c}",label,"Participant hourly profiles","supplement")

    # Clock-period densities provide interpretable distributional views beyond mean profiles.
    periods={"night_00_06":list(range(0,6)),"morning_06_12":list(range(6,12)),"afternoon_12_18":list(range(12,18)),"evening_18_24":list(range(18,24))}
    for name,hours in periods.items():
        hourly[name]=hourly[[hcols[h] for h in hours]].mean(axis=1)
        fig,ax=new_fig()
        for target,g,color in [(0,"Control",PALETTE["control"]),(1,"Incident dementia",PALETTE["incident"])]:sns.kdeplot(data=hourly[hourly.target.eq(target)],x=name,fill=True,alpha=.18,lw=2,color=color,label=g,ax=ax)
        finish(ax,f"Activity distribution: {name.replace('_','–').replace('–00–06',' 00:00–06:00').replace('–06–12',' 06:00–12:00').replace('–12–18',' 12:00–18:00').replace('–18–24',' 18:00–24:00')}","Mean acceleration","Density");save_figure(fig,module,f"clock_period_density_{name}",f"Clock-period density {name}","Participant hourly profiles","candidate")
    # Day-specific outcome distributions and a daily difference confidence band.
    a=daily.loc[daily.target.eq(0),dcols].to_numpy(float);b=daily.loc[daily.target.eq(1),dcols].to_numpy(float);delta=np.nanmean(b,0)-np.nanmean(a,0);se=np.sqrt(np.nanvar(a,0,ddof=1)/len(a)+np.nanvar(b,0,ddof=1)/len(b))
    fig,ax=new_fig();x=np.arange(1,8);ax.errorbar(x,delta,yerr=1.96*se,fmt="o-",capsize=4,lw=2,color=PALETTE["incident"]);ax.axhline(0,color="#68747C",lw=.8);finish(ax,"Day-specific activity difference","Monitoring day","Incident minus control (95% CI)",False);save_figure(fig,module,"daily_difference_ci","Daily PA difference with 95% CI","Participant daily profiles","main candidate")
    long=daily.melt(id_vars=["eid","target"],value_vars=dcols,var_name="day",value_name="activity")
    fig,ax=new_fig(8,5.5);sns.boxenplot(data=long,x="day",y="activity",hue="target",palette=[PALETTE["control"],PALETTE["incident"]],showfliers=False,ax=ax);ax.legend(["Control","Incident dementia"],frameon=False);finish(ax,"Day-specific activity distributions","Monitoring day","Mean acceleration");save_figure(fig,module,"daily_boxen_outcome","Day-specific PA distributions","Participant daily profiles","candidate")
    # Correlation structure demonstrates same-clock consistency and day-to-day stability.
    ch=hourly[hcols].corr(method="spearman");fig,ax=new_fig(8,7);sns.heatmap(ch,cmap=SEQ_CMAP,vmin=0,vmax=1,square=True,cbar_kws={"label":"Spearman r"},ax=ax);ax.set_title("Correlation structure across clock hours",loc="left",fontweight="bold");save_figure(fig,module,"hourly_correlation_heatmap","Hourly PA correlation structure","Participant hourly profiles","supplement")
    cd=daily[dcols].corr(method="spearman");fig,ax=new_fig(6.5,5.5);sns.heatmap(cd,cmap=SEQ_CMAP,vmin=0,vmax=1,annot=True,fmt=".2f",square=True,cbar_kws={"label":"Spearman r"},ax=ax);ax.set_title("Correlation structure across monitoring days",loc="left",fontweight="bold");save_figure(fig,module,"daily_correlation_heatmap","Daily PA correlation structure","Participant daily profiles","candidate")
    # Subtype-specific contrast curves with uncertainty.
    for col,label,color in subtype_specs:
        if col not in hourly:continue
        aa=hourly.loc[hourly[col].eq(0),hcols].to_numpy(float);bb=hourly.loc[hourly[col].eq(1),hcols].to_numpy(float)
        if len(bb)<3:continue
        dd=np.nanmean(bb,0)-np.nanmean(aa,0);ss=np.sqrt(np.nanvar(aa,0,ddof=1)/len(aa)+np.nanvar(bb,0,ddof=1)/len(bb));fig,ax=new_fig(8,5);ax.plot(range(24),dd,lw=2,color=color);ax.fill_between(range(24),dd-1.96*ss,dd+1.96*ss,color=color,alpha=.18);ax.axhline(0,color="#68747C",lw=.8);ax.set_xticks(range(0,24,3),[f"{h:02d}:00" for h in range(0,24,3)]);finish(ax,f"Hourly contrast: {label}","Clock hour","Subtype minus comparison (95% CI)",False);save_figure(fig,module,f"hourly_contrast_{col}",f"Hourly contrast {label}","UKB subtype labels and hourly profiles","supplement")


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
    save_figure(fig,module,f"{stem}_{kind}",f"{kind.upper()} curve for {model_from_probability(col)}",source,"candidate")


def metric_forest(frame: pd.DataFrame, metric: str, module: str, stem: str, title: str, source: str, label_col: str="model", use: str="candidate") -> None:
    q=frame.dropna(subset=[metric]).copy().sort_values(metric);labels=q[label_col].map(metric_label) if label_col=="model" else q[label_col].astype(str)
    lo=q.get(f"{metric}_ci_low",q[metric]);hi=q.get(f"{metric}_ci_high",q[metric]);x=q[metric].to_numpy();y=np.arange(len(q))
    fig,ax=new_fig(8,max(4.8,.43*len(q)+1.7));ax.errorbar(x,y,xerr=[np.maximum(0,x-np.asarray(lo)),np.maximum(0,np.asarray(hi)-x)],fmt="o",ms=6,color=PALETTE["hybrid"],ecolor="#6B7280",capsize=3);ax.set_yticks(y,labels);finish(ax,title,metric.upper(),"",False);save_figure(fig,module,stem,title,source,use)


def calibration_panel(pred: pd.DataFrame, col: str, module: str, stem: str, source: str) -> None:
    q=pred[["target",col]].dropna().sort_values(col);q["bin"]=pd.qcut(q[col],10,duplicates="drop");g=q.groupby("bin",observed=True).agg(predicted=(col,"mean"),events=("target","sum"),n=("target","size")).reset_index();g["observed"]=g.events/g.n
    cis=np.array([proportion_confint(e,n,method="wilson") for e,n in zip(g.events,g.n)])
    lower=np.maximum(0,g.observed.to_numpy(float)-cis[:,0]);upper=np.maximum(0,cis[:,1]-g.observed.to_numpy(float));fig,ax=new_fig();ax.errorbar(g.predicted,g.observed,yerr=np.vstack([lower,upper]),fmt="o-",color=PALETTE["hybrid"],capsize=3,label="Deciles (95% CI)");mx=max(g.predicted.max(),g.observed.max(),1e-6)*1.08;ax.plot([0,mx],[0,mx],"--",color="#999",label="Ideal");finish(ax,f"Calibration: {metric_label(model_from_probability(col))}","Mean predicted risk","Observed incidence");save_figure(fig,module,stem,"Calibration with Wilson confidence intervals",source,"main candidate")


def internal_module(d: pd.DataFrame) -> None:
    module="02_core_model";root=V13/"02_core_model";met=pd.read_csv(root/"ukb_20pct_final_test_metrics.csv");pred=pd.read_csv(root/"ukb_20pct_final_test_predictions.csv")
    metric_forest(met,"auprc",module,"auprc_forest","Internal discrimination by AUPRC",str(root),use="main candidate");metric_forest(met,"auroc",module,"auroc_forest","Internal discrimination by AUROC",str(root),use="main candidate");metric_forest(met,"brier",module,"brier_forest","Internal prediction error by Brier score",str(root))
    for c in probability_columns(pred):curve_panel(pred,c,module,clean_name(model_from_probability(c)),"roc",str(root),SERIES[probability_columns(pred).index(c)%len(SERIES)]);curve_panel(pred,c,module,clean_name(model_from_probability(c)),"pr",str(root),SERIES[probability_columns(pred).index(c)%len(SERIES)])
    calibration_panel(pred,"probability_final_hybrid",module,"final_hybrid_calibration",str(root))
    q=pred[["target","probability_final_hybrid"]].copy();q["decile"]=pd.qcut(q.probability_final_hybrid,10,labels=False,duplicates="drop")+1;g=q.groupby("decile").agg(predicted=("probability_final_hybrid","mean"),events=("target","sum"),n=("target","size"));g["observed"]=g.events/g.n;cis=np.array([proportion_confint(e,n,method="wilson") for e,n in zip(g.events,g.n)])
    fig,ax=new_fig(7,5.2);ax.bar(g.index,g.observed*100,color=SEQ_CMAP(np.linspace(.25,.9,len(g))),label="Observed");ax.errorbar(g.index,g.observed*100,yerr=[np.maximum(0,(g.observed-cis[:,0])*100),np.maximum(0,(cis[:,1]-g.observed)*100)],fmt="none",ecolor="#333",capsize=3);ax.plot(g.index,g.predicted*100,"o-",color="#243B53",label="Predicted");finish(ax,"Observed dementia incidence across predicted-risk deciles","Predicted-risk decile","Risk (%)");save_figure(fig,module,"risk_decile_gradient","Risk decile gradient",str(root),"main candidate")
    for target,label,color in [(0,"Control",PALETTE["control"]),(1,"Incident dementia",PALETTE["incident"])]:
        pass
    fig,ax=new_fig();
    for target,label,color in [(0,"Control",PALETTE["control"]),(1,"Incident dementia",PALETTE["incident"])]:sns.kdeplot(data=pred[pred.target.eq(target)],x="probability_final_hybrid",fill=True,alpha=.2,color=color,label=label,ax=ax)
    ax.set_xscale("log");finish(ax,"Predicted-risk separation in the locked 20% test set","Predicted risk (log scale)","Density");save_figure(fig,module,"risk_density_final_hybrid","Risk distribution by outcome",str(root),"main candidate")
    top=pd.read_csv(root/"ukb_20pct_top_risk_enrichment.csv");top=top[top.model.eq("final_hybrid")]
    for ycol,ylabel in [("event_capture_fraction","Events captured"),("enrichment","Enrichment over prevalence"),("ppv","Positive predictive value")]:
        fig,ax=new_fig();ax.plot(top.top_fraction*100,top[ycol]*(100 if ycol!="enrichment" else 1),"o-",lw=2.2,color=PALETTE["hybrid"]);finish(ax,f"Top-risk targeting: {ylabel.lower()}","Highest-risk population selected (%)",ylabel+(" (%)" if ycol!="enrichment" else " (×)"),False);save_figure(fig,module,f"top_risk_{ycol}",ylabel,str(root),"candidate")
    dc=pd.read_csv(V13/"10_decision_analysis"/"decision_curve_data.csv");fig,ax=new_fig(7.5,5.5)
    for i,(name,z) in enumerate(dc.groupby("model")):
        if name in ["final_hybrid","age_sex","dual_pa_feature","treat_all","treat_none"]:ax.plot(z.threshold,z.net_benefit,lw=2 if name=="final_hybrid" else 1.4,label=metric_label(name),color=SERIES[i%len(SERIES)])
    finish(ax,"Decision-curve analysis in the locked test set","Risk threshold","Net benefit");save_figure(fig,module,"decision_curve","Clinical net benefit",str(V13/"10_decision_analysis"),"main candidate")
    hist=pd.read_csv(root/"development_training_history.csv")
    for name,z in hist.groupby("model"):
        fig,ax=new_fig();ax.plot(z.epoch,z.train_loss,"o-",color=PALETTE["gru"],label="Training loss");ax2=ax.twinx();ax2.plot(z.epoch,z.validation_auprc,"s--",color=PALETTE["incident"],label="Validation AUPRC");finish(ax,f"Development learning curve: {metric_label(name)}","Epoch","Training loss",False);ax2.set_ylabel("Validation AUPRC");save_figure(fig,module,f"learning_curve_{name}",f"Learning curve {name}",str(root),"supplement")


def algorithm_module() -> None:
    module="01_algorithm_pool";root=V13/"01_algorithm_pool";met=pd.read_csv(root/"fixed_20pct_algorithm_metrics.csv");pred=pd.read_csv(root/"fixed_20pct_algorithm_predictions.csv");hist=pd.read_csv(root/"development_history.csv")
    for metric,title in [("auprc","Algorithm benchmark: AUPRC"),("auroc","Algorithm benchmark: AUROC"),("brier","Algorithm benchmark: Brier score")]:metric_forest(met,metric,module,f"algorithm_{metric}_forest",title,str(root),use="main candidate")
    fig,ax=new_fig(7,5.5);ax.scatter(met.auroc,met.auprc,s=70,c=np.arange(len(met)),cmap=SEQ_CMAP,edgecolor="white");
    for _,r in met.iterrows():ax.annotate(metric_label(r.model),(r.auroc,r.auprc),xytext=(4,3),textcoords="offset points",fontsize=8.5)
    finish(ax,"Discrimination trade-off across temporal architectures","AUROC","AUPRC",False);save_figure(fig,module,"algorithm_pareto_auroc_auprc","Algorithm Pareto plot",str(root),"main candidate")
    ranks=met.set_index("model")[["auprc","auroc","brier","calibration_slope"]].copy();ranks["brier"]=-ranks.brier;ranks=ranks.rank(ascending=False)
    fig,ax=new_fig(7,6);sns.heatmap(ranks,cmap=SEQ_CMAP.reversed(),annot=True,fmt=".0f",cbar_kws={"label":"Rank (1 = best)"},ax=ax);ax.set_title("Cross-metric architecture ranking",loc="left",fontweight="bold");save_figure(fig,module,"algorithm_rank_heatmap","Cross-metric algorithm ranks",str(root),"candidate")
    pc=pred[probability_columns(pred)].corr(method="spearman");pc.index=[metric_label(model_from_probability(x)) for x in pc.index];pc.columns=pc.index
    fig,ax=new_fig(9,7.5);sns.heatmap(pc,cmap=SEQ_CMAP,vmin=0,vmax=1,annot=True,fmt=".2f",square=True,cbar_kws={"label":"Spearman r"},ax=ax);ax.set_title("Algorithm prediction concordance",loc="left",fontweight="bold");save_figure(fig,module,"algorithm_prediction_correlation","Prediction correlation across algorithms",str(root),"candidate")
    for c in probability_columns(pred):curve_panel(pred,c,module,clean_name(model_from_probability(c)),"pr",str(root),SERIES[probability_columns(pred).index(c)%len(SERIES)])
    for name,z in hist.groupby("model"):
        fig,ax=new_fig();ax.plot(z.epoch,z.validation_auprc,"o-",color=PALETTE["mamba"]);best=z.loc[z.validation_auprc.idxmax()];ax.scatter(best.epoch,best.validation_auprc,s=80,color=PALETTE["incident"],zorder=4,label=f"Selected epoch {int(best.epoch)}");finish(ax,f"Validation trajectory: {metric_label(name)}","Epoch","Validation AUPRC");save_figure(fig,module,f"validation_trajectory_{name}",f"Validation trajectory {name}",str(root),"supplement")


def ablation_module() -> None:
    module="08_ablation";root=V13/"08_ablation";met=pd.read_csv(root/"ablation_test_metrics.csv");pred=pd.read_csv(root/"ablation_test_predictions.csv")
    for metric,title in [("auprc","Component ablation by AUPRC"),("auroc","Component ablation by AUROC"),("brier","Component ablation by Brier score")]:metric_forest(met,metric,module,f"ablation_{metric}_forest",title,str(root),use="main candidate")
    base=float(met.loc[met.model.eq("age_sex"),"auprc"].iloc[0]);q=met.copy();q["increment"]=q.auprc-base;q=q.sort_values("increment")
    fig,ax=new_fig(8,6);ax.barh(q.model.map(metric_label),q.increment,color=np.where(q.increment>=0,PALETTE["hybrid"],"#BCC5CF"));ax.axvline(0,color="#333",lw=.8);finish(ax,"AUPRC increment over age and sex","AUPRC difference","",False);save_figure(fig,module,"ablation_increment_waterfall","AUPRC gain over age-sex baseline",str(root),"main candidate")
    for c in probability_columns(pred):curve_panel(pred,c,module,clean_name(model_from_probability(c)),"pr",str(root),SERIES[probability_columns(pred).index(c)%len(SERIES)])
    shap=pd.read_csv(V13/"13_hierarchical_shapley"/"modality_shapley_summary.csv");
    for g,z in shap.groupby("group"):
        fig,ax=new_fig(7,4.8);z=z.sort_values("mean_absolute_shapley");ax.errorbar(z.mean_shapley_probability,np.arange(len(z)),xerr=[np.maximum(0,z.mean_shapley_probability-z.ci_low),np.maximum(0,z.ci_high-z.mean_shapley_probability)],fmt="o",capsize=3,color=PALETTE["hybrid"]);ax.axvline(0,color="#888",lw=.8);ax.set_yticks(np.arange(len(z)),z.source.map(metric_label));finish(ax,f"Modality Shapley effects: {g}","Mean probability contribution (95% CI)","",False);save_figure(fig,module,f"modality_shapley_forest_{g}",f"Modality Shapley {g}",str(V13/"13_hierarchical_shapley"),"main candidate")
    ind=pd.read_csv(V13/"13_hierarchical_shapley"/"modality_shapley_individual.csv");long=ind.melt(id_vars=["eid","target"],value_vars=["shapley_gru","shapley_mamba","shapley_clinical"],var_name="source",value_name="shapley");long["source"]=long.source.str.replace("shapley_","",regex=False)
    for source,z in long.groupby("source"):
        fig,ax=new_fig();sns.violinplot(data=z,x="target",y="shapley",palette=[PALETTE["control"],PALETTE["incident"]],inner="quart",cut=0,ax=ax);ax.axhline(0,color="#777",lw=.8);ax.set_xticks([0,1],["Control","Incident dementia"]);finish(ax,f"Individual Shapley effects: {metric_label(source)}","","Probability contribution",False);save_figure(fig,module,f"individual_shapley_{source}",f"Individual Shapley effects {source}",str(V13/"13_hierarchical_shapley"),"supplement")
    paired=pd.read_csv(V13/"18_final_paired"/"final_paired_increment_tests.csv")
    for metric in ["delta_auprc","delta_auroc","delta_brier"]:
        q=paired.sort_values(metric);fig,ax=new_fig(8,5.5);x=q[metric].to_numpy();lo=q[f"{metric}_ci_low"].to_numpy();hi=q[f"{metric}_ci_high"].to_numpy();y=np.arange(len(q));ax.errorbar(x,y,xerr=[np.maximum(0,x-lo),np.maximum(0,hi-x)],fmt="o",capsize=3,color=PALETTE["hybrid"]);ax.axvline(0,color="#777",ls="--");ax.set_yticks(y,q.comparison.str.replace("_"," ").str.title());finish(ax,f"Paired incremental comparison: {metric.replace('delta_','').upper()}","Model minus reference","",False);save_figure(fig,module,f"paired_increment_{metric}",f"Paired {metric}",str(V13/"18_final_paired"),"candidate")


def external_module() -> None:
    module="04_nhats_external";root=V13/"04_nhats_external";met=pd.read_csv(root/"nhats_external_metrics.csv");pred=pd.read_csv(root/"nhats_external_predictions.csv")
    primary=met[met.analysis.eq("primary_complete_demographics")].copy()
    for metric,title in [("auprc","NHATS external AUPRC"),("auroc","NHATS external AUROC")]:
        q=primary.copy();q["label"]=q.harmonization.str.replace("_"," ").str.title()+" | "+q.model.map(metric_label);metric_forest(q,metric,module,f"external_{metric}_forest",title,str(root),label_col="label",use="main candidate")
    for metric in ["auprc","auroc"]:
        pivot=primary.pivot(index="model",columns="harmonization",values=metric)
        fig,ax=new_fig(9,5.5);sns.heatmap(pivot,annot=True,fmt=".3f",cmap=SEQ_CMAP,cbar_kws={"label":metric.upper()},ax=ax);ax.set_xlabel("Device harmonization");ax.set_ylabel("");ax.set_yticklabels([metric_label(x.get_text()) for x in ax.get_yticklabels()],rotation=0);ax.set_title(f"External {metric.upper()} across device harmonization strategies",loc="left",fontweight="bold");save_figure(fig,module,f"harmonization_{metric}_heatmap",f"NHATS harmonization {metric}",str(root),"main candidate")
    cols=[c for c in pred if "__" in c]
    for c in cols:
        q=pred[["target",c]].rename(columns={c:"probability_"+c}).dropna();curve_panel(q,"probability_"+c,module,clean_name(c),"pr",str(root),SERIES[cols.index(c)%len(SERIES)])
    best="learned_adapter__final_hybrid" if "learned_adapter__final_hybrid" in pred else "positive_robust__final_hybrid"
    q=pred[["target",best]].rename(columns={best:"probability_final_hybrid"});calibration_panel(q,"probability_final_hybrid",module,"nhats_final_calibration",str(root))
    fig,ax=new_fig();
    for target,label,color in [(0,"No incident dementia",PALETTE["control"]),(1,"Incident dementia",PALETTE["incident"])]:sns.kdeplot(data=pred[pred.target.eq(target)],x=best,fill=True,alpha=.2,color=color,label=label,ax=ax)
    finish(ax,"NHATS predicted-risk distribution","Predicted risk","Density");save_figure(fig,module,"nhats_risk_density","NHATS risk distribution",str(root),"main candidate")
    enrich=pd.read_csv(root/"risk_enrichment.csv")
    for endpoint,z0 in enrich.groupby("endpoint"):
        for measure,label in [("event_capture_fraction","Events captured"),("enrichment","Risk enrichment")]:
            fig,ax=new_fig();
            for i,(score,z) in enumerate(z0.groupby("score")):ax.plot(z.top_fraction*100,z[measure]*(100 if measure=="event_capture_fraction" else 1),"o-",label=score.replace("_"," ").title(),color=SERIES[i%len(SERIES)])
            finish(ax,f"NHATS top-risk analysis: {label.lower()}","Highest-risk population selected (%)",label+(" (%)" if measure=="event_capture_fraction" else " (×)"));save_figure(fig,module,f"enrichment_{clean_name(endpoint)}_{measure}",f"NHATS {label}",str(root),"candidate")
    # External endpoint and age-overlap sensitivity.
    sens=met[~met.analysis.eq("primary_complete_demographics") & met.model.eq("final_hybrid")]
    for metric in ["auprc","auroc"]:
        q=sens.copy();q["label"]=q.analysis.str.replace("_"," ").str.title()+" | "+q.harmonization.str.replace("_"," ").str.title();metric_forest(q,metric,module,f"external_sensitivity_{metric}",f"External sensitivity analyses: {metric.upper()}",str(root),label_col="label")
    # Dataset-shift panels based on participant hourly profiles and NHATS harmonized raw series.
    candidates=list(EXT_BASE.rglob("nhats_r11_daily_7x1440_positive_robust.csv")) if EXT_BASE.exists() else []
    nh_path=candidates[0] if candidates else EXT_BASE/"nhats_r11_daily_7x1440_positive_robust.csv"
    if nh_path.exists():
        nh=pd.read_csv(nh_path,low_memory=False);idcol=nh.columns[0];minute=nh.iloc[:,1:1441].apply(pd.to_numeric,errors="coerce").to_numpy(float)
        # The harmonized NHATS file is in person-day format: seven consecutive 1,440-minute rows per eid.
        blocks=[]
        for _,idx in nh.groupby(idcol,sort=False).groups.items():
            a=minute[np.asarray(list(idx),int)]
            if len(a)>=7:blocks.append(a[:7])
        arr=np.asarray(blocks,float);nh_hour=np.nanmean(arr.reshape(len(arr),7,24,60),axis=(1,3));nh_day=np.nanmean(arr,axis=2)
        uk,_=load_profiles();hcols=[f"hour_{i:02d}" for i in range(24)];uk_a=uk[hcols].to_numpy(float)
        for normalize in [False,True]:
            ua=np.nanmean(uk_a,0);na=np.nanmean(nh_hour,axis=0);
            if normalize:ua=(ua-ua.mean())/ua.std();na=(na-na.mean())/na.std()
            fig,ax=new_fig(8,5.2);ax.plot(range(24),ua,lw=2.3,color=PALETTE["control"],label="UK Biobank");ax.plot(range(24),na,lw=2.3,color=PALETTE["external"],label="NHATS");ax.set_xticks(range(0,24,3),[f"{h:02d}:00" for h in range(0,24,3)]);finish(ax,"UKB–NHATS 24-hour activity profiles"+(" (standardized)" if normalize else ""),"Clock hour","Standardized activity" if normalize else "Device-specific activity scale");save_figure(fig,module,"ukb_nhats_hourly_"+("standardized" if normalize else "raw_scale"),"UKB versus NHATS hourly profiles",str(nh_path),"main candidate")
        fig,ax=new_fig(7,5);ud=np.nanmean(pd.read_csv(DESC/"participant_daily_means.csv").iloc[:,2:9].to_numpy(float),0);nd=np.nanmean(nh_day,0);ax.plot(range(1,8),(ud-ud.mean())/ud.std(),"o-",color=PALETTE["control"],label="UK Biobank");ax.plot(range(1,8),(nd-nd.mean())/nd.std(),"s-",color=PALETTE["external"],label="NHATS");finish(ax,"Standardized seven-day profiles across cohorts","Monitoring day","Within-cohort standardized activity");save_figure(fig,module,"ukb_nhats_daily_standardized","UKB versus NHATS daily profiles",str(nh_path),"candidate")


def interpretability_module(d: pd.DataFrame) -> None:
    module="13_hierarchical_shapley";root=V13/"13_hierarchical_shapley";temp=pd.read_csv(root/"temporal_shapley_7day_x_4period.csv")
    for group,z in temp.groupby("group"):
        for metric,title in [("mean_shapley_probability","Signed temporal Shapley contribution"),("mean_absolute_shapley","Absolute temporal Shapley importance")]:
            p=z.pivot(index="day",columns="clock_start",values=metric);fig,ax=new_fig(8,5.2);sns.heatmap(p,cmap=DIV_CMAP if "absolute" not in metric else SEQ_CMAP,center=0 if "absolute" not in metric else None,annot=True,fmt=".4f",cbar_kws={"label":"Probability contribution"},ax=ax);ax.set_xlabel("Clock period start (hour)");ax.set_ylabel("Monitoring day");ax.set_title(f"{title}: {group}",loc="left",fontweight="bold");save_figure(fig,module,f"temporal_heatmap_{clean_name(group)}_{metric}",f"{title} {group}",str(root),"main candidate")
    byday=pd.read_csv(root/"temporal_shapley_by_day.csv");byclock=pd.read_csv(root/"temporal_shapley_by_clock.csv")
    for frame,xcol,prefix in [(byday,"day","day"),(byclock,"clock_start","clock")]:
        for metric in ["mean_shapley_probability","mean_absolute_shapley"]:
            fig,ax=new_fig();
            for i,(g,z) in enumerate(frame.groupby("group")):ax.plot(z[xcol],z[metric],"o-",lw=2,label=g,color=SERIES[i%len(SERIES)])
            ax.axhline(0,color="#888",lw=.8);finish(ax,f"Temporal attribution by {prefix}: {metric.replace('_',' ')}",prefix.title(),"Probability contribution");save_figure(fig,module,f"shapley_by_{prefix}_{metric}",f"Shapley by {prefix}",str(root),"candidate")
    ind=pd.read_csv(root/"temporal_shapley_individual.csv");scols=[c for c in ind if c.startswith("shapley_day")]
    for target,label in [(0,"control"),(1,"incident_dementia")]:
        q=ind[ind.target.eq(target)].copy();q=q.sort_values("full_probability",ascending=False).head(200);a=q[scols].to_numpy(float);fig,ax=new_fig(10,6);sns.heatmap(a,cmap=DIV_CMAP,center=0,xticklabels=[c.replace("shapley_","").replace("_"," ") for c in scols],yticklabels=False,cbar_kws={"label":"Shapley contribution"},ax=ax);ax.tick_params(axis="x",rotation=70);ax.set_ylabel("Participants ranked by risk");ax.set_title(f"Participant temporal attributions: {label.replace('_',' ')}",loc="left",fontweight="bold");save_figure(fig,module,f"individual_temporal_heatmap_{label}",f"Individual temporal Shapley {label}",str(root),"candidate")
    mod=pd.read_csv(root/"modality_shapley_individual.csv")
    for c in ["shapley_gru","shapley_mamba","shapley_clinical"]:
        fig,ax=new_fig();sns.scatterplot(data=mod,x="full_probability",y=c,hue="target",palette=[PALETTE["control"],PALETTE["incident"]],alpha=.35,s=18,ax=ax);ax.axhline(0,color="#888",lw=.8);finish(ax,f"Risk dependence of {c.replace('shapley_','').upper()} attribution","Final predicted risk","Shapley probability contribution");save_figure(fig,module,f"dependence_{c}",f"Shapley dependence {c}",str(root),"supplement")
        fig,ax=new_fig();
        for target,label,color in [(0,"Control",PALETTE["control"]),(1,"Incident dementia",PALETTE["incident"])]:sns.kdeplot(data=mod[mod.target.eq(target)],x=c,fill=True,alpha=.18,lw=2,color=color,label=label,ax=ax)
        ax.axvline(0,color="#68747C",lw=.8);finish(ax,f"Distribution of {c.replace('shapley_','').upper()} contributions","Shapley probability contribution","Density");save_figure(fig,module,f"density_{c}",f"Shapley contribution density {c}",str(root),"candidate")
    fig,ax=new_fig();sns.scatterplot(data=mod,x="baseline_reference_probability",y="full_probability",hue="target",palette=[PALETTE["control"],PALETTE["incident"]],alpha=.4,s=20,ax=ax);mx=max(mod.baseline_reference_probability.max(),mod.full_probability.max());ax.plot([0,mx],[0,mx],"--",color="#A5ADB1");finish(ax,"Risk revision after multimodal fusion","Reference probability","Full-model probability");save_figure(fig,module,"risk_revision_after_fusion","Risk revision after fusion",str(root),"main candidate")
    fig,ax=new_fig();sns.histplot(data=mod,x="efficiency_error",bins=40,color=PALETTE["neutral"],ax=ax);finish(ax,"Shapley efficiency diagnostic","Reconstruction error","Participants",False);save_figure(fig,module,"shapley_efficiency_error","Shapley efficiency error",str(root),"supplement")
    occ=pd.read_csv(V13/"09_interpretability"/"temporal_occlusion_summary.csv")
    for group,z in occ.groupby("group"):
        q=z.sort_values("mean_delta_probability").tail(15);fig,ax=new_fig(8,6);ax.barh(q.window,q.mean_delta_probability,color=SEQ_CMAP(np.linspace(.2,.9,len(q))));finish(ax,f"Temporal occlusion importance: {group}","Mean probability decrease","",False);save_figure(fig,module,f"occlusion_importance_{clean_name(group)}",f"Occlusion importance {group}",str(V13/"09_interpretability"),"candidate")
    var=pd.read_csv(V13/"11_representations"/"latent_pca_variance.csv");fig,ax=new_fig();ax.bar(var.component,var.explained_variance_ratio*100,color=PALETTE["mamba"]);ax2=ax.twinx();ax2.plot(var.component,var.cumulative_explained_variance*100,"o-",color=PALETTE["incident"]);ax.tick_params(axis="x",rotation=45);finish(ax,"Latent representation variance","Latent principal component","Explained variance (%)",False);ax2.set_ylabel("Cumulative variance (%)");save_figure(fig,module,"latent_pca_variance","Latent PCA variance",str(V13/"11_representations"),"candidate")
    pcs=[f"latent_pc{i:02d}" for i in range(1,17)];q=d[d.split.eq("test")].dropna(subset=pcs[:2]).copy();sample=q.sample(min(5000,len(q)),random_state=20260820);fig,ax=new_fig(7,6);sns.scatterplot(data=sample,x="latent_pc01",y="latent_pc02",hue="target",palette=[PALETTE["control"],PALETTE["incident"]],alpha=.45,s=20,ax=ax);finish(ax,"Latent PA representation in the locked test set","Latent PC1","Latent PC2");save_figure(fig,module,"latent_pc_scatter_outcome","Latent PC map by outcome",str(V13/"11_representations"),"main candidate")
    X=StandardScaler().fit_transform(q[pcs].to_numpy());q["cluster"]=KMeans(5,random_state=20260820,n_init=20).fit_predict(X);g=q.groupby("cluster").agg(n=("target","size"),events=("target","sum"),risk=("final_probability","mean"));g["rate"]=g.events/g.n
    fig,ax=new_fig();ax.bar(g.index.astype(str),g.rate*100,color=SEQ_CMAP(np.linspace(.25,.9,len(g))));finish(ax,"Dementia incidence across latent PA phenotypes","Latent phenotype cluster","Incident dementia (%)",False);save_figure(fig,module,"latent_cluster_event_rate","Latent cluster event rate",str(V13/"11_representations"),"candidate")
    corr=pd.read_csv(V13/"14_rhythm_knowledge"/"rhythm_concept_latent_correlations.csv");test=corr[corr.split.eq("test")];pivot=test.pivot(index="concept",columns="latent_component",values="spearman_r");fig,ax=new_fig(11,8);sns.heatmap(pivot,cmap=DIV_CMAP,center=0,vmin=-.6,vmax=.6,cbar_kws={"label":"Spearman r"},ax=ax);ax.set_title("Biological rhythm concepts encoded in latent PA dimensions",loc="left",fontweight="bold");save_figure(fig,module,"rhythm_latent_correlation_heatmap","Rhythm-latent correlation map",str(V13/"14_rhythm_knowledge"),"main candidate")


def clinical_module(d: pd.DataFrame) -> None:
    module="08_clinical_extensions";rhroot=V13/"14_rhythm_knowledge";assoc=pd.read_csv(rhroot/"rhythm_concept_dementia_associations.csv")
    for split,z in assoc.groupby("split"):
        q=z.dropna(subset=["odds_ratio","ci_low","ci_high"]).sort_values("odds_ratio");fig,ax=new_fig(8,max(6,.34*len(q)+1.5));y=np.arange(len(q));ax.errorbar(q.odds_ratio,y,xerr=[q.odds_ratio-q.ci_low,q.ci_high-q.odds_ratio],fmt="o",capsize=2,color=PALETTE["hybrid"]);ax.axvline(1,color="#888",ls="--");ax.set_xscale("log");ax.set_yticks(y,q.concept.str.replace("_"," ").str.title());finish(ax,f"Rhythm concepts associated with incident dementia: {split}","Odds ratio per standardized increment (95% CI)","",False);save_figure(fig,module,f"rhythm_association_forest_{split}",f"Rhythm associations {split}",str(rhroot),"main candidate")
        q2=z.sort_values("spearman_with_final_risk");fig,ax=new_fig(8,max(6,.3*len(q2)+1.5));ax.barh(q2.concept.str.replace("_"," ").str.title(),q2.spearman_with_final_risk,color=np.where(q2.spearman_with_final_risk>=0,PALETTE["incident"],PALETTE["control"]));ax.axvline(0,color="#444",lw=.8);finish(ax,f"Rhythm concepts correlated with model risk: {split}","Spearman r","",False);save_figure(fig,module,f"rhythm_risk_correlation_{split}",f"Rhythm-risk correlations {split}",str(rhroot),"candidate")
    concepts=pd.read_csv(rhroot/"posthoc_rhythm_concepts.csv",dtype={"eid":"string"});q=d[["eid","target","split","final_probability"]].merge(concepts,on="eid",how="inner")
    radar_cols=[c for c in ["overall_activity","interdaily_stability","intradaily_variability","relative_amplitude","day_night_difference","weekend_minus_weekday"] if c in q]
    med=q[radar_cols].median();scale=q[radar_cols].quantile(.75)-q[radar_cols].quantile(.25);theta=np.linspace(0,2*np.pi,len(radar_cols),endpoint=False);theta=np.r_[theta,theta[0]]
    fig=plt.figure(figsize=(7,6.5),constrained_layout=True);ax=fig.add_subplot(111,projection="polar")
    for target,label,color in [(0,"Control",PALETTE["control"]),(1,"Incident dementia",PALETTE["incident"])]:v=((q.loc[q.target.eq(target),radar_cols].median()-med)/scale).to_numpy();v=np.r_[v,v[0]];ax.plot(theta,v,lw=2.2,label=label,color=color);ax.fill(theta,v,alpha=.12,color=color)
    ax.set_xticks(theta[:-1],[x.replace("_"," ").title() for x in radar_cols]);ax.set_title("Multidimensional rhythm phenotype",fontweight="bold",pad=18);ax.legend(frameon=False,loc="upper right",bbox_to_anchor=(1.25,1.15));save_figure(fig,module,"rhythm_phenotype_radar","Rhythm phenotype radar",str(rhroot),"main candidate")
    surv=V13/"15_survival";km=pd.read_csv(surv/"landmark_km_by_risk_quartile.csv")
    fig,ax=new_fig(7.5,5.7)
    for i,(g,z) in enumerate(km.groupby("group")):ax.step(z.time_years,1-z.survival,where="post",lw=2,label=str(g),color=SERIES[i%len(SERIES)])
    finish(ax,"Cumulative dementia incidence by predicted-risk quartile","Years from accelerometer assessment","Cumulative incidence");save_figure(fig,module,"survival_cumulative_incidence_quartiles","Cumulative incidence by risk quartile",str(surv),"main candidate")
    fig,ax=new_fig(7.5,5.4)
    for i,(g,z) in enumerate(km.groupby("group")):ax.step(z.time_years,z.n_at_risk,where="post",lw=2,label=str(g),color=SERIES[i%len(SERIES)])
    finish(ax,"Number at risk over follow-up","Years from accelerometer assessment","Participants at risk");save_figure(fig,module,"survival_number_at_risk","Number at risk",str(surv),"supplement")
    aucd=pd.read_csv(surv/"time_dependent_auc.csv");fig,ax=new_fig();ax.plot(aucd.horizon_years_from_accelerometer,aucd.dynamic_auroc,"o-",lw=2.2,color=PALETTE["hybrid"]);ax.axhline(.5,color="#999",ls="--");finish(ax,"Time-dependent discrimination across prediction horizons","Prediction horizon (years)","Dynamic AUROC",False);save_figure(fig,module,"time_dependent_auroc","Time-dependent AUROC",str(surv),"main candidate")
    cox=pd.read_csv(surv/"landmark_cox_models.csv");cox=cox.dropna(subset=["hazard_ratio"]);fig,ax=new_fig(8,max(5,.38*len(cox)+1.5));q0=cox.sort_values("hazard_ratio");y=np.arange(len(q0));ax.errorbar(q0.hazard_ratio,y,xerr=[q0.hazard_ratio-q0.ci_low,q0.ci_high-q0.hazard_ratio],fmt="o",capsize=3,color=PALETTE["hybrid"]);ax.axvline(1,color="#888",ls="--");ax.set_xscale("log");ax.set_yticks(y,(q0.model+" | "+q0.term).str.replace("_"," "));finish(ax,"Landmark Cox survival models","Hazard ratio (95% CI)","",False);save_figure(fig,module,"landmark_cox_forest","Landmark Cox models",str(surv),"main candidate")
    subroot=V13/"12_subtypes";sm=pd.read_csv(subroot/"subtype_test_metrics.csv")
    for metric in ["auprc","auroc"]:
        q0=sm.copy();q0["label"]=q0.subtype.str.replace("_"," ").str.title()+" | "+q0.model.map(metric_label);metric_forest(q0,metric,module,f"subtype_{metric}_forest",f"Dementia subtype discrimination: {metric.upper()}",str(subroot),label_col="label",use="main candidate")
    counts=pd.read_csv(subroot/"subtype_cohort_counts.csv");fig,ax=new_fig(8,5);x=np.arange(len(counts));ax.bar(x-.18,counts.development_events,.36,label="Development",color=PALETTE["control"]);ax.bar(x+.18,counts.test_events,.36,label="Locked test",color=PALETTE["incident"]);ax.set_xticks(x,counts.subtype.str.replace("_"," ").str.title(),rotation=20,ha="right");finish(ax,"Dementia subtype event counts","Subtype","Events");save_figure(fig,module,"subtype_event_counts","Subtype counts",str(subroot),"candidate")
    spl=pd.read_csv(subroot/"subtype_test_predictions_long.csv")
    for subtype,z0 in spl.groupby("subtype"):
        for model,z in z0.groupby("model"):
            if z.target.nunique()<2:continue
            q1=z.rename(columns={"probability_or_score":"probability_"+model})[["target","probability_"+model]];curve_panel(q1,"probability_"+model,module,f"subtype_{clean_name(subtype)}_{clean_name(model)}","pr",str(subroot),SERIES[list(z0.model.unique()).index(model)%len(SERIES)])
    aporoot=V13/"03_apoe_secondary";apm=pd.read_csv(aporoot/"apoe_complete_test_metrics.csv")
    for metric in ["auprc","auroc"]:metric_forest(apm,metric,module,f"apoe_{metric}_forest",f"APOE secondary analysis: {metric.upper()}",str(aporoot),use="candidate")
    interactions=pd.read_csv(V13/"16_interactions"/"risk_score_interaction_tests.csv").dropna(subset=["odds_ratio"]);q0=interactions.sort_values("odds_ratio");fig,ax=new_fig(8,max(5,.4*len(q0)+1.5));y=np.arange(len(q0));ax.errorbar(q0.odds_ratio,y,xerr=[q0.odds_ratio-q0.ci_low,q0.ci_high-q0.odds_ratio],fmt="o",capsize=3,color=PALETTE["hybrid"]);ax.axvline(1,color="#888",ls="--");ax.set_xscale("log");ax.set_yticks(y,q0.modifier.str.replace("_"," ").str.title());finish(ax,"Risk-score interaction tests","Interaction odds ratio (95% CI)","",False);save_figure(fig,module,"subgroup_interaction_forest","Risk score interactions",str(V13/"16_interactions"),"candidate")
    clin=V13/"17_clinical_associations";probe=pd.read_csv(clin/"clinical_linear_probe_test_metrics.csv")
    for metric in ["auroc","auprc","r2","spearman_r"]:
        q0=probe.dropna(subset=[metric]).sort_values(metric);fig,ax=new_fig(8,max(4.8,.36*len(q0)+1.5));ax.barh((q0.outcome+" | "+q0.analysis).str.replace("_"," ").str.title(),q0[metric],color=SEQ_CMAP(np.linspace(.25,.9,len(q0))));finish(ax,f"Latent clinical probes: {metric.upper()}",metric.upper(),"",False);save_figure(fig,module,f"clinical_probe_{metric}",f"Clinical probe {metric}",str(clin),"candidate")
    lat=pd.read_csv(clin/"latent_clinical_associations.csv");lat=lat[lat.split.eq("test")].dropna(subset=["coefficient"]);top=lat.assign(abscoef=lat.coefficient.abs()).sort_values("abscoef").tail(25);fig,ax=new_fig(8,8);ax.barh((top.outcome+" | "+top.feature).str.replace("_"," "),top.coefficient,color=np.where(top.coefficient>=0,PALETTE["incident"],PALETTE["control"]));ax.axvline(0,color="#444",lw=.8);finish(ax,"Strongest latent–clinical associations in the locked test set","Standardized coefficient","",False);save_figure(fig,module,"latent_clinical_association_top25","Latent clinical associations",str(clin),"candidate")
    mort=pd.read_csv(clin/"mortality_cox_associations.csv");mort=mort[mort.split.eq("test")].dropna(subset=["hazard_ratio"]);fig,ax=new_fig(8,max(4.8,.4*len(mort)+1.5));q0=mort.sort_values("hazard_ratio");y=np.arange(len(q0));ax.errorbar(q0.hazard_ratio,y,xerr=[q0.hazard_ratio-q0.ci_low,q0.ci_high-q0.hazard_ratio],fmt="o",capsize=3,color=PALETTE["incident"]);ax.axvline(1,color="#888",ls="--");ax.set_yticks(y,(q0.model+" | "+q0.term).str.replace("_"," "));finish(ax,"Association of PA-derived risk with mortality","Hazard ratio (95% CI)","",False);save_figure(fig,module,"mortality_cox_forest","Mortality Cox associations",str(clin),"candidate")


def write_manifest() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    current=pd.DataFrame(MANIFEST); old_path=OUT/"figure_manifest.csv"
    if old_path.exists():
        old=pd.read_csv(old_path)
        current=pd.concat([old,current],ignore_index=True).drop_duplicates(["module","figure"],keep="last")
    current.to_csv(old_path,index=False,encoding="utf-8-sig")
    summary=current.groupby("module").size().rename("n_figures").reset_index() if len(current) else pd.DataFrame(columns=["module","n_figures"])
    summary.to_csv(OUT/"figure_counts_by_module.csv",index=False,encoding="utf-8-sig")


def main():
    global BASE,V13,DESC,EXT_BASE,OUT
    modules=["demographics","patterns","internal","algorithm","ablation","external","interpretability","clinical"]
    p=argparse.ArgumentParser(description="Publication-grade V13 figure atlas")
    p.add_argument("--module",default="all",choices=["all"]+modules)
    p.add_argument("--base-dir",default=str(BASE),help="Dementia project directory")
    p.add_argument("--v13-dir",default="",help="V13 result directory containing 00_protocol...18_final_paired")
    p.add_argument("--descriptive-dir",default="",help="Directory containing participant_hourly_profiles.csv")
    p.add_argument("--external-dir",default="",help="NHATS external-validation directory")
    p.add_argument("--output-dir",default="",help="New output directory; original results remain read-only")
    a=p.parse_args();BASE=Path(a.base_dir)
    if a.v13_dir:V13=Path(a.v13_dir)
    else:
        cand=BASE/"Results"/"v13_final";V13=cand if (cand/"02_core_model").exists() else cand/"v13_final"
    DESC=Path(a.descriptive_dir) if a.descriptive_dir else BASE/"descriptive_nature_results"
    EXT_BASE=Path(a.external_dir) if a.external_dir else BASE/"external_validation"
    OUT=Path(a.output_dir) if a.output_dir else V13/"publication_atlas_muted"
    configure_style();OUT.mkdir(parents=True,exist_ok=True);logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s")
    d=load_fixed_cohort();logging.info("Loaded fixed cohort N=%d events=%d",len(d),int(d.target.sum()))
    export_publication_tables(d)
    if a.module in ["all","demographics"]:demographic_module(d)
    if a.module in ["all","patterns"]:pa_pattern_module(d)
    if a.module in ["all","internal"]:internal_module(d)
    if a.module in ["all","algorithm"]:algorithm_module()
    if a.module in ["all","ablation"]:ablation_module()
    if a.module in ["all","external"]:external_module()
    if a.module in ["all","interpretability"]:interpretability_module(d)
    if a.module in ["all","clinical"]:clinical_module(d)
    write_manifest();logging.info("Completed module=%s figures=%d",a.module,len(MANIFEST))


if __name__=="__main__":main()
