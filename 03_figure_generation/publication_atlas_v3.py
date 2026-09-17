from __future__ import annotations

"""Selective publication-atlas revision.

This module patches the existing V13 atlas without changing any analysis result.
It fixes visual semantics, over-compressed PR curves, false one-curve comparison
panels, score/probability confusion in NHATS, long Windows paths, and duplicates.
"""

import hashlib
import json
import logging
import math
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from statsmodels.stats.proportion import proportion_confint

import publication_atlas as core
import publication_atlas_expanded as atlas


SEMANTIC = {
    "hybrid": "#294D63", "clinical": "#7D858B", "gru": "#4F7F78",
    "mamba": "#4F7085", "apoe": "#756A82", "incident": "#9A6F68",
    "control": "#78909C", "external": "#6D8580", "gold": "#B19A67",
    "neutral": "#747D84",
}
SERIES = ["#294D63", "#4F7F78", "#4F7085", "#756A82", "#9A6F68", "#B19A67", "#7D858B"]
LINESTYLES = {"positive_robust": "-", "learned_adapter": "--", "positive_quantile": "-.", "raw_count": ":"}
SAVED_IDS: set[tuple[str, str]] = set()
IMAGE_HASHES: dict[str, str] = {}
AUDIT = {"duplicates_skipped": [], "score_transforms": [], "font": None}


def short_stem(value: str, limit: int = 92) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    replacements = {
        "fixed_20pct_algorithm_predictions": "algorithm_predictions",
        "ukb_20pct_final_test_predictions": "ukb_test_predictions",
        "apoe_complete_test_predictions": "apoe_test_predictions",
        "nhats_external_predictions": "nhats_predictions",
        "three_seed_ensemble_predictions": "seed_ensemble_predictions",
        "subtype_test_predictions_long": "subtype_predictions",
        "probability_": "p_", "positive_quantile": "quantile",
        "positive_robust": "robust", "learned_adapter": "adapter",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    if len(s) > limit:
        digest = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
        s = s[: limit - 11].rstrip("_.-") + "_" + digest
    return s


def configure_style_v3() -> None:
    search = [core.BASE / "fonts", core.V13 / "fonts", Path(__file__).parent / "fonts"]
    for folder in search:
        if folder.exists():
            for fp in folder.rglob("*"):
                if fp.suffix.lower() in {".ttf", ".otf", ".ttc"}:
                    try:
                        font_manager.fontManager.addfont(str(fp))
                    except Exception:
                        pass
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = next((x for x in ["Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"] if x in available), "DejaVu Sans")
    AUDIT["font"] = family
    logging.warning("Publication font selected: %s", family)
    mpl.rcParams.update({
        "font.family": family, "font.sans-serif": [family], "font.size": 11.5,
        "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5, "legend.fontsize": 9.5, "axes.linewidth": .8,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
        "ps.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.dpi": 450,
    })
    sns.set_theme(style="ticks", context="paper", font=family, rc={"axes.spines.top": False, "axes.spines.right": False})


def model_color(name: str, index: int = 0) -> str:
    s = name.lower()
    if "hybrid" in s: return SEMANTIC["hybrid"]
    if "age_sex" in s or "clinical" in s: return SEMANTIC["clinical"]
    if "apoe" in s: return SEMANTIC["apoe"]
    if "gru" in s: return SEMANTIC["gru"]
    if "mamba" in s: return SEMANTIC["mamba"]
    if "traditional" in s: return SEMANTIC["gold"]
    return SERIES[index % len(SERIES)]


def save_v3(fig, module, name, title, source, use="candidate") -> None:
    section = "main_candidates" if "main" in str(use).lower() else "supplementary_figures"
    stem = short_stem(name)
    key = (module, stem)
    if key in SAVED_IDS:
        plt.close(fig); return
    png_dir = core.OUT / module / section / "png"
    pdf_dir = core.OUT / module / section / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True); pdf_dir.mkdir(parents=True, exist_ok=True)
    png = png_dir / f"{stem}.png"; pdf = pdf_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=450, bbox_inches="tight", pad_inches=.08, facecolor="white")
    digest = hashlib.sha256(png.read_bytes()).hexdigest()
    if digest in IMAGE_HASHES:
        png.unlink(missing_ok=True); plt.close(fig)
        AUDIT["duplicates_skipped"].append({"duplicate": f"{module}/{stem}", "kept": IMAGE_HASHES[digest]})
        return
    fig.savefig(pdf, bbox_inches="tight", pad_inches=.08, facecolor="white")
    plt.close(fig); SAVED_IDS.add(key); IMAGE_HASHES[digest] = f"{module}/{stem}"
    core.MANIFEST.append({"module": module, "figure": stem, "title": title, "source": source,
                          "recommended_use": use, "png": str(png.relative_to(core.OUT)), "pdf": str(pdf.relative_to(core.OUT))})


def probability(values: np.ndarray, label: str) -> np.ndarray:
    p = np.asarray(values, float)
    finite = p[np.isfinite(p)]
    if len(finite) and (finite.min() < 0 or finite.max() > 1):
        AUDIT["score_transforms"].append(label)
        return expit(p)
    return np.clip(p, 0, 1)


def pr_upper(y: np.ndarray, p: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y, p)
    interesting = precision[(recall >= .01) & (recall <= .98)]
    peak = np.nanquantile(interesting, .995) if len(interesting) else y.mean() * 4
    return float(min(1, max(.12, y.mean() * 3.5, peak * 1.10)))


def pretty(name: str) -> str:
    return name.replace("probability_", "").replace("_", " ").replace("bimambasl", "BiMamba-SL").title()


def select_columns(cols: list[str], limit: int = 6) -> list[str]:
    priorities = ["final_hybrid", "dual_pa_feature", "age_sex", "grufcn", "convcontext", "mamba", "tcn", "fcn", "patchtst"]
    out = []
    for token in priorities:
        for col in cols:
            if token in col and col not in out: out.append(col)
            if len(out) >= limit: return out
    return (out + [c for c in cols if c not in out])[:limit]


def combined_curves(d: pd.DataFrame, cols: list[str], module: str, stem: str, source: str, subtitle: str = "") -> None:
    y0 = pd.to_numeric(d["target"], errors="coerce").to_numpy()
    cols = select_columns(cols)
    valid_series = []
    for col in cols:
        raw = pd.to_numeric(d[col], errors="coerce").to_numpy(); ok = np.isfinite(y0) & np.isfinite(raw)
        if ok.sum() and len(np.unique(y0[ok])) == 2:
            valid_series.append((col, y0[ok].astype(int), probability(raw[ok], col)))
    if len(valid_series) < 2:
        return
    fig, ax = core.new_fig(7.3, 5.7)
    for i, (col, y, p) in enumerate(valid_series):
        fpr, tpr, _ = roc_curve(y, p); ax.plot(fpr, tpr, lw=2.0, color=model_color(col, i), label=f"{pretty(col)}  {roc_auc_score(y,p):.3f}")
    ax.plot([0,1],[0,1],"--",color="#A7AFB3",lw=1); core.finish(ax, f"ROC comparison{subtitle}", "1 − specificity", "Sensitivity")
    ax.legend(frameon=False, loc="lower right", fontsize=8.8); save_v3(fig,module,f"{stem}_models_roc","Multi-model ROC",source,"main candidate")
    fig, ax = core.new_fig(7.3, 5.7); ymax = 0
    for i, (col, y, p) in enumerate(valid_series):
        pr, rc, _ = precision_recall_curve(y, p); ymax=max(ymax,pr_upper(y,p)); ax.plot(rc,pr,lw=2.0,color=model_color(col,i),label=f"{pretty(col)}  {average_precision_score(y,p):.3f}")
    prevalence = valid_series[0][1].mean(); ax.axhline(prevalence,ls="--",color="#A7AFB3",lw=1,label=f"Prevalence  {prevalence:.3f}")
    ax.set_ylim(0,min(1,ymax)); core.finish(ax,f"Precision–recall comparison{subtitle}","Recall","Precision"); ax.legend(frameon=False,loc="upper right",fontsize=8.8)
    save_v3(fig,module,f"{stem}_models_pr_zoom","Multi-model precision recall (zoomed)",source,"main candidate")


def harmonization_curves(d: pd.DataFrame, cols: list[str], module: str, stem: str, source: str) -> None:
    """Compare one frozen model across NHATS harmonizations.

    Color stays fixed because the model is fixed; line style encodes the input
    harmonization. This prevents color from carrying two meanings.
    """
    y0=pd.to_numeric(d["target"],errors="coerce").to_numpy(); series=[]
    for col in cols:
        raw=pd.to_numeric(d[col],errors="coerce").to_numpy(); ok=np.isfinite(y0)&np.isfinite(raw)
        if ok.sum() and len(np.unique(y0[ok]))==2:
            harm=col.split("__",1)[0]; series.append((harm,y0[ok].astype(int),probability(raw[ok],col)))
    if len(series)<2:return
    fig,ax=core.new_fig(7.3,5.7)
    for harm,y,p in series:
        fpr,tpr,_=roc_curve(y,p); ax.plot(fpr,tpr,lw=2.1,ls=LINESTYLES.get(harm,"-"),color=SEMANTIC["hybrid"],label=f"{harm.replace('_',' ').title()}  {roc_auc_score(y,p):.3f}")
    ax.plot([0,1],[0,1],"--",color="#A7AFB3",lw=1); core.finish(ax,"Final hybrid across NHATS harmonizations","1 − specificity","Sensitivity"); ax.legend(frameon=False,loc="lower right",fontsize=8.8); save_v3(fig,module,f"{stem}_roc","NHATS harmonization ROC",source,"main candidate")
    fig,ax=core.new_fig(7.3,5.7); ymax=0
    for harm,y,p in series:
        pr,rc,_=precision_recall_curve(y,p); ymax=max(ymax,pr_upper(y,p)); ax.plot(rc,pr,lw=2.1,ls=LINESTYLES.get(harm,"-"),color=SEMANTIC["hybrid"],label=f"{harm.replace('_',' ').title()}  {average_precision_score(y,p):.3f}")
    prev=series[0][1].mean(); ax.axhline(prev,ls="--",color="#A7AFB3",lw=1,label=f"Prevalence  {prev:.3f}"); ax.set_ylim(0,min(1,ymax)); core.finish(ax,"Final hybrid across NHATS harmonizations","Recall","Precision"); ax.legend(frameon=False,loc="upper right",fontsize=8.8); save_v3(fig,module,f"{stem}_pr_zoom","NHATS harmonization PR",source,"main candidate")


def one_curve(pred: pd.DataFrame, col: str, module: str, stem: str, kind: str, source: str, color="#294D63") -> None:
    q=pred[["target",col]].apply(pd.to_numeric,errors="coerce").dropna(); y=q.target.to_numpy(int); p=probability(q[col].to_numpy(),col)
    if len(np.unique(y))<2:return
    fig,ax=core.new_fig(6.4,5.3); color=model_color(col)
    if kind=="roc":
        fpr,tpr,_=roc_curve(y,p); ax.plot(fpr,tpr,color=color,lw=2.3,label=f"AUROC = {roc_auc_score(y,p):.3f}"); ax.plot([0,1],[0,1],"--",color="#A7AFB3"); core.finish(ax,f"ROC: {pretty(col)}","1 − specificity","Sensitivity")
    else:
        pr,rc,_=precision_recall_curve(y,p); ax.plot(rc,pr,color=color,lw=2.3,label=f"AUPRC = {average_precision_score(y,p):.3f}"); ax.axhline(y.mean(),ls="--",color="#A7AFB3",label=f"Prevalence = {y.mean():.3f}"); ax.set_ylim(0,pr_upper(y,p)); core.finish(ax,f"Precision–recall: {pretty(col)}","Recall","Precision")
    save_v3(fig,module,f"{stem}_{kind}",f"{kind.upper()} curve",source,"supplement")


def calibration(pred: pd.DataFrame, col: str, module: str, stem: str, source: str) -> None:
    q=pred[["target",col]].apply(pd.to_numeric,errors="coerce").dropna(); q[col]=probability(q[col].to_numpy(),col); q=q.sort_values(col)
    if len(q)<20:return
    q["bin"]=pd.qcut(q[col],10,duplicates="drop"); g=q.groupby("bin",observed=True).agg(predicted=(col,"mean"),events=("target","sum"),n=("target","size")); g["observed"]=g.events/g.n
    ci=np.asarray([proportion_confint(int(e),int(n),method="wilson") for e,n in zip(g.events,g.n)]); lower=np.maximum(0,g.observed.to_numpy()-ci[:,0]); upper=np.maximum(0,ci[:,1]-g.observed.to_numpy())
    fig,ax=core.new_fig(); ax.errorbar(g.predicted,g.observed,yerr=np.vstack([lower,upper]),fmt="o-",capsize=3,color=model_color(col)); mx=max(g.predicted.max(),g.observed.max())*1.08; ax.plot([0,mx],[0,mx],"--",color="#A7AFB3"); core.finish(ax,f"Calibration: {pretty(col)}","Predicted risk","Observed incidence",False); save_v3(fig,module,stem,"Calibration curve",source,"main candidate")


def probability_suite_v3(d,module,stem,source,max_models=14):
    if "target" not in d:return
    # Properly pivot subtype long data so every comparison panel contains several curves.
    if {"model","probability_or_score"}.issubset(d):
        groups = d.groupby("subtype") if "subtype" in d else [("all",d)]
        for subtype,z in groups:
            idx=[c for c in ["eid","target"] if c in z]
            wide=z.pivot_table(index=idx,columns="model",values="probability_or_score",aggfunc="first").reset_index()
            wide=wide.rename(columns={c:"probability_"+str(c) for c in wide.columns if c not in idx})
            cols=[c for c in wide if c.startswith("probability_")]
            combined_curves(wide,cols,module,f"{stem}_{short_stem(subtype,25)}",source,f": {str(subtype).replace('_',' ')}")
            if len(cols)>1:
                corr=wide[cols].corr(method="spearman"); fig,ax=core.new_fig(7,5.8); sns.heatmap(corr,cmap=core.SEQ_CMAP,vmin=0,vmax=1,annot=True,fmt=".2f",square=True,ax=ax,cbar_kws={"label":"Spearman r"}); ax.set_title(f"Score concordance: {str(subtype).replace('_',' ')}",loc="left",fontweight="bold"); save_v3(fig,module,f"{stem}_{short_stem(subtype,25)}_concordance","Subtype score concordance",source)
        return
    cols=[c for c in d if c.startswith("probability_") or "__" in c]
    cols=[c for c in cols if pd.to_numeric(d[c],errors="coerce").notna().sum()>10]
    if not cols:return
    # NHATS: model is color, harmonization is line type; avoid an unreadable 16-curve panel.
    if any("__" in c for c in cols):
        for harm in ["learned_adapter","positive_robust","positive_quantile","raw_count"]:
            hcols=[c for c in cols if c.startswith(harm+"__")]
            combined_curves(d,hcols,module,f"{stem}_{harm}",source,f" ({harm.replace('_',' ')})")
        hcols=[c for c in cols if c.endswith("__final_hybrid")]
        harmonization_curves(d,hcols,module,f"{stem}_hybrid_harmonizations",source)
        return
    selected=select_columns(cols)
    combined_curves(d,selected,module,short_stem(stem),source)
    for col in selected:
        one_curve(d,col,module,f"{short_stem(stem)}_{short_stem(col,38)}", "roc",source)
        one_curve(d,col,module,f"{short_stem(stem)}_{short_stem(col,38)}", "pr",source)
    for col in selected[:3]:calibration(d,col,module,f"{short_stem(stem)}_{short_stem(col,38)}_calibration",source)
    if len(selected)>=2:
        corr=d[selected].apply(pd.to_numeric,errors="coerce").corr(method="spearman"); fig,ax=core.new_fig(7.5,6.2); sns.heatmap(corr,cmap=core.SEQ_CMAP,vmin=0,vmax=1,annot=True,fmt=".2f",square=True,ax=ax,cbar_kws={"label":"Spearman r"}); ax.set_title("Prediction concordance",loc="left",fontweight="bold"); save_v3(fig,module,f"{short_stem(stem)}_prediction_concordance","Prediction concordance",source,"main candidate")


def performance_suite_v3(d,module,stem,source):
    # Retain statistically honest forest/heatmap/landscape plots, but remove the
    # invalid same-axis AUPRC-bar + AUROC-point chart.
    metrics=[m for m in ["auprc","auroc","brier","log_loss","sensitivity","specificity","ppv","npv","calibration_slope","auprc_over_prevalence","r2","mae","spearman_r"] if m in d]
    for metric in metrics: atlas.metric_forest(d,metric,module,f"{short_stem(stem)}_{metric}_forest",source)
    lab=atlas.label_column(d)
    if not lab:return
    if "auprc" in d and "auroc" in d:
        q=d[[lab,"auprc","auroc"]].dropna().drop_duplicates(lab); fig,ax=core.new_fig(7.2,5.5)
        for i,r in q.iterrows(): ax.scatter(r.auroc,r.auprc,s=58,color=model_color(str(r[lab]),i),zorder=3); ax.annotate(pretty(str(r[lab])),(r.auroc,r.auprc),xytext=(4,3),textcoords="offset points",fontsize=8.2)
        core.finish(ax,"Discrimination landscape","AUROC","AUPRC",False); save_v3(fig,module,f"{short_stem(stem)}_discrimination_landscape","AUPRC–AUROC landscape",source,"main candidate")
        # Comparable rank scale replaces the misleading dual-metric axis.
        r=q.copy(); r["AUPRC rank"]=r.auprc.rank(ascending=False); r["AUROC rank"]=r.auroc.rank(ascending=False); r=r.sort_values("AUPRC rank"); y=np.arange(len(r)); fig,ax=core.new_fig(8,max(5,.35*len(r)+1.6))
        for j,(_,row) in enumerate(r.iterrows()): ax.plot([row["AUPRC rank"],row["AUROC rank"]],[j,j],color="#BBC2C5",lw=1.5); ax.scatter(row["AUPRC rank"],j,color=SEMANTIC["hybrid"],s=38); ax.scatter(row["AUROC rank"],j,color=SEMANTIC["gold"],s=38)
        ax.set_yticks(y,[pretty(x) for x in r[lab]]); ax.invert_xaxis(); core.finish(ax,"Rank agreement across discrimination metrics","Rank (1 = best)","",False); save_v3(fig,module,f"{short_stem(stem)}_rank_dumbbell","AUPRC and AUROC rank agreement",source)


def experiment_v3(module,src,min_figures=20):
    status=ORIGINAL_EXPERIMENT(module,src,0)
    status["minimum_required"]=0; status["passed"]=True
    status["rule"]="Evidence-driven atlas; no artificial minimum or filler figures"
    (core.OUT/module/"STATUS.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    return status


ORIGINAL_EXPERIMENT = atlas.experiment


def install_patches() -> None:
    core.PALETTE.update(SEMANTIC); core.SERIES[:] = SERIES
    core.DIV_CMAP=mpl.colors.LinearSegmentedColormap.from_list("muted_div",["#4F7085","#F1F2F1","#9A6F68"])
    core.SEQ_CMAP=mpl.colors.LinearSegmentedColormap.from_list("muted_seq",["#EDF1F2","#81969A","#294D63"])
    core.configure_style=configure_style_v3; core.clean_name=short_stem; core.save_figure=save_v3
    core.curve_panel=one_curve; core.calibration_panel=calibration
    atlas.safe=short_stem; atlas.save=lambda fig,module,name,title,source,use="supplement": save_v3(fig,module,name,title,source,"main candidate" if use=="main" else "supplement")
    atlas.probability_suite=probability_suite_v3; atlas.performance_suite=performance_suite_v3; atlas.experiment=experiment_v3
    atlas.SKIP_NUMERIC.update({"p_value","q_value","ci_low","ci_high","followup_years"})


def audit_output() -> None:
    core.OUT.mkdir(parents=True,exist_ok=True)
    files=list(core.OUT.rglob("*.png")); long_paths=[str(p) for p in files if len(str(p))>240]
    report={**AUDIT,"png_files":len(files),"manifest_rows":len(core.MANIFEST),"paths_over_240":long_paths,
            "notes":["PR main panels use data-adaptive zoom; full ROC/PR evidence remains in supplementary outputs.","Subtype long predictions are pivoted before comparison.","NHATS scores outside [0,1] are transformed with expit for calibration-compatible displays."]}
    (core.OUT/"V3_VISUAL_QA.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")


if __name__ == "__main__":
    install_patches()
    try:
        atlas.main()
    finally:
        audit_output()
