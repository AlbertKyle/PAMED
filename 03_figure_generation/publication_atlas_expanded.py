from __future__ import annotations

import argparse, json, logging, math, re, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from statsmodels.stats.proportion import proportion_confint

import publication_atlas as core


EXPERIMENTS = {
    "01_algorithm_pool": "01_algorithm_pool",
    "02_core_model": "02_core_model",
    "03_apoe_secondary": "03_apoe_secondary",
    "04_nhats_external": "04_nhats_external",
    "05_stability": "05_stability",
    "06_paired_comparisons": "06_paired_comparisons",
    "07_subgroups": "07_subgroups",
    "08_ablation": "08_ablation",
    "09_interpretability": "09_interpretability",
    "10_decision_analysis": "10_decision_analysis",
    "11_representations": "11_representations",
    "12_subtypes": "12_subtypes",
    "13_hierarchical_shapley": "13_hierarchical_shapley",
    "14_rhythm_knowledge": "14_rhythm_knowledge",
    "15_survival": "15_survival",
    "16_interactions": "16_interactions",
    "17_clinical_associations": "17_clinical_associations",
    "18_final_paired": "18_final_paired",
}

SKIP_NUMERIC = {"eid","target","seed","epoch","tp","fp","fn","tn","n","events","selected_n"}


def safe(s): return re.sub(r"[^A-Za-z0-9_.-]+","_",str(s)).strip("_")


def save(fig, module, name, title, source, use="supplement"):
    core.save_figure(fig,module,name,title,source,"main candidate" if use=="main" else "supplement")


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
    for extra in ["harmonization","endpoint","split","analysis","level","term","subtype"]:
        if extra in q and extra!=lab:q["_label"]+= " | "+q[extra].astype(str)
    q=q.drop_duplicates("_label").sort_values(metric).tail(35)
    lo=q[f"{metric}_ci_low"] if f"{metric}_ci_low" in q else q[metric]
    hi=q[f"{metric}_ci_high"] if f"{metric}_ci_high" in q else q[metric]
    x=q[metric].to_numpy(float);y=np.arange(len(q));fig,ax=core.new_fig(8.4,max(5,.34*len(q)+1.8))
    ax.errorbar(x,y,xerr=[np.maximum(0,x-lo.to_numpy(float)),np.maximum(0,hi.to_numpy(float)-x)],fmt="o",ms=5.5,capsize=2.5,color=core.PALETTE["hybrid"],ecolor="#88959D")
    ref=1 if metric in ["odds_ratio","hazard_ratio"] else 0
    if metric in ["odds_ratio","hazard_ratio"]:ax.set_xscale("log")
    if metric.startswith("delta_") or metric in ["coefficient","spearman_r"]:ax.axvline(ref,color="#879198",ls="--",lw=.9)
    ax.set_yticks(y,q._label.str.replace("_"," ").str.title());core.finish(ax,stem.replace("_"," ").title(),metric.replace("_"," ").upper(),"",False);save(fig,module,stem,stem,source,"main")


def performance_suite(d,module,stem,source):
    metrics=[m for m in ["auprc","auroc","brier","log_loss","sensitivity","specificity","ppv","npv","calibration_slope","calibration_intercept","auprc_over_prevalence","r2","mae","spearman_r"] if m in d]
    for m in metrics:metric_forest(d,m,module,f"{stem}_{m}_forest",source)
    lab=label_column(d)
    if lab and len(metrics)>=2:
        q=d[[lab]+metrics].drop_duplicates(lab).set_index(lab).apply(pd.to_numeric,errors="coerce");q=q.loc[:,q.notna().any()]
        if len(q) and len(q.columns):
            z=(q-q.mean())/q.std(ddof=0).replace(0,1);fig,ax=core.new_fig(9,max(5,.38*len(z)+2));sns.heatmap(z,cmap=core.DIV_CMAP,center=0,annot=len(z)<=15,fmt=".2f",cbar_kws={"label":"Standardized value"},ax=ax);ax.set_title("Multimetric performance profile",loc="left",fontweight="bold");save(fig,module,f"{stem}_multimetric_heatmap","Multimetric performance heatmap",source,"main")
            rank=q.copy()
            for c in rank:
                rank[c]=rank[c].rank(ascending=c in ["brier","log_loss","mae"])
            fig,ax=core.new_fig(9,max(5,.38*len(rank)+2));sns.heatmap(rank,cmap=core.SEQ_CMAP.reversed(),annot=True,fmt=".0f",cbar_kws={"label":"Rank"},ax=ax);ax.set_title("Cross-metric ranking",loc="left",fontweight="bold");save(fig,module,f"{stem}_rank_heatmap","Cross-metric ranking",source)
        if "auprc" in d and "auroc" in d:
            fig,ax=core.new_fig(7,5.5);ax.scatter(d.auroc,d.auprc,s=65,color=core.PALETTE["hybrid"],alpha=.85)
            for _,r in d.dropna(subset=["auprc","auroc"]).head(25).iterrows():ax.annotate(str(r[lab]).replace("_"," "),(r.auroc,r.auprc),xytext=(4,3),textcoords="offset points",fontsize=8)
            core.finish(ax,"AUPRC–AUROC discrimination landscape","AUROC","AUPRC",False);save(fig,module,f"{stem}_auprc_auroc_landscape","AUPRC AUROC landscape",source,"main")
            q=d[[lab,"auprc","auroc"]+(["auprc_ci_low","auprc_ci_high"] if {"auprc_ci_low","auprc_ci_high"}.issubset(d) else [])].dropna(subset=["auprc","auroc"]).drop_duplicates(lab).sort_values("auprc").tail(25);y=np.arange(len(q));fig,ax=core.new_fig(8.5,max(5,.36*len(q)+1.8));ax.barh(y,q.auprc,color=core.PALETTE["control"],alpha=.72,label="AUPRC")
            if {"auprc_ci_low","auprc_ci_high"}.issubset(q):ax.errorbar(q.auprc,y,xerr=[np.maximum(0,q.auprc-q.auprc_ci_low),np.maximum(0,q.auprc_ci_high-q.auprc)],fmt="none",ecolor="#59666D",capsize=2)
            ax.plot(q.auroc,y,"o-",color=core.PALETTE["incident"],lw=1.5,ms=5,label="AUROC");ax.set_yticks(y,q[lab].astype(str).str.replace("_"," "));core.finish(ax,"Discrimination summary: bars and points","Metric value","",True);save(fig,module,f"{stem}_auprc_bar_auroc_points","AUPRC bars and AUROC points",source,"main")
            for metric,color in [("auprc",core.PALETTE["hybrid"]),("auroc",core.PALETTE["external"])]:
                q=d[[lab,metric]+([f"{metric}_ci_low",f"{metric}_ci_high"] if {f"{metric}_ci_low",f"{metric}_ci_high"}.issubset(d) else [])].dropna(subset=[metric]).drop_duplicates(lab).sort_values(metric).tail(30);y=np.arange(len(q));fig,ax=core.new_fig(8.2,max(5,.34*len(q)+1.7));ax.barh(y,q[metric],color=color,alpha=.82)
                if {f"{metric}_ci_low",f"{metric}_ci_high"}.issubset(q):ax.errorbar(q[metric],y,xerr=[np.maximum(0,q[metric]-q[f"{metric}_ci_low"]),np.maximum(0,q[f"{metric}_ci_high"]-q[metric])],fmt="none",ecolor="#4D5960",capsize=2.5)
                ax.set_yticks(y,q[lab].astype(str).str.replace("_"," "));core.finish(ax,f"Model comparison by {metric.upper()}",metric.upper(),"",False);save(fig,module,f"{stem}_{metric}_horizontal_bars","Horizontal {metric} bars",source,"main")
        if "sensitivity" in d and "specificity" in d:
            fig,ax=core.new_fig(7,5.5);ax.scatter(d.specificity,d.sensitivity,s=70,color=core.PALETTE["external"])
            for _,r in d.dropna(subset=["sensitivity","specificity"]).head(25).iterrows():ax.annotate(str(r[lab]).replace("_"," "),(r.specificity,r.sensitivity),fontsize=8,xytext=(4,3),textcoords="offset points")
            core.finish(ax,"Operating-point trade-off","Specificity","Sensitivity",False);save(fig,module,f"{stem}_sensitivity_specificity","Sensitivity specificity trade-off",source)
            q=d[[lab,"sensitivity","specificity"]].dropna().drop_duplicates(lab).sort_values("sensitivity");y=np.arange(len(q));fig,ax=core.new_fig(8,max(5,.34*len(q)+1.7))
            for i,r in q.reset_index(drop=True).iterrows():ax.plot([r.sensitivity,r.specificity],[i,i],color="#AAB3B8",lw=2)
            ax.scatter(q.sensitivity,y,color=core.PALETTE["incident"],s=45,label="Sensitivity",zorder=3);ax.scatter(q.specificity,y,color=core.PALETTE["control"],s=45,label="Specificity",zorder=3);ax.set_yticks(y,q[lab].astype(str).str.replace("_"," "));core.finish(ax,"Sensitivity–specificity dumbbell comparison","Operating characteristic","",True);save(fig,module,f"{stem}_sensitivity_specificity_dumbbell","Sensitivity specificity dumbbell",source,"main")


def probability_suite(d,module,stem,source,max_models=14):
    if "target" not in d:return
    if {"model","probability_or_score"}.issubset(d):
        group_cols=[c for c in ["subtype","model"] if c in d]
        for keys,z in d.groupby(group_cols):
            key="_".join(map(str,keys if isinstance(keys,tuple) else [keys]));tmp=z[["target","probability_or_score"]].rename(columns={"probability_or_score":"probability_"+safe(key)})
            probability_suite(tmp,module,f"{stem}_{safe(key)}",source,max_models=1)
        return
    cols=[c for c in d if c.startswith("probability_") or "__" in c]
    cols=[c for c in cols if pd.to_numeric(d[c],errors="coerce").notna().sum()>10][:max_models]
    if not cols:return
    y=pd.to_numeric(d.target,errors="coerce").to_numpy();valid_y=np.isfinite(y)
    # Combined comparison curves are main-figure candidates; individual curves remain available for panel assembly.
    fig_roc,ax_roc=core.new_fig(9.0,6.4);fig_pr,ax_pr=core.new_fig(9.0,6.4)
    combined=0
    for j,c in enumerate(cols):
        p=pd.to_numeric(d[c],errors="coerce").to_numpy();ok=valid_y&np.isfinite(p)
        if len(np.unique(y[ok]))<2:continue
        yy=y[ok].astype(int);pp=p[ok];color=core.SERIES[j%len(core.SERIES)];fpr,tpr,_=roc_curve(yy,pp);pr,rc,_=precision_recall_curve(yy,pp);name=c.replace("probability_","").replace("_"," ")
        ax_roc.plot(fpr,tpr,lw=1.8,color=color,label=f"{name} ({roc_auc_score(yy,pp):.3f})");ax_pr.plot(rc,pr,lw=1.8,color=color,label=f"{name} ({average_precision_score(yy,pp):.3f})");combined+=1
    if combined:
        ax_roc.plot([0,1],[0,1],"--",color="#9BA5AA",lw=1);core.finish(ax_roc,"ROC comparison across models","1 − specificity","Sensitivity");ax_roc.legend(frameon=False,ncol=2,fontsize=9.5);save(fig_roc,module,f"{stem}_all_models_roc","Combined ROC comparison",source,"main")
        ax_pr.axhline(np.nanmean(y),ls="--",color="#9BA5AA",lw=1,label=f"Prevalence = {np.nanmean(y):.3f}");core.finish(ax_pr,"Precision–recall comparison across models","Recall","Precision");ax_pr.legend(frameon=False,ncol=2,fontsize=9.5);save(fig_pr,module,f"{stem}_all_models_pr","Combined precision recall comparison",source,"main")
    for j,c in enumerate(cols):
        p=pd.to_numeric(d[c],errors="coerce").to_numpy();ok=valid_y&np.isfinite(p)
        if len(np.unique(y[ok]))<2:continue
        yy=y[ok].astype(int);pp=p[ok];color=core.SERIES[j%len(core.SERIES)]
        fpr,tpr,_=roc_curve(yy,pp);fig,ax=core.new_fig();grid=np.linspace(0,1,101);lo,hi=core.bootstrap_curve(yy,pp,"roc",grid,n_boot=80,seed=20260820+j);ax.fill_between(grid,lo,hi,color=color,alpha=.14,label="95% bootstrap band");ax.plot(fpr,tpr,lw=2.2,color=color,label=f"AUROC = {roc_auc_score(yy,pp):.3f}");ax.plot([0,1],[0,1],"--",color="#9BA5AA");core.finish(ax,f"ROC: {c.replace('probability_','').replace('_',' ')}","1 − specificity","Sensitivity");save(fig,module,f"{stem}_{safe(c)}_roc","ROC curve",source,"main")
        pr,rc,_=precision_recall_curve(yy,pp);fig,ax=core.new_fig();lo,hi=core.bootstrap_curve(yy,pp,"pr",grid,n_boot=80,seed=20261820+j);ax.fill_between(grid,lo,hi,color=color,alpha=.14,label="95% bootstrap band");ax.plot(rc,pr,lw=2.2,color=color,label=f"AUPRC = {average_precision_score(yy,pp):.3f}");ax.axhline(yy.mean(),ls="--",color="#9BA5AA",label=f"Prevalence = {yy.mean():.3f}");core.finish(ax,f"Precision–recall: {c.replace('probability_','').replace('_',' ')}","Recall","Precision");save(fig,module,f"{stem}_{safe(c)}_pr","Precision recall curve",source,"main")
        if j<5:
            q=pd.DataFrame({"y":yy,"p":pp}).sort_values("p");q["bin"]=pd.qcut(q.p,10,duplicates="drop");g=q.groupby("bin",observed=True).agg(pred=("p","mean"),events=("y","sum"),n=("y","size"));g["obs"]=g.events/g.n;ci=np.array([proportion_confint(int(e),int(n),method="wilson") for e,n in zip(g.events,g.n)],dtype=float);lower=np.maximum(0,g.obs.to_numpy(float)-ci[:,0]);upper=np.maximum(0,ci[:,1]-g.obs.to_numpy(float));fig,ax=core.new_fig();ax.errorbar(g.pred,g.obs,yerr=np.vstack([lower,upper]),fmt="o-",capsize=3,color=color);mx=max(float(g.pred.max()),float(g.obs.max()),1e-6)*1.1;ax.plot([0,mx],[0,mx],"--",color="#9BA5AA");core.finish(ax,f"Calibration: {c.replace('probability_','').replace('_',' ')}","Predicted risk","Observed incidence",False);save(fig,module,f"{stem}_{safe(c)}_calibration","Calibration curve",source,"main")
    # Prediction concordance.
    corr=d[cols].apply(pd.to_numeric,errors="coerce").corr(method="spearman");fig,ax=core.new_fig(9,7);sns.heatmap(corr,cmap=core.SEQ_CMAP,vmin=0,vmax=1,square=True,cbar_kws={"label":"Spearman r"},ax=ax);ax.set_title("Prediction concordance",loc="left",fontweight="bold");save(fig,module,f"{stem}_prediction_concordance","Prediction correlation heatmap",source,"main")
    # Risk-stratum transition alluvial (Sankey-like) between first and last model.
    if len(cols)>=2:alluvial(d,cols[0],cols[-1],module,f"{stem}_risk_strata_alluvial",source)
    # Risk density, lift, top-risk capture and decision curve for final score.
    c=cols[-1];q=d[["target",c]].apply(pd.to_numeric,errors="coerce").dropna();fig,ax=core.new_fig()
    for t,label,color in [(0,"Control",core.PALETTE["control"]),(1,"Incident dementia",core.PALETTE["incident"])]:sns.kdeplot(data=q[q.target.eq(t)],x=c,fill=True,alpha=.18,lw=2,color=color,label=label,ax=ax)
    core.finish(ax,"Predicted-risk distributions", "Predicted risk","Density");save(fig,module,f"{stem}_risk_density","Risk density",source,"main")
    top=[]
    for frac in [.01,.02,.05,.1,.2,.3,.5]:
        z=q.nlargest(max(1,math.ceil(len(q)*frac)),c);top.append((frac,z.target.sum()/max(1,q.target.sum()),z.target.mean()/q.target.mean()))
    top=pd.DataFrame(top,columns=["fraction","capture","enrichment"])
    for val,ylabel in [("capture","Events captured"),("enrichment","Enrichment")]:
        fig,ax=core.new_fig();ax.plot(top.fraction*100,top[val]*(100 if val=="capture" else 1),"o-",lw=2.2,color=core.PALETTE["hybrid"]);core.finish(ax,f"Top-risk {ylabel.lower()}","Population selected (%)",ylabel+(" (%)" if val=="capture" else " (×)"),False);save(fig,module,f"{stem}_top_risk_{val}",f"Top risk {val}",source,"main")
    thresholds=np.linspace(max(0,np.nanquantile(q[c],.05)),np.nanquantile(q[c],.98),50);nb=[]
    y=q.target.to_numpy(int);p=q[c].to_numpy(float);n=len(q)
    for t in thresholds:
        pred=p>=t;tp=((pred)&(y==1)).sum();fp=((pred)&(y==0)).sum();nb.append(tp/n-fp/n*t/max(1e-8,1-t))
    fig,ax=core.new_fig();ax.plot(thresholds,nb,lw=2.2,color=core.PALETTE["hybrid"],label="Model");ax.axhline(0,color="#9BA5AA",ls="--",label="Treat none");core.finish(ax,"Decision-curve analysis","Risk threshold","Net benefit");save(fig,module,f"{stem}_decision_curve","Decision curve",source,"main")


def alluvial(d,left,right,module,stem,source):
    q=d[[left,right]].apply(pd.to_numeric,errors="coerce").dropna()
    if len(q)<20:return
    try:q["L"]=pd.qcut(q[left],3,labels=["Low","Middle","High"]);q["R"]=pd.qcut(q[right],3,labels=["Low","Middle","High"])
    except ValueError:return
    tab=pd.crosstab(q.L,q.R,normalize=True).reindex(index=["Low","Middle","High"],columns=["Low","Middle","High"],fill_value=0)
    fig,ax=core.new_fig(8,6);colors=[core.SERIES[0],core.SERIES[2],core.SERIES[4]];left_base=np.zeros(3);right_base=np.zeros(3);x=np.linspace(0,1,80);smooth=3*x*x-2*x*x*x
    for i in range(3):
        for j in range(3):
            w=tab.iloc[i,j];y0=(1-smooth)*left_base[i]+smooth*right_base[j];y1=y0+w;ax.fill_between(x,y0,y1,color=colors[i],alpha=.5,lw=0);left_base[i]+=w;right_base[j]+=w
    ax.set_xlim(-.05,1.05);ax.set_ylim(0,tab.values.sum()/3*1.08);ax.set_xticks([0,1],[left.replace("probability_","").replace("_"," "),right.replace("probability_","").replace("_"," ")]);ax.set_ylabel("Participant proportion");ax.set_title("Risk-stratum transition across models",loc="left",fontweight="bold");save(fig,module,stem,"Risk-stratum alluvial",source,"main")


def association_suite(d,module,stem,source):
    for metric in ["odds_ratio","hazard_ratio","coefficient","spearman_r","delta_auprc","delta_auroc","delta_brier"]:
        if metric in d:
            metric_forest(d,metric,module,f"{stem}_{metric}",source)
            lo,hi=f"{metric}_ci_low",f"{metric}_ci_high";lab=label_column(d)
            if lab and lo in d and hi in d:
                q=d[[lab,metric,lo,hi]].dropna().copy();q["width"]=q[hi]-q[lo];q=q.sort_values("width");fig,ax=core.new_fig(8,max(4.8,.34*len(q)+1.5));ax.hlines(np.arange(len(q)),0,q.width,color="#AAB3B8");ax.scatter(q.width,np.arange(len(q)),color=core.PALETTE["external"],s=42);ax.set_yticks(np.arange(len(q)),q[lab].astype(str).str.replace("_"," "));core.finish(ax,f"Uncertainty width: {metric.replace('_',' ')}","95% CI width","",False);save(fig,module,f"{stem}_{metric}_ci_width",f"CI width {metric}",source)
    pcol="p_value" if "p_value" in d else None
    ecol="coefficient" if "coefficient" in d else ("spearman_r" if "spearman_r" in d else None)
    if pcol and ecol:
        q=d[[pcol,ecol]+([label_column(d)] if label_column(d) else [])].copy();q[pcol]=pd.to_numeric(q[pcol],errors="coerce");q[ecol]=pd.to_numeric(q[ecol],errors="coerce");q=q.dropna();fig,ax=core.new_fig(7,5.5);ax.scatter(q[ecol],-np.log10(q[pcol].clip(1e-300)),s=30,color=core.PALETTE["external"],alpha=.65);ax.axhline(-np.log10(.05),color="#9BA5AA",ls="--");ax.axvline(0,color="#9BA5AA",lw=.8);core.finish(ax,"Association evidence map",ecol.replace("_"," "),"−log10(P)",False);save(fig,module,f"{stem}_evidence_map","Association evidence map",source,"main")


def temporal_special(d,module,stem,source):
    if {"day","clock_start","mean_shapley_probability"}.issubset(d):
        for group,z in d.groupby("group") if "group" in d else [("all",d)]:
            for val in ["mean_shapley_probability","mean_absolute_shapley"]:
                if val not in z:continue
                p=z.pivot(index="day",columns="clock_start",values=val);fig,ax=core.new_fig(8,5);sns.heatmap(p,cmap=core.DIV_CMAP if "absolute" not in val else core.SEQ_CMAP,center=0 if "absolute" not in val else None,annot=True,fmt=".4f",ax=ax,cbar_kws={"label":"Probability contribution"});ax.set_title(f"Day × clock attribution: {group}",loc="left",fontweight="bold");save(fig,module,f"{stem}_{safe(group)}_{val}_heatmap","Temporal Shapley heatmap",source,"main")
    if {"time_years","survival","group"}.issubset(d):
        fig,ax=core.new_fig(7.5,5.5)
        for i,(g,z) in enumerate(d.groupby("group")):ax.step(z.time_years,1-z.survival,where="post",lw=2,label=str(g),color=core.SERIES[i%len(core.SERIES)])
        core.finish(ax,"Cumulative dementia incidence by risk group","Years from accelerometry","Cumulative incidence");save(fig,module,f"{stem}_cumulative_incidence","Cumulative incidence",source,"main")
        for g,z in d.groupby("group"):
            fig,ax=core.new_fig();ax.step(z.time_years,1-z.survival,where="post",lw=2.3,color=core.PALETTE["hybrid"]);core.finish(ax,f"Cumulative incidence: {g}","Years from accelerometry","Cumulative incidence",False);save(fig,module,f"{stem}_cumulative_incidence_{safe(g)}",f"Cumulative incidence {g}",source)
    if {"threshold","net_benefit","model"}.issubset(d):
        fig,ax=core.new_fig(7.5,5.5)
        for i,(g,z) in enumerate(d.groupby("model")):ax.plot(z.threshold,z.net_benefit,lw=1.8,label=str(g).replace("_"," "),color=core.SERIES[i%len(core.SERIES)])
        core.finish(ax,"Decision-curve analysis","Risk threshold","Net benefit");save(fig,module,f"{stem}_all_models_decision_curve","Decision curve",source,"main")
        for g,z in d.groupby("model"):
            fig,ax=core.new_fig();ax.plot(z.threshold,z.net_benefit,lw=2.2,color=core.PALETTE["hybrid"]);ax.axhline(0,color="#9BA5AA",ls="--");core.finish(ax,f"Decision curve: {str(g).replace('_',' ')}","Risk threshold","Net benefit",False);save(fig,module,f"{stem}_decision_curve_{safe(g)}",f"Decision curve {g}",source)


def generic_diagnostics(d,module,stem,source,limit=12):
    nums=[c for c in d.select_dtypes(include=np.number).columns if c not in SKIP_NUMERIC and not c.endswith("_ci_low") and not c.endswith("_ci_high")]
    lab=label_column(d)
    if len(nums)>=2:
        corr=d[nums[:16]].corr(method="spearman");fig,ax=core.new_fig(8,7);sns.heatmap(corr,cmap=core.DIV_CMAP,center=0,cbar_kws={"label":"Spearman r"},ax=ax);ax.set_title("Numeric evidence correlation map",loc="left",fontweight="bold");save(fig,module,f"{stem}_numeric_correlation","Numeric correlation heatmap",source)
    for c in nums[:limit]:
        z=pd.to_numeric(d[c],errors="coerce").dropna()
        if len(z)<4 or z.nunique()<2:continue
        fig,ax=core.new_fig();sns.ecdfplot(x=z,color=core.PALETTE["hybrid"],lw=2.2,ax=ax);core.finish(ax,f"Empirical distribution: {c.replace('_',' ')}",c.replace("_"," "),"Cumulative probability",False);save(fig,module,f"{stem}_{safe(c)}_ecdf",f"ECDF {c}",source)
        fig,ax=core.new_fig();sns.histplot(x=z,bins=min(30,max(6,int(np.sqrt(len(z))))),stat="density",color=core.PALETTE["control"],alpha=.42,edgecolor=None,ax=ax);sns.kdeplot(x=z,color=core.PALETTE["hybrid"],lw=2,ax=ax);core.finish(ax,f"Density: {c.replace('_',' ')}",c.replace("_"," "),"Density",False);save(fig,module,f"{stem}_{safe(c)}_density",f"Density {c}",source)
        if lab and len(d)<=60:
            q=d[[lab,c]].dropna().drop_duplicates(lab).sort_values(c).tail(30);fig,ax=core.new_fig(8,max(4.8,.32*len(q)+1.6));ax.hlines(np.arange(len(q)),0,q[c],color="#AAB3B8",lw=1.2);ax.scatter(q[c],np.arange(len(q)),s=40,color=core.PALETTE["hybrid"]);ax.set_yticks(np.arange(len(q)),q[lab].astype(str).str.replace("_"," "));core.finish(ax,f"Ranked estimates: {c.replace('_',' ')}",c.replace("_"," "),"",False);save(fig,module,f"{stem}_{safe(c)}_lollipop",f"Lollipop {c}",source)
        if "target" in d and d.target.nunique()==2:
            q=d[["target",c]].apply(pd.to_numeric,errors="coerce").dropna();fig,ax=core.new_fig();sns.violinplot(data=q,x="target",y=c,palette=[core.PALETTE["control"],core.PALETTE["incident"]],inner="quart",cut=0,ax=ax);ax.set_xticks([0,1],["Control","Incident dementia"]);core.finish(ax,f"Outcome distribution: {c.replace('_',' ')}","",c.replace("_"," "),False);save(fig,module,f"{stem}_{safe(c)}_outcome_violin",f"Outcome violin {c}",source)
    for i,(x,y) in enumerate([(nums[i],nums[j]) for i in range(min(3,len(nums))) for j in range(i+1,min(4,len(nums)))][:4]):
        q=d[[x,y]].apply(pd.to_numeric,errors="coerce").dropna();
        if len(q)<3:continue
        fig,ax=core.new_fig();ax.scatter(q[x],q[y],s=35,color=core.PALETTE["external"],alpha=.65);core.finish(ax,f"Evidence relationship: {x.replace('_',' ')} vs {y.replace('_',' ')}",x.replace("_"," "),y.replace("_"," "),False);save(fig,module,f"{stem}_{safe(x)}_vs_{safe(y)}",f"Scatter {x} {y}",source)
    if lab and len(nums)>=3 and 2<=d[lab].nunique()<=20:
        q=d[[lab]+nums[:6]].drop_duplicates(lab).set_index(lab).apply(pd.to_numeric,errors="coerce");z=(q-q.min())/(q.max()-q.min()).replace(0,1);fig,ax=core.new_fig(9,5.8)
        for i,(name,row) in enumerate(z.iterrows()):ax.plot(range(len(z.columns)),row,"o-",lw=1.4,alpha=.75,color=core.SERIES[i%len(core.SERIES)],label=str(name).replace("_"," "))
        ax.set_xticks(range(len(z.columns)),[c.replace("_"," ") for c in z.columns],rotation=25,ha="right");core.finish(ax,"Normalized multimetric profiles","Metric","Within-metric normalized value");save(fig,module,f"{stem}_parallel_metric_profiles","Parallel metric profiles",source)


def copy_tables(src,module):
    td=core.OUT/module/"supplementary_tables";td.mkdir(parents=True,exist_ok=True);rows=[]
    for p in sorted(src.rglob("*.csv")):
        if ".ipynb_checkpoints" in str(p):continue
        size=p.stat().st_size
        if size<=25_000_000:
            dest=td/p.name
            if dest.exists():dest=td/(safe(str(p.relative_to(src)).replace("/","__").replace("\\","__")))
            shutil.copy2(p,dest);rows.append({"file":dest.name,"source":str(p.relative_to(src)),"bytes":size})
        else:rows.append({"file":"not_copied_large_participant_file","source":str(p.relative_to(src)),"bytes":size})
    pd.DataFrame(rows).to_csv(td/"TABLE_INDEX.csv",index=False,encoding="utf-8-sig")


def experiment(module,src,min_figures=20):
    before=len(core.MANIFEST);copy_tables(src,module)
    csvs=[p for p in sorted(src.rglob("*.csv")) if ".ipynb_checkpoints" not in str(p)]
    frames=[]
    for p in csvs:
        d=read_csv(p)
        if d is None or not len(d):continue
        frames.append((p,d));stem=safe(p.stem)
        performance_suite(d,module,stem,str(p));association_suite(d,module,stem,str(p));temporal_special(d,module,stem,str(p))
        if "target" in d:probability_suite(d,module,stem,str(p))
    # Add diverse diagnostic panels only until the prespecified candidate minimum is reached.
    for p,d in frames:
        if sum(x["module"]==module for x in core.MANIFEST)>=min_figures:break
        generic_diagnostics(d,module,safe(p.stem),str(p),limit=10)
    count=sum(x["module"]==module for x in core.MANIFEST)
    status={"module":module,"figures":count,"minimum_required":min_figures,"passed":count>=min_figures,"source":str(src)}
    (core.OUT/module).mkdir(parents=True,exist_ok=True);(core.OUT/module/"STATUS.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    logging.info("%s generated %d figures",module,count)
    return status


def main():
    p=argparse.ArgumentParser(description="Expanded 20-folder V13 publication atlas")
    p.add_argument("--base-dir",required=True);p.add_argument("--v13-dir",required=True);p.add_argument("--descriptive-dir",required=True);p.add_argument("--external-dir",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--module",default="all")
    a=p.parse_args();core.BASE=Path(a.base_dir);core.V13=Path(a.v13_dir);core.DESC=Path(a.descriptive_dir);core.EXT_BASE=Path(a.external_dir);core.OUT=Path(a.output_dir);core.MANIFEST=[];core.configure_style();logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s")
    d=core.load_fixed_cohort();core.export_publication_tables(d);statuses=[]
    requested=[a.module] if a.module!="all" else ["00A_demographics_mining","00B_pa_pattern_mining"]+list(EXPERIMENTS)
    if "00A_demographics_mining" in requested:
        old=len(core.MANIFEST);core.demographic_module(d)
        for x in core.MANIFEST[old:]:x["module"]="00A_demographics_mining"
        st={"module":"00A_demographics_mining","figures":len(core.MANIFEST)-old,"minimum_required":20,"passed":len(core.MANIFEST)-old>=20};statuses.append(st);(core.OUT/st["module"]/"STATUS.json").write_text(json.dumps(st,indent=2),encoding="utf-8")
    if "00B_pa_pattern_mining" in requested:
        old=len(core.MANIFEST);core.pa_pattern_module(d)
        for x in core.MANIFEST[old:]:x["module"]="00B_pa_pattern_mining"
        st={"module":"00B_pa_pattern_mining","figures":len(core.MANIFEST)-old,"minimum_required":20,"passed":len(core.MANIFEST)-old>=20};statuses.append(st);(core.OUT/st["module"]/"STATUS.json").write_text(json.dumps(st,indent=2),encoding="utf-8")
    for module,rel in EXPERIMENTS.items():
        if module not in requested:continue
        if module=="01_algorithm_pool":core.algorithm_module()
        elif module=="02_core_model":core.internal_module(d)
        elif module=="04_nhats_external":core.external_module()
        elif module=="08_ablation":core.ablation_module()
        elif module=="13_hierarchical_shapley":core.interpretability_module(d)
        statuses.append(experiment(module,core.V13/rel,20))
    core.write_manifest();mf=pd.read_csv(core.OUT/"figure_manifest.csv");mf[mf.recommended_use.astype(str).str.contains("main",case=False,na=False)].to_csv(core.OUT/"MAIN_FIGURE_CANDIDATE_INDEX.csv",index=False,encoding="utf-8-sig");mf[~mf.recommended_use.astype(str).str.contains("main",case=False,na=False)].to_csv(core.OUT/"SUPPLEMENTARY_FIGURE_INDEX.csv",index=False,encoding="utf-8-sig");pd.DataFrame(statuses).to_csv(core.OUT/"FOLDER_COMPLETENESS.csv",index=False,encoding="utf-8-sig")
    passed=all(x["passed"] for x in statuses);(core.OUT/"COMPLETED.json").write_text(json.dumps({"completed":passed,"folders":statuses,"rule":"Each requested folder must contain at least 20 candidate figures"},indent=2),encoding="utf-8")
    if not passed:raise SystemExit("One or more folders did not reach the 20-figure minimum; inspect FOLDER_COMPLETENESS.csv")


if __name__=="__main__":main()
