#!/usr/bin/env python3
"""Local, publication-oriented descriptive analysis for the UKB dementia cohort.

Reads the original 7-day minute-level acceleration CSV in streaming blocks.
It is independent of model training and writes audit-ready numeric tables,
manuscript-formatted tables, single-panel figures, and a multipage PDF atlas.
"""
from __future__ import annotations

import argparse, json, logging, math, sys, warnings
from datetime import datetime
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA_ROOT, RESULTS_ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import stats

CONTROL = "#287C8E"
CASE = "#D96C75"
NAVY = "#20364B"
GOLD = "#D6A84B"
GREY = "#68737D"
PALE = "#E8EEF0"
CMAP = LinearSegmentedColormap.from_list("activity", ["#F6F8F8", "#BFD9D8", CONTROL, NAVY])

CONT = ["enrollment_age", "BMI", "Townsend_index", "Followingyears_birthday",
        "Light_Total_Hour", "MVPA_Total_Hour", "Sedentary_Total_Hour", "Sleep_Total_Hour"]
CAT = ["gender", "Disable_long", "education_level", "smoking", "alcohol_use",
       "ethic", "overall_health", "is_dead"]
LABEL = {
    "enrollment_age":"Age at accelerometry, years", "BMI":"Body mass index, kg m$^{-2}$",
    "Townsend_index":"Townsend deprivation index", "Followingyears_birthday":"Age at end of follow-up, years",
    "Light_Total_Hour":"Light activity, h day$^{-1}$", "MVPA_Total_Hour":"MVPA, h day$^{-1}$",
    "Sedentary_Total_Hour":"Sedentary time, h day$^{-1}$", "Sleep_Total_Hour":"Sleep duration, h day$^{-1}$",
    "gender":"Sex code", "Disable_long":"Longstanding disability code", "education_level":"Education code",
    "smoking":"Smoking code", "alcohol_use":"Alcohol-use code", "ethic":"Ethnicity code",
    "overall_health":"Self-rated health code", "is_dead":"Death status"
}
METRIC_LABEL = {
    "mean_acceleration":"Mean acceleration", "sd_acceleration":"Acceleration s.d.",
    "valid_minute_fraction":"Valid minute fraction", "day_to_day_cv":"Day-to-day coefficient of variation",
    "morning_mean_0600_1200":"Morning mean (06:00–12:00)", "afternoon_mean_1200_1800":"Afternoon mean (12:00–18:00)",
    "evening_mean_1800_2200":"Evening mean (18:00–22:00)", "night_mean_2200_0600":"Night mean (22:00–06:00)",
    "day_night_ratio":"Day-to-night ratio", "minute_fragmentation":"Minute-to-minute fragmentation",
    "inactive_transitions_per_day":"Inactive-state transitions per day", "relative_amplitude":"Relative amplitude (M10–L5)/(M10+L5)",
    "M10_mean":"M10 mean acceleration", "L5_mean":"L5 mean acceleration", "M10_onset_hour":"M10 onset, clock hour",
    "L5_onset_hour":"L5 onset, clock hour", "cosinor_amplitude":"First-harmonic amplitude",
    "cosinor_acrophase_hour":"First-harmonic acrophase, clock hour", "interday_profile_correlation":"Mean interday profile correlation"
}

def setup(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for p in [out/"tables_numeric", out/"tables_manuscript", out/"figures_png", out/"figures_pdf"]: p.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(out/"analysis.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)], force=True)
    plt.rcParams.update({"font.family":"Arial", "font.size":8.5, "axes.titlesize":10, "axes.labelsize":9,
        "axes.titleweight":"bold", "axes.spines.top":False, "axes.spines.right":False,
        "axes.linewidth":0.7, "xtick.major.width":0.7, "ytick.major.width":0.7,
        "legend.frameon":False, "figure.facecolor":"white", "axes.facecolor":"white",
        "pdf.fonttype":42, "ps.fonttype":42})

def clean_ax(ax):
    ax.grid(axis="y", color="#D9E0E3", linewidth=.55, alpha=.7, zorder=0)
    ax.tick_params(length=3, color="#3E4A50")

def save_single(fig, stem: str, out: Path):
    fig.tight_layout(pad=.8)
    png=out/"figures_png"/(stem+".png"); pdf=out/"figures_pdf"/(stem+".pdf")
    fig.savefig(png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig); return png

def fmt_p(p):
    if pd.isna(p): return "—"
    if p < 1e-300: return "<1.0 × 10⁻³⁰⁰"
    if p < .001:
        e=int(math.floor(math.log10(p))); m=p/(10**e)
        supers=str(e).translate(str.maketrans("-0123456789","⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
        return f"{m:.1f} × 10{supers}"
    return f"{p:.3f}"

def bh(p):
    a=np.asarray(p,float); out=np.full(len(a),np.nan); ok=np.isfinite(a); x=a[ok]
    if not len(x): return out
    order=np.argsort(x); ranked=x[order]; q=np.minimum.accumulate((ranked*len(x)/np.arange(1,len(x)+1))[::-1])[::-1]
    tmp=np.empty(len(x)); tmp[order]=np.minimum(q,1); out[ok]=tmp; return out

def mean_sd(s, digits=1):
    x=pd.to_numeric(s,errors="coerce").dropna(); return "—" if not len(x) else f"{x.mean():.{digits}f} ({x.std(ddof=1):.{digits}f})"

def smd(a,b):
    a=pd.to_numeric(a,errors="coerce").dropna(); b=pd.to_numeric(b,errors="coerce").dropna()
    den=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2); return (a.mean()-b.mean())/den if den>0 else np.nan

def rank_biserial(a,b):
    a=pd.to_numeric(a,errors="coerce").dropna(); b=pd.to_numeric(b,errors="coerce").dropna()
    if not len(a) or not len(b): return np.nan
    u=stats.mannwhitneyu(a,b,alternative="two-sided").statistic
    return 2*u/(len(a)*len(b))-1

def cat_smd(a,b):
    aa=a.astype("string").fillna("Missing"); bb=b.astype("string").fillna("Missing")
    lev=sorted(set(aa)|set(bb)); pa=np.array([(aa==v).mean() for v in lev]); pb=np.array([(bb==v).mean() for v in lev])
    den=np.maximum((pa+pb)*(2-pa-pb)/2,1e-12); return float(np.sqrt(np.sum((pa-pb)**2/den)))

def table1(df):
    case=df.target.eq(1); rows=[]
    for c in CONT:
        a=df.loc[case,c]; b=df.loc[~case,c]
        aa=pd.to_numeric(a,errors="coerce").dropna(); bb=pd.to_numeric(b,errors="coerce").dropna()
        p=stats.ttest_ind(aa,bb,equal_var=False,nan_policy="omit").pvalue
        rows.append({"variable":c,"level":"","type":"continuous","overall":mean_sd(df[c]),"control":mean_sd(b),"incident":mean_sd(a),
            "missing_n":int(df[c].isna().sum()),"smd":smd(a,b),"p_value":p})
    for c in CAT:
        allv=df[c].astype("string").fillna("Missing"); tab=pd.crosstab(allv,df.target)
        p=stats.chi2_contingency(tab,correction=False).pvalue if min(tab.shape)>0 and tab.shape[0]>1 else np.nan
        effect=cat_smd(df.loc[case,c],df.loc[~case,c])
        for j,level in enumerate(sorted(allv.unique())):
            def f(s):
                z=s.astype("string").fillna("Missing"); n=int((z==level).sum()); return f"{n:,} ({100*n/len(z):.1f}%)"
            rows.append({"variable":c,"level":str(level),"type":"categorical","overall":f(df[c]),"control":f(df.loc[~case,c]),
                "incident":f(df.loc[case,c]),"missing_n":int(df[c].isna().sum()),"smd":effect if j==0 else np.nan,"p_value":p if j==0 else np.nan})
    raw=pd.DataFrame(rows)
    clean_label={k:v.replace("$^{-2}$","m⁻²").replace("$^{-1}$","⁻¹") for k,v in LABEL.items()}
    seen=set(); shown=[]
    for r in raw.itertuples():
        if not r.level: shown.append(clean_label[r.variable])
        elif r.variable not in seen: shown.append(clean_label[r.variable]+": "+r.level); seen.add(r.variable)
        else: shown.append("    "+r.level)
    raw["label"]=shown
    pub=raw[["label","overall","control","incident","missing_n","smd","p_value"]].copy()
    pub.columns=["Characteristic","Overall","No recorded dementia","Incident dementia >1 year","Missing, n","SMD","P value"]
    pub["SMD"]=pub["SMD"].map(lambda x:"—" if pd.isna(x) else f"{x:.2f}")
    pub["P value"]=pub["P value"].map(fmt_p)
    return raw,pub

def rolling_window(profile, width, mode):
    n=len(profile); ext=np.r_[profile,profile[:width-1]]
    good=np.isfinite(ext); vals=np.nan_to_num(ext); cs=np.r_[0,np.cumsum(vals)]; cn=np.r_[0,np.cumsum(good)]
    sums=cs[width:]-cs[:-width]; nums=cn[width:]-cn[:-width]; means=np.divide(sums,nums,out=np.full(n,np.nan),where=nums>width*.8)
    k=int(np.nanargmax(means) if mode=="max" else np.nanargmin(means)); return means[k], k/60

def interday_corr(safe):
    vals=[]
    for i in range(7):
        for j in range(i+1,7):
            ok=np.isfinite(safe[i])&np.isfinite(safe[j])
            if ok.sum()>300 and np.nanstd(safe[i,ok])>0 and np.nanstd(safe[j,ok])>0: vals.append(np.corrcoef(safe[i,ok],safe[j,ok])[0,1])
    return np.nanmean(vals) if vals else np.nan

def participant_features(a):
    # a: 7 x 1440; missing represented by NaN.
    valid=np.isfinite(a); flat=a.ravel(); okflat=np.isfinite(flat); daily=np.nanmean(a,axis=1)
    prof=np.nanmean(a,axis=0); grand=np.nanmean(a)
    morning=np.nanmean(a[:,360:720]); afternoon=np.nanmean(a[:,720:1080]); evening=np.nanmean(a[:,1080:1320])
    night=np.nanmean(np.concatenate([a[:,:360],a[:,1320:]],axis=1)); day=np.nanmean(a[:,360:1320])
    adjacent=okflat[1:]&okflat[:-1]; frag=np.mean(np.abs(np.diff(flat)[adjacent])) if adjacent.any() else np.nan
    inactive=flat<5; transitions=np.sum((inactive[1:]!=inactive[:-1])&adjacent)/max(valid.sum()/1440,1e-6)
    m10,m10t=rolling_window(prof,600,"max"); l5,l5t=rolling_window(prof,300,"min")
    ra=(m10-l5)/(m10+l5) if np.isfinite(m10+l5) and (m10+l5)!=0 else np.nan
    t=np.arange(1440); cc=np.cos(2*np.pi*t/1440); ss=np.sin(2*np.pi*t/1440); good=np.isfinite(prof)
    X=np.c_[np.ones(good.sum()),cc[good],ss[good]]; beta=np.linalg.lstsq(X,prof[good],rcond=None)[0] if good.sum()>50 else [np.nan]*3
    amp=np.hypot(beta[1],beta[2]); phase=(np.arctan2(-beta[2],beta[1])%(2*np.pi))*24/(2*np.pi)
    hourly=np.nanmean(a.reshape(7,24,60),axis=(0,2)); daymean=np.nanmean(a,axis=1)
    return ({"mean_acceleration":grand,"sd_acceleration":np.nanstd(a),"valid_minute_fraction":valid.mean(),
        "day_to_day_cv":np.nanstd(daily)/grand if grand else np.nan,"morning_mean_0600_1200":morning,
        "afternoon_mean_1200_1800":afternoon,"evening_mean_1800_2200":evening,"night_mean_2200_0600":night,
        "day_night_ratio":day/night if night else np.nan,"minute_fragmentation":frag,"inactive_transitions_per_day":transitions,
        "relative_amplitude":ra,"M10_mean":m10,"L5_mean":l5,"M10_onset_hour":m10t,"L5_onset_hour":l5t,
        "cosinor_amplitude":amp,"cosinor_acrophase_hour":phase,"interday_profile_correlation":interday_corr(a)}, hourly, daymean, prof)

def batch_features(a):
    """Vectorized participant features for a B x 7 x 1440 block."""
    b=a.shape[0]; valid=np.isfinite(a); flat=a.reshape(b,-1); okflat=np.isfinite(flat)
    daily=np.nanmean(a,axis=2); prof=np.nanmean(a,axis=1); grand=np.nanmean(a,axis=(1,2))
    morning=np.nanmean(a[:,:,360:720],axis=(1,2)); afternoon=np.nanmean(a[:,:,720:1080],axis=(1,2)); evening=np.nanmean(a[:,:,1080:1320],axis=(1,2))
    night=np.nanmean(np.concatenate([a[:,:,:360],a[:,:,1320:]],axis=2),axis=(1,2)); day=np.nanmean(a[:,:,360:1320],axis=(1,2))
    adj=okflat[:,1:]&okflat[:,:-1]; delta=np.abs(np.diff(flat,axis=1)); frag=np.divide(np.where(adj,delta,0).sum(1),adj.sum(1),out=np.full(b,np.nan),where=adj.sum(1)>0)
    inactive=flat<5; trans=((inactive[:,1:]!=inactive[:,:-1])&adj).sum(1)/np.maximum(valid.sum((1,2))/1440,1e-6)
    def roll(width, mode):
        ext=np.concatenate([prof,prof[:,:width-1]],axis=1); good=np.isfinite(ext); vals=np.nan_to_num(ext)
        cs=np.concatenate([np.zeros((b,1)),np.cumsum(vals,axis=1)],axis=1); cn=np.concatenate([np.zeros((b,1)),np.cumsum(good,axis=1)],axis=1)
        sm=cs[:,width:]-cs[:,:-width]; nn=cn[:,width:]-cn[:,:-width]; mm=np.divide(sm,nn,out=np.full_like(sm,np.nan),where=nn>width*.8)[:,:1440]
        safe=np.where(np.isfinite(mm),mm,-np.inf if mode=="max" else np.inf); k=np.argmax(safe,axis=1) if mode=="max" else np.argmin(safe,axis=1)
        return mm[np.arange(b),k],k/60
    m10,m10t=roll(600,"max"); l5,l5t=roll(300,"min"); ra=np.divide(m10-l5,m10+l5,out=np.full(b,np.nan),where=np.isfinite(m10+l5)&((m10+l5)!=0))
    t=np.arange(1440); cc=np.cos(2*np.pi*t/1440); ss=np.sin(2*np.pi*t/1440); centered=prof-np.nanmean(prof,axis=1,keepdims=True)
    bc=np.nansum(centered*cc,axis=1)/np.sum(cc*cc); bs=np.nansum(centered*ss,axis=1)/np.sum(ss*ss); amp=np.hypot(bc,bs); phase=(np.arctan2(-bs,bc)%(2*np.pi))*24/(2*np.pi)
    cors=[]
    for i in range(7):
        for j in range(i+1,7):
            ok=np.isfinite(a[:,i])&np.isfinite(a[:,j]); n=ok.sum(1); xi=np.where(ok,a[:,i],0); xj=np.where(ok,a[:,j],0)
            sx=xi.sum(1); sy=xj.sum(1); sxx=(xi*xi).sum(1); syy=(xj*xj).sum(1); sxy=(xi*xj).sum(1)
            num=n*sxy-sx*sy; den=np.sqrt(np.maximum((n*sxx-sx*sx)*(n*syy-sy*sy),0)); cors.append(np.divide(num,den,out=np.full(b,np.nan),where=(n>300)&(den>0)))
    icorr=np.nanmean(np.stack(cors,axis=1),axis=1); hourly=np.nanmean(a.reshape(b,7,24,60),axis=(1,3))
    data={"mean_acceleration":grand,"sd_acceleration":np.nanstd(a,axis=(1,2)),"valid_minute_fraction":valid.mean((1,2)),
        "day_to_day_cv":np.nanstd(daily,axis=1)/grand,"morning_mean_0600_1200":morning,"afternoon_mean_1200_1800":afternoon,
        "evening_mean_1800_2200":evening,"night_mean_2200_0600":night,"day_night_ratio":day/night,"minute_fragmentation":frag,
        "inactive_transitions_per_day":trans,"relative_amplitude":ra,"M10_mean":m10,"L5_mean":l5,"M10_onset_hour":m10t,
        "L5_onset_hour":l5t,"cosinor_amplitude":amp,"cosinor_acrophase_hour":phase,"interday_profile_correlation":icorr}
    return pd.DataFrame(data),hourly,daily,prof

def derive_stream(raw, eligible, cache_dir, chunksize=350):
    phenotype=cache_dir/"derived_activity_phenotypes.csv"; hourly_path=cache_dir/"participant_hourly_profiles.csv"
    daily_path=cache_dir/"participant_daily_means.csv"; agg_path=cache_dir/"profile_aggregates.npz"
    if all(p.exists() for p in [phenotype,hourly_path,daily_path,agg_path]):
        d=pd.read_csv(phenotype,dtype={"eid":"string"})
        if len(d)==len(eligible) and set(d.eid)==set(eligible.eid):
            logging.info("Reusing validated local activity cache (%d participants)",len(d)); return d,pd.read_csv(hourly_path,dtype={"eid":"string"}),pd.read_csv(daily_path,dtype={"eid":"string"}),np.load(agg_path)
    lookup=eligible.set_index("eid")["target"].to_dict(); rows=[]; hours=[]; days=[]
    sums=np.zeros((2,1440)); sums2=np.zeros((2,1440)); nums=np.zeros((2,1440)); nraw=0
    pending=pd.DataFrame()
    for chunk in pd.read_csv(raw,chunksize=chunksize*7,dtype={"eid":"string"},low_memory=False):
        if len(pending): chunk=pd.concat([pending,chunk],ignore_index=True); pending=pd.DataFrame()
        # Retain a trailing incomplete eid group.
        last=chunk.eid.iloc[-1]; start=np.where(chunk.eid.to_numpy()!=last)[0]
        if len(start): cut=start[-1]+1; pending=chunk.iloc[cut:].copy(); chunk=chunk.iloc[:cut]
        else: pending=chunk; continue
        ids=chunk.eid.iloc[::7].to_numpy(); repeated=chunk.eid.to_numpy().reshape(-1,7)
        if not np.all(repeated==repeated[:,[0]]): raise ValueError("Raw activity rows are not contiguous seven-row eid groups")
        keep=np.array([eid in lookup for eid in ids]); ids=ids[keep]
        if len(ids):
            a=chunk.iloc[:,1:].to_numpy(np.float32).reshape(-1,7,1440)[keep]; feat,h,d,p=batch_features(a); targets=np.array([int(lookup[eid]) for eid in ids])
            feat.insert(0,"target",targets); feat.insert(0,"eid",ids); rows.append(feat)
            hf=pd.DataFrame(h,columns=[f"hour_{i:02d}" for i in range(24)]); hf.insert(0,"target",targets); hf.insert(0,"eid",ids); hours.append(hf)
            dd=pd.DataFrame(d,columns=[f"day_{i+1}" for i in range(7)]); dd.insert(0,"target",targets); dd.insert(0,"eid",ids); days.append(dd)
            for target in (0,1):
                pp=p[targets==target]; good=np.isfinite(pp); sums[target]+=np.nansum(pp,0); sums2[target]+=np.nansum(pp*pp,0); nums[target]+=good.sum(0)
        nraw+=len(chunk)
        done=sum(len(z) for z in rows)
        if done%3500==0 or done<400: logging.info("Derived activity for %d/%d participants",done,len(eligible))
    if len(pending):
        for eid,g in pending.groupby("eid",sort=False):
            if eid in lookup:
                if len(g)!=7: raise ValueError(f"Final eid {eid} has {len(g)} rows; expected 7")
                a=g.iloc[:,1:].to_numpy(np.float32)[None,:,:]; feat,h,d,p=batch_features(a); target=int(lookup[eid]); feat.insert(0,"target",[target]); feat.insert(0,"eid",[eid]); rows.append(feat)
                hf=pd.DataFrame(h,columns=[f"hour_{i:02d}" for i in range(24)]); hf.insert(0,"target",[target]); hf.insert(0,"eid",[eid]); hours.append(hf)
                dd=pd.DataFrame(d,columns=[f"day_{i+1}" for i in range(7)]); dd.insert(0,"target",[target]); dd.insert(0,"eid",[eid]); days.append(dd)
                pp=p[0]; good=np.isfinite(pp); sums[target,good]+=pp[good]; sums2[target,good]+=pp[good]**2; nums[target,good]+=1
    derived=pd.concat(rows,ignore_index=True); hourly=pd.concat(hours,ignore_index=True); daily=pd.concat(days,ignore_index=True)
    if len(derived)!=len(eligible) or derived.eid.nunique()!=len(eligible): raise ValueError(f"Derived {len(derived)} unique {derived.eid.nunique()}, expected {len(eligible)}")
    derived.to_csv(phenotype,index=False,encoding="utf-8-sig"); hourly.to_csv(hourly_path,index=False,encoding="utf-8-sig"); daily.to_csv(daily_path,index=False,encoding="utf-8-sig")
    np.savez_compressed(agg_path,sums=sums,sums2=sums2,nums=nums); logging.info("Activity extraction complete")
    return derived,hourly,daily,np.load(agg_path)

def derive_day_hour_aggregates(raw, eligible, cache_dir, chunksize=700):
    path=cache_dir/"day_hour_aggregates.npz"
    if path.exists(): return np.load(path)
    lookup=eligible.set_index("eid")["target"].to_dict(); sums=np.zeros((2,7,24)); sums2=np.zeros((2,7,24)); nums=np.zeros((2,7,24)); pending=pd.DataFrame()
    for chunk in pd.read_csv(raw,chunksize=chunksize*7,dtype={"eid":"string"},low_memory=False):
        if len(pending): chunk=pd.concat([pending,chunk],ignore_index=True); pending=pd.DataFrame()
        last=chunk.eid.iloc[-1]; pos=np.where(chunk.eid.to_numpy()!=last)[0]
        if len(pos): cut=pos[-1]+1; pending=chunk.iloc[cut:].copy(); chunk=chunk.iloc[:cut]
        else: pending=chunk; continue
        ids=chunk.eid.iloc[::7].to_numpy(); repeated=chunk.eid.to_numpy().reshape(-1,7)
        if not np.all(repeated==repeated[:,[0]]): raise ValueError("Raw rows fail seven-day contiguity check")
        keep=np.array([eid in lookup for eid in ids]); ids=ids[keep]
        if len(ids):
            a=chunk.iloc[:,1:].to_numpy(np.float32).reshape(-1,7,24,60)[keep]; h=np.nanmean(a,axis=3); target=np.array([lookup[e] for e in ids])
            for g in (0,1):
                z=h[target==g]; sums[g]+=np.nansum(z,axis=0); sums2[g]+=np.nansum(z*z,axis=0); nums[g]+=np.isfinite(z).sum(axis=0)
    if len(pending):
        for eid,g in pending.groupby("eid",sort=False):
            if eid in lookup:
                z=np.nanmean(g.iloc[:,1:].to_numpy(np.float32).reshape(7,24,60),axis=2); target=int(lookup[eid]); good=np.isfinite(z); sums[target,good]+=z[good]; sums2[target,good]+=z[good]**2; nums[target,good]+=1
    np.savez_compressed(path,sums=sums,sums2=sums2,nums=nums); logging.info("Day-by-hour aggregates complete"); return np.load(path)

def metric_table(metrics):
    rows=[]; case=metrics.target.eq(1)
    for c in METRIC_LABEL:
        a=metrics.loc[case,c].dropna(); b=metrics.loc[~case,c].dropna(); test=stats.mannwhitneyu(a,b,alternative="two-sided")
        rows.append({"metric":c,"overall_mean":metrics[c].mean(),"overall_sd":metrics[c].std(),"control_mean":b.mean(),"control_sd":b.std(),
            "incident_mean":a.mean(),"incident_sd":a.std(),"smd":smd(a,b),"rank_biserial":rank_biserial(a,b),"p_value":test.pvalue,
            "control_n":len(b),"incident_n":len(a)})
    raw=pd.DataFrame(rows); raw["q_value_bh"]=bh(raw.p_value)
    pub=pd.DataFrame({"Seven-day activity phenotype":raw.metric.map(METRIC_LABEL),
        "No recorded dementia":raw.apply(lambda r:f"{r.control_mean:.2f} ({r.control_sd:.2f})",axis=1),
        "Incident dementia >1 year":raw.apply(lambda r:f"{r.incident_mean:.2f} ({r.incident_sd:.2f})",axis=1),
        "SMD":raw.smd.map(lambda x:f"{x:.2f}"),"Rank-biserial r":raw.rank_biserial.map(lambda x:f"{x:.2f}"),
        "P value":raw.p_value.map(fmt_p),"FDR-adjusted q value":raw.q_value_bh.map(fmt_p)})
    return raw,pub

def export_table(raw,pub,stem,out):
    raw.to_csv(out/"tables_numeric"/(stem+"_numeric.csv"),index=False,encoding="utf-8-sig")
    pub.to_csv(out/"tables_manuscript"/(stem+"_manuscript.csv"),index=False,encoding="utf-8-sig")

def make_figures(df,metrics,hourly,daily,agg,dayhour,subtypes,events,out):
    paths=[]; case=df.target.eq(1); nc=int((~case).sum()); ni=int(case.sum())
    # 01 flow
    full=pd.read_csv(ARGS.labels,usecols=["dementia_state"]); states=full.dementia_state.value_counts()
    labels=["7-day accelerometer cohort","Prevalent dementia excluded","1–365-day washout excluded","Eligible incident cohort","Incident dementia >1 year"]
    vals=[len(full),int(states.get("prevalent_at_accelerometer",0)),int(states.get("washout_1_365_days",0)),len(df),ni]
    fig,ax=plt.subplots(figsize=(5.8,3.5)); y=np.arange(len(vals)); ax.barh(y,vals,color=[NAVY,GREY,GOLD,CONTROL,CASE],height=.62,zorder=2)
    ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlabel("Participants"); ax.set_title("Analytical cohort derivation",loc="left")
    for yy,v in zip(y,vals): ax.text(v+max(vals)*.018,yy,f"{v:,}",va="center",fontsize=8,weight="bold")
    ax.set_xlim(0,max(vals)*1.18); clean_ax(ax); paths.append(save_single(fig,"01_cohort_derivation",out))
    # 02 covariate SMD
    t1,_=table1(df); q=t1.dropna(subset=["smd"]).drop_duplicates("variable").sort_values("smd")
    fig,ax=plt.subplots(figsize=(5.8,4.5)); yy=np.arange(len(q)); ax.hlines(yy,0,q.smd,color=PALE,lw=3); ax.scatter(q.smd,yy,c=np.where(abs(q.smd)>=.1,CASE,CONTROL),s=28,zorder=3)
    ax.axvline(0,color=NAVY,lw=.7); ax.axvline(.1,color=GREY,ls="--",lw=.7); ax.axvline(-.1,color=GREY,ls="--",lw=.7); ax.set_yticks(yy,[LABEL[x] for x in q.variable]); ax.set_xlabel("Standardized mean difference\n(incident dementia minus control)"); ax.set_title("Baseline covariate balance",loc="left"); clean_ax(ax); paths.append(save_single(fig,"02_baseline_covariate_smd",out))
    # 03 exact minute profile
    sums,sums2,nums=agg["sums"],agg["sums2"],agg["nums"]; x=np.arange(1440)/60
    fig,ax=plt.subplots(figsize=(6.8,3.7))
    for g,col,lab in [(0,CONTROL,f"No recorded dementia (n={nc:,})"),(1,CASE,f"Incident dementia (n={ni:,})")]:
        mean=sums[g]/nums[g]; var=np.maximum((sums2[g]-sums[g]**2/nums[g])/(nums[g]-1),0); se=np.sqrt(var/nums[g])
        ax.plot(x,mean,color=col,lw=1.45,label=lab); ax.fill_between(x,mean-1.96*se,mean+1.96*se,color=col,alpha=.18,lw=0)
    ax.axvspan(0,6,color=NAVY,alpha=.06); ax.axvspan(22,24,color=NAVY,alpha=.06); ax.set(xlim=(0,24),xticks=np.arange(0,25,4),xlabel="Clock time, h",ylabel="Mean acceleration",title="Twenty-four-hour activity profile")
    ax.legend(ncol=1,loc="upper left"); clean_ax(ax); paths.append(save_single(fig,"03_24h_activity_profile",out))
    # 04 difference profile
    m0=sums[0]/nums[0]; m1=sums[1]/nums[1]; diff=m1-m0
    fig,ax=plt.subplots(figsize=(6.8,3.1)); ax.fill_between(x,0,diff,where=diff<0,color=CONTROL,alpha=.75); ax.fill_between(x,0,diff,where=diff>=0,color=CASE,alpha=.75); ax.axhline(0,color=NAVY,lw=.7)
    ax.set(xlim=(0,24),xticks=np.arange(0,25,4),xlabel="Clock time, h",ylabel="Mean difference",title="Minute-level activity difference: incident dementia minus control"); clean_ax(ax); paths.append(save_single(fig,"04_24h_activity_difference",out))
    # 05-07 exact 7 x 24 heatmaps.
    for stem,g,title in [("05_control_hourly_profile",0,"Control 24-hour activity"),("06_incident_hourly_profile",1,"Incident dementia 24-hour activity")]:
        h=dayhour["sums"][g]/dayhour["nums"][g]
        fig,ax=plt.subplots(figsize=(6.6,3.0)); im=ax.imshow(h,aspect="auto",cmap=CMAP); ax.set_yticks(range(7),[f"Day {i}" for i in range(1,8)]); ax.set_xticks(np.arange(0,24,3),[f"{i:02d}:00" for i in range(0,24,3)]); ax.set_title(title,loc="left"); fig.colorbar(im,ax=ax,label="Mean acceleration",fraction=.035,pad=.025); paths.append(save_single(fig,stem,out))
    hdiff=dayhour["sums"][1]/dayhour["nums"][1]-dayhour["sums"][0]/dayhour["nums"][0]; lim=np.nanmax(abs(hdiff)); fig,ax=plt.subplots(figsize=(6.6,3.0)); im=ax.imshow(hdiff,aspect="auto",cmap="RdBu_r",vmin=-lim,vmax=lim); ax.set_yticks(range(7),[f"Day {i}" for i in range(1,8)]); ax.set_xticks(np.arange(0,24,3),[f"{i:02d}:00" for i in range(0,24,3)]); ax.set_title("Day-by-hour activity difference",loc="left"); fig.colorbar(im,ax=ax,label="Incident minus control",fraction=.035,pad=.025); paths.append(save_single(fig,"07_hourly_difference_heatmap",out))
    # 08 activity effect forest
    mt,_=metric_table(metrics); q=mt.sort_values("smd"); fig,ax=plt.subplots(figsize=(6.2,6.0)); yy=np.arange(len(q)); ax.hlines(yy,0,q.smd,color=PALE,lw=3); ax.scatter(q.smd,yy,c=np.where(q.q_value_bh<.05,CASE,GREY),s=25,zorder=3); ax.axvline(0,color=NAVY,lw=.7); ax.set_yticks(yy,[METRIC_LABEL[x] for x in q.metric]); ax.set_xlabel("Standardized mean difference"); ax.set_title("Seven-day activity phenotype differences",loc="left"); clean_ax(ax); paths.append(save_single(fig,"08_activity_phenotype_smd",out))
    # 09-14 selected distribution figures, each single panel and control downsampled for rendering only
    rng=np.random.default_rng(20260813); control_idx=np.where(~metrics.target.eq(1))[0]; sample=np.r_[rng.choice(control_idx,min(5000,len(control_idx)),False),np.where(metrics.target.eq(1))[0]]
    selected=["mean_acceleration","night_mean_2200_0600","day_night_ratio","relative_amplitude","minute_fragmentation","interday_profile_correlation"]
    for i,c in enumerate(selected,9):
        plot=metrics.iloc[sample]; a=plot.loc[plot.target.eq(0),c].dropna(); b=plot.loc[plot.target.eq(1),c].dropna()
        fig,ax=plt.subplots(figsize=(4.1,3.6)); vp=ax.violinplot([a,b],positions=[0,1],showextrema=False,widths=.75)
        for body,col in zip(vp["bodies"],[CONTROL,CASE]): body.set_facecolor(col); body.set_edgecolor("none"); body.set_alpha(.55)
        for xpos,z,col in [(0,a,CONTROL),(1,b,CASE)]:
            ax.boxplot(z,positions=[xpos],widths=.20,showfliers=False,patch_artist=True,boxprops=dict(facecolor="white",edgecolor=col),medianprops=dict(color=NAVY),whiskerprops=dict(color=col),capprops=dict(color=col))
        combined=np.r_[a.to_numpy(),b.to_numpy()]; lo,hi=np.nanpercentile(combined,[.5,99.5]); pad=(hi-lo)*.08
        ax.set_ylim(lo-pad,hi+pad); ax.text(.99,.02,"Display range: 0.5th–99.5th percentile",transform=ax.transAxes,ha="right",va="bottom",fontsize=6.5,color=GREY)
        ax.set_xticks([0,1],["Control","Incident\ndementia"]); ax.set_ylabel(METRIC_LABEL[c]); ax.set_title(METRIC_LABEL[c],loc="left"); clean_ax(ax); paths.append(save_single(fig,f"{i:02d}_{c}_distribution",out))
    # 15 time to diagnosis
    years=pd.to_numeric(df.loc[case,"days_accel_to_dementia"],errors="coerce")/365.25
    fig,ax=plt.subplots(figsize=(5.5,3.6)); bins=np.arange(1,math.ceil(years.max())+1,.5); ax.hist(years,bins=bins,color=CASE,edgecolor="white",lw=.5); ax.axvline(years.median(),color=NAVY,ls="--",lw=1,label=f"Median {years.median():.1f} years"); ax.set(xlabel="Years from accelerometry to first dementia record",ylabel="Participants",title="Time to incident dementia ascertainment"); ax.legend(); clean_ax(ax); paths.append(save_single(fig,"15_time_to_diagnosis",out))
    # 16 age distribution
    fig,ax=plt.subplots(figsize=(5.5,3.6)); bins=np.arange(38,77,2)
    for g,col,lab in [(0,CONTROL,"Control"),(1,CASE,"Incident dementia")]: ax.hist(df.loc[df.target.eq(g),"enrollment_age"],bins=bins,density=True,histtype="step",lw=1.7,color=col,label=lab)
    ax.set(xlabel="Age at accelerometry, years",ylabel="Density",title="Age distribution"); ax.legend(); clean_ax(ax); paths.append(save_single(fig,"16_age_distribution",out))
    # 17 age-band risk (descriptive proportion)
    agebin=pd.cut(df.enrollment_age,[35,50,55,60,65,70,80],right=False); z=df.groupby(agebin,observed=True).target.agg(["sum","count"]); z["pct"]=100*z["sum"]/z["count"]
    fig,ax=plt.subplots(figsize=(5.5,3.6)); ax.bar(np.arange(len(z)),z.pct,color=CASE,width=.7); ax.set_xticks(np.arange(len(z)),[str(x) for x in z.index],rotation=30,ha="right"); ax.set(xlabel="Age band, years",ylabel="Incident dementia, %",title="Incident dementia proportion by baseline age"); clean_ax(ax); paths.append(save_single(fig,"17_incident_proportion_by_age",out))
    # 18 subtype composition, mutually interpretable
    inc=subtypes[subtypes.eid.isin(set(df.loc[case,"eid"]))].copy(); fl=["incident_alzheimers_disease_after_1y","incident_vascular_dementia_after_1y","incident_frontotemporal_dementia_after_1y"]
    s=inc[fl].fillna(0).astype(int); counts=pd.Series({"Alzheimer's disease":int(s.iloc[:,0].sum()),"Vascular dementia":int(s.iloc[:,1].sum()),"Frontotemporal dementia":int(s.iloc[:,2].sum()),"Other or unspecified":int((s.sum(1)==0).sum())})
    fig,ax=plt.subplots(figsize=(5.5,3.5)); yy=np.arange(len(counts)); ax.barh(yy,counts.values,color=[CASE,GOLD,"#8B7BAE",GREY]); ax.set_yticks(yy,counts.index); ax.invert_yaxis(); ax.set_xlabel("Participants"); ax.set_title("Recorded dementia subtypes",loc="left");
    for y,v in zip(yy,counts): ax.text(v+5,y,str(v),va="center",weight="bold",fontsize=8)
    ax.set_xlim(0,max(counts)*1.18); clean_ax(ax); paths.append(save_single(fig,"18_dementia_subtypes",out))
    # 19 daily means
    fig,ax=plt.subplots(figsize=(5.5,3.5)); xx=np.arange(1,8)
    for g,col,lab in [(0,CONTROL,"Control"),(1,CASE,"Incident dementia")]:
        z=daily.loc[daily.target.eq(g),[f"day_{i}" for i in range(1,8)]].to_numpy(float); mm=np.nanmean(z,0); se=np.nanstd(z,0,ddof=1)/np.sqrt(np.sum(np.isfinite(z),0)); ax.plot(xx,mm,marker="o",ms=4,color=col,label=lab); ax.fill_between(xx,mm-1.96*se,mm+1.96*se,color=col,alpha=.16)
    ax.set(xticks=xx,xlabel="Ordered accelerometer day",ylabel="Mean acceleration",title="Day-specific activity across the seven-day assessment"); ax.legend(); clean_ax(ax); paths.append(save_single(fig,"19_day_specific_activity",out))
    # 20 hourly SMD
    hh=[f"hour_{i:02d}" for i in range(24)]; vals=[smd(hourly.loc[hourly.target.eq(1),c],hourly.loc[hourly.target.eq(0),c]) for c in hh]
    fig,ax=plt.subplots(figsize=(6.0,3.4)); ax.plot(np.arange(24),vals,color=CASE,lw=1.5,marker="o",ms=3); ax.axhline(0,color=NAVY,lw=.7); ax.set(xticks=np.arange(0,24,3),xlabel="Clock hour",ylabel="Standardized mean difference",title="Hourly activity effect size"); clean_ax(ax); paths.append(save_single(fig,"20_hourly_activity_smd",out))
    # 21 covariate missingness
    miss=(100*df.drop(columns=["target"]).isna().mean()).sort_values(ascending=True); fig,ax=plt.subplots(figsize=(5.8,5.5)); ax.barh(np.arange(len(miss)),miss,color=np.where(miss>5,CASE,CONTROL)); ax.set_yticks(np.arange(len(miss)),miss.index); ax.set(xlabel="Missing, %",title="Cohort variable completeness"); clean_ax(ax); paths.append(save_single(fig,"21_variable_missingness",out))
    # 22 PA correlation matrix
    cols=["enrollment_age","BMI","Townsend_index","Light_Total_Hour","MVPA_Total_Hour","Sedentary_Total_Hour","Sleep_Total_Hour"]
    corr=df[cols].corr(method="spearman"); fig,ax=plt.subplots(figsize=(5.4,4.8)); im=ax.imshow(corr,cmap="RdBu_r",vmin=-1,vmax=1); labs=[LABEL[c].replace(", years","").replace(", h day$^{-1}$","") for c in cols]; ax.set_xticks(range(len(cols)),labs,rotation=45,ha="right"); ax.set_yticks(range(len(cols)),labs); ax.set_title("Spearman correlations among baseline measures",loc="left"); fig.colorbar(im,ax=ax,label="Spearman ρ",fraction=.045,pad=.03); paths.append(save_single(fig,"22_baseline_correlation_matrix",out))
    return paths

def supplementary_tables(df,metrics,hourly,daily,subtypes,events,out):
    full=pd.read_csv(ARGS.labels,dtype={"eid":"string"},low_memory=False); states=full.dementia_state.value_counts()
    flow=pd.DataFrame([("Initial 7-day cohort",len(full)),("Prevalent dementia at accelerometry",int(states.get("prevalent_at_accelerometer",0))),
        ("Dementia within 1–365 days (washout)",int(states.get("washout_1_365_days",0))),("Eligible incident cohort",len(df)),
        ("Incident dementia >365 days",int(df.target.sum())),("No hospital dementia observed",int((df.target==0).sum()))],columns=["stage","n"])
    flow["percent_initial"]=100*flow.n/len(full); flow.to_csv(out/"tables_numeric"/"Table_S1_cohort_flow_numeric.csv",index=False,encoding="utf-8-sig")
    miss=pd.DataFrame({"variable":df.columns,"missing_n":df.isna().sum().values,"missing_percent":100*df.isna().mean().values}); miss.to_csv(out/"tables_numeric"/"Table_S2_missingness_numeric.csv",index=False,encoding="utf-8-sig")
    cases=df[df.target.eq(1)].copy(); years=pd.to_numeric(cases.days_accel_to_dementia)/365.25
    timing=pd.DataFrame({"statistic":["n","mean","s.d.","median","25th percentile","75th percentile","minimum","maximum"],"years":[len(years),years.mean(),years.std(),years.median(),years.quantile(.25),years.quantile(.75),years.min(),years.max()]}); timing.to_csv(out/"tables_numeric"/"Table_S3_time_to_diagnosis_numeric.csv",index=False,encoding="utf-8-sig")
    bands=pd.cut(df.enrollment_age,[35,50,55,60,65,70,80],right=False); age=df.groupby(bands,observed=True).target.agg(cases="sum",total="count").reset_index(); age["controls"]=age.total-age.cases; age["incident_percent"]=100*age.cases/age.total; age.to_csv(out/"tables_numeric"/"Table_S4_age_band_incidence_numeric.csv",index=False,encoding="utf-8-sig")
    inc=subtypes[subtypes.eid.isin(set(cases.eid))].copy(); fl=["incident_alzheimers_disease_after_1y","incident_vascular_dementia_after_1y","incident_frontotemporal_dementia_after_1y"]; s=inc[fl].fillna(0).astype(int)
    subtype=pd.DataFrame({"category":["All-cause incident dementia","Any AD/vascular/FTD","Alzheimer's disease","Vascular dementia","Frontotemporal dementia","Multiple specific subtypes","Other or unspecified"],
        "unique_participants":[len(cases),int((s.sum(1)>0).sum()),int(s.iloc[:,0].sum()),int(s.iloc[:,1].sum()),int(s.iloc[:,2].sum()),int((s.sum(1)>1).sum()),int((s.sum(1)==0).sum())],
        "interpretation":["Primary outcome","Unique participant count","Non-mutually-exclusive","Non-mutually-exclusive","Non-mutually-exclusive","At least two of AD/vascular/FTD","No AD/vascular/FTD flag"]}); subtype.to_csv(out/"tables_numeric"/"Table_S5_subtypes_numeric.csv",index=False,encoding="utf-8-sig")
    # ICD-10 first-record distribution and all-event distribution among primary cases.
    first=cases.first_dementia_icd10.astype("string").fillna("Missing").value_counts().rename_axis("first_icd10").reset_index(name="participants"); first["percent_cases"]=100*first.participants/len(cases); first.to_csv(out/"tables_numeric"/"Table_S6_first_ICD10_distribution_numeric.csv",index=False,encoding="utf-8-sig")
    if events is not None and len(events):
        ecol=next((c for c in events.columns if "icd" in c.lower() and "description" not in c.lower()),None)
        if ecol:
            ev=events[events.eid.astype(str).isin(set(cases.eid.astype(str)))]; ec=ev[ecol].astype("string").value_counts().rename_axis("icd10").reset_index(name="event_records"); ec.to_csv(out/"tables_numeric"/"Table_S7_all_dementia_ICD10_events_numeric.csv",index=False,encoding="utf-8-sig")
    # Hourly and daily exact summary tables.
    hr=[]
    for h in range(24):
        c=f"hour_{h:02d}"; a=hourly.loc[hourly.target.eq(1),c]; b=hourly.loc[hourly.target.eq(0),c]; p=stats.mannwhitneyu(a.dropna(),b.dropna()).pvalue
        hr.append({"hour":h,"control_mean":b.mean(),"control_sd":b.std(),"incident_mean":a.mean(),"incident_sd":a.std(),"smd":smd(a,b),"p_value":p})
    hr=pd.DataFrame(hr); hr["q_value_bh"]=bh(hr.p_value); hr.to_csv(out/"tables_numeric"/"Table_S8_hourly_activity_numeric.csv",index=False,encoding="utf-8-sig")
    dd=[]
    for d in range(1,8):
        c=f"day_{d}"; a=daily.loc[daily.target.eq(1),c]; b=daily.loc[daily.target.eq(0),c]; dd.append({"ordered_day":d,"control_mean":b.mean(),"control_sd":b.std(),"incident_mean":a.mean(),"incident_sd":a.std(),"smd":smd(a,b),"p_value":stats.mannwhitneyu(a.dropna(),b.dropna()).pvalue})
    dd=pd.DataFrame(dd); dd["q_value_bh"]=bh(dd.p_value); dd.to_csv(out/"tables_numeric"/"Table_S9_day_specific_activity_numeric.csv",index=False,encoding="utf-8-sig")
    return flow,miss,timing,age,subtype,first

def atlas(paths,out,df):
    pdf=out/"UKB_dementia_descriptive_atlas.pdf"
    with PdfPages(pdf) as pp:
        fig=plt.figure(figsize=(8.27,11.69)); fig.text(.10,.88,"UKB dementia cohort",fontsize=24,color=NAVY,weight="bold"); fig.text(.10,.835,"Seven-day physical activity descriptive atlas",fontsize=16,color=CONTROL)
        fig.text(.10,.73,f"Eligible cohort  n = {len(df):,}\nIncident dementia >1 year  n = {int(df.target.sum()):,}\nNo hospital dementia observed  n = {int((df.target==0).sum()):,}",fontsize=13,linespacing=1.7)
        fig.text(.10,.58,"Local analysis of original minute-level acceleration data.\nNo model predictions, checkpoints, embeddings or pretrained representations were used.",fontsize=10.5,color=GREY,linespacing=1.5)
        fig.text(.10,.12,f"Generated {datetime.now():%Y-%m-%d %H:%M}",fontsize=9,color=GREY); pp.savefig(fig); plt.close(fig)
        for p in paths:
            img=plt.imread(p); fig,ax=plt.subplots(figsize=(8.27,11.69)); ax.imshow(img); ax.axis("off"); pp.savefig(fig,bbox_inches="tight"); plt.close(fig)
    return pdf

def main():
    global ARGS
    ap=argparse.ArgumentParser(); ap.add_argument("--labels",type=Path,required=True); ap.add_argument("--subtypes",type=Path,required=True); ap.add_argument("--events",type=Path); ap.add_argument("--activity",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); ARGS=ap.parse_args(); setup(ARGS.out)
    from study_design import fixed_window_cohort
    df=fixed_window_cohort(pd.read_csv(ARGS.labels,dtype={"eid":"string"},low_memory=False))
    logging.info("Eligible cohort N=%d; incident cases=%d; controls=%d",len(df),df.target.sum(),(df.target==0).sum())
    raw1,pub1=table1(df); export_table(raw1,pub1,"Table_1_baseline_characteristics",ARGS.out)
    metrics,hourly,daily,agg=derive_stream(ARGS.activity,df[["eid","target"]],ARGS.out)
    metrics=metrics.set_index("eid").loc[df.eid].reset_index(); hourly=hourly.set_index("eid").loc[df.eid].reset_index(); daily=daily.set_index("eid").loc[df.eid].reset_index()
    raw2,pub2=metric_table(metrics); export_table(raw2,pub2,"Table_2_seven_day_activity_phenotypes",ARGS.out)
    sub=pd.read_csv(ARGS.subtypes,dtype={"eid":"string"},low_memory=False); events=pd.read_csv(ARGS.events,dtype={"eid":"string"},low_memory=False) if ARGS.events and ARGS.events.exists() else None
    supp=supplementary_tables(df,metrics,hourly,daily,sub,events,ARGS.out)
    dayhour=derive_day_hour_aggregates(ARGS.activity,df[["eid","target"]],ARGS.out)
    paths=make_figures(df,metrics,hourly,daily,agg,dayhour,sub,events,ARGS.out); pdf=atlas(paths,ARGS.out,df)
    manifest={"generated":datetime.now().isoformat(),"cohort_n":len(df),"incident_n":int(df.target.sum()),"control_n":int((df.target==0).sum()),"figure_count":len(paths),"figures":[p.name for p in paths],"atlas":pdf.name,
        "analysis_scope":"Descriptive only; hospital inpatient ICD-10 ascertainment; no model outputs used.","p_value_policy":"Two-sided; Welch t test for Table 1 continuous variables, chi-square for categorical variables; Mann-Whitney U for activity phenotypes; BH FDR for activity families.","display_policy":"Continuous values mean (s.d.); percentages one decimal; SMD two decimals; P >=0.001 three decimals; smaller P values scientific notation."}
    (ARGS.out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    logging.info("Completed %d single figures and atlas %s",len(paths),pdf)

if __name__=="__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="Mean of empty slice")
        main()
