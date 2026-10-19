#!/usr/bin/env python3
"""Frozen fusion dementia-model validation in NHATS R11--R14.

Primary endpoint: incident possible-or-probable dementia from R11 to R14,
excluding possible/probable dementia at R11. No NHATS outcome is used for
model fitting, checkpoint selection, harmonization or calibration.
"""
from __future__ import annotations
import argparse, json, logging, math, os, sys
from datetime import datetime
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA_ROOT, RESULTS_ROOT
from types import SimpleNamespace

for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
 _x=os.environ.get(_v,"").strip()
 if not _x.isdigit() or int(_x)<1:os.environ[_v]="4"

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

import cohort
import pamed_models
import fusion

DEFAULT_INPUT=DATA_ROOT/"external_validation"
DEFAULT_OUT=RESULTS_ROOT/"04_nhats_external"
MINUTES=[str(i) for i in range(1,1441)]
AGE_MAP={2:72.0,3:77.0,4:82.0,5:87.0,6:92.5}
AGE_SEX=["enrollment_age","gender"]
TRAD_PA=["Light_Total_Hour","MVPA_Total_Hour","Sedentary_Total_Hour","Sleep_Total_Hour"]
PRIMARY_COMPARATOR="age_sex_traditional_pa"
ACTIVITY_FILES={
 "positive_robust":"nhats_r11_daily_7x1440_positive_robust.csv",
 "raw_count":"nhats_r11_daily_7x1440_raw_count.csv",
 "positive_quantile":"nhats_r11_daily_7x1440_positive_quantile.csv",
 "learned_adapter":"nhats_r11_adapter_daily_7x1440_raw.csv",
}

def atomic_json(path,obj):
 tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf-8");os.replace(tmp,path)

def setup(out):
 out.mkdir(parents=True,exist_ok=True);logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",handlers=[logging.FileHandler(out/"run.log",mode="w",encoding="utf-8"),logging.StreamHandler(sys.stdout)],force=True)

def find_one(folder,exact=None,pattern=None,required=True):
 p=folder/exact if exact else None
 if p is not None and p.exists():return p
 hits=sorted(folder.glob(pattern)) if pattern else []
 if len(hits)==1:return hits[0]
 if required:raise FileNotFoundError(f"Expected one input: exact={exact!r} pattern={pattern!r}; found={hits}")
 return None

def normalize_eid(values):
 s=pd.Series(values,dtype="string").str.strip();digits=s.str.extract(r"(\d{6,})",expand=False)
 return digits.fillna(s.str.replace(r"\.0+$","",regex=True)).astype("string")

def coded_number(values):
 s=pd.Series(values,dtype="string").str.strip();code=s.str.extract(r"^\s*(-?\d+(?:\.\d+)?)",expand=False)
 return pd.to_numeric(code,errors="coerce")

def load_demographics(path):
 d=pd.read_csv(path,low_memory=False);cols={str(c).strip().lower():c for c in d.columns}
 idcol=cols.get("spid",cols.get("eid",d.columns[0]));agecol=cols.get("r11d2intvrage");sexcol=cols.get("r5dgender")
 if agecol is None or sexcol is None:raise ValueError(f"Demographic fields missing; need r11d2intvrage and r5dgender. Columns={list(d.columns)}")
 q=d[[idcol,agecol,sexcol]].rename(columns={idcol:"eid",agecol:"age_group",sexcol:"gender"}).copy();age_text=q.age_group.astype("string").str.lower();sex_text=q.gender.astype("string").str.lower();q["eid"]=normalize_eid(q.eid);q["age_group"]=coded_number(q.age_group);q.loc[age_text.str.contains(r"70\s*(?:to|[-–])\s*74",regex=True,na=False),"age_group"]=2;q.loc[age_text.str.contains(r"75\s*(?:to|[-–])\s*79",regex=True,na=False),"age_group"]=3;q.loc[age_text.str.contains(r"80\s*(?:to|[-–])\s*84",regex=True,na=False),"age_group"]=4;q.loc[age_text.str.contains(r"85\s*(?:to|[-–])\s*89",regex=True,na=False),"age_group"]=5;q.loc[age_text.str.contains(r"90\s*\+",regex=True,na=False),"age_group"]=6;q.loc[age_text.str.contains("inapplicable",na=False),"age_group"]=-1;q["gender"]=coded_number(q.gender);q.loc[sex_text.str.contains("male",na=False)&~sex_text.str.contains("female",na=False),"gender"]=1;q.loc[sex_text.str.contains("female",na=False),"gender"]=2;q["enrollment_age"]=q.age_group.map(AGE_MAP);q.loc[~q.gender.isin([1,2]),"gender"]=np.nan
 if q.eid.duplicated().any():raise ValueError("Duplicate IDs in NHATS demographics")
 return q

def load_labels(path):
 d=pd.read_csv(path,dtype={"eid":"string"},low_memory=False);d["eid"]=normalize_eid(d.eid);need=["dementia_binary_r11","incident_dementia_binary","incident_probable_dementia"]
 if not set(need).issubset(d.columns):raise ValueError(f"NHATS dementia label columns missing: {sorted(set(need)-set(d.columns))}")
 for c in need:d[c]=pd.to_numeric(d[c],errors="coerce")
 if d.eid.duplicated().any():raise ValueError("Duplicate IDs in NHATS labels")
 return d[["eid",*need]]

def load_pa_summaries(path):
 """Read externally prepared daily-duration summaries; do not derive them from counts."""
 if not path.is_file():
  raise FileNotFoundError(f"NHATS traditional-PA summaries are required: {path}. Supply --pa-summary-csv with eid and {TRAD_PA}, in hours/day. Raw counts cannot substitute for these inputs.")
 d=pd.read_csv(path,dtype={"eid":"string"},low_memory=False)
 missing=sorted({"eid",*TRAD_PA}-set(d.columns))
 if missing:raise ValueError(f"{path}: required PA-summary columns missing: {missing}")
 q=d[["eid",*TRAD_PA]].copy();q["eid"]=normalize_eid(q.eid)
 if q.eid.isna().any() or q.eid.eq("").any() or q.eid.duplicated().any():
  raise ValueError(f"{path}: PA-summary eid values must be nonempty and unique")
 for column in TRAD_PA:q[column]=pd.to_numeric(q[column],errors="coerce")
 return q

def activity_paths(folder,manifest_path=None):
 if manifest_path:
  manifest=Path(manifest_path).expanduser().resolve()
  spec=json.loads(manifest.read_text(encoding="utf-8"))
  if not isinstance(spec,dict) or set(spec)-set(ACTIVITY_FILES):
   raise ValueError(f"{manifest}: expected an object with activity modes from {sorted(ACTIVITY_FILES)}")
  activity={}
  for mode,value in spec.items():
   if not isinstance(value,str):raise ValueError(f"{manifest}: {mode} must map to a CSV path")
   path=Path(value).expanduser();activity[mode]=path if path.is_absolute() else manifest.parent/path
 else:activity={mode:folder/name for mode,name in ACTIVITY_FILES.items() if (folder/name).is_file()}
 for mode,path in activity.items():
  if not path.is_file():raise FileNotFoundError(f"{mode}: {path}")
 if not {"positive_robust","raw_count"}.issubset(activity):
  raise FileNotFoundError("Provide positive_robust and raw_count NHATS 7x1440 activity CSVs; optional sensitivity mappings must be prepared without dementia outcomes.")
 return activity

def read_week(path):
 d=pd.read_csv(path,dtype={"eid":"string"},usecols=["eid",*MINUTES]);d["eid"]=normalize_eid(d.eid);counts=d.eid.value_counts()
 if not counts.eq(7).all():raise ValueError(f"{path.name}: every participant must have exactly seven rows; distribution={counts.value_counts().to_dict()}")
 eids=d.eid.drop_duplicates().to_numpy();groups=[]
 for eid in eids:
  a=d.loc[d.eid.eq(eid),MINUTES].to_numpy(np.float32)
  if a.shape!=(7,1440):raise ValueError(f"Bad week shape for {eid}: {a.shape}")
  groups.append(a)
 return eids,np.stack(groups)

class WeekArray(Dataset):
 def __init__(self,x,loc,scale):self.x=np.asarray(x,np.float32);self.loc=float(loc);self.scale=float(scale)
 def __len__(self):return len(self.x)
 def __getitem__(self,j):
  a=np.asarray(self.x[j],np.float32);m=np.isfinite(a);z=(np.log1p(np.clip(np.nan_to_num(a,nan=0.),0,None))-self.loc)/self.scale;z=np.where(m,z,0.);return torch.from_numpy(np.stack([z.reshape(-1),m.astype(np.float32).reshape(-1)],0).astype(np.float32))

def load_models(cfg,tab_dim,device):
 root=Path(cfg.core_dir);gp=root/"grufcn_80pct_refit.pt";cp=root/"convcontext_mamba_80pct_refit.pt";hp=root/"final_hybrid_80pct_refit.pt";pp=root/"dual_pa_feature_80pct_refit.pt"
 for p in [gp,cp,hp,pp]:
  if not p.exists() or p.stat().st_size==0:raise FileNotFoundError(p)
 gs=torch.load(gp,map_location=device);cs=torch.load(cp,map_location=device)
 if abs(float(gs["loc"])-float(cs["loc"]))>1e-6 or abs(float(gs["scale"])-float(cs["scale"]))>1e-6:raise ValueError("GRU-FCN and ConvContext scaling mismatch")
 base=gs.get("config",{});mcfg=SimpleNamespace(d_model=int(base.get("d_model",64)),dropout=float(base.get("dropout",.30)))
 gm=pamed_models.build("grufcn",mcfg).to(device);cm=pamed_models.build("convcontext_mamba",mcfg).to(device);gm.load_state_dict(gs["model"]);cm.load_state_dict(cs["model"]);gm.eval();cm.eval()
 hm=fusion.HybridFusion(tab_dim).to(device);hm.load_state_dict(torch.load(hp,map_location=device)["model"]);hm.eval();pm=fusion.DualPAFeature().to(device);pm.load_state_dict(torch.load(pp,map_location=device)["model"]);pm.eval()
 pa_path=root/"age_sex_traditional_pa_80pct.joblib"
 if not pa_path.is_file():raise FileNotFoundError(f"Frozen UKB age-sex-PA comparator is required: {pa_path}")
 pa_pipe=joblib.load(pa_path);age_path=root/"age_sex_80pct.joblib"
 agepipe=joblib.load(age_path) if age_path.is_file() else None
 return {"gru":gm,"conv":cm,"hybrid":hm,"pa_fusion":pm,"pa_pipe":pa_pipe,"agepipe":agepipe,"loc":float(gs["loc"]),"scale":float(gs["scale"])}

def infer_mode(x,tab,models,cfg,device):
 loader=DataLoader(WeekArray(x,models["loc"],models["scale"]),batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,pin_memory=device.type=="cuda",persistent_workers=cfg.num_workers>0);ge=[];ce=[];gl=[];cl=[]
 with torch.no_grad():
  for xb in loader:
   xb=xb.to(device,non_blocking=True)
   with cohort.amp_ctx(device,cfg.amp):
    g=models["gru"].encoder(xb);c=models["conv"].encoder(xb);a=models["gru"].risk(g).squeeze(-1);b=models["conv"].risk(c).squeeze(-1)
   ge.append(g.float().cpu());ce.append(c.float().cpu());gl.append(a.float().cpu());cl.append(b.float().cpu())
 g=torch.cat(ge);c=torch.cat(ce);a=torch.cat(gl);b=torch.cat(cl);t=torch.from_numpy(np.asarray(tab,np.float32));e=torch.zeros((len(t),1),dtype=torch.float32);scores={"grufcn_pa":a.numpy(),"convcontext_pa":b.numpy()}
 with torch.no_grad():
  out=[];pout=[]
  for st in range(0,len(t),cfg.eval_batch_size):
   ix=slice(st,min(st+cfg.eval_batch_size,len(t)));args=[q[ix].to(device) for q in (g,c,a,b,t,e)]
   with cohort.amp_ctx(device,cfg.amp):pout.append(models["pa_fusion"](*args).float().cpu());out.append(models["hybrid"](*args).float().cpu())
 scores["dual_pa_feature"]=torch.cat(pout).numpy();scores["final_hybrid"]=torch.cat(out).numpy();return scores

def metrics(y,s,repeats,seed):
 y=np.asarray(y,int);s=np.asarray(s,float);ap=average_precision_score(y,s);auc=roc_auc_score(y,s);rng=np.random.default_rng(seed);pos=np.where(y==1)[0];neg=np.where(y==0)[0];boot=[]
 for _ in range(repeats):
  ix=np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)];boot.append([average_precision_score(y[ix],s[ix]),roc_auc_score(y[ix],s[ix])])
 z=np.asarray(boot);return {"n":len(y),"events":int(y.sum()),"prevalence":float(y.mean()),"auprc":float(ap),"auprc_ci_low":float(np.quantile(z[:,0],.025)),"auprc_ci_high":float(np.quantile(z[:,0],.975)),"auprc_over_prevalence":float(ap/y.mean()),"auroc":float(auc),"auroc_ci_low":float(np.quantile(z[:,1],.025)),"auroc_ci_high":float(np.quantile(z[:,1],.975))}

def paired(y,pred,ref,repeats,seed):
 pos=np.where(y==1)[0];neg=np.where(y==0)[0];rows=[]
 for name,s in pred.items():
  if name==ref:continue
  rng=np.random.default_rng(seed);dap=[];dau=[]
  for _ in range(repeats):
   ix=np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)];dap.append(average_precision_score(y[ix],s[ix])-average_precision_score(y[ix],pred[ref][ix]));dau.append(roc_auc_score(y[ix],s[ix])-roc_auc_score(y[ix],pred[ref][ix]))
  rows.append({"model":name,"reference":ref,"delta_definition":"model_minus_reference","delta_auprc":average_precision_score(y,s)-average_precision_score(y,pred[ref]),"delta_auprc_ci_low":np.quantile(dap,.025),"delta_auprc_ci_high":np.quantile(dap,.975),"delta_auroc":roc_auc_score(y,s)-roc_auc_score(y,pred[ref]),"delta_auroc_ci_low":np.quantile(dau,.025),"delta_auroc_ci_high":np.quantile(dau,.975)})
 return pd.DataFrame(rows)

def enrich(y,s):
 rows=[]
 for frac in [.05,.10,.20]:
  k=max(1,math.ceil(len(y)*frac));ix=np.argsort(s,kind="stable")[-k:];rows.append({"top_fraction":frac,"selected_fraction":k/len(y),"selected_n":k,"events_captured":int(y[ix].sum()),"event_capture_fraction":float(y[ix].sum()/y.sum()),"ppv":float(y[ix].mean()),"enrichment":float(y[ix].mean()/y.mean())})
 return rows

def report(out,metric,pred):
 primary=metric[(metric.endpoint.eq("incident_possible_or_probable"))&(metric.analysis.eq("primary_complete_demographics"))];fig,ax=plt.subplots(figsize=(9,6));z=primary.sort_values("auprc");ax.barh(z.model,z.auprc,color="#668DB7");ax.axvline(z.prevalence.iloc[0],color="black",ls="--",lw=.8);ax.set(xlabel="NHATS R11-R14 AUPRC",title="Frozen UKB dementia model in NHATS");fig.tight_layout();fig.savefig(out/"01_external_auprc.png",dpi=300,bbox_inches="tight");plt.close(fig)
 y=pred.target.to_numpy(int);fig,ax=plt.subplots(figsize=(7,6))
 for c in [x for x in pred if x.endswith("__final_hybrid")]:
  f,t,_=roc_curve(y,pred[c]);ax.plot(f,t,label=f"{c.split('__')[0]}: {roc_auc_score(y,pred[c]):.3f}")
 ax.plot([0,1],[0,1],"k--",lw=.8);ax.set(xlabel="False-positive rate",ylabel="True-positive rate",title="Final Hybrid external ROC");ax.legend(fontsize=8,frameon=False);fig.tight_layout();fig.savefig(out/"02_external_roc.png",dpi=300,bbox_inches="tight");plt.close(fig)
 pdf=out/"PAMED_NHATS_EXTERNAL_REPORT.pdf"
 with PdfPages(pdf) as pages:
  fig=plt.figure(figsize=(11.69,8.27));fig.text(.06,.86,"PAMED external validation",fontsize=24,weight="bold");fig.text(.06,.76,"Frozen external validation in NHATS R11–R14",fontsize=16);fig.text(.06,.57,"Primary endpoint: incident possible-or-probable dementia over approximately 3 years.\nR11 prevalent dementia excluded; zero refitting and zero outcome-informed calibration.\nUKB model horizon is 7 years, so NHATS validates ranking, not 3-year absolute calibration.",fontsize=11.5,linespacing=1.8);pages.savefig(fig);plt.close(fig)
  for f in ["01_external_auprc.png","02_external_roc.png"]:
   fig,ax=plt.subplots(figsize=(11.69,8.27));ax.imshow(plt.imread(out/f));ax.axis("off");pages.savefig(fig,bbox_inches="tight");plt.close(fig)
 return pdf

def run(cfg):
 folder=Path(cfg.input_dir).expanduser();out=Path(cfg.output_dir).expanduser();setup(out)
 if cfg.bootstrap_repeats<1:raise ValueError("--bootstrap-repeats must be positive")
 pa_path=Path(cfg.pa_summary_csv).expanduser() if cfg.pa_summary_csv else folder/"nhats_pa_summaries.csv"
 pa=load_pa_summaries(pa_path);activity=activity_paths(folder,cfg.activity_manifest)
 atomic_json(out/"config.json",{**vars(cfg),"pa_summary_csv":str(pa_path),"activity_csvs":{k:str(v) for k,v in activity.items()}})
 demo=find_one(folder,"NHATS_Round_11_SP_File_V2.csv")
 labels=find_one(folder,"NHATS_R11_BEAT384_revised_longitudinal_labels_R11_R14.csv","*longitudinal_labels_R11_R14.csv")
 dem=load_demographics(demo);lab=load_labels(labels)
 first_eids,first_x=read_week(activity["positive_robust"])
 base=(pd.DataFrame({"eid":normalize_eid(first_eids)})
       .merge(lab,on="eid",how="left",validate="one_to_one")
       .merge(dem,on="eid",how="left",validate="one_to_one")
       .merge(pa,on="eid",how="left",validate="one_to_one",indicator="pa_summary_merge"))
 flow=[{"stage":"PA QC cohort","n":len(base),"events":int(base.dementia_binary_r11.eq(1).sum())},
       {"stage":"label intersection","n":int(base.dementia_binary_r11.notna().sum()),"events":int(base.dementia_binary_r11.eq(1).sum())}]
 eligible=base.dementia_binary_r11.eq(0)&base.incident_dementia_binary.notna()
 q0=base.loc[eligible].copy()
 if not q0.incident_dementia_binary.isin([0,1]).all():raise ValueError("Incident dementia endpoint must be binary")
 q0["target"]=q0.incident_dementia_binary.astype(np.int8)
 flow.append({"stage":"incident endpoint observed; R11 dementia excluded","n":len(q0),"events":int(q0.target.sum())})
 complete=q0.enrollment_age.notna()&q0.gender.notna()
 diag=pd.DataFrame([{"activity_unique_ids":len(first_eids),"label_unique_ids":lab.eid.nunique(),
                     "demographic_unique_ids":dem.eid.nunique(),"pa_summary_unique_ids":pa.eid.nunique(),
                     "eligible_endpoint_n":len(q0),"eligible_age_recognized":int(q0.enrollment_age.notna().sum()),
                     "eligible_gender_recognized":int(q0.gender.notna().sum()),"eligible_complete_demographics":int(complete.sum()),
                     "eligible_pa_summary_matches":int(q0.pa_summary_merge.eq("both").sum()),
                     "recognized_age_groups":json.dumps(q0.age_group.value_counts(dropna=False).sort_index().to_dict(),default=str),
                     "recognized_gender_codes":json.dumps(q0.gender.value_counts(dropna=False).sort_index().to_dict(),default=str)}])
 diag.to_csv(out/"demographic_merge_diagnostics.csv",index=False)
 q=q0.loc[complete].reset_index(drop=True)
 flow.append({"stage":"complete age and sex primary cohort","n":len(q),"events":int(q.target.sum())})
 pd.DataFrame(flow).to_csv(out/"cohort_flow.csv",index=False)
 if q.empty:raise ValueError("No complete NHATS age and sex inputs; inspect demographic_merge_diagnostics.csv")
 pa_values=q[TRAD_PA].to_numpy(float)
 invalid=~np.isfinite(pa_values)|(pa_values<0)|(pa_values>24)
 if invalid.any():
  bad=q.loc[invalid.any(axis=1),["eid",*TRAD_PA,"pa_summary_merge"]]
  bad.to_csv(out/"invalid_pa_summaries.csv",index=False)
  raise ValueError(f"{len(bad)} primary-cohort participants lack valid traditional-PA summaries in hours/day. Inspect {out/'invalid_pa_summaries.csv'} and supply --pa-summary-csv. The age-sex-PA comparator cannot fall back to age and sex.")
 logging.info("Primary NHATS cohort N=%d events=%d prevalence=%.4f",len(q),int(q.target.sum()),q.target.mean())
 y=q.target.to_numpy(int)
 if np.unique(y).size!=2:raise ValueError("Primary endpoint requires cases and non-cases")
 age_input=q[AGE_SEX];pa_input=q[AGE_SEX+TRAD_PA]
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 pre=joblib.load(Path(cfg.core_dir)/"age_sex_preprocessor_80pct.joblib")
 tab=np.asarray(pre.transform(age_input),np.float32);models=load_models(cfg,tab.shape[1],device)
 baselines={PRIMARY_COMPARATOR:cohort.logit(models["pa_pipe"].predict_proba(pa_input)[:,1])}
 if models["agepipe"] is not None:baselines["age_sex"]=cohort.logit(models["agepipe"].predict_proba(age_input)[:,1])
 pred=q[["eid","target","age_group","gender",*TRAD_PA]].copy()
 pred["strict_probable"]=q.incident_probable_dementia;pred["age_midpoint"]=q.enrollment_age
 for name,s in baselines.items():pred[name]=s
 for mode,path in activity.items():
  if mode=="positive_robust":eids,x=first_eids,first_x
  else:eids,x=read_week(path)
  lookup={e:i for i,e in enumerate(eids)}
  if not q.eid.isin(lookup).all():raise ValueError(f"{mode}: missing primary-cohort activity IDs")
  xx=x[[lookup[e] for e in q.eid]];scores=infer_mode(xx,tab,models,cfg,device)
  for name,s in scores.items():
   if not np.isfinite(s).all():raise ValueError(f"Non-finite inference scores: {mode}/{name}")
   pred[f"{mode}__{name}"]=s
  diagnostic={"mode":mode,"n":len(xx),"missing_fraction":float((~np.isfinite(xx)).mean()),
              "zero_fraction":float(np.nanmean(xx==0)),
              "positive_median":float(np.nanmedian(xx[xx>0])) if np.any(xx>0) else np.nan,
              "positive_q90":float(np.nanquantile(xx[xx>0],.9)) if np.any(xx>0) else np.nan}
  pd.DataFrame([diagnostic]).to_csv(out/f"diagnostic_{mode}.csv",index=False)
 all_scores={**baselines,**{c:pred[c].to_numpy(float) for c in pred if "__" in c}}
 if any(not np.isfinite(s).all() for s in all_scores.values()):raise ValueError("Comparator scores must be finite")
 rows=[];enrichment=[]
 def collect(analysis,endpoint,target,mask,selected_scores,seed_offset):
  target=np.asarray(target)[mask].astype(int)
  for i,(name,score) in enumerate(selected_scores.items()):
   mode,model=name.split("__",1) if "__" in name else ("not_applicable",name)
   fields={"analysis":analysis,"endpoint":endpoint,"harmonization":mode,"model":model,
           "primary_comparator":model==PRIMARY_COMPARATOR}
   if len(target)==0 or np.unique(target).size!=2:
    rows.append({**fields,"n":len(target),"events":int(target.sum()),"status":"insufficient_outcome_classes"});continue
   score=np.asarray(score)[mask];row={**fields,"status":"evaluated"}
   row.update(metrics(target,score,cfg.bootstrap_repeats,cfg.seed+seed_offset+i));rows.append(row)
   if analysis=="primary_complete_demographics":
    for values in enrich(target,score):enrichment.append({**fields,**values,"score":name})
 collect("primary_complete_demographics","incident_possible_or_probable",y,np.ones(len(y),bool),all_scores,0)
 # Sensitivity sets retain the same frozen comparator and disease-model scores.
 sensitivity_scores={**baselines,**{f"{mode}__final_hybrid":pred[f"{mode}__final_hybrid"].to_numpy(float) for mode in activity}}
 strict=q.incident_probable_dementia.to_numpy(float);strict_mask=np.isfinite(strict)
 if not np.isin(strict[strict_mask],[0,1]).all():raise ValueError("Strict probable-dementia endpoint must be binary")
 collect("endpoint_sensitivity","incident_probable_strict",strict,strict_mask,sensitivity_scores,200)
 overlap=q.enrollment_age.le(79).to_numpy()
 collect("UKB_age_overlap_70_79","incident_possible_or_probable",y,overlap,sensitivity_scores,300)
 metric=pd.DataFrame(rows);metric.to_csv(out/"nhats_external_metrics.csv",index=False)
 pred.to_csv(out/"nhats_external_predictions.csv",index=False)
 pd.DataFrame(enrichment).to_csv(out/"risk_enrichment.csv",index=False)
 for mode in activity:
  names={**baselines,**{name:pred[f"{mode}__{name}"].to_numpy(float) for name in ["grufcn_pa","convcontext_pa","dual_pa_feature","final_hybrid"]}}
  paired(y,names,"final_hybrid",cfg.bootstrap_repeats,cfg.seed+500).assign(harmonization=mode).to_csv(out/f"paired_vs_final_{mode}.csv",index=False)
  paired(y,names,PRIMARY_COMPARATOR,cfg.bootstrap_repeats,cfg.seed+500).assign(harmonization=mode).to_csv(out/f"paired_vs_age_sex_traditional_pa_{mode}.csv",index=False)
 pdf=report(out,metric,pred)
 audit={"external_cohort":"NHATS R11-R14","zero_refit":True,"zero_outcome_informed_calibration":True,
        "UKB_training_horizon_years":7,"NHATS_followup_years_approx":3,
        "primary_endpoint":"incident possible or probable dementia","baseline_exclusion":"possible or probable dementia at R11",
        "age_mapping":AGE_MAP,"age_minus1":"excluded as inapplicable","sex_mapping":{"1":"male","2":"female"},
        "primary_harmonization":"positive_robust","sensitivity_harmonizations":sorted(set(activity)-{"positive_robust"}),
        "primary_comparator":PRIMARY_COMPARATOR,"primary_comparator_checkpoint":"age_sex_traditional_pa_80pct.joblib",
        "primary_comparator_features":AGE_SEX+TRAD_PA,"pa_summary_units":"hours/day","pa_summary_csv":str(pa_path),
        "activity_csvs":{k:str(v) for k,v in activity.items()},"apoe_used":False,
        "probability_warning":"Scores are used for ranking; not interpreted as calibrated NHATS 3-year probabilities."}
 atomic_json(out/"external_validation_audit.json",audit)
 atomic_json(out/"COMPLETED.json",{"completed":True,"time":datetime.now().isoformat(),"n":len(q),"events":int(y.sum()),"report":pdf.name})
 logging.info("NHATS external validation completed N=%d events=%d report=%s",len(q),int(y.sum()),pdf)

def args():
 a=argparse.ArgumentParser(description=__doc__)
 a.add_argument("--input-dir",default=str(DEFAULT_INPUT))
 a.add_argument("--core-dir",required=True,help="Directory containing frozen UKB 80%% development-fit checkpoints")
 a.add_argument("--pa-summary-csv",help="Participant-level NHATS eid and four traditional PA durations (hours/day); defaults to INPUT_DIR/nhats_pa_summaries.csv")
 a.add_argument("--activity-manifest",help="Optional JSON mapping harmonization names to prepared activity CSVs; relative paths resolve beside the manifest")
 a.add_argument("--output-dir",default=str(DEFAULT_OUT))
 a.add_argument("--batch-size",type=int,default=64)
 a.add_argument("--eval-batch-size",type=int,default=512)
 a.add_argument("--num-workers",type=int,default=4)
 a.add_argument("--bootstrap-repeats",type=int,default=2000)
 a.add_argument("--seed",type=int,default=20260819)
 a.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True)
 return a.parse_args()
if __name__=="__main__":run(args())

