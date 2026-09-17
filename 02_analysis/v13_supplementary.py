#!/usr/bin/env python3
"""Frozen supplementary experiments for Dementia v13 final.

Stages never redraw the outer split and never choose a best random seed.  They
produce tidy CSV/JSON outputs only; publication figures are intentionally left
to the later figure-design stage.
"""
from __future__ import annotations

import argparse, json, logging, os, random, shutil, subprocess, sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch.utils.data import DataLoader

import v13_cohort_engine as v6
import v13_core_80_20 as core
import v13_fusion_engine as fusion
import v13_mamba_engine as models

ROOT=Path("/root/autodl-tmp/PA_CR_CNN/Project_new/PA_Foundation_Model/dementia")
RESULTS=ROOT/"Results"/"v13_final"
SEEDS=(20260820,20260821,20260822)


def setup(out:Path):
 out.mkdir(parents=True,exist_ok=True)
 logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",handlers=[logging.FileHandler(out/"run.log",encoding="utf-8"),logging.StreamHandler(sys.stdout)],force=True)


def atomic_json(path:Path,obj):
 tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf-8");os.replace(tmp,path)


def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def load_predictions(core_dir:Path):
 p=core_dir/"ukb_20pct_final_test_predictions.csv"
 if not p.exists():raise FileNotFoundError(f"Run v13 core first: {p}")
 x=pd.read_csv(p,dtype={"eid":"string"});x.eid=x.eid.str.strip();return x


def load_cohort(cfg):
 return v6.load_cohort(v6.Config(cfg.labels_csv,cfg.apoe_csv,cfg.cache_dir,cfg.output_dir,horizon_years=7.))


def split_indices(d,protocol:Path):
 outer=pd.read_csv(protocol/"ukb_fixed_80_20_split.csv",dtype={"eid":"string"});outer.eid=outer.eid.str.strip()
 inner=pd.read_csv(protocol/"development_internal_epoch_selection.csv",dtype={"eid":"string"});inner.eid=inner.eid.str.strip()
 if outer.eid.tolist()!=d.eid.tolist():raise ValueError("Protocol/cohort EID order mismatch")
 dev=np.where(outer.split.eq("development"))[0];test=np.where(outer.split.eq("test"))[0];role=dict(zip(inner.eid,inner.role));itr=np.array([i for i in dev if role.get(d.eid.iloc[i])=="inner_training"]);iva=np.array([i for i in dev if role.get(d.eid.iloc[i])=="epoch_selection"])
 return dev,test,itr,iva


def simple_metric(name,y,p):
 y=np.asarray(y,int);p=np.clip(np.asarray(p,float),1e-7,1-1e-7);prev=y.mean()
 return {"model":name,"n":len(y),"events":int(y.sum()),"prevalence":prev,"auprc":average_precision_score(y,p),"auroc":roc_auc_score(y,p),"auprc_over_prevalence":average_precision_score(y,p)/prev,"brier":brier_score_loss(y,p),"log_loss":log_loss(y,p,labels=[0,1])}


def stratified_indices(y,rng):
 pos=np.where(y==1)[0];neg=np.where(y==0)[0];ix=np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)];rng.shuffle(ix);return ix


def bootstrap_metrics(rows,y,preds,repeats,seed):
 rng=np.random.default_rng(seed);vals={k:{"auprc":[],"auroc":[],"brier":[]} for k in preds}
 for b in range(repeats):
  ix=stratified_indices(y,rng);yy=y[ix]
  for k,p in preds.items():
   vals[k]["auprc"].append(average_precision_score(yy,p[ix]));vals[k]["auroc"].append(roc_auc_score(yy,p[ix]));vals[k]["brier"].append(brier_score_loss(yy,p[ix]))
  if (b+1)%500==0:logging.info("Bootstrap %d/%d",b+1,repeats)
 for r in rows:
  for metric,a in vals[r["model"]].items():r[metric+"_ci_low"],r[metric+"_ci_high"]=np.quantile(a,[.025,.975])
 return rows


def paired_table(y,preds,reference,repeats,seed):
 rng=np.random.default_rng(seed);rows=[];ref=preds[reference]
 for name,p in preds.items():
  if name==reference:continue
  da=[];du=[];db=[]
  for _ in range(repeats):
   ix=stratified_indices(y,rng);yy=y[ix];da.append(average_precision_score(yy,p[ix])-average_precision_score(yy,ref[ix]));du.append(roc_auc_score(yy,p[ix])-roc_auc_score(yy,ref[ix]));db.append(brier_score_loss(yy,p[ix])-brier_score_loss(yy,ref[ix]))
  rows.append({"model":name,"reference":reference,"delta_definition":"model minus reference","delta_auprc":average_precision_score(y,p)-average_precision_score(y,ref),"delta_auprc_ci_low":np.quantile(da,.025),"delta_auprc_ci_high":np.quantile(da,.975),"delta_auroc":roc_auc_score(y,p)-roc_auc_score(y,ref),"delta_auroc_ci_low":np.quantile(du,.025),"delta_auroc_ci_high":np.quantile(du,.975),"delta_brier":brier_score_loss(y,p)-brier_score_loss(y,ref),"delta_brier_ci_low":np.quantile(db,.025),"delta_brier_ci_high":np.quantile(db,.975)})
 return pd.DataFrame(rows)


def run_paired(cfg):
 out=Path(cfg.output_dir);setup(out);x=load_predictions(Path(cfg.core_dir));y=x.target.to_numpy(int);preds={c.removeprefix("probability_"):x[c].to_numpy(float) for c in x if c.startswith("probability_")}
 paired_table(y,preds,"final_hybrid",cfg.bootstrap_repeats,cfg.seed).to_csv(out/"paired_all_models_vs_final_hybrid.csv",index=False)
 paired_table(y,preds,"age_sex",cfg.bootstrap_repeats,cfg.seed+1).to_csv(out/"paired_all_models_vs_age_sex.csv",index=False)
 ep=Path(cfg.stability_dir)/"three_seed_ensemble_predictions.csv"
 if ep.exists():
  e=pd.read_csv(ep,dtype={"eid":"string"});e.eid=e.eid.str.strip()
  if e.eid.tolist()!=x.eid.tolist() or not np.array_equal(e.target,x.target):raise ValueError("Stability ensemble/core EID mismatch")
  pe={c.removeprefix("probability_ensemble_"):e[c].to_numpy(float) for c in e if c.startswith("probability_ensemble_")}
  paired_table(y,pe,"final_hybrid",cfg.bootstrap_repeats,cfg.seed+2).to_csv(out/"paired_three_seed_ensemble_vs_final_hybrid.csv",index=False)
 atomic_json(out/"COMPLETED.json",{"completed":True,"method":"stratified paired participant bootstrap; not 1:1 case-control matching","n":len(y),"events":int(y.sum()),"repeats":cfg.bootstrap_repeats})
 logging.info("COMPLETED paired comparisons N=%d events=%d",len(y),int(y.sum()))


def subgroup_metric(group,level,model,y,p,repeats,seed):
 row={"group":group,"level":str(level),**simple_metric(model,y,p)}
 if y.sum()<5 or (y==0).sum()<5:return row
 return bootstrap_metrics([row],y,{model:p},repeats,seed)[0]


def run_subgroups(cfg):
 out=Path(cfg.output_dir);setup(out);pred=load_predictions(Path(cfg.core_dir));ep=Path(cfg.stability_dir)/"three_seed_ensemble_predictions.csv"
 if ep.exists():
  e=pd.read_csv(ep,dtype={"eid":"string"});e.eid=e.eid.str.strip();pred=pred.merge(e[["eid","probability_ensemble_final_hybrid"]],on="eid",how="left",validate="one_to_one")
 d=load_cohort(cfg);q=pred.merge(d.drop(columns=["target"],errors="ignore"),on="eid",how="left",validate="one_to_one");models_use=["final_hybrid","age_sex","dual_pa_feature","grufcn_pa","convcontext_pa"]
 if "probability_ensemble_final_hybrid" in q:models_use.insert(0,"ensemble_final_hybrid")
 age=pd.to_numeric(q.enrollment_age,errors="coerce");q["age_group"]=pd.cut(age,[-np.inf,59,64,69,np.inf],labels=["<60","60-64","65-69","70+"])
 q["sex_group"]=q.gender.astype("string").fillna("Missing");e4=pd.to_numeric(q.apoe_e4,errors="coerce");q["apoe_group"]=e4.map({0:"e4=0",1:"e4=1",2:"e4=2"}).fillna("APOE missing")
 rows=[];k=0
 for group in ["sex_group","age_group","apoe_group"]:
  for level,z in q.groupby(group,observed=True,dropna=False):
   for name in models_use:
    rows.append(subgroup_metric(group,level,name,z.target.to_numpy(int),z[f"probability_{name}"].to_numpy(float),cfg.subgroup_bootstrap_repeats,cfg.seed+k));k+=1
 pd.DataFrame(rows).to_csv(out/"subgroup_performance.csv",index=False)
 # Sex-by-age is supplied only when at least ten events permit interpretable estimates.
 q["sex_by_age"]=q.sex_group.astype(str)+" | "+q.age_group.astype(str);rows=[]
 for level,z in q.groupby("sex_by_age"):
  if z.target.sum()>=10:
   for name in ["final_hybrid","age_sex"]:rows.append(subgroup_metric("sex_by_age",level,name,z.target.to_numpy(int),z[f"probability_{name}"].to_numpy(float),cfg.subgroup_bootstrap_repeats,cfg.seed+k));k+=1
 pd.DataFrame(rows).to_csv(out/"intersectional_subgroups_min10events.csv",index=False)
 atomic_json(out/"COMPLETED.json",{"completed":True,"subgroups":["sex","age","APOE dosage","sex-by-age when >=10 events"],"warning":"Subgroup comparisons are exploratory; sparse-event intervals must be reported."})
 logging.info("COMPLETED subgroup analysis")


def run_decision(cfg):
 out=Path(cfg.output_dir);setup(out);x=load_predictions(Path(cfg.core_dir));ep=Path(cfg.stability_dir)/"three_seed_ensemble_predictions.csv"
 if ep.exists():
  e=pd.read_csv(ep,dtype={"eid":"string"});e.eid=e.eid.str.strip();x=x.merge(e[["eid","probability_ensemble_final_hybrid"]],on="eid",how="left",validate="one_to_one")
 y=x.target.to_numpy(int);n=len(y);rows=[];names=["age_sex","grufcn_pa","convcontext_pa","dual_pa_feature","final_hybrid"]
 if "probability_ensemble_final_hybrid" in x:names.append("ensemble_final_hybrid")
 for pt in np.r_[np.linspace(.001,.01,37),np.linspace(.011,.05,40)]:
  rows.append({"model":"treat_none","threshold":pt,"net_benefit":0.0});rows.append({"model":"treat_all","threshold":pt,"net_benefit":y.mean()-(1-y.mean())*pt/(1-pt)})
  for name in names:
   z=x[f"probability_{name}"].to_numpy(float)>=pt;tp=np.sum(z&(y==1));fp=np.sum(z&(y==0));rows.append({"model":name,"threshold":pt,"net_benefit":tp/n-fp/n*pt/(1-pt),"selected":int(z.sum()),"tp":int(tp),"fp":int(fp)})
 pd.DataFrame(rows).to_csv(out/"decision_curve_data.csv",index=False)
 atomic_json(out/"COMPLETED.json",{"completed":True,"threshold_range":[.001,.05],"probability_source":"development-calibrated fixed UKB test probabilities"})
 logging.info("COMPLETED decision-curve data")


class NoResidual(nn.Module):
 def __init__(self,tab_dim,d=64,drop=.2):super().__init__();self.g=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU());self.c=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU());self.gate=nn.Sequential(nn.Linear(2*d,d),nn.Sigmoid());self.t=fusion.TabTower(tab_dim,d,drop);self.head=nn.Sequential(nn.Linear(4*d,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,1))
 def forward(self,g,c,gl,cl,t,e):g=self.g(g);c=self.c(c);w=self.gate(torch.cat([g,c],1));z=w*g+(1-w)*c;q=self.t(t);return self.head(torch.cat([z,q,z*q,torch.abs(z-q)],1)).squeeze(-1)


class DecisionOnly(nn.Module):
 def __init__(self,tab_dim,d=64,drop=.2):super().__init__();self.t=fusion.TabTower(tab_dim,d,drop);self.clin=nn.Linear(d,1);self.weight=nn.Parameter(torch.zeros(3));self.scale=nn.Parameter(torch.ones(3));self.bias=nn.Parameter(torch.zeros(()))
 def forward(self,g,c,gl,cl,t,e):cw=self.clin(self.t(t)).squeeze(-1);w=torch.softmax(self.weight,0);return w[0]*self.scale[0]*gl+w[1]*self.scale[1]*cl+w[2]*self.scale[2]*cw+self.bias


class SinglePAClinical(nn.Module):
 def __init__(self,which,tab_dim,d=64,drop=.2):super().__init__();self.which=which;self.p=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU());self.t=fusion.TabTower(tab_dim,d,drop);self.head=nn.Sequential(nn.Linear(4*d,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,1));self.rw=nn.Parameter(torch.zeros(2));self.rs=nn.Parameter(torch.tensor(-1.));self.clin=nn.Linear(d,1)
 def forward(self,g,c,gl,cl,t,e):p=self.p(g if self.which=="gru" else c);q=self.t(t);inter=self.head(torch.cat([p,q,p*q,torch.abs(p-q)],1)).squeeze(-1);cw=self.clin(q).squeeze(-1);base=gl if self.which=="gru" else cl;w=torch.softmax(self.rw,0);return inter+torch.sigmoid(self.rs)*(w[0]*base+w[1]*cw)


def experiment_cfg(cfg,out):
 return SimpleNamespace(labels_csv=cfg.labels_csv,apoe_csv=cfg.apoe_csv,cache_dir=cfg.cache_dir,output_dir=str(out),horizon_years=7.,mode="full",d_model=64,dropout=.30,extract_batch_size=128,eval_batch_size=256,num_workers=cfg.num_workers,amp=cfg.amp,fusion_batch_size=512,fusion_epochs=60,fusion_min_epochs=10,fusion_patience=12,fusion_lr=3e-4,fusion_weight_decay=1e-3,sampled_positive_fraction=.08,seed=cfg.seed,target_sensitivity=.80)


def run_ablation(cfg):
 out=Path(cfg.output_dir);setup(out);ec=experiment_cfg(cfg,out);d=load_cohort(ec);y=d.target.to_numpy(np.int8);dev,test,itr,iva=split_indices(d,Path(cfg.protocol_dir));sel=Path(cfg.core_dir)/"epoch_selection"
 ztr=core.extract_pair(ec,d,y,itr,sel/"grufcn_best.pt",sel/"convcontext_mamba_best.pt");zva=core.extract_pair(ec,d,y,iva,sel/"grufcn_best.pt",sel/"convcontext_mamba_best.pt")
 pp=v6.tab_preprocessor(core.AGE_SEX);pp.fit(d.iloc[itr][core.AGE_SEX]);tab=np.asarray(pp.transform(d[core.AGE_SEX]),np.float32);trds=core.fusion_dataset(ztr,tab[itr]);vads=core.fusion_dataset(zva,tab[iva]);specs=[("no_residual_interaction_only",lambda:NoResidual(tab.shape[1])),("decision_only_no_feature_interaction",lambda:DecisionOnly(tab.shape[1])),("gru_plus_age_sex",lambda:SinglePAClinical("gru",tab.shape[1])),("mamba_plus_age_sex",lambda:SinglePAClinical("mamba",tab.shape[1]))]
 locked={};hist=[];cal={};threshold={};selection=out/"selection";selection.mkdir(exist_ok=True)
 for j,(name,factory) in enumerate(specs):
  score,ep,h,_,_=fusion.train_head(name,factory(),trds,vads,ec,selection,cfg.seed+50+j);locked[name]=ep;hist.extend(h);cal[name],threshold[name]=core.fit_platt_threshold(y[iva],score,.80)
 fg=Path(cfg.core_dir)/"grufcn_80pct_refit.pt";fc=Path(cfg.core_dir)/"convcontext_mamba_80pct_refit.pt";zd=core.extract_pair(ec,d,y,dev,fg,fc);zt=core.extract_pair(ec,d,y,test,fg,fc);pp2=joblib.load(Path(cfg.core_dir)/"age_sex_preprocessor_80pct.joblib");tab2=np.asarray(pp2.transform(d[core.AGE_SEX]),np.float32);devds=core.fusion_dataset(zd,tab2[dev]);testds=core.fusion_dataset(zt,tab2[test]);pred={};device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 for j,(name,factory) in enumerate(specs):
  model,_=core.refit_head(name,factory(),devds,locked[name],ec,out,cfg.seed+150+j);ty,score=fusion.infer(model,testds,device,ec.eval_batch_size);pred[name]=cal[name].predict_proba(score.reshape(-1,1))[:,1]
 existing=load_predictions(Path(cfg.core_dir));
 if existing.eid.tolist()!=d.eid.iloc[test].astype(str).tolist():raise ValueError("Existing core predictions do not match fixed test EID order")
 for name in ["age_sex","grufcn_pa","convcontext_pa","dual_pa_feature","final_hybrid"]:pred[name]=existing[f"probability_{name}"].to_numpy(float)
 rows=[simple_metric(k,y[test],p) for k,p in pred.items()];rows=bootstrap_metrics(rows,y[test],pred,cfg.bootstrap_repeats,cfg.seed);pd.DataFrame(rows).sort_values("auprc",ascending=False).to_csv(out/"ablation_test_metrics.csv",index=False);pd.DataFrame({"eid":d.eid.iloc[test].to_numpy(),"target":y[test],**{f"probability_{k}":p for k,p in pred.items()}}).to_csv(out/"ablation_test_predictions.csv",index=False);pd.DataFrame(hist).to_csv(out/"ablation_selection_history.csv",index=False);atomic_json(out/"COMPLETED.json",{"completed":True,"locked_epochs":locked,"principle":"epochs selected in development; complete 80% refit; identical fixed 20% evaluation"})
 logging.info("COMPLETED ablation analysis")


def run_stability(cfg):
 out=Path(cfg.output_dir);setup(out);protocol=Path(cfg.protocol_dir);source=Path(cfg.core_dir);runs=[]
 for seed in SEEDS:
  target=source if seed==SEEDS[0] else out/f"seed_{seed}"
  if seed!=SEEDS[0] and not (target/"COMPLETED.json").exists():
   target.mkdir(parents=True,exist_ok=True);shutil.copy2(protocol/"ukb_fixed_80_20_split.csv",target/"ukb_fixed_80_20_split.csv")
   cmd=[sys.executable,str(Path(__file__).with_name("v13_core_80_20.py")),"--mode","full","--labels-csv",cfg.labels_csv,"--apoe-csv",cfg.apoe_csv,"--cache-dir",cfg.cache_dir,"--output-dir",str(target),"--seed",str(seed),"--split-seed","20260820","--bootstrap-repeats","500"]
   logging.info("Starting prespecified stability seed %d",seed);subprocess.run(cmd,cwd=Path(__file__).parent,check=True)
  x=load_predictions(target);x["seed"]=seed;runs.append(x)
 base=runs[0][["eid","target"]].copy();models_available=[c.removeprefix("probability_") for c in runs[0] if c.startswith("probability_")];per=[]
 for x in runs:
  if x.eid.tolist()!=base.eid.tolist() or not np.array_equal(x.target,base.target):raise ValueError("Seed prediction EIDs/targets differ")
  for name in models_available:per.append({"seed":int(x.seed.iloc[0]),**simple_metric(name,x.target.to_numpy(int),x[f"probability_{name}"].to_numpy(float))})
 per=pd.DataFrame(per);per.to_csv(out/"per_seed_test_metrics.csv",index=False);summary=per.groupby("model")[["auprc","auroc","brier"]].agg(["mean","std","min","max"]);summary.columns=["_".join(c) for c in summary.columns];summary.reset_index().to_csv(out/"three_seed_mean_sd.csv",index=False)
 ensemble={name:np.mean([x[f"probability_{name}"].to_numpy(float) for x in runs],axis=0) for name in models_available};y=base.target.to_numpy(int);rows=[simple_metric("ensemble_"+k,y,p) for k,p in ensemble.items()];rows=bootstrap_metrics(rows,y,{"ensemble_"+k:p for k,p in ensemble.items()},cfg.bootstrap_repeats,cfg.seed);pd.DataFrame(rows).sort_values("auprc",ascending=False).to_csv(out/"three_seed_ensemble_metrics.csv",index=False);pd.DataFrame({"eid":base.eid,"target":y,**{f"probability_ensemble_{k}":p for k,p in ensemble.items()}}).to_csv(out/"three_seed_ensemble_predictions.csv",index=False);atomic_json(out/"COMPLETED.json",{"completed":True,"seeds":SEEDS,"selection":"No best seed selected; arithmetic probability ensemble is primary stability summary","reused_seed_20260820":True})
 logging.info("COMPLETED three-seed stability and probability ensemble")


def infer_occluded(cfg,d,test,tab,window,device,gm,cm,head,cal,loc,scale):
 act=np.load(Path(cfg.cache_dir)/"activity_7d.npy",mmap_mode="r");obs=np.load(Path(cfg.cache_dir)/"observed_7d.npy",mmap_mode="r");cache_idx=d.cache_index.to_numpy(int)[test];scores=[];bs=cfg.eval_batch_size
 for st in range(0,len(test),bs):
  en=min(st+bs,len(test));a=np.asarray(act[cache_idx[st:en]],np.float32).copy();m=np.asarray(obs[cache_idx[st:en]],bool).copy()
  if window is not None:
   day,start,end=window
   if day is None:a[:, :, start:end]=0;m[:, :, start:end]=False
   else:a[:,day,start:end]=0;m[:,day,start:end]=False
  z=(np.log1p(np.clip(np.nan_to_num(a,nan=0.),0,None))-loc)/scale;z=np.where(m,z,0.);x=torch.from_numpy(np.stack([z.reshape(len(a),-1),m.astype(np.float32).reshape(len(a),-1)],1).astype(np.float32)).to(device);tb=torch.from_numpy(tab[test[st:en]]).to(device);e=torch.zeros((len(x),1),device=device)
  with torch.no_grad(),v6.amp_ctx(device,cfg.amp):g=gm.encoder(x);c=cm.encoder(x);gl=gm.risk(g).squeeze(-1);cl=cm.risk(c).squeeze(-1);s=head(g,c,gl,cl,tb,e)
  scores.append(s.float().cpu().numpy())
 raw=np.concatenate(scores);return raw,cal.predict_proba(raw.reshape(-1,1))[:,1]


def run_interpretability(cfg):
 out=Path(cfg.output_dir);setup(out);ec=experiment_cfg(cfg,out);d=load_cohort(ec);y=d.target.to_numpy(int);_,test,_,_=split_indices(d,Path(cfg.protocol_dir));device=torch.device("cuda" if torch.cuda.is_available() else "cpu");gp=torch.load(Path(cfg.core_dir)/"grufcn_80pct_refit.pt",map_location=device);cp=torch.load(Path(cfg.core_dir)/"convcontext_mamba_80pct_refit.pt",map_location=device);gm=models.build("grufcn",ec).to(device);cm=models.build("convcontext_mamba",ec).to(device);gm.load_state_dict(gp["model"]);cm.load_state_dict(cp["model"]);gm.eval();cm.eval();pp=joblib.load(Path(cfg.core_dir)/"age_sex_preprocessor_80pct.joblib");tab=np.asarray(pp.transform(d[core.AGE_SEX]),np.float32);head=fusion.HybridFusion(tab.shape[1]).to(device);head.load_state_dict(torch.load(Path(cfg.core_dir)/"final_hybrid_80pct_refit.pt",map_location=device)["model"]);head.eval();cal=joblib.load(Path(cfg.core_dir)/"development_only_calibrators.joblib")["final_hybrid"]
 base_logit,base_prob=infer_occluded(ec,d,test,tab,None,device,gm,cm,head,cal,float(gp["loc"]),float(gp["scale"]));windows=[]
 for day in range(7):windows.append((f"day_{day+1}",(day,0,1440)))
 for label,s,e in [("clock_00_06",0,360),("clock_06_12",360,720),("clock_12_18",720,1080),("clock_18_24",1080,1440)]:windows.append((label,(None,s,e)))
 for day in range(7):
  for label,s,e in [("00_06",0,360),("06_12",360,720),("12_18",720,1080),("18_24",1080,1440)]:windows.append((f"day_{day+1}_{label}",(day,s,e)))
 wide=pd.DataFrame({"eid":d.eid.iloc[test].to_numpy(),"target":y[test],"baseline_probability":base_prob});rows=[]
 for j,(name,window) in enumerate(windows):
  logit,p=infer_occluded(ec,d,test,tab,window,device,gm,cm,head,cal,float(gp["loc"]),float(gp["scale"]));dp=base_prob-p;dl=base_logit-logit;wide["delta_probability_"+name]=dp
  for group,mask in [("all",np.ones(len(test),bool)),("incident",y[test]==1),("control",y[test]==0)]:rows.append({"window":name,"group":group,"n":int(mask.sum()),"mean_delta_probability":float(dp[mask].mean()),"median_delta_probability":float(np.median(dp[mask])),"mean_delta_logit":float(dl[mask].mean()),"positive_delta_fraction":float(np.mean(dp[mask]>0))})
  logging.info("Occlusion %d/%d %s",j+1,len(windows),name)
 wide.to_csv(out/"temporal_occlusion_individual.csv",index=False);pd.DataFrame(rows).to_csv(out/"temporal_occlusion_summary.csv",index=False);saved=load_predictions(Path(cfg.core_dir));maxdiff=float(np.max(np.abs(base_prob-saved.probability_final_hybrid.to_numpy(float))));atomic_json(out/"COMPLETED.json",{"completed":True,"windows":len(windows),"baseline_probability_max_abs_difference_vs_saved":maxdiff,"interpretation":"Positive delta means the removed interval supported higher predicted risk; associational, not causal."})
 logging.info("COMPLETED temporal occlusion windows=%d baseline_max_abs_diff=%.3g",len(windows),maxdiff)


def smoke(cfg):
 out=Path(cfg.output_dir);setup(out);required=[Path(cfg.core_dir)/"COMPLETED.json",Path(cfg.protocol_dir)/"V13_PROTOCOL_LOCK.json",Path(cfg.core_dir)/"ukb_20pct_final_test_predictions.csv"]
 missing=[str(p) for p in required if not p.exists()]
 if missing:raise FileNotFoundError("Missing prerequisites:\n"+"\n".join(missing))
 atomic_json(out/"SMOKE_COMPLETED.json",{"completed":True,"stage":cfg.stage,"prerequisites":required})


if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("stage",choices=["supplement_smoke","stability","paired","subgroups","ablation","interpretability","decision"]);a.add_argument("--labels-csv",default=str(ROOT/"ukb_dementia_labels_covariates_N80990.csv"));a.add_argument("--apoe-csv",default=str(ROOT/"APOE_phenotype.csv"));a.add_argument("--cache-dir",default=str(ROOT/"supervised_results"/"cache"));a.add_argument("--core-dir",default=str(RESULTS/"02_core_model"));a.add_argument("--protocol-dir",default=str(RESULTS/"00_protocol"));a.add_argument("--stability-dir",default=str(RESULTS/"05_stability"));a.add_argument("--output-dir",required=True);a.add_argument("--bootstrap-repeats",type=int,default=2000);a.add_argument("--subgroup-bootstrap-repeats",type=int,default=1000);a.add_argument("--eval-batch-size",type=int,default=256);a.add_argument("--num-workers",type=int,default=6);a.add_argument("--seed",type=int,default=20260820);a.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True);cfg=a.parse_args()
 if cfg.stage=="supplement_smoke":smoke(cfg)
 elif cfg.stage=="stability":run_stability(cfg)
 elif cfg.stage=="paired":run_paired(cfg)
 elif cfg.stage=="subgroups":run_subgroups(cfg)
 elif cfg.stage=="ablation":run_ablation(cfg)
 elif cfg.stage=="interpretability":run_interpretability(cfg)
 elif cfg.stage=="decision":run_decision(cfg)
