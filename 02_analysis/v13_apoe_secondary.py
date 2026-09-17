#!/usr/bin/env python3
"""APOE-complete secondary analysis on the fixed UKB 80:20 EID split."""
from __future__ import annotations
import argparse,json,logging,os,sys
from datetime import datetime
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import v13_cohort_engine as v6
import v13_fusion_engine as v12
import v13_core_80_20 as core

BASE=Path("/root/autodl-tmp/PA_CR_CNN/Project_new/PA_Foundation_Model");DEM=BASE/"dementia";DEFAULT_MASTER=DEM/"Results"/"v13_final"/"02_core_model";DEFAULT_OUT=DEM/"Results"/"v13_final"/"03_apoe_secondary"
AGE=v12.AGE_SEX

def setup(out):
 out.mkdir(parents=True,exist_ok=True);logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",handlers=[logging.FileHandler(out/"run.log",encoding="utf-8"),logging.StreamHandler(sys.stdout)],force=True)
def fit_tab(cols,d,y,tr,pred):
 p=Pipeline([("prep",v6.tab_preprocessor(cols)),("model",LogisticRegression(C=.25,max_iter=4000,solver="lbfgs"))]);p.fit(d.iloc[tr][cols],y[tr]);return p,p.predict_proba(d.iloc[pred][cols])[:,1]
def dosage(d,fit,ids):
 x=pd.to_numeric(d.apoe_e4,errors="coerce").to_numpy(float);med=float(np.nanmedian(x[fit]));scale=max(float(np.nanstd(x[fit])),.5);return ((x[ids]-med)/scale).astype(np.float32)[:,None],{"median":med,"scale":scale}

def run(cfg):
 out=Path(cfg.output_dir);setup(out);master=Path(cfg.master_dir);d=v6.load_cohort(v6.Config(cfg.labels_csv,cfg.apoe_csv,cfg.cache_dir,cfg.output_dir,horizon_years=cfg.horizon_years));y=d.target.to_numpy(np.int8);outer=pd.read_csv(master/"ukb_fixed_80_20_split.csv",dtype={"eid":"string"});inner=pd.read_csv(master/"development_internal_epoch_selection.csv",dtype={"eid":"string"});outer.eid=outer.eid.str.strip();inner.eid=inner.eid.str.strip()
 if outer.eid.tolist()!=d.eid.tolist():raise ValueError("Fixed split/cohort mismatch")
 role=dict(zip(inner.eid,inner.role));dev=np.where(outer.split.eq("development"))[0];test=np.where(outer.split.eq("test"))[0];ok=d.apoe_e4.notna().to_numpy();dev=dev[ok[dev]];test=test[ok[test]];itr=np.array([i for i in dev if role.get(d.eid.iloc[i])=="inner_training"]);iva=np.array([i for i in dev if role.get(d.eid.iloc[i])=="epoch_selection"])
 logging.info("APOE complete development N=%d events=%d test N=%d events=%d",len(dev),int(y[dev].sum()),len(test),int(y[test].sum()))
 if cfg.mode=="smoke":
  if (len(dev),len(test),int(y[dev].sum()),int(y[test].sum()))!=(41727,10501,304,84):raise ValueError("Unexpected APOE complete subset")
  (out/"SMOKE_COMPLETED.json").write_text(json.dumps({"completed":True,"development_n":len(dev),"test_n":len(test)}),encoding="utf-8");return
 # Selection representations from the original development-only backbone checkpoints.
 sg=master/"epoch_selection"/"grufcn_best.pt";sc=master/"epoch_selection"/"convcontext_mamba_best.pt";ztr=core.extract_pair(cfg,d,y,itr,sg,sc);zva=core.extract_pair(cfg,d,y,iva,sg,sc);pp=v6.tab_preprocessor(AGE);pp.fit(d.iloc[itr][AGE]);tab=np.asarray(pp.transform(d[AGE]),np.float32);etr,apoe_sel=dosage(d,itr,itr);eva,_=dosage(d,itr,iva);trds=v12.tensor_dataset(ztr,tab[itr],etr);vads=v12.tensor_dataset(zva,tab[iva],eva)
 score,epoch,h,_,_=v12.train_head("final_hybrid_apoe",v12.APOEHybrid(tab.shape[1]),trds,vads,cfg,out,cfg.seed+200);cal={};threshold={};cal["final_hybrid_apoe"],threshold["final_hybrid_apoe"]=core.fit_platt_threshold(y[iva],score,cfg.target_sensitivity)
 # Tabular comparator selection/calibration on exactly the same APOE-complete development subset.
 for name,cols in [("age_sex_completecase",AGE),("apoe_only",["apoe_e4"]),("age_sex_apoe",AGE+["apoe_e4"])]:
  _,p=fit_tab(cols,d,y,itr,iva);cal[name],threshold[name]=core.fit_platt_threshold(y[iva],v6.logit(p),cfg.target_sensitivity)
 core.atomic_json(out/"locked_apoe_design.json",{"hybrid_epoch":epoch,"thresholds":threshold,"selection_source":"APOE-complete subset of the fixed 80% development partition","test_role":"secondary complete-case analysis only"});joblib.dump(cal,out/"development_only_calibrators.joblib");pd.DataFrame(h).to_csv(out/"apoe_hybrid_development_history.csv",index=False)
 # Full 80% complete-case refit using the already frozen full-development PA backbones.
 fg=master/"grufcn_80pct_refit.pt";fc=master/"convcontext_mamba_80pct_refit.pt";zd=core.extract_pair(cfg,d,y,dev,fg,fc);zt=core.extract_pair(cfg,d,y,test,fg,fc);pp2=v6.tab_preprocessor(AGE);pp2.fit(d.iloc[dev][AGE]);tab2=np.asarray(pp2.transform(d[AGE]),np.float32);ed,apoe_full=dosage(d,dev,dev);et,_=dosage(d,dev,test);devds=v12.tensor_dataset(zd,tab2[dev],ed);testds=v12.tensor_dataset(zt,tab2[test],et);model,path=core.refit_head("final_hybrid_apoe",v12.APOEHybrid(tab2.shape[1]),devds,epoch,cfg,out,cfg.seed+300);joblib.dump(pp2,out/"age_sex_preprocessor_apoe_complete.joblib");core.atomic_json(out/"apoe_dosage_preprocessor.json",apoe_full)
 pipes={};raw={}
 for name,cols in [("age_sex_completecase",AGE),("apoe_only",["apoe_e4"]),("age_sex_apoe",AGE+["apoe_e4"])]:pipes[name],raw[name]=fit_tab(cols,d,y,dev,test);joblib.dump(pipes[name],out/f"{name}.joblib")
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu");ty,s=v12.infer(model,testds,device,cfg.eval_batch_size);raw["final_hybrid_apoe"]=sig=s
 pred={name:cal[name].predict_proba((score if name=="final_hybrid_apoe" else v6.logit(score)).reshape(-1,1))[:,1] for name,score in raw.items()}
 # Restrict the already frozen full-cohort predictions for apples-to-apples subset comparison.
 prior=pd.read_csv(master/"ukb_20pct_final_test_predictions.csv",dtype={"eid":"string"}).set_index("eid");ids=d.eid.iloc[test].astype(str);pred["age_sex_fullcohort_restricted"]=prior.loc[ids,"probability_age_sex"].to_numpy();pred["final_hybrid_fullcohort_restricted"]=prior.loc[ids,"probability_final_hybrid"].to_numpy();master_thresholds=json.loads((master/"locked_epochs_and_thresholds.json").read_text(encoding="utf-8"))["thresholds"];threshold["age_sex_fullcohort_restricted"]=master_thresholds["age_sex"];threshold["final_hybrid_fullcohort_restricted"]=master_thresholds["final_hybrid"]
 rows=[]
 for name,p in pred.items():rows.append(core.metrics(name,ty.astype(int),p,threshold[name]))
 rows=core.bootstrap(rows,ty.astype(int),pred,cfg.bootstrap_repeats,cfg.seed);md=pd.DataFrame(rows).sort_values("auprc",ascending=False);md.to_csv(out/"apoe_complete_test_metrics.csv",index=False);pd.DataFrame({"eid":ids.to_numpy(),"target":ty.astype(int),**{f"probability_{k}":p for k,p in pred.items()}}).to_csv(out/"apoe_complete_test_predictions.csv",index=False);core.top_risk(ty.astype(int),pred).to_csv(out/"apoe_complete_top_risk.csv",index=False);core.atomic_json(out/"COMPLETED.json",{"completed":True,"time":datetime.now().isoformat(),"test_n":len(test),"test_events":int(ty.sum()),"role":"secondary APOE-complete analysis; not comparable as a replacement for the full-cohort primary"});logging.info("Completed APOE secondary analysis")

if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--mode",choices=["smoke","full"],default="full");a.add_argument("--labels-csv",default=str(v6.DEFAULT_LABELS));a.add_argument("--apoe-csv",default=str(v6.DEFAULT_APOE));a.add_argument("--cache-dir",default=str(v6.DEFAULT_CACHE));a.add_argument("--master-dir",default=str(DEFAULT_MASTER));a.add_argument("--output-dir",default=str(DEFAULT_OUT));a.add_argument("--horizon-years",type=float,default=7.);a.add_argument("--d-model",type=int,default=64);a.add_argument("--dropout",type=float,default=.30);a.add_argument("--extract-batch-size",type=int,default=128);a.add_argument("--eval-batch-size",type=int,default=256);a.add_argument("--num-workers",type=int,default=6);a.add_argument("--fusion-batch-size",type=int,default=512);a.add_argument("--fusion-epochs",type=int,default=60);a.add_argument("--fusion-min-epochs",type=int,default=10);a.add_argument("--fusion-patience",type=int,default=12);a.add_argument("--fusion-lr",type=float,default=3e-4);a.add_argument("--fusion-weight-decay",type=float,default=1e-3);a.add_argument("--sampled-positive-fraction",type=float,default=.08);a.add_argument("--target-sensitivity",type=float,default=.80);a.add_argument("--bootstrap-repeats",type=int,default=2000);a.add_argument("--seed",type=int,default=20260820);a.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True);run(a.parse_args())

