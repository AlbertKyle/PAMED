#!/usr/bin/env python3
"""Representative raw-PA benchmark on the already fixed UKB 80:20 split.

This is a supplementary benchmark, not a mechanism for replacing the
prespecified final Hybrid. Each architecture selects its epoch using only the
existing development-internal validation subset, is reinitialized/refit on all
80% development participants, and is then scored on the identical 20% test.
"""
from __future__ import annotations
import argparse,json,logging,os,sys
from datetime import datetime
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA_ROOT, RESULTS_ROOT
import joblib,numpy as np,pandas as pd,torch,torch.nn as nn
from sklearn.metrics import average_precision_score,roc_auc_score

import cohort
import temporal_models as arena
import pamed_models
import train_evaluate as core

BASE=DATA_ROOT.parent;DEM=DATA_ROOT
DEFAULT_MASTER=RESULTS_ROOT/"00_protocol";DEFAULT_OUT=RESULTS_ROOT/"01_algorithm_pool"
DEFAULT_MODELS=["fcn","resnet1d","tcn","xceptiontime","inception","patchtst","sameclock","mamba_fcn","bimambasl_fcn","chronoview_mamba_fcn","gru_mamba_fcn"]

def setup(out):
 out.mkdir(parents=True,exist_ok=True);logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",handlers=[logging.FileHandler(out/"run.log",encoding="utf-8"),logging.StreamHandler(sys.stdout)],force=True)
def build(name,cfg):return arena.build(name,cfg) if name in arena.ARCHS else pamed_models.build(name,cfg)

def selection(name,cfg,d,y,tr,va,out,seed):
 core.seed_all(seed);idx=d.cache_index.to_numpy(int)
 if name in arena.ARCHS:
  acfg=arena.Config(
   labels_csv=str(cfg.labels_csv),apoe_csv=str(cfg.apoe_csv),cache_dir=str(cfg.cache_dir),
   output_dir=str(out),architectures=name,horizon_years=cfg.horizon_years,repeats=1,
   validation_fraction=.15,ssl_epochs=cfg.ssl_epochs,epochs=cfg.epochs,
   min_epochs=cfg.min_epochs,patience=cfg.patience,batch_size=cfg.batch_size,
   eval_batch_size=cfg.eval_batch_size,num_workers=cfg.num_workers,lr=cfg.lr,
   ssl_lr=cfg.ssl_lr,weight_decay=cfg.weight_decay,dropout=cfg.dropout,
   d_model=cfg.d_model,sampled_positive_fraction=cfg.sampled_positive_fraction,
   temperature=cfg.temperature,seed=seed,amp=cfg.amp,mode=cfg.mode,overwrite=True)
  r=arena.train_model(name,acfg,d,y,idx,tr,va,out);return r["best_epoch"],r["score"],r["history"]
 r=pamed_models.train_one(name,cfg,d,y,tr,va,out,seed);return r["best_epoch"],r["score"],r["history"]

def refit(name,epochs,cfg,d,y,dev,out,seed):
 core.seed_all(seed);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");idx=d.cache_index.to_numpy(int);loc,scale=cohort.robust_raw_scale(cfg.cache_dir,idx,dev,cfg.seed);ssl=core.make_week_loader(cfg,d,y,dev,loc,scale,cfg.batch_size,shuffle=True,augment=True);tr=core.make_week_loader(cfg,d,y,dev,loc,scale,cfg.batch_size,sampler=core.weighted_sampler(y,dev,cfg.sampled_positive_fraction,seed));model=build(name,cfg).to(device)
 logging.info("REFIT %-24s N=%d cases=%d epochs=%d parameters=%d",name,len(dev),int(y[dev].sum()),epochs,sum(p.numel() for p in model.parameters()))
 if cfg.ssl_epochs:
  opt=torch.optim.AdamW(model.parameters(),lr=cfg.ssl_lr,weight_decay=cfg.weight_decay);gs=cohort.grad_scaler(device,cfg.amp)
  for ep in range(1,cfg.ssl_epochs+1):
   model.train();ls=[]
   for a,b,_,_ in ssl:
    a=a.to(device);b=b.to(device);opt.zero_grad(set_to_none=True)
    with cohort.amp_ctx(device,cfg.amp):loss=cohort.nt_xent(model.project(a),model.project(b),cfg.temperature)
    gs.scale(loss).backward();gs.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);gs.step(opt);gs.update();ls.append(float(loss.detach().cpu()))
   logging.info("%s refit SSL epoch=%d loss=%.4f",name,ep,np.mean(ls))
 opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(min(cfg.cosine_epochs,epochs),1),eta_min=cfg.lr*.1);gs=cohort.grad_scaler(device,cfg.amp);lossfn=nn.BCEWithLogitsLoss()
 for ep in range(1,epochs+1):
  model.train();ls=[]
  for x,yy,_ in tr:
   x=x.to(device);yy=yy.to(device);opt.zero_grad(set_to_none=True)
   with cohort.amp_ctx(device,cfg.amp):loss=lossfn(model(x),yy)
   gs.scale(loss).backward();gs.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);gs.step(opt);gs.update();ls.append(float(loss.detach().cpu()))
  if ep<=cfg.cosine_epochs:sch.step()
  logging.info("%s refit epoch=%d/%d loss=%.4f",name,ep,epochs,np.mean(ls))
 path=out/f"{name}_80pct_refit.pt";torch.save({"model":model.state_dict(),"architecture":name,"fixed_epochs":epochs,"loc":loc,"scale":scale,"seed":seed},path);return path

def test_score(name,path,cfg,d,y,test):
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu");s=torch.load(path,map_location=device);m=build(name,cfg).to(device);m.load_state_dict(s["model"]);m.eval();loader=core.make_week_loader(cfg,d,y,test,float(s["loc"]),float(s["scale"]),cfg.eval_batch_size);yy=[];ss=[]
 with torch.no_grad():
  for x,t,_ in loader:
   with cohort.amp_ctx(device,cfg.amp):q=m(x.to(device))
   yy.append(t.numpy());ss.append(q.float().cpu().numpy())
 return np.concatenate(yy).astype(int),np.concatenate(ss)

def run(cfg):
 out=Path(cfg.output_dir);setup(out);models=[x.strip() for x in cfg.models.split(",") if x.strip()];allowed=set(DEFAULT_MODELS)
 if not models or not set(models)<=allowed:raise ValueError(f"models must be a subset of {sorted(allowed)}")
 d=cohort.load_cohort(cohort.Config(cfg.labels_csv,cfg.apoe_csv,cfg.cache_dir,cfg.output_dir,horizon_years=cfg.horizon_years));y=d.target.to_numpy(np.int8);master=Path(cfg.master_dir);outer=pd.read_csv(master/"ukb_fixed_80_20_split.csv",dtype={"eid":"string"});inner=pd.read_csv(master/"development_internal_epoch_selection.csv",dtype={"eid":"string"});outer.eid=outer.eid.str.strip();inner.eid=inner.eid.str.strip()
 if outer.eid.tolist()!=d.eid.tolist():raise ValueError("Master 80:20 split/cohort mismatch")
 dev=np.where(outer.split.eq("development"))[0];test=np.where(outer.split.eq("test"))[0];role=dict(zip(inner.eid,inner.role));itr=np.array([i for i in dev if role.get(d.eid.iloc[i])=="inner_training"]);iva=np.array([i for i in dev if role.get(d.eid.iloc[i])=="epoch_selection"])
 if (len(dev),len(test))!=(49885,12472):raise ValueError("Unexpected fixed split size")
 if cfg.mode=="smoke":
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu");x=torch.randn(2,2,10080,device=device)
  for n in models:
   m=build(n,cfg).to(device);q=m(x);assert q.shape==(2,) and torch.isfinite(q).all();del m,q
  (out/"SMOKE_COMPLETED.json").write_text(json.dumps({"completed":True,"models":models}),encoding="utf-8");return
 sel=out/"epoch_selection";sel.mkdir(exist_ok=True);epochs={};cal={};thresholds={};paths={};hist=[]
 for j,name in enumerate(models):
  ep,score,h=selection(name,cfg,d,y,itr,iva,sel,cfg.seed+10+j);epochs[name]=ep;cal[name],thresholds[name]=core.fit_platt_threshold(y[iva],score,cfg.target_sensitivity);hist.extend([{"model":name,**z} for z in h]);logging.info("LOCKED %s epoch=%d",name,ep)
 for j,name in enumerate(models):paths[name]=refit(name,epochs[name],cfg,d,y,dev,out,cfg.seed+110+j)
 joblib.dump(cal,out/"development_only_calibrators.joblib");core.atomic_json(out/"locked_epochs_thresholds.json",{"epochs":epochs,"thresholds":thresholds,"models":models,"test_role":"supplementary fixed-split benchmark; no primary-model reselection"});pd.DataFrame(hist).to_csv(out/"development_history.csv",index=False)
 core.atomic_json(out/"TEST_INFERENCE_STARTED.json",{"time":datetime.now().isoformat(),"models_locked":models});pred=pd.DataFrame({"eid":d.eid.iloc[test].to_numpy(),"target":y[test]});rows=[]
 for j,name in enumerate(models):
  ty,score=test_score(name,paths[name],cfg,d,y,test);p=cal[name].predict_proba(score.reshape(-1,1))[:,1];pred[f"probability_{name}"]=p;row=core.metrics(name,ty,p,thresholds[name]);rows.append(row);logging.info("TEST %-24s AUPRC=%.4f AUROC=%.4f",name,row["auprc"],row["auroc"])
 rows=core.bootstrap(rows,y[test],{n:pred[f"probability_{n}"].to_numpy() for n in models},cfg.bootstrap_repeats,cfg.seed);pd.DataFrame(rows).sort_values("auprc",ascending=False).to_csv(out/"fixed_20pct_algorithm_metrics.csv",index=False);pred.to_csv(out/"fixed_20pct_algorithm_predictions.csv",index=False)
 # The protocol directory intentionally contains only split files.  Pool
 # results are kept separate from the later core-model results; combining
 # them is a reporting step, never a reason to select a model on test AUPRC.
 core.atomic_json(out/"COMPLETED.json",{"completed":True,"time":datetime.now().isoformat(),"models":models,"warning":"Descriptive benchmark only; final_hybrid remains prespecified primary regardless of test ranking"})

if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--mode",choices=["smoke","full"],default="full");a.add_argument("--labels-csv",default=str(cohort.DEFAULT_LABELS));a.add_argument("--apoe-csv",default=str(cohort.DEFAULT_APOE));a.add_argument("--cache-dir",default=str(cohort.DEFAULT_CACHE));a.add_argument("--master-dir",default=str(DEFAULT_MASTER));a.add_argument("--output-dir",default=str(DEFAULT_OUT));a.add_argument("--models",default=",".join(DEFAULT_MODELS));a.add_argument("--horizon-years",type=float,default=7.);a.add_argument("--ssl-epochs",type=int,default=1);a.add_argument("--epochs",type=int,default=24);a.add_argument("--cosine-epochs",type=int,default=12);a.add_argument("--min-epochs",type=int,default=8);a.add_argument("--patience",type=int,default=6);a.add_argument("--early-stopping",action=argparse.BooleanOptionalAction,default=True);a.add_argument("--batch-size",type=int,default=64);a.add_argument("--eval-batch-size",type=int,default=256);a.add_argument("--num-workers",type=int,default=6);a.add_argument("--lr",type=float,default=1e-4);a.add_argument("--ssl-lr",type=float,default=2e-4);a.add_argument("--weight-decay",type=float,default=1e-3);a.add_argument("--dropout",type=float,default=.30);a.add_argument("--d-model",type=int,default=64);a.add_argument("--sampled-positive-fraction",type=float,default=.08);a.add_argument("--temperature",type=float,default=.15);a.add_argument("--target-sensitivity",type=float,default=.80);a.add_argument("--bootstrap-repeats",type=int,default=2000);a.add_argument("--seed",type=int,default=20260820);a.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True);run(a.parse_args())

