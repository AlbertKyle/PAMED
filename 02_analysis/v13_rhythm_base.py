#!/usr/bin/env python3
"""Dementia v8: GRU-FCN baseline and physiology-guided 7-day PA model.

Primary comparison on one fixed train/validation/test split:
  1. traditional_pa_only                         (tabular comparator)
  2. clinical_apoe                               (baseline information)
  3. grufcn_raw                                  (arena winner, raw PA only)
  4. rhythm_grufcn_raw                           (proposed raw-PA model)
  5. rhythm_grufcn_plus_clinical_apoe             (validation-fit late fusion)

The proposed encoder never receives handcrafted PA/rhythm summaries. It learns
from activity and observation-mask channels and adds three structural priors:
same-clock cross-day aggregation, day/night phase tokens, and interdaily change.
All preprocessing is fitted on training data, epochs are selected on validation
AUPRC, Platt calibration and late-fusion weights are fitted on validation data,
and test labels are used only for final metrics. Followingyears_birthday and all
outcome/follow-up/comorbidity fields are prohibited predictors.
"""
from __future__ import annotations

import argparse, json, logging, os, sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Some AutoDL images expose empty/non-integer OpenMP settings. Sanitize them
# before importing NumPy/PyTorch so libgomp always receives a positive integer.
for _thread_var in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
 _thread_value=os.environ.get(_thread_var,"").strip()
 if not _thread_value.isdigit() or int(_thread_value)<1:os.environ[_thread_var]="4"

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
 confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler

import v13_cohort_engine as v6
import v13_classic_engine as arena

BASE=Path("/root/autodl-tmp/PA_CR_CNN/Project_new/PA_Foundation_Model")
DEM=BASE/"dementia"
DEFAULT_OUT=DEM/"supervised_v8_grufcn_rhythm"

# Baseline information measured independently of the outcome.
CLINICAL=v6.CLINICAL
APOE=v6.APOE
TRAD_PA=v6.TRAD_PA
FUSION_TAB=CLINICAL+APOE
PROHIBITED=set(v6.BLOCKED)|{"Followingyears_birthday","comorbidity_index","charlson_index"}

@dataclass
class Config:
 labels_csv:str=str(v6.DEFAULT_LABELS);apoe_csv:str=str(v6.DEFAULT_APOE)
 cache_dir:str=str(v6.DEFAULT_CACHE);split_csv:str=""
 output_dir:str=str(DEFAULT_OUT);horizon_years:float=7.;seed:int=20260818
 validation_fraction:float=.15;test_fraction:float=.15
 ssl_epochs:int=1;epochs:int=12;min_epochs:int=3;patience:int=3
 batch_size:int=64;eval_batch_size:int=128;num_workers:int=6
 lr:float=1e-4;ssl_lr:float=2e-4;weight_decay:float=1e-3
 dropout:float=.30;d_model:int=64;sampled_positive_fraction:float=.08
 temperature:float=.15;bootstrap_repeats:int=1000;target_sensitivity:float=.80
 amp:bool=True;mode:str="full";overwrite:bool=False

def atomic_json(path,obj):
 tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf-8");os.replace(tmp,path)

def setup(out):
 out.mkdir(parents=True,exist_ok=True)
 logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",handlers=[logging.FileHandler(out/"run.log",encoding="utf-8"),logging.StreamHandler(sys.stdout)],force=True)

def load_data(cfg):
 d=v6.load_cohort(v6.Config(cfg.labels_csv,cfg.apoe_csv,cfg.cache_dir,cfg.output_dir,horizon_years=cfg.horizon_years))
 y=d.target.to_numpy(np.int8);out=Path(cfg.output_dir);generated=out/"v8_internal_split.csv"
 source=Path(cfg.split_csv) if cfg.split_csv else generated
 if source.exists() and not cfg.overwrite:
  s=pd.read_csv(source,dtype={"eid":"string"});s["eid"]=s.eid.str.strip();logging.info("Reusing fixed v8 split: %s",source)
  if set(s.eid)!=set(d.eid):raise ValueError("Reference split and v8 cohort differ")
  d=d.merge(s[["eid","target","split"]].rename(columns={"target":"split_target"}),on="eid",how="left",validate="one_to_one")
  if d.split.isna().any() or not np.array_equal(d.target.to_numpy(int),d.split_target.to_numpy(int)):raise ValueError("Split alignment/target check failed")
  d=d.drop(columns="split_target")
 else:
  # Use the complete UKB analytical cohort. Joint stratification prevents a
  # chance imbalance in outcome, age and sex in this rare-event setting.
  age=pd.to_numeric(d.enrollment_age,errors="coerce");agebin=pd.cut(age,[-np.inf,60,70,np.inf],labels=["lt60","60_69","ge70"]).astype(str)
  key=d.target.astype(str)+"|"+agebin+"|"+d.gender.astype(str)
  rare=set(key.value_counts()[lambda z:z<3].index);key=key.where(~key.isin(rare),d.target.astype(str)+"|"+agebin)
  first=StratifiedShuffleSplit(1,test_size=cfg.test_fraction,random_state=cfg.seed);remain,test=next(first.split(np.zeros(len(d)),key))
  rel=cfg.validation_fraction/(1-cfg.test_fraction);second=StratifiedShuffleSplit(1,test_size=rel,random_state=cfg.seed+1);tr0,va0=next(second.split(np.zeros(len(remain)),key.iloc[remain]));train,validation=remain[tr0],remain[va0]
  split=np.full(len(d),"",dtype="U10");split[train]="train";split[validation]="validation";split[test]="test";d["split"]=split
  s=pd.DataFrame({"eid":d.eid,"target":y,"split":split});s.to_csv(generated,index=False);logging.info("Created target-age-sex stratified v8 split: %s",generated)
 ix=tuple(np.where(d.split.eq(q))[0] for q in ["train","validation","test"])
 for q,z in zip(["train","validation","test"],ix):logging.info("Split %-10s N=%d cases=%d prevalence=%.5f",q,len(z),y[z].sum(),y[z].mean())
 used=set(FUSION_TAB+TRAD_PA)
 bad=used&PROHIBITED
 if bad:raise RuntimeError(f"Leakage predictors requested: {sorted(bad)}")
 return d,y,ix

def tabular_model(name,cols,d,y,train,val,test):
 pipe=v6.Pipeline([("prep",v6.tab_preprocessor(cols)),("model",LogisticRegression(C=.25,max_iter=4000,solver="lbfgs"))])
 pipe.fit(d.iloc[train][cols],y[train])
 return {"name":name,"pipeline":pipe,
  "val_score":v6.logit(pipe.predict_proba(d.iloc[val][cols])[:,1]),
  "test_score":v6.logit(pipe.predict_proba(d.iloc[test][cols])[:,1])}

class MaskedAttnPool(nn.Module):
 def __init__(self,d):super().__init__();self.score=nn.Sequential(nn.Linear(d,d//2),nn.Tanh(),nn.Linear(d//2,1))
 def forward(self,x,valid=None):
  s=self.score(x).squeeze(-1)
  if valid is not None:s=s.masked_fill(~valid,-1e4)
  w=torch.softmax(s,-1);return (x*w.unsqueeze(-1)).sum(1),w

class RhythmGRUFCNEncoder(nn.Module):
 """GRU-FCN with learned circadian organization, not handcrafted features.

 Local branch: reproduces GRU-FCN minute-pattern encoder.
 Same-clock branch: aligns seven days at the same clock time and learns their
   mean, variability and observation support.
 Phase branch: aggregates learnable representations in four physiological
   periods (night, morning, daytime, evening), with cyclic clock embeddings.
 Interday branch: encodes day embeddings and consecutive-day changes.
 """
 def __init__(self,d=64,drop=.30):
  super().__init__();self.output_dim=d
  self.base=arena.GRUFCNEncoder(d,drop)
  q=d//2
  self.minute=nn.Sequential(nn.Conv1d(2,q,15,stride=10,padding=7,bias=False),nn.BatchNorm1d(q),nn.GELU(),nn.Conv1d(q,q,5,padding=2,bias=False),nn.BatchNorm1d(q),nn.GELU())
  # Clock representation is fused at dimension d, so the cyclic clock
  # embedding must also project to d (not q=d//2).
  self.clock_embed=nn.Sequential(nn.Linear(2,16),nn.GELU(),nn.Linear(16,d))
  self.clock_mix=nn.Sequential(nn.Conv1d(2*q+1,d,5,padding=2,bias=False),nn.BatchNorm1d(d),nn.GELU(),nn.Conv1d(d,d,3,padding=1,bias=False),nn.BatchNorm1d(d),nn.GELU())
  self.phase_pool=MaskedAttnPool(d);self.phase_id=nn.Parameter(torch.randn(1,4,d)*.02)
  phase_layer=nn.TransformerEncoderLayer(d,4,2*d,drop,batch_first=True,norm_first=True,activation="gelu")
  self.phase_encoder=nn.TransformerEncoder(phase_layer,1);self.phase_final=MaskedAttnPool(d)
  self.day_pool=MaskedAttnPool(q);self.day_gru=nn.GRU(q,d//2,1,batch_first=True,bidirectional=True)
  self.change=nn.Sequential(nn.Linear(q,d),nn.GELU(),nn.Dropout(drop))
  self.fuse=nn.Sequential(nn.Linear(d*4,d*2),nn.LayerNorm(d*2),nn.GELU(),nn.Dropout(drop),nn.Linear(d*2,d),nn.LayerNorm(d))
  self.last_phase_attention=None;self.last_day_attention=None

 def forward(self,x):
  b=x.shape[0];base=self.base(x)
  days=x.reshape(b,2,7,1440).permute(0,2,1,3).reshape(b*7,2,1440)
  z=self.minute(days).transpose(1,2).reshape(b,7,144,self.output_dim//2)
  support=days[:,1].unfold(-1,10,10).mean(-1)[:,:144].reshape(b,7,144)
  valid=support.gt(.20)
  # Seven aligned days are treated as repeated observations of clock time.
  # Calculate cross-day moments in FP32. sqrt(var) at exactly zero has an
  # infinite derivative; the epsilon is required for stable AMP/SSL training.
  z32=z.float();denom=valid.sum(1).clamp_min(1).unsqueeze(-1).float()
  mean=(z32*valid.unsqueeze(-1)).sum(1)/denom
  var=((z32-mean.unsqueeze(1)).square()*valid.unsqueeze(-1)).sum(1)/denom
  clock=torch.arange(144,device=x.device,dtype=x.dtype)/144.*2*torch.pi
  cyc=self.clock_embed(torch.stack([torch.sin(clock),torch.cos(clock)],-1)).unsqueeze(0).expand(b,-1,-1)
  stats=torch.cat([mean,torch.sqrt(var.clamp_min(1e-4)),valid.float().mean(1).unsqueeze(-1)],-1)
  stats=torch.nan_to_num(stats,nan=0.,posinf=20.,neginf=-20.).clamp(-20.,20.)
  clock_rep=self.clock_mix(stats.transpose(1,2)).transpose(1,2)+cyc
  # Fixed boundaries express interpretable physiology; contents and weights are learned.
  hour=torch.arange(144,device=x.device)/6.
  phase_masks=torch.stack([(hour<6),((hour>=6)&(hour<12)),((hour>=12)&(hour<18)),(hour>=18)])
  phases=[]
  for pm in phase_masks:phases.append(self.phase_pool(clock_rep,pm.unsqueeze(0).expand(b,-1))[0])
  phases=self.phase_encoder(torch.stack(phases,1)+self.phase_id);phase,phase_w=self.phase_final(phases)
  # Daily organization and consecutive-day instability are learned from embeddings.
  day=[]
  for j in range(7):day.append(self.day_pool(z[:,j],valid[:,j])[0])
  day=torch.stack(day,1);_,h=self.day_gru(day);interday=torch.cat([h[-2],h[-1]],-1)
  change=self.change((day[:,1:]-day[:,:-1]).abs().mean(1))
  self.last_phase_attention=phase_w;self.last_day_attention=valid.float().mean(2)
  return self.fuse(torch.cat([base,phase,interday,change],-1))

class Classifier(nn.Module):
 def __init__(self,encoder,d=64):super().__init__();self.encoder=encoder;self.risk=nn.Linear(d,1);self.proj=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,32))
 def forward(self,x):return self.risk(self.encoder(x)).squeeze(-1)
 def project(self,x):return F.normalize(self.proj(self.encoder(x)),dim=-1)

def build(name,cfg):
 enc=arena.GRUFCNEncoder(cfg.d_model,cfg.dropout) if name=="grufcn_raw" else RhythmGRUFCNEncoder(cfg.d_model,cfg.dropout)
 return Classifier(enc,cfg.d_model)

def loaders(cfg,d,y,train,val,test):
 idx=d.cache_index.to_numpy(int);dummy=np.zeros((len(d),1),np.float32);loc,scale=v6.robust_raw_scale(cfg.cache_dir,idx,train,cfg.seed)
 sets=[v6.WeekDataset(cfg.cache_dir,idx[z],y[z],dummy[z],loc,scale,False) for z in [train,val,test]]
 ssl=v6.WeekDataset(cfg.cache_dir,idx[train],y[train],dummy[train],loc,scale,True)
 pos=max(y[train].sum(),1);neg=max((y[train]==0).sum(),1);q=cfg.sampled_positive_fraction;w=np.where(y[train]==1,q/pos,(1-q)/neg)
 sampler=WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),len(train),replacement=True)
 nw=0 if cfg.mode=="smoke" else cfg.num_workers;kw={"num_workers":nw,"pin_memory":torch.cuda.is_available(),"persistent_workers":nw>0}
 return (DataLoader(ssl,cfg.batch_size,shuffle=True,drop_last=True,**kw),DataLoader(sets[0],cfg.batch_size,sampler=sampler,**kw),DataLoader(sets[1],cfg.eval_batch_size,shuffle=False,**kw),DataLoader(sets[2],cfg.eval_batch_size,shuffle=False,**kw),loc,scale)

def predict(model,loader,device,amp):
 model.eval();ys=[];ss=[]
 with torch.no_grad():
  for x,y,_ in loader:
   with v6.amp_ctx(device,amp):s=model(x.to(device,non_blocking=True))
   ys.append(y.numpy());ss.append(s.float().cpu().numpy())
 return np.concatenate(ys),np.concatenate(ss)

def train_neural(name,cfg,d,y,train,val,test,out):
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu");ssl,tr,va,te,loc,scale=loaders(cfg,d,y,train,val,test)
 model=build(name,cfg).to(device);npar=sum(p.numel() for p in model.parameters());logging.info("Training %s parameters=%d device=%s",name,npar,device)
 if cfg.ssl_epochs:
  opt=torch.optim.AdamW(model.parameters(),lr=cfg.ssl_lr,weight_decay=cfg.weight_decay);gs=v6.grad_scaler(device,cfg.amp)
  for ep in range(1,cfg.ssl_epochs+1):
   model.train();ls=[]
   for a,b,_,_ in ssl:
    a=a.to(device);b=b.to(device);opt.zero_grad(set_to_none=True)
    with v6.amp_ctx(device,cfg.amp):loss=v6.nt_xent(model.project(a),model.project(b),cfg.temperature)
    if not torch.isfinite(loss):raise FloatingPointError(f"{name} SSL produced non-finite loss at epoch {ep}")
    gs.scale(loss).backward();gs.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);gs.step(opt);gs.update();ls.append(float(loss.detach().cpu()))
   logging.info("%s SSL epoch=%d loss=%.4f",name,ep,np.mean(ls))
 opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(cfg.epochs,1),eta_min=cfg.lr*.1);gs=v6.grad_scaler(device,cfg.amp);bce=nn.BCEWithLogitsLoss();best=-1.;stale=0;hist=[];path=out/f"{name}_best.pt"
 for ep in range(1,cfg.epochs+1):
  model.train();losses=[]
  for x,yy,_ in tr:
   x=x.to(device);yy=yy.to(device);opt.zero_grad(set_to_none=True)
   with v6.amp_ctx(device,cfg.amp):loss=bce(model(x),yy)
   if not torch.isfinite(loss):raise FloatingPointError(f"{name} supervised training produced non-finite loss at epoch {ep}")
   gs.scale(loss).backward();gs.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);gs.step(opt);gs.update();losses.append(float(loss.detach().cpu()))
  sch.step();vy,vs=predict(model,va,device,cfg.amp);ap=average_precision_score(vy,vs);auc=roc_auc_score(vy,vs);hist.append({"model":name,"epoch":ep,"loss":np.mean(losses),"validation_auprc":ap,"validation_auroc":auc});logging.info("%s epoch=%d loss=%.4f val_AUPRC=%.4f val_AUROC=%.4f",name,ep,np.mean(losses),ap,auc)
  if ap>best+1e-5:best=ap;stale=0;torch.save({"model":model.state_dict(),"name":name,"epoch":ep,"loc":loc,"scale":scale,"config":asdict(cfg)},path)
  else:stale+=1
  if ep>=cfg.min_epochs and stale>=cfg.patience:break
 state=torch.load(path,map_location=device);model.load_state_dict(state["model"]);vy,vs=predict(model,va,device,cfg.amp);ty,ts=predict(model,te,device,cfg.amp)
 return {"name":name,"val_score":vs,"test_score":ts,"history":hist,"best_epoch":state["epoch"],"parameters":npar,"model":model,"state":state}

def calibrate(result,yval,target=.80):
 cal,thr=v6.fit_calibrator(yval,result["val_score"],target);result["calibrator"]=cal;result["threshold"]=thr;result["test_probability"]=cal.predict_proba(np.asarray(result["test_score"])[:,None])[:,1];return result

def fit_late_fusion(yval,raw,clinical):
 # Inputs are out-of-training validation scores; L2 regularization limits the
 # instability caused by the small number of validation events.
 x=np.column_stack([raw,clinical]);return LogisticRegression(C=.10,solver="lbfgs",max_iter=4000).fit(x,yval)

def sensitivity_threshold(y,p,target=.80):
 precision,recall,thresholds=precision_recall_curve(y,p);ok=np.where(recall[:-1]>=target)[0]
 return float(thresholds[ok[np.argmax(precision[:-1][ok])]]) if len(ok) else .5

def paired_bootstrap(y,pred,reference,n,seed):
 rng=np.random.default_rng(seed);ca=np.where(y==1)[0];co=np.where(y==0)[0];rows=[]
 for name,p in pred.items():
  if name==reference:continue
  vals=[]
  for _ in range(n):
   z=np.r_[rng.choice(ca,len(ca),True),rng.choice(co,len(co),True)]
   vals.append([average_precision_score(y[z],p[z])-average_precision_score(y[z],pred[reference][z]),roc_auc_score(y[z],p[z])-roc_auc_score(y[z],pred[reference][z])])
  a=np.asarray(vals);rows.append({"model":name,"reference":reference,"delta_auprc":average_precision_score(y,p)-average_precision_score(y,pred[reference]),"delta_auprc_ci_low":np.quantile(a[:,0],.025),"delta_auprc_ci_high":np.quantile(a[:,0],.975),"delta_auroc":roc_auc_score(y,p)-roc_auc_score(y,pred[reference]),"delta_auroc_ci_low":np.quantile(a[:,1],.025),"delta_auroc_ci_high":np.quantile(a[:,1],.975)})
 return pd.DataFrame(rows)

def report(out,y,results,metrics,history):
 figs=[];colors=plt.cm.Set2(np.linspace(0,1,len(results)))
 def save(fig,name):fig.tight_layout();p=out/name;fig.savefig(p,dpi=300,bbox_inches="tight");plt.close(fig);figs.append(p)
 fig,ax=plt.subplots(figsize=(7,5))
 for (name,r),c in zip(results.items(),colors):f,t,_=roc_curve(y,r["test_probability"]);ax.plot(f,t,color=c,label=f"{name}: {roc_auc_score(y,r['test_probability']):.3f}")
 ax.plot([0,1],[0,1],"k--",lw=.8);ax.set(xlabel="False-positive rate",ylabel="True-positive rate",title="v8 test ROC");ax.legend(fontsize=7);save(fig,"01_roc.png")
 fig,ax=plt.subplots(figsize=(7,5))
 for (name,r),c in zip(results.items(),colors):p,re,_=precision_recall_curve(y,r["test_probability"]);ax.plot(re,p,color=c,label=f"{name}: {average_precision_score(y,r['test_probability']):.3f}")
 ax.axhline(y.mean(),color="k",ls="--",lw=.8);ax.set(xlabel="Recall",ylabel="Precision",title="v8 test precision-recall");ax.legend(fontsize=7);save(fig,"02_pr.png")
 fig,ax=plt.subplots(figsize=(7,5))
 for (name,r),c in zip(results.items(),colors):o,m=calibration_curve(y,r["test_probability"],n_bins=8,strategy="quantile");ax.plot(m,o,"o-",color=c,label=name)
 lim=max(.05,ax.get_xlim()[1]);ax.plot([0,lim],[0,lim],"k--",lw=.8);ax.set(xlabel="Predicted probability",ylabel="Observed proportion",title="Validation-only calibrated probabilities");ax.legend(fontsize=7);save(fig,"03_calibration.png")
 fig,ax=plt.subplots(figsize=(7,4))
 for name,g in history.groupby("model"):ax.plot(g.epoch,g.validation_auprc,"o-",label=name)
 ax.set(xlabel="Epoch",ylabel="Validation AUPRC",title="Neural model selection");ax.legend();save(fig,"04_training.png")
 pdf=out/"DEMENTIA_GRUFCN_V8_REPORT.pdf"
 with PdfPages(pdf) as pages:
  fig=plt.figure(figsize=(11.69,8.27));fig.text(.06,.88,"Dementia GRU-FCN v8",fontsize=24,weight="bold");fig.text(.06,.79,"Physiology-guided end-to-end 7-day activity modelling",fontsize=14);fig.text(.06,.62,f"Test N={len(y):,}; cases={int(y.sum())}; prevalence={y.mean():.3%}\nRaw baseline, rhythm-guided raw model, and clinical/APOE late fusion.",fontsize=12,linespacing=1.6);fig.text(.06,.42,"Training-only preprocessing; validation-only epoch selection, calibration and fusion; test-only final evaluation.\nNo follow-up age, outcome-derived variables, comorbidity index, traditional PA or handcrafted rhythm features enter the proposed neural encoder.",fontsize=10.5,linespacing=1.6);pages.savefig(fig);plt.close(fig)
  show=metrics[["model","auroc","auroc_ci_low","auroc_ci_high","auprc","auprc_ci_low","auprc_ci_high","brier","sensitivity","specificity","ppv"]].round(4);fig,ax=plt.subplots(figsize=(11.69,8.27));ax.axis("off");tb=ax.table(cellText=show.values,colLabels=show.columns,loc="center",cellLoc="center");tb.auto_set_font_size(False);tb.set_fontsize(7);tb.scale(1,1.5);ax.set_title("Test performance",fontsize=16);pages.savefig(fig,bbox_inches="tight");plt.close(fig)
  for p in figs:fig,ax=plt.subplots(figsize=(11.69,8.27));ax.imshow(plt.imread(p));ax.axis("off");pages.savefig(fig,bbox_inches="tight");plt.close(fig)
 return pdf

def self_test(cfg):
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 for name in ["grufcn_raw","rhythm_grufcn_raw"]:
  m=build(name,cfg).to(device);x=torch.randn(2,2,10080,device=device);o=m(x);assert o.shape==(2,) and torch.isfinite(o).all();o.sum().backward();logging.info("Self-test %s OK parameters=%d",name,sum(p.numel() for p in m.parameters()));del m,x,o
 if torch.cuda.is_available():torch.cuda.empty_cache()

def real_data_smoke(cfg,d,y,train,val,test):
 def take(z,ncase=8,ncontrol=120):return np.r_[z[y[z]==1][:ncase],z[y[z]==0][:ncontrol]]
 tr=take(train);va=take(val,4,60);te=take(test,4,60)
 pp=v6.tab_preprocessor(FUSION_TAB);pp.fit(d.iloc[tr][FUSION_TAB]);tab=np.asarray(pp.transform(d.iloc[va][FUSION_TAB]),np.float32)
 if len(tab)!=len(va) or not np.isfinite(tab).all():raise RuntimeError("Smoke tabular preprocessing failed")
 ssl_loader,loader,_,_,_,_=loaders(cfg,d,y,tr,va,te);x,yy,_=next(iter(loader));device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 for name in ["grufcn_raw","rhythm_grufcn_raw"]:
  model=build(name,cfg).to(device);model.train();opt=torch.optim.AdamW(model.parameters(),lr=cfg.ssl_lr)
  for step,(a,b,_,_) in enumerate(ssl_loader,1):
   opt.zero_grad(set_to_none=True);loss_ssl=v6.nt_xent(model.project(a.to(device)),model.project(b.to(device)),cfg.temperature)
   if not torch.isfinite(loss_ssl):raise RuntimeError(f"Non-finite real-data SSL smoke loss: {name} step={step}")
   loss_ssl.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
   if step>=3:break
  opt.zero_grad(set_to_none=True);score=model(x.to(device));loss=nn.BCEWithLogitsLoss()(score,yy.to(device));loss.backward()
  if not torch.isfinite(loss):raise RuntimeError(f"Non-finite real-data smoke loss: {name}")
  logging.info("Real-data smoke %s OK batch=%s ssl_loss=%.4f supervised_loss=%.4f",name,tuple(x.shape),float(loss_ssl.detach().cpu()),float(loss.detach().cpu()))
  del model
 if torch.cuda.is_available():torch.cuda.empty_cache()

def run(cfg):
 out=Path(cfg.output_dir);setup(out);v6.seed_all(cfg.seed);atomic_json(out/"config.json",asdict(cfg))
 if (out/"COMPLETED.json").exists() and not cfg.overwrite:raise RuntimeError("Output already completed; use --overwrite or a new output directory")
 with v6.run_lock(out):
  self_test(cfg)
  d,y,(train,val,test)=load_data(cfg);results={}
  if cfg.mode=="smoke":real_data_smoke(cfg,d,y,train,val,test);logging.info("Smoke end-to-end test completed successfully");atomic_json(out/"SMOKE_COMPLETED.json",{"completed":True,"time":datetime.now().isoformat(),"cohort_n":len(d),"cases":int(y.sum())});return
  pd.DataFrame([{"split":name,"n":len(z),"cases":int(y[z].sum()),"prevalence":float(y[z].mean())} for name,z in zip(["train","validation","test"],[train,val,test])]).to_csv(out/"cohort_flow.csv",index=False)
  for name,cols in [("traditional_pa_only",TRAD_PA),("clinical_apoe",FUSION_TAB)]:
   logging.info("Training comparator %s columns=%s",name,cols);results[name]=calibrate(tabular_model(name,cols,d,y,train,val,test),y[val],cfg.target_sensitivity);joblib.dump(results[name],out/f"{name}.joblib")
  histories=[]
  for name in ["grufcn_raw","rhythm_grufcn_raw"]:
   r=train_neural(name,cfg,d,y,train,val,test,out);histories.extend(r.pop("history"));results[name]=calibrate(r,y[val],cfg.target_sensitivity);joblib.dump(results[name]["calibrator"],out/f"{name}_platt.joblib")
  # Late fusion is deliberately fit only after raw-PA and clinical models have
  # produced validation scores; the test labels are not accessed here.
  meta=fit_late_fusion(y[val],results["rhythm_grufcn_raw"]["val_score"],results["clinical_apoe"]["val_score"])
  fs_val=meta.decision_function(np.column_stack([results["rhythm_grufcn_raw"]["val_score"],results["clinical_apoe"]["val_score"]]));fs_test=meta.decision_function(np.column_stack([results["rhythm_grufcn_raw"]["test_score"],results["clinical_apoe"]["test_score"]]))
  # The L2 logistic meta-model itself supplies the calibrated probability;
  # fitting a second Platt layer on the same small validation set is avoided.
  fp_val=meta.predict_proba(np.column_stack([results["rhythm_grufcn_raw"]["val_score"],results["clinical_apoe"]["val_score"]]))[:,1]
  fp_test=meta.predict_proba(np.column_stack([results["rhythm_grufcn_raw"]["test_score"],results["clinical_apoe"]["test_score"]]))[:,1]
  results["rhythm_grufcn_plus_clinical_apoe"]={"name":"rhythm_grufcn_plus_clinical_apoe","val_score":fs_val,"test_score":fs_test,"test_probability":fp_test,"threshold":sensitivity_threshold(y[val],fp_val,cfg.target_sensitivity)};joblib.dump(meta,out/"late_fusion_model.joblib")
  selection=[]
  for name,r in results.items():
   selection.append({"model":name,"validation_auroc":roc_auc_score(y[val],r["val_score"]),"validation_auprc":average_precision_score(y[val],r["val_score"]),"best_epoch":r.get("best_epoch",np.nan),"parameters":r.get("parameters",np.nan)})
  pd.DataFrame(selection).sort_values("validation_auprc",ascending=False).to_csv(out/"validation_model_selection.csv",index=False)
  rows=[];pred=pd.DataFrame({"eid":d.iloc[test].eid.to_numpy(),"target":y[test]})
  for i,(name,r) in enumerate(results.items()):
   p=r["test_probability"];pred[name]=p;row=v6.metric_row(name,y[test],p,r["threshold"]);rows.append(v6.bootstrap(row,y[test],p,cfg.bootstrap_repeats,cfg.seed+i))
  metrics=pd.DataFrame(rows).sort_values("auprc",ascending=False);metrics.to_csv(out/"test_metrics.csv",index=False);pred.to_csv(out/"test_predictions.csv",index=False)
  pd.DataFrame(histories).to_csv(out/"training_history.csv",index=False)
  paired_bootstrap(y[test],{k:r["test_probability"] for k,r in results.items()},"grufcn_raw",cfg.bootstrap_repeats,cfg.seed+99).to_csv(out/"paired_differences_vs_grufcn.csv",index=False)
  v6.top_risk(out,y[test],results)
  coef=pd.DataFrame({"source":["rhythm_grufcn_raw","clinical_apoe"],"coefficient":meta.coef_[0]});coef.to_csv(out/"late_fusion_coefficients.csv",index=False)
  pdf=report(out,y[test],results,metrics,pd.DataFrame(histories))
  audit={"design":"complete UKB cohort with fixed target-age-sex stratified 70/15/15 internal split","split_source":str(Path(cfg.output_dir)/"v8_internal_split.csv") if not cfg.split_csv else cfg.split_csv,"horizon_years":cfg.horizon_years,"primary_metric":"AUPRC","epoch_selection":"validation AUPRC only","calibration":"validation-only Platt","late_fusion":"validation-only L2 logistic fusion","test_role":"final internal metric computation only","external_validation":"NHATS planned","neural_input":["minute activity","observation mask"],"proposed_priors":["same-clock cross-day aggregation","four day/night phase tokens","interdaily sequence and change"],"handcrafted_rhythm_entered_neural_model":False,"traditional_pa_role":"comparator only","fusion_predictors":FUSION_TAB,"prohibited_predictors":sorted(PROHIBITED)}
  atomic_json(out/"design_and_leakage_audit.json",audit);atomic_json(out/"COMPLETED.json",{"completed":True,"time":datetime.now().isoformat(),"best_test_model":metrics.iloc[0].model,"best_test_auprc":metrics.iloc[0].auprc,"report":pdf.name});logging.info("v8 completed. Best test AUPRC=%s %.4f",metrics.iloc[0].model,metrics.iloc[0].auprc)

def parse_args():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument("--mode",choices=["smoke","full"],default="full");a.add_argument("--labels-csv",type=Path,default=v6.DEFAULT_LABELS);a.add_argument("--apoe-csv",type=Path,default=v6.DEFAULT_APOE);a.add_argument("--cache-dir",type=Path,default=v6.DEFAULT_CACHE);a.add_argument("--split-csv",type=Path,default=None,help="Optional existing split; default creates a new fixed v8 split from the complete UKB cohort");a.add_argument("--output-dir",type=Path,default=DEFAULT_OUT);a.add_argument("--ssl-epochs",type=int,default=1);a.add_argument("--epochs",type=int,default=12);a.add_argument("--batch-size",type=int,default=64);a.add_argument("--num-workers",type=int,default=6);a.add_argument("--bootstrap-repeats",type=int,default=1000);a.add_argument("--seed",type=int,default=20260818);a.add_argument("--overwrite",action="store_true");z=a.parse_args()
 return Config(str(z.labels_csv),str(z.apoe_csv),str(z.cache_dir),str(z.split_csv) if z.split_csv else "",str(z.output_dir),ssl_epochs=z.ssl_epochs,epochs=z.epochs,batch_size=z.batch_size,num_workers=z.num_workers,bootstrap_repeats=z.bootstrap_repeats,seed=z.seed,mode=z.mode,overwrite=z.overwrite)

if __name__=="__main__":run(parse_args())

