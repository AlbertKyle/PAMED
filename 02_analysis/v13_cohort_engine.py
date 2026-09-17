#!/usr/bin/env python3
"""Dementia-TSC v6: end-to-end multiview time-series classification benchmark.

No handcrafted activity/rhythm phenotype is used by any neural predictor or
auxiliary task. Inputs are only the 7-day minute-level activity series, its
observation mask, baseline variables, and optional APOE e4 dosage.

Three complementary encoders are compared on one newly locked split:
  1) InceptionTime: established convolutional time-series classifier.
  2) PatchTST: patch-token Transformer classifier.
  3) SameClockFormer: study-specific day-aligned multivariate classifier.

Each encoder receives task-internal, training-set-only contrastive pretraining.
Model/epoch selection is validation-AUPRC-only. The locked test labels are read
only after every candidate has finished and the validation winner is declared.
"""
from __future__ import annotations

import argparse, hashlib, json, logging, math, os, random, socket, sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
 confusion_matrix, log_loss, precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

BASE=Path("/root/autodl-tmp/PA_CR_CNN/Project_new/PA_Foundation_Model")
DEM=BASE/"dementia"
DEFAULT_LABELS=DEM/"ukb_dementia_labels_covariates_N80990.csv"
DEFAULT_APOE=DEM/"APOE_phenotype.csv"
DEFAULT_CACHE=DEM/"supervised_results"/"cache"
DEFAULT_OUT=DEM/"supervised_v6_tsc"

CLIN_CONT=["enrollment_age","BMI","Townsend_index"]
CLIN_CAT=["gender","education_level","smoking","alcohol_use","ethic"]
CLINICAL=CLIN_CONT+CLIN_CAT
TRAD_PA=["Light_Total_Hour","MVPA_Total_Hour","Sedentary_Total_Hour","Sleep_Total_Hour"]
APOE=["apoe_e4"]
NEURAL_TAB=CLINICAL+APOE
BLOCKED={"Followingyears_birthday","is_dead","Disable_long","overall_health",
 "first_dementia_date","first_dementia_icd10","first_dementia_description",
 "days_accel_to_dementia","dementia_state","prevalent_dementia",
 "incident_dementia_after_1y","incident_analysis_eligible",
 "dementia_definition_version","dementia_ascertainment_source",
 "comorbidity_index","charlson_index"}

try:
 from torch.amp import GradScaler as AmpScaler
 from torch.amp import autocast as amp_autocast
 AMP_NEW=True
except ImportError:
 from torch.cuda.amp import GradScaler as AmpScaler
 from torch.cuda.amp import autocast as amp_autocast
 AMP_NEW=False

@dataclass
class Config:
 labels_csv:str; apoe_csv:str; cache_dir:str; output_dir:str
 horizon_years:float=7.; landmark_days:int=365; seed:int=20260815
 validation_fraction:float=.15; test_fraction:float=.15
 models:str="inception,patchtst,sameclock"; ssl_epochs:int=2; epochs:int=12
 min_epochs:int=3; patience:int=3; batch_size:int=64; eval_batch_size:int=128
 num_workers:int=6; lr:float=1e-4; ssl_lr:float=2e-4; weight_decay:float=1e-3
 dropout:float=.30; d_model:int=64; sampled_positive_fraction:float=.08
 temperature:float=.15; bootstrap_repeats:int=1000; target_sensitivity:float=.80
 amp:bool=True; mode:str="full"; overwrite:bool=False

def atomic_json(path,obj):
 tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf-8");os.replace(tmp,path)
def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def setup_log(out):
 out.mkdir(parents=True,exist_ok=True);logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",handlers=[logging.FileHandler(out/"run.log",encoding="utf-8"),logging.StreamHandler(sys.stdout)],force=True)
@contextmanager
def run_lock(out):
 p=out/"RUNNING.lock";payload={"pid":os.getpid(),"host":socket.gethostname(),"started":datetime.now().isoformat()}
 try:fd=os.open(str(p),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
 except FileExistsError:raise RuntimeError(f"Another process owns {p}: {p.read_text(errors='replace')}")
 try:os.write(fd,json.dumps(payload).encode());os.close(fd);yield
 finally:
  if p.exists():p.unlink()
def amp_ctx(device,enabled):
 use=bool(enabled and device.type=="cuda")
 if AMP_NEW:return amp_autocast(device_type=device.type,dtype=torch.float16,enabled=use)
 return amp_autocast(dtype=torch.float16,enabled=use)
def grad_scaler(device,enabled):
 use=bool(enabled and device.type=="cuda")
 if AMP_NEW:
  try:return AmpScaler("cuda",enabled=use)
  except TypeError:return AmpScaler(enabled=use)
 return AmpScaler(enabled=use)

def load_cohort(cfg):
 d=pd.read_csv(cfg.labels_csv,dtype={"eid":"string"},low_memory=False);d["eid"]=d.eid.str.strip()
 accel=pd.to_datetime(d.accelerometer_start_date,errors="coerce");dx=pd.to_datetime(d.first_dementia_date,errors="coerce")
 days=(dx-accel).dt.days.astype(float).fillna(pd.to_numeric(d.days_accel_to_dementia,errors="coerce"))
 follow=pd.to_numeric(d.Followingyears_birthday,errors="coerce")-pd.to_numeric(d.enrollment_age,errors="coerce")
 horizon=int(round(365.25*cfg.horizon_years));base=d.incident_analysis_eligible.eq(1)&accel.notna()
 case=base&days.gt(cfg.landmark_days)&days.le(horizon)
 control=base&(~case)&(days.gt(horizon)|(days.isna()&follow.ge(cfg.horizon_years)))
 keep=case|control;d=d.loc[keep].copy();d["target"]=case.loc[keep].astype(np.int8).to_numpy()
 ap=pd.read_csv(cfg.apoe_csv,dtype={"eid":"string"},low_memory=False);ap["eid"]=ap.eid.str.strip();ap["apoe_e4"]=pd.to_numeric(ap.e4,errors="coerce");ap.loc[~ap.apoe_e4.isin([0,1,2]),"apoe_e4"]=np.nan
 d=d.merge(ap[["eid","apoe_e4"]],on="eid",how="left",validate="one_to_one").reset_index(drop=True)
 if BLOCKED.intersection(NEURAL_TAB):raise RuntimeError("Outcome/follow-up leakage in predictors")
 missing=sorted(set(NEURAL_TAB+TRAD_PA)-set(d.columns))
 if missing:raise ValueError(f"Missing predictors: {missing}")
 cache=Path(cfg.cache_dir);req=[cache/"activity_7d.npy",cache/"observed_7d.npy",cache/"eids.npy"]
 if not all(p.exists() for p in req):raise FileNotFoundError(f"Raw cache incomplete: {cache}")
 ce=np.load(cache/"eids.npy",mmap_mode="r").astype(str);lookup=pd.Series(np.arange(len(ce)),index=ce);d["cache_index"]=d.eid.map(lookup)
 if d.cache_index.isna().any() or d.eid.duplicated().any():raise ValueError("Cohort/cache EID alignment failed")
 logging.info("v6 fixed-horizon cohort N=%d cases=%d prevalence=%.5f APOE=%d",len(d),d.target.sum(),d.target.mean(),d.apoe_e4.notna().sum())
 return d

def locked_split(d,cfg,out):
 p=out/"v6_locked_split.csv";lock=out/"v6_split_lock.json"
 if p.exists() and not cfg.overwrite:
  z=pd.read_csv(p,dtype={"eid":"string"})
  if z.eid.tolist()!=d.eid.tolist() or z.target.astype(int).tolist()!=d.target.astype(int).tolist():raise ValueError("Existing v6 split mismatch")
  logging.info("Reusing locked v6 split")
 else:
  y=d.target.to_numpy();a=StratifiedShuffleSplit(1,test_size=cfg.test_fraction,random_state=cfg.seed);rem,test=next(a.split(np.zeros(len(y)),y));rel=cfg.validation_fraction/(1-cfg.test_fraction)
  b=StratifiedShuffleSplit(1,test_size=rel,random_state=cfg.seed+1);trr,var=next(b.split(np.zeros(len(rem)),y[rem]));train,val=rem[trr],rem[var]
  s=np.full(len(y),"",dtype="U10");s[train]="train";s[val]="validation";s[test]="test";z=pd.DataFrame({"eid":d.eid,"target":y,"split":s});z.to_csv(p,index=False)
  digest=hashlib.sha256("\n".join(z.loc[z.split.eq("test"),"eid"]).encode()).hexdigest();atomic_json(lock,{"created":datetime.now().isoformat(),"seed":cfg.seed,"test_eid_sha256":digest,"rule":"Test is inaccessible during training and validation model selection."})
 return tuple(np.where(z.split.eq(s))[0] for s in ["train","validation","test"])

def ohe():
 try:return OneHotEncoder(handle_unknown="ignore",sparse_output=False)
 except TypeError:return OneHotEncoder(handle_unknown="ignore",sparse=False)
def tab_preprocessor(cols):
 cats=[c for c in cols if c in CLIN_CAT+APOE];nums=[c for c in cols if c not in cats];parts=[]
 if nums:parts.append(("num",Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),("sc",RobustScaler())]),nums))
 if cats:parts.append(("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent",add_indicator=True)),("ohe",ohe())]),cats))
 return ColumnTransformer(parts,sparse_threshold=0)
def make_tabular(d,train):
 pp=tab_preprocessor(NEURAL_TAB);pp.fit(d.iloc[train][NEURAL_TAB]);x=np.asarray(pp.transform(d[NEURAL_TAB]),np.float32);return pp,x
def logit(p):p=np.clip(np.asarray(p,float),1e-6,1-1e-6);return np.log(p/(1-p))
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-30,30)))
def tabular_model(name,cols,d,y,train,val,test):
 pipe=Pipeline([("prep",tab_preprocessor(cols)),("model",LogisticRegression(C=.25,max_iter=4000,solver="lbfgs"))]);pipe.fit(d.iloc[train][cols],y[train])
 return {"name":name,"pipeline":pipe,"val_score":logit(pipe.predict_proba(d.iloc[val][cols])[:,1]),"test_score":logit(pipe.predict_proba(d.iloc[test][cols])[:,1])}

def robust_raw_scale(cache,cache_index,train,seed):
 x=np.load(Path(cache)/"activity_7d.npy",mmap_mode="r");m=np.load(Path(cache)/"observed_7d.npy",mmap_mode="r");rng=np.random.default_rng(seed);vals=[]
 for j in rng.choice(train,min(2000,len(train)),False):
  a=np.asarray(x[cache_index[j]],np.float32);o=np.asarray(m[cache_index[j]],bool);vals.append(np.log1p(np.clip(a[o],0,None)))
 v=np.concatenate(vals);return float(np.median(v)),float(max(np.quantile(v,.75)-np.quantile(v,.25),.1))

class WeekDataset(Dataset):
 def __init__(self,cache,cache_index,y,tab,loc,scale,augment=False):
  self.x=np.load(Path(cache)/"activity_7d.npy",mmap_mode="r");self.m=np.load(Path(cache)/"observed_7d.npy",mmap_mode="r")
  self.i=np.asarray(cache_index,np.int64);self.y=np.asarray(y,np.float32);self.t=np.asarray(tab,np.float32);self.loc=float(loc);self.scale=float(scale);self.augment=augment
 def __len__(self):return len(self.i)
 def base(self,j):
  a=np.asarray(self.x[self.i[j]],np.float32);m=np.asarray(self.m[self.i[j]],bool);z=(np.log1p(np.clip(np.nan_to_num(a,nan=0.),0,None))-self.loc)/self.scale;z=np.where(m,z,0.)
  return np.stack([z.reshape(-1),m.astype(np.float32).reshape(-1)],0).astype(np.float32)
 def view(self,x):
  x=x.copy();length=x.shape[-1];rng=np.random.default_rng()
  # Contiguous masks, amplitude jitter and occasional complete-day dropout.
  for _ in range(rng.integers(2,6)):
   width=int(rng.integers(30,241));start=int(rng.integers(0,max(1,length-width)));x[:,start:start+width]=0
  x[0]*=float(rng.uniform(.85,1.15));x[0]+=rng.normal(0,.025,length).astype(np.float32)*x[1]
  if rng.random()<.2:
   day=int(rng.integers(0,7));x[:,day*1440:(day+1)*1440]=0
  return x.astype(np.float32)
 def __getitem__(self,j):
  x=self.base(j)
  if self.augment:return torch.from_numpy(self.view(x)),torch.from_numpy(self.view(x)),torch.tensor(self.y[j]),torch.from_numpy(self.t[j])
  return torch.from_numpy(x),torch.tensor(self.y[j]),torch.from_numpy(self.t[j])

class InceptionModule(nn.Module):
 def __init__(self,ni,nf=32,bottleneck=32):
  super().__init__();self.b=nn.Conv1d(ni,bottleneck,1,bias=False) if ni>1 else nn.Identity();self.convs=nn.ModuleList([nn.Conv1d(bottleneck if ni>1 else ni,nf,k,padding=k//2,bias=False) for k in [9,19,39]]);self.pool=nn.Sequential(nn.MaxPool1d(3,1,1),nn.Conv1d(ni,nf,1,bias=False));self.bn=nn.BatchNorm1d(4*nf)
 def forward(self,x):return F.gelu(self.bn(torch.cat([c(self.b(x)) for c in self.convs]+[self.pool(x)],1)))
class InceptionEncoder(nn.Module):
 def __init__(self,d=64,drop=.3):
  super().__init__();self.stem=nn.Sequential(nn.Conv1d(2,32,9,stride=5,padding=4),nn.BatchNorm1d(32),nn.GELU());self.i1=InceptionModule(32,16,32);self.i2=InceptionModule(64,16,32);self.i3=InceptionModule(64,16,32);self.res=nn.Conv1d(32,64,1);self.out=nn.Sequential(nn.Linear(128,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop));self.output_dim=d
 def forward(self,x):
  s=self.stem(x);h=self.i3(self.i2(self.i1(s)))+self.res(s);return self.out(torch.cat([h.mean(-1),h.amax(-1)],-1))

class PatchTSTEncoder(nn.Module):
 def __init__(self,d=64,drop=.3):
  super().__init__();self.patch=nn.Conv1d(2,d,kernel_size=60,stride=30);self.pos=nn.Parameter(torch.randn(1,335,d)*.01);layer=nn.TransformerEncoderLayer(d,4,2*d,drop,batch_first=True,norm_first=True,activation="gelu");self.enc=nn.TransformerEncoder(layer,2);self.norm=nn.LayerNorm(d);self.output_dim=d
 def forward(self,x):
  z=self.patch(x).transpose(1,2);z=self.enc(z+self.pos[:,:z.shape[1]]);return self.norm(z.mean(1))

class AttentionPool(nn.Module):
 def __init__(self,d):super().__init__();self.score=nn.Sequential(nn.Linear(d,d//2),nn.Tanh(),nn.Linear(d//2,1))
 def forward(self,x):w=torch.softmax(self.score(x).squeeze(-1),-1);return (w.unsqueeze(-1)*x).sum(-2),w
class SameClockEncoder(nn.Module):
 """Treats seven aligned days as multivariate replicates at each clock time."""
 def __init__(self,d=64,drop=.3):
  super().__init__();self.local=nn.Sequential(nn.Conv1d(2,d//2,9,stride=5,padding=4),nn.BatchNorm1d(d//2),nn.GELU(),nn.Conv1d(d//2,d,5,padding=2),nn.BatchNorm1d(d),nn.GELU())
  self.day_id=nn.Parameter(torch.randn(1,7,d)*.02);day_layer=nn.TransformerEncoderLayer(d,4,2*d,drop,batch_first=True,norm_first=True,activation="gelu");self.across_days=nn.TransformerEncoder(day_layer,1);self.day_pool=AttentionPool(d)
  self.clock_pos=nn.Parameter(torch.randn(1,288,d)*.01);clock_layer=nn.TransformerEncoderLayer(d,4,2*d,drop,batch_first=True,norm_first=True,activation="gelu");self.across_clock=nn.TransformerEncoder(clock_layer,2);self.clock_pool=AttentionPool(d);self.norm=nn.LayerNorm(d);self.output_dim=d;self.last_day_attention=None;self.last_clock_attention=None
 def forward(self,x):
  b=x.shape[0];z=self.local(x.reshape(b,2,7,1440).permute(0,2,1,3).reshape(b*7,2,1440)).transpose(1,2).reshape(b,7,288,self.output_dim)
  q=z.permute(0,2,1,3).reshape(b*288,7,self.output_dim)+self.day_id;q=self.across_days(q);clock,dayw=self.day_pool(q);clock=clock.reshape(b,288,self.output_dim)
  clock=self.across_clock(clock+self.clock_pos);week,clockw=self.clock_pool(clock);self.last_day_attention=dayw.reshape(b,288,7).mean(1);self.last_clock_attention=clockw
  return self.norm(week)

class FusionClassifier(nn.Module):
 def __init__(self,encoder,tab_dim,d=64,drop=.3,use_tab=True):
  super().__init__();self.encoder=encoder;self.use_tab=use_tab;self.tab=nn.Sequential(nn.Linear(tab_dim,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,d),nn.GELU())
  self.raw_head=nn.Linear(d,1);self.fusion=nn.Sequential(nn.Linear(4*d,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,1));self.projection=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,32))
 def encode(self,x):return self.encoder(x)
 def project(self,x):return F.normalize(self.projection(self.encode(x)),dim=-1)
 def forward_both(self,x,tab):
  r=self.encode(x);raw=self.raw_head(r).squeeze(-1)
  if not self.use_tab:return raw,raw
  t=self.tab(tab);fused=self.fusion(torch.cat([r,t,r*t,torch.abs(r-t)],-1)).squeeze(-1);return fused,raw
 def forward(self,x,tab,raw_only=False):
  fused,raw=self.forward_both(x,tab);return raw if raw_only else fused

def build_model(name,tab_dim,cfg,use_tab=True):
 if name=="inception":enc=InceptionEncoder(cfg.d_model,cfg.dropout)
 elif name=="patchtst":enc=PatchTSTEncoder(cfg.d_model,cfg.dropout)
 elif name=="sameclock":enc=SameClockEncoder(cfg.d_model,cfg.dropout)
 else:raise ValueError(name)
 return FusionClassifier(enc,tab_dim,cfg.d_model,cfg.dropout,use_tab)

def make_loaders(cfg,d,y,idx,train,val,test,tab):
 loc,scale=robust_raw_scale(cfg.cache_dir,idx,train,cfg.seed);base=[WeekDataset(cfg.cache_dir,idx[ix],y[ix],tab[ix],loc,scale,False) for ix in [train,val,test]];ssl=WeekDataset(cfg.cache_dir,idx[train],y[train],tab[train],loc,scale,True)
 pos=max(y[train].sum(),1);neg=max((y[train]==0).sum(),1);q=cfg.sampled_positive_fraction;w=np.where(y[train]==1,q/pos,(1-q)/neg);sampler=WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),len(train),True)
 nw=0 if cfg.mode=="smoke" else cfg.num_workers;kw={"num_workers":nw,"pin_memory":torch.cuda.is_available(),"persistent_workers":nw>0}
 return DataLoader(ssl,cfg.batch_size,shuffle=True,drop_last=True,**kw),DataLoader(base[0],cfg.batch_size,sampler=sampler,**kw),DataLoader(base[1],cfg.eval_batch_size,shuffle=False,**kw),DataLoader(base[2],cfg.eval_batch_size,shuffle=False,**kw),loc,scale

def nt_xent(a,b,temp):
 n=a.shape[0];z=torch.cat([a,b],0);sim=z@z.T/temp;sim.fill_diagonal_(-1e4);target=torch.arange(2*n,device=z.device);target=(target+n)%(2*n);return F.cross_entropy(sim,target)
def predict(model,loader,device,amp,raw_only=False):
 model.eval();ys=[];ss=[];days=[];clocks=[]
 with torch.no_grad():
  for x,y,t in loader:
   with amp_ctx(device,amp):s=model(x.to(device,non_blocking=True),t.to(device,non_blocking=True),raw_only=raw_only)
   ys.append(y.numpy());ss.append(s.float().cpu().numpy())
   if isinstance(model.encoder,SameClockEncoder):days.append(model.encoder.last_day_attention.float().cpu().numpy());clocks.append(model.encoder.last_clock_attention.float().cpu().numpy())
 detail={}
 if days:detail={"day_attention":np.concatenate(days),"clock_attention":np.concatenate(clocks)}
 return np.concatenate(ys),np.concatenate(ss),detail

def train_one(name,cfg,d,y,idx,train,val,test,tab,out,use_tab=True):
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu");ssl,tr,va,te,loc,scale=make_loaders(cfg,d,y,idx,train,val,test,tab);model=build_model(name,tab.shape[1],cfg,use_tab).to(device)
 logging.info("%s use_tab=%s parameters=%d device=%s",name,use_tab,sum(p.numel() for p in model.parameters()),device)
 if cfg.ssl_epochs>0:
  opt=torch.optim.AdamW(model.parameters(),lr=cfg.ssl_lr,weight_decay=cfg.weight_decay);gs=grad_scaler(device,cfg.amp)
  for ep in range(1,cfg.ssl_epochs+1):
   model.train();ls=[]
   for a,b,_,_ in ssl:
    a=a.to(device,non_blocking=True);b=b.to(device,non_blocking=True);opt.zero_grad(set_to_none=True)
    with amp_ctx(device,cfg.amp):loss=nt_xent(model.project(a),model.project(b),cfg.temperature)
    gs.scale(loss).backward();gs.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);gs.step(opt);gs.update();ls.append(float(loss.detach().cpu()))
   logging.info("%s SSL epoch=%d contrastive_loss=%.4f",name,ep,np.mean(ls))
 opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(cfg.epochs,1),eta_min=cfg.lr*.1);gs=grad_scaler(device,cfg.amp);bce=nn.BCEWithLogitsLoss();best=-1.;stale=0;hist=[];path=out/f"{name}_best.pt"
 for ep in range(1,cfg.epochs+1):
  model.train();ls=[]
  for x,yy,t in tr:
   x=x.to(device,non_blocking=True);yy=yy.to(device);t=t.to(device);opt.zero_grad(set_to_none=True)
   with amp_ctx(device,cfg.amp):
    s,raw=model.forward_both(x,t);loss=bce(s,yy)+.30*bce(raw,yy)
   gs.scale(loss).backward();gs.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);gs.step(opt);gs.update();ls.append(float(loss.detach().cpu()))
  sch.step();vy,vs,_=predict(model,va,device,cfg.amp);ap=average_precision_score(vy,vs);auc=roc_auc_score(vy,vs);hist.append({"model":name,"epoch":ep,"loss":np.mean(ls),"val_auprc":ap,"val_auroc":auc,"lr":opt.param_groups[0]["lr"]});logging.info("%s epoch=%d loss=%.4f val_AUPRC=%.4f val_AUROC=%.4f",name,ep,np.mean(ls),ap,auc)
  if ap>best+1e-5:best=ap;stale=0;torch.save({"model":model.state_dict(),"epoch":ep,"name":name,"use_tab":use_tab,"tab_dim":tab.shape[1],"loc":loc,"scale":scale,"config":asdict(cfg)},path)
  else:stale+=1
  if ep>=cfg.min_epochs and stale>=cfg.patience:break
 state=torch.load(path,map_location=device);model.load_state_dict(state["model"]);vy,vs,_=predict(model,va,device,cfg.amp);_,vraw,_=predict(model,va,device,cfg.amp,raw_only=True)
 # Deliberately do not call test loader here. Test inference occurs after validation selection.
 return {"name":name,"val_score":vs,"val_raw_score":vraw,"validation_auprc":average_precision_score(vy,vs),"validation_raw_auprc":average_precision_score(vy,vraw),"history":hist,"model":model,"test_loader":te,"state":state}

def fit_calibrator(val_y,val_score,val_probability_target=.80):
 cal=LogisticRegression(C=1e6,solver="lbfgs").fit(np.asarray(val_score)[:,None],val_y);vp=cal.predict_proba(np.asarray(val_score)[:,None])[:,1];pr,re,th=precision_recall_curve(val_y,vp);ok=np.where(re[:-1]>=val_probability_target)[0];threshold=float(th[ok[np.argmax(pr[:-1][ok])]]) if len(ok) else .5;return cal,threshold
def metric_row(name,y,p,t):
 z=p>=t;tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel();prev=y.mean();recal=LogisticRegression(C=1e6,solver="lbfgs").fit(logit(p)[:,None],y)
 return {"model":name,"n":len(y),"cases":int(y.sum()),"prevalence":prev,"auroc":roc_auc_score(y,p),"auprc":average_precision_score(y,p),"auprc_over_prevalence":average_precision_score(y,p)/prev,"threshold":t,"sensitivity":tp/max(tp+fn,1),"specificity":tn/max(tn+fp,1),"ppv":tp/max(tp+fp,1),"npv":tn/max(tn+fn,1),"brier":brier_score_loss(y,p),"log_loss":log_loss(y,p,labels=[0,1]),"tp":int(tp),"fp":int(fp),"fn":int(fn),"tn":int(tn),"calibration_intercept":float(recal.intercept_[0]),"calibration_slope":float(recal.coef_[0,0])}
def bootstrap(row,y,p,n,seed):
 rng=np.random.default_rng(seed);ca=np.where(y==1)[0];co=np.where(y==0)[0];v=[]
 for _ in range(n):
  ix=np.r_[rng.choice(ca,len(ca),True),rng.choice(co,len(co),True)];v.append([roc_auc_score(y[ix],p[ix]),average_precision_score(y[ix],p[ix]),brier_score_loss(y[ix],p[ix])])
 a=np.asarray(v)
 for j,m in enumerate(["auroc","auprc","brier"]):row[m+"_ci_low"]=np.quantile(a[:,j],.025);row[m+"_ci_high"]=np.quantile(a[:,j],.975)
 return row

def top_risk(out,y,results):
 rows=[]
 for name,r in results.items():
  p=r["test_probability"]
  for q in [.01,.05,.10,.20]:
   k=max(1,int(np.ceil(len(y)*q)));ix=np.argsort(p)[-k:];rows.append({"model":name,"top_fraction":q,"selected_n":k,"cases_captured":int(y[ix].sum()),"case_capture_fraction":y[ix].sum()/y.sum(),"ppv":y[ix].mean(),"enrichment":y[ix].mean()/y.mean()})
 pd.DataFrame(rows).to_csv(out/"top_risk_enrichment.csv",index=False)

def report(out,y,results,metrics,history):
 figs=[];colors=plt.cm.Set2(np.linspace(0,1,len(results)))
 def save(fig,n):fig.tight_layout();p=out/n;fig.savefig(p,dpi=300,bbox_inches="tight");plt.close(fig);figs.append(p)
 fig,ax=plt.subplots(figsize=(7,5))
 for (n,r),c in zip(results.items(),colors):f,t,_=roc_curve(y,r["test_probability"]);ax.plot(f,t,color=c,label=f"{n}: {roc_auc_score(y,r['test_probability']):.3f}")
 ax.plot([0,1],[0,1],"k--",lw=.8);ax.set(xlabel="False-positive rate",ylabel="True-positive rate",title="v6 locked-test ROC");ax.legend(fontsize=7);save(fig,"01_roc.png")
 fig,ax=plt.subplots(figsize=(7,5))
 for (n,r),c in zip(results.items(),colors):p,re,_=precision_recall_curve(y,r["test_probability"]);ax.plot(re,p,color=c,label=f"{n}: {average_precision_score(y,r['test_probability']):.3f}")
 ax.axhline(y.mean(),color="k",ls="--",lw=.8);ax.set(xlabel="Recall",ylabel="Precision",title="v6 locked-test precision-recall");ax.legend(fontsize=7);save(fig,"02_pr.png")
 fig,ax=plt.subplots(figsize=(7,5))
 for (n,r),c in zip(results.items(),colors):o,m=calibration_curve(y,r["test_probability"],n_bins=8,strategy="quantile");ax.plot(m,o,"o-",color=c,label=n)
 mx=max(.06,ax.get_xlim()[1]);ax.plot([0,mx],[0,mx],"k--",lw=.8);ax.set(xlabel="Predicted probability",ylabel="Observed proportion",title="v6 locked-test calibration");ax.legend(fontsize=7);save(fig,"03_calibration.png")
 fig,ax=plt.subplots(figsize=(7,4))
 for n,g in history.groupby("model",sort=False):ax.plot(g.epoch,g.val_auprc,"o-",label=n)
 ax.set(xlabel="Epoch",ylabel="Validation AUPRC",title="Time-series classifier selection");ax.legend();save(fig,"04_training.png")
 if (out/"sameclock_attention_clock.csv").exists():
  a=pd.read_csv(out/"sameclock_attention_clock.csv");fig,ax=plt.subplots(figsize=(8,4));ax.plot(a.clock_hour,a.mean_attention,color="#5B8DB8");ax.set(xlabel="Clock hour",ylabel="Mean attention",title="SameClockFormer learned clock attention");save(fig,"05_sameclock_attention.png")
 pdf=out/"DEMENTIA_TSC_V6_REPORT.pdf"
 with PdfPages(pdf) as pages:
  fig=plt.figure(figsize=(11.69,8.27));fig.text(.06,.88,"Dementia-TSC v6",fontsize=24,weight="bold");fig.text(.06,.79,"End-to-end multivariate time-series classification",fontsize=15);fig.text(.06,.63,f"Locked test N={len(y):,}; cases={int(y.sum())}; prevalence={y.mean():.3%}\nInceptionTime, PatchTST and SameClockFormer; validation-only model selection.",fontsize=12,linespacing=1.6);fig.text(.06,.43,"Neural inference inputs: minute-level activity, observation mask, baseline information and APOE.\nNo handcrafted PA/rhythm phenotype, endpoint field, comorbidity index, BEAT representation or external checkpoint is used.",fontsize=10.5,linespacing=1.6);pages.savefig(fig);plt.close(fig)
  show=metrics[["model","auroc","auroc_ci_low","auroc_ci_high","auprc","auprc_ci_low","auprc_ci_high","brier","sensitivity","specificity","ppv"]].round(4);fig,ax=plt.subplots(figsize=(11.69,8.27));ax.axis("off");tb=ax.table(cellText=show.values,colLabels=show.columns,loc="center",cellLoc="center");tb.auto_set_font_size(False);tb.set_fontsize(7);tb.scale(1,1.5);ax.set_title("Locked-test performance",fontsize=16);pages.savefig(fig,bbox_inches="tight");plt.close(fig)
  for p in figs:fig,ax=plt.subplots(figsize=(11.69,8.27));ax.imshow(plt.imread(p));ax.axis("off");pages.savefig(fig,bbox_inches="tight");plt.close(fig)
 return pdf

def self_test(cfg,models):
 dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 for name in models:
  m=build_model(name,20,cfg,True).to(dev);x=torch.randn(2,2,10080,device=dev);t=torch.randn(2,20,device=dev);s=m(x,t);assert s.shape==(2,);m.project(x).sum().backward();logging.info("Self-test %s OK parameters=%d",name,sum(p.numel() for p in m.parameters()));del m,x,t,s
 if torch.cuda.is_available():torch.cuda.empty_cache()

def run(cfg):
 out=Path(cfg.output_dir);setup_log(out);seed_all(cfg.seed);models=[x.strip().lower() for x in cfg.models.split(",") if x.strip()];allowed={"inception","patchtst","sameclock"}
 if not models or not set(models)<=allowed:raise ValueError(f"models must be subset of {sorted(allowed)}")
 if BLOCKED.intersection(NEURAL_TAB):raise RuntimeError("Static leakage audit failed")
 atomic_json(out/"config.json",asdict(cfg))
 with run_lock(out):
  if (out/"COMPLETED.json").exists() and not cfg.overwrite:raise RuntimeError(f"Completed output exists: {out}")
  logging.info("NO HANDCRAFTED RHYTHM INPUTS. NO BEAT/PRETRAINED CHECKPOINT.");self_test(cfg,models);d=load_cohort(cfg);train,val,test=locked_split(d,cfg,out);y=d.target.to_numpy(np.int8);idx=d.cache_index.to_numpy(np.int64)
  for n,ix in [("train",train),("validation",val),("test",test)]:logging.info("Split %-10s N=%d cases=%d",n,len(ix),y[ix].sum())
  if cfg.mode=="smoke":
   rng=np.random.default_rng(cfg.seed)
   def sub(ix,np_,nn_):z=np.r_[rng.choice(ix[y[ix]==1],min(np_,sum(y[ix]==1)),False),rng.choice(ix[y[ix]==0],min(nn_,sum(y[ix]==0)),False)];rng.shuffle(z);return z
   train,val,test=sub(train,24,96),sub(val,12,48),sub(test,12,48);cfg.ssl_epochs=1;cfg.epochs=2;cfg.min_epochs=1;cfg.patience=1;cfg.batch_size=8;cfg.eval_batch_size=16;cfg.num_workers=0;cfg.bootstrap_repeats=30
  pp,tab=make_tabular(d,train);joblib.dump(pp,out/"neural_baseline_apoe_preprocessor.joblib")
  # Traditional PA is retained only as a non-neural benchmark.
  tab_specs={"clinical_baseline":CLINICAL,"clinical_plus_apoe":CLINICAL+APOE,"clinical_plus_traditional_pa":CLINICAL+TRAD_PA}
  results={};val_y=y[val]
  for name,cols in tab_specs.items():
   r=tabular_model(name,cols,d,y,train,val,test);cal,t=fit_calibrator(val_y,r["val_score"],cfg.target_sensitivity);r["calibrator"]=cal;r["threshold"]=t;r["test_probability"]=cal.predict_proba(r["test_score"][:,None])[:,1];results[name]=r;joblib.dump({"columns":cols,"pipeline":r["pipeline"],"calibrator":cal,"threshold":t},out/(name+".joblib"))
  trained=[];history=[]
  for name in models:
   r=train_one(name,cfg,d,y,idx,train,val,test,tab,out,True);trained.append(r);history.extend(r["history"])
  hist=pd.DataFrame(history);hist.to_csv(out/"training_history.csv",index=False)
  selection=pd.DataFrame([{"model":r["name"],"fusion_validation_auprc":r["validation_auprc"],"raw_pa_validation_auprc":r["validation_raw_auprc"],"best_epoch":r["state"]["epoch"]} for r in trained]).sort_values("fusion_validation_auprc",ascending=False);selection["selected_primary"]=False;selection.iloc[0,selection.columns.get_loc("selected_primary")]=True;selection.to_csv(out/"validation_model_selection.csv",index=False);primary=selection.iloc[0].model;logging.info("Validation-selected primary=%s fusion_AUPRC=%.4f; unlocking test inference now",primary,selection.iloc[0].fusion_validation_auprc)
  # Test inference begins only after the preceding selection table is fixed.
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu");attention=None
  for r in trained:
   ty,ts,detail=predict(r["model"],r["test_loader"],device,cfg.amp);_,raw_ts,_=predict(r["model"],r["test_loader"],device,cfg.amp,raw_only=True);assert np.array_equal(ty,y[test])
   cal,t=fit_calibrator(val_y,r["val_score"],cfg.target_sensitivity);prob=cal.predict_proba(ts[:,None])[:,1];results[r["name"]+"_fusion"]={"name":r["name"]+"_fusion","val_score":r["val_score"],"test_score":ts,"test_probability":prob,"calibrator":cal,"threshold":t}
   raw_cal,raw_t=fit_calibrator(val_y,r["val_raw_score"],cfg.target_sensitivity);raw_prob=raw_cal.predict_proba(raw_ts[:,None])[:,1];results[r["name"]+"_raw_pa"]={"name":r["name"]+"_raw_pa","val_score":r["val_raw_score"],"test_score":raw_ts,"test_probability":raw_prob,"calibrator":raw_cal,"threshold":raw_t}
   joblib.dump(cal,out/(r["name"]+"_fusion_calibrator.joblib"));joblib.dump(raw_cal,out/(r["name"]+"_raw_pa_calibrator.joblib"))
   if r["name"]=="sameclock" and detail:attention=detail
   # Save deployable package after inference; no manual phenotype dependency.
   torch.save({"model":r["model"].state_dict(),"architecture":r["name"],"tab_dim":tab.shape[1],"raw_scaler":{"location":r["state"]["loc"],"scale":r["state"]["scale"]},"best_epoch":r["state"]["epoch"],"config":r["state"]["config"]},out/(r["name"]+"_deployable.pt"))
  rows=[bootstrap(metric_row(n,y[test],r["test_probability"],r["threshold"]),y[test],r["test_probability"],cfg.bootstrap_repeats,cfg.seed+i) for i,(n,r) in enumerate(results.items())];metrics=pd.DataFrame(rows).sort_values("auprc",ascending=False);metrics.to_csv(out/"locked_test_metrics.csv",index=False)
  pred=pd.DataFrame({"eid":d.iloc[test].eid.to_numpy(),"target":y[test]});
  for n,r in results.items():pred["probability_"+n]=r["test_probability"]
  pred.to_csv(out/"locked_test_predictions.csv",index=False);top_risk(out,y[test],results)
  if attention is not None:
   pd.DataFrame({"day":np.arange(1,8),"mean_attention":attention["day_attention"].mean(0)}).to_csv(out/"sameclock_attention_day.csv",index=False)
   pd.DataFrame({"clock_hour":np.arange(288)/12.,"mean_attention":attention["clock_attention"].mean(0)}).to_csv(out/"sameclock_attention_clock.csv",index=False)
  pdf=report(out,y[test],results,metrics,hist)
  audit={"task":"multivariate time-series classification","fixed_horizon_years":cfg.horizon_years,"landmark_days":cfg.landmark_days,"new_v6_locked_split":True,"test_access":"after validation-only architecture/epoch selection","candidate_models":models,"selected_primary":primary,"dual_head":"Every encoder reports raw-PA-only and PA+baseline/APOE fusion from shared weights","task_internal_ssl":"training participants only; two masked/jittered views; NT-Xent","neural_time_series_channels":["log-scaled minute activity","observation mask"],"neural_tabular_predictors":NEURAL_TAB,"handcrafted_rhythm_predictors":[],"traditional_pa_role":"benchmark only, never neural input","pretraining_used":False,"BEAT_used":False,"blocked_predictors":sorted(BLOCKED),"warning":"Do not tune v7 against these v6 locked-test results."}
  atomic_json(out/"design_and_leakage_audit.json",audit);atomic_json(out/"COMPLETED.json",{"completed":True,"time":datetime.now().isoformat(),"validation_selected_primary":primary,"best_test_auprc_model_descriptive_only":metrics.iloc[0].model,"report":pdf.name});logging.info("Completed v6. Validation primary=%s; descriptive best locked-test=%s AUPRC=%.4f",primary,metrics.iloc[0].model,metrics.iloc[0].auprc)

def parse_args():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument("--mode",choices=["smoke","full"],default="full");a.add_argument("--labels-csv",type=Path,default=DEFAULT_LABELS);a.add_argument("--apoe-csv",type=Path,default=DEFAULT_APOE);a.add_argument("--cache-dir",type=Path,default=DEFAULT_CACHE);a.add_argument("--output-dir",type=Path,default=DEFAULT_OUT);a.add_argument("--horizon-years",type=float,default=7.);a.add_argument("--models",default="inception,patchtst,sameclock");a.add_argument("--ssl-epochs",type=int,default=2);a.add_argument("--epochs",type=int,default=12);a.add_argument("--batch-size",type=int,default=64);a.add_argument("--num-workers",type=int,default=6);a.add_argument("--bootstrap-repeats",type=int,default=1000);a.add_argument("--seed",type=int,default=20260815);a.add_argument("--overwrite",action="store_true");z=a.parse_args()
 return Config(str(z.labels_csv),str(z.apoe_csv),str(z.cache_dir),str(z.output_dir),horizon_years=z.horizon_years,models=z.models,ssl_epochs=z.ssl_epochs,epochs=z.epochs,batch_size=z.batch_size,num_workers=z.num_workers,bootstrap_repeats=z.bootstrap_repeats,seed=z.seed,mode=z.mode,overwrite=z.overwrite)
if __name__=="__main__":run(parse_args())

