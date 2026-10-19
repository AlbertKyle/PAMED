#!/usr/bin/env python3
"""Cohort loading, observed-week preprocessing and shared model utilities."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA_ROOT, RESULTS_ROOT

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from torch.utils.data import Dataset

BASE=DATA_ROOT.parent
DEM=DATA_ROOT
DEFAULT_LABELS=DEM/"ukb_dementia_labels_covariates_N80990.csv"
DEFAULT_APOE=DEM/"APOE_phenotype.csv"
DEFAULT_CACHE=DEM/"cache"
DEFAULT_OUT=RESULTS_ROOT/"02_core_model"

CLIN_CONT=["enrollment_age"]
CLIN_CAT=["gender"]
CLINICAL=CLIN_CONT+CLIN_CAT
TRAD_PA=["Light_Total_Hour","MVPA_Total_Hour","Sedentary_Total_Hour","Sleep_Total_Hour"]
APOE=["apoe_e4"]
NEURAL_TAB=CLINICAL
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
 labels_csv:str
 apoe_csv:str
 cache_dir:str
 output_dir:str
 horizon_years:float=7.0
 landmark_days:int=365


def atomic_json(path,obj):
 tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf-8");os.replace(tmp,path)
def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
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
 from study_design import fixed_window_cohort
 d=fixed_window_cohort(d,cfg.horizon_years,cfg.landmark_days)
 ap=pd.read_csv(cfg.apoe_csv,dtype={"eid":"string"},low_memory=False);ap["eid"]=ap.eid.str.strip();ap["apoe_e4"]=pd.to_numeric(ap.e4,errors="coerce");ap.loc[~ap.apoe_e4.isin([0,1,2]),"apoe_e4"]=np.nan
 d=d.merge(ap[["eid","apoe_e4"]],on="eid",how="left",validate="one_to_one").reset_index(drop=True)
 if BLOCKED.intersection(NEURAL_TAB):raise RuntimeError("Outcome/follow-up leakage in predictors")
 missing=sorted(set(NEURAL_TAB+TRAD_PA)-set(d.columns))
 if missing:raise ValueError(f"Missing predictors: {missing}")
 cache=Path(cfg.cache_dir);req=[cache/"activity_7d.npy",cache/"observed_7d.npy",cache/"eids.npy"]
 if not all(p.exists() for p in req):raise FileNotFoundError(f"Raw cache incomplete: {cache}")
 ce=np.load(cache/"eids.npy",mmap_mode="r").astype(str);lookup=pd.Series(np.arange(len(ce)),index=ce);d["cache_index"]=d.eid.map(lookup)
 if d.cache_index.isna().any() or d.eid.duplicated().any():raise ValueError("Cohort/cache EID alignment failed")
 logging.info("Fixed-horizon cohort N=%d cases=%d prevalence=%.5f APOE=%d",len(d),d.target.sum(),d.target.mean(),d.apoe_e4.notna().sum())
 return d


def ohe():
 try:return OneHotEncoder(handle_unknown="ignore",sparse_output=False)
 except TypeError:return OneHotEncoder(handle_unknown="ignore",sparse=False)
def tab_preprocessor(cols):
 cats=[c for c in cols if c in CLIN_CAT+APOE];nums=[c for c in cols if c not in cats];parts=[]
 if nums:parts.append(("num",Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),("sc",RobustScaler())]),nums))
 if cats:parts.append(("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent",add_indicator=True)),("ohe",ohe())]),cats))
 return ColumnTransformer(parts,sparse_threshold=0)
def logit(p):p=np.clip(np.asarray(p,float),1e-6,1-1e-6);return np.log(p/(1-p))
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-30,30)))

def robust_raw_scale(cache,cache_index,train,seed):
 x=np.load(Path(cache)/"activity_7d.npy",mmap_mode="r");m=np.load(Path(cache)/"observed_7d.npy",mmap_mode="r");rng=np.random.default_rng(seed);vals=[]
 for j in rng.choice(train,min(2000,len(train)),False):
  a=np.asarray(x[cache_index[j]],np.float32);o=np.asarray(m[cache_index[j]],bool);vals.append(np.log1p(np.clip(a[o],0,None)))
 v=np.concatenate(vals);return float(np.median(v)),float(max(np.quantile(v,.75)-np.quantile(v,.25),.1))

class WeekDataset(Dataset):
 def __init__(self,cache,cache_index,y,tab,loc,scale,augment=False):
  self.x=np.load(Path(cache)/"activity_7d.npy",mmap_mode="r");self.m=np.load(Path(cache)/"observed_7d.npy",mmap_mode="r")
  self.i=np.asarray(cache_index,np.int64);self.y=np.asarray(y,np.float32);self.t=np.asarray(tab,np.float32);self.loc=float(loc);self.scale=float(scale);self.augment=augment;self.rng=None
 def __len__(self):return len(self.i)
 def base(self,j):
  a=np.asarray(self.x[self.i[j]],np.float32);m=np.asarray(self.m[self.i[j]],bool);z=(np.log1p(np.clip(np.nan_to_num(a,nan=0.),0,None))-self.loc)/self.scale;z=np.where(m,z,0.)
  return np.stack([z.reshape(-1),m.astype(np.float32).reshape(-1)],0).astype(np.float32)
 def view(self,x):
  x=x.copy();length=x.shape[-1]
  if self.rng is None:self.rng=np.random.default_rng(torch.initial_seed() % (2**32))
  rng=self.rng
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


def nt_xent(a,b,temp):
 n=a.shape[0];z=torch.cat([a,b],0);sim=z@z.T/temp;sim.fill_diagonal_(-1e4);target=torch.arange(2*n,device=z.device);target=(target+n)%(2*n);return F.cross_entropy(sim,target)


