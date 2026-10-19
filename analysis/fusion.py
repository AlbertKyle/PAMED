#!/usr/bin/env python3
"""Activity representation fusion, demographic interaction and APOE extension."""
from __future__ import annotations
import argparse, json, logging, os, random, sys, time
from datetime import datetime
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA_ROOT, RESULTS_ROOT

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
import torch.nn as nn
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import cohort

BASE=DATA_ROOT.parent
DEM=DATA_ROOT
AGE_SEX=["enrollment_age","gender"]

def atomic_json(path,obj):
 tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf-8");os.replace(tmp,path)


def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

class TabTower(nn.Module):
 def __init__(self,n,d=64,drop=.20):super().__init__();self.net=nn.Sequential(nn.Linear(n,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,d),nn.LayerNorm(d),nn.GELU())
 def forward(self,x):return self.net(x)

class DualPAFeature(nn.Module):
 def __init__(self,d=64,drop=.20):
  super().__init__();self.g=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU());self.c=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU());self.gate=nn.Sequential(nn.Linear(2*d,d),nn.Sigmoid());self.head=nn.Sequential(nn.Linear(4*d,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,1))
 def forward(self,g,c,gl,cl,t,e):
  g=self.g(g);c=self.c(c);w=self.gate(torch.cat([g,c],1));z=w*g+(1-w)*c;return self.head(torch.cat([z,g*c,torch.abs(g-c),g+c],1)).squeeze(-1)

class DualPADecision(nn.Module):
 def __init__(self):super().__init__();self.weight=nn.Parameter(torch.zeros(2));self.scale=nn.Parameter(torch.ones(2));self.bias=nn.Parameter(torch.zeros(()))
 def forward(self,g,c,gl,cl,t,e):w=torch.softmax(self.weight,0);return w[0]*self.scale[0]*gl+w[1]*self.scale[1]*cl+self.bias

class ClinicalFeature(nn.Module):
 def __init__(self,tab_dim,d=64,drop=.20):
  super().__init__();self.p=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU());self.t=TabTower(tab_dim,d,drop);self.head=nn.Sequential(nn.Linear(4*d,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,1))
 def forward(self,g,c,gl,cl,t,e):p=self.p(c);q=self.t(t);return self.head(torch.cat([p,q,p*q,torch.abs(p-q)],1)).squeeze(-1)

class ClinicalDecision(nn.Module):
 def __init__(self,tab_dim,d=64,drop=.20):super().__init__();self.t=TabTower(tab_dim,d,drop);self.clin=nn.Linear(d,1);self.weight=nn.Parameter(torch.zeros(2));self.scale=nn.Parameter(torch.ones(2));self.bias=nn.Parameter(torch.zeros(()))
 def forward(self,g,c,gl,cl,t,e):q=self.clin(self.t(t)).squeeze(-1);w=torch.softmax(self.weight,0);return w[0]*self.scale[0]*cl+w[1]*self.scale[1]*q+self.bias

class HybridFusion(nn.Module):
 def __init__(self,tab_dim,d=64,drop=.20):
  super().__init__();self.g=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU());self.c=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU());self.pg=nn.Sequential(nn.Linear(2*d,d),nn.Sigmoid());self.t=TabTower(tab_dim,d,drop);self.inter=nn.Sequential(nn.Linear(4*d,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,1));self.residual_weight=nn.Parameter(torch.zeros(3));self.residual_scale=nn.Parameter(torch.tensor(-1.0));self.clin=nn.Linear(d,1)
 def pa(self,g,c):g=self.g(g);c=self.c(c);w=self.pg(torch.cat([g,c],1));return w*g+(1-w)*c
 def forward(self,g,c,gl,cl,t,e):
  z=self.pa(g,c);q=self.t(t);interaction=self.inter(torch.cat([z,q,z*q,torch.abs(z-q)],1)).squeeze(-1);cw=self.clin(q).squeeze(-1);w=torch.softmax(self.residual_weight,0);res=w[0]*gl+w[1]*cl+w[2]*cw;return interaction+torch.sigmoid(self.residual_scale)*res

class APOEHybrid(HybridFusion):
 def __init__(self,tab_dim,d=64,drop=.20):super().__init__(tab_dim,d,drop);self.a=nn.Sequential(nn.Linear(1,16),nn.GELU(),nn.Linear(16,d),nn.LayerNorm(d),nn.GELU());self.apoe_head=nn.Linear(d,1);self.apoe_gate=nn.Parameter(torch.tensor(-1.0));self.apoe_inter=nn.Sequential(nn.Linear(3*d,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,1))
 def forward(self,g,c,gl,cl,t,e):
  base=super().forward(g,c,gl,cl,t,e);z=self.pa(g,c);a=self.a(e);q=self.t(t);return base+self.apoe_inter(torch.cat([z,a,z*a],1)).squeeze(-1)+torch.sigmoid(self.apoe_gate)*self.apoe_head(a).squeeze(-1)


def tensor_dataset(z,tab,e4,select=None):
 ix=np.arange(len(z["y"])) if select is None else np.asarray(select);return TensorDataset(torch.from_numpy(z["g"][ix]),torch.from_numpy(z["c"][ix]),torch.from_numpy(z["gl"][ix]),torch.from_numpy(z["cl"][ix]),torch.from_numpy(tab[ix]),torch.from_numpy(e4[ix]),torch.from_numpy(z["y"][ix]))

def infer(model,ds,device,batch):
 model.eval();ys=[];ss=[]
 with torch.no_grad():
  for q in DataLoader(ds,batch,shuffle=False,num_workers=0):
   *x,y=q;x=[a.to(device) for a in x];s=model(*x);ys.append(y.numpy());ss.append(s.float().cpu().numpy())
 return np.concatenate(ys),np.concatenate(ss)

def train_head(name,model,tr_ds,va_ds,cfg,out,seed):
 seed_all(seed);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model=model.to(device);y=tr_ds.tensors[-1].numpy();pos=max(y.sum(),1);neg=max((y==0).sum(),1);q=cfg.sampled_positive_fraction;w=np.where(y==1,q/pos,(1-q)/neg);gen=torch.Generator().manual_seed(seed);sampler=WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),len(y),True,generator=gen);loader=DataLoader(tr_ds,cfg.fusion_batch_size,sampler=sampler,num_workers=0);opt=torch.optim.AdamW(model.parameters(),lr=cfg.fusion_lr,weight_decay=cfg.fusion_weight_decay);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg.fusion_epochs,eta_min=cfg.fusion_lr*.05);lossfn=nn.BCEWithLogitsLoss();best=-1.;stale=0;hist=[];path=out/f"{name}_best.pt";t0=time.time()
 for ep in range(1,cfg.fusion_epochs+1):
  model.train();ls=[]
  for qv in loader:
   *x,yy=qv;x=[a.to(device) for a in x];yy=yy.to(device);opt.zero_grad(set_to_none=True);loss=lossfn(model(*x),yy);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();ls.append(float(loss.detach().cpu()))
  sch.step();vy,vs=infer(model,va_ds,device,cfg.eval_batch_size);ap=average_precision_score(vy,vs);auc=roc_auc_score(vy,vs);hist.append({"model":name,"epoch":ep,"loss":np.mean(ls),"validation_auprc":ap,"validation_auroc":auc})
  if ep==1 or ep%5==0:logging.info("%-28s epoch=%d loss=%.4f val_AUPRC=%.4f val_AUROC=%.4f",name,ep,np.mean(ls),ap,auc)
  if ap>best+1e-5:best=ap;stale=0;torch.save({"model":model.state_dict(),"epoch":ep,"seed":seed},path)
  else:stale+=1
  if ep>=cfg.fusion_min_epochs and stale>=cfg.fusion_patience:break
 state=torch.load(path,map_location=device);model.load_state_dict(state["model"]);vy,vs=infer(model,va_ds,device,cfg.eval_batch_size);logging.info("BEST %-24s epoch=%d AUPRC=%.4f AUROC=%.4f",name,state["epoch"],average_precision_score(vy,vs),roc_auc_score(vy,vs));return vs,state["epoch"],hist,sum(p.numel() for p in model.parameters()),time.time()-t0


