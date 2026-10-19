#!/usr/bin/env python3
"""Convolutional, recurrent and patch-based activity sequence comparators."""
from __future__ import annotations

import argparse, json, logging, random, sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA_ROOT, RESULTS_ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

import cohort

BASE=DATA_ROOT.parent
DEM=DATA_ROOT
DEFAULT_OUT=RESULTS_ROOT/"01_algorithm_pool"
ARCHS=["fcn","resnet1d","tcn","xceptiontime","grufcn","inception","patchtst","sameclock"]

@dataclass
class Config:
 labels_csv:str=str(cohort.DEFAULT_LABELS);apoe_csv:str=str(cohort.DEFAULT_APOE)
 cache_dir:str=str(cohort.DEFAULT_CACHE)
 output_dir:str=str(DEFAULT_OUT);architectures:str=",".join(ARCHS)
 horizon_years:float=7.;repeats:int=2;validation_fraction:float=.20
 ssl_epochs:int=1;epochs:int=8;min_epochs:int=2;patience:int=2
 batch_size:int=64;eval_batch_size:int=128;num_workers:int=6
 lr:float=1e-4;ssl_lr:float=2e-4;weight_decay:float=1e-3
 dropout:float=.30;d_model:int=64;sampled_positive_fraction:float=.08
 temperature:float=.15;seed:int=20260817;amp:bool=True
 mode:str="full";overwrite:bool=False

def seed_all(seed):cohort.seed_all(seed)
def setup(out):
 out.mkdir(parents=True,exist_ok=True);logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",handlers=[logging.FileHandler(out/"run.log",encoding="utf-8"),logging.StreamHandler(sys.stdout)],force=True)


class ConvBNAct(nn.Module):
 def __init__(self,ni,nf,k,stride=1,dilation=1,groups=1):super().__init__();self.net=nn.Sequential(nn.Conv1d(ni,nf,k,stride,padding=(k//2)*dilation,dilation=dilation,groups=groups,bias=False),nn.BatchNorm1d(nf),nn.GELU())
 def forward(self,x):return self.net(x)

class FCNEncoder(nn.Module):
 def __init__(self,d=64,drop=.3):super().__init__();self.net=nn.Sequential(ConvBNAct(2,64,9,5),ConvBNAct(64,128,7,2),ConvBNAct(128,128,5,2),nn.AdaptiveAvgPool1d(1));self.head=nn.Sequential(nn.Flatten(),nn.Linear(128,d),nn.Dropout(drop));self.output_dim=d
 def forward(self,x):return self.head(self.net(x))

class ResBlock(nn.Module):
 def __init__(self,ni,nf,k=7,stride=1):super().__init__();self.a=ConvBNAct(ni,nf,k,stride);self.b=nn.Sequential(nn.Conv1d(nf,nf,k,padding=k//2,bias=False),nn.BatchNorm1d(nf));self.skip=nn.Sequential(nn.Conv1d(ni,nf,1,stride,bias=False),nn.BatchNorm1d(nf)) if ni!=nf or stride!=1 else nn.Identity()
 def forward(self,x):return F.gelu(self.b(self.a(x))+self.skip(x))
class ResNet1DEncoder(nn.Module):
 def __init__(self,d=64,drop=.3):super().__init__();self.net=nn.Sequential(ConvBNAct(2,32,11,5),ResBlock(32,64,7,2),ResBlock(64,64),ResBlock(64,128,5,2),ResBlock(128,128));self.head=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(128,d),nn.LayerNorm(d),nn.Dropout(drop));self.output_dim=d
 def forward(self,x):return self.head(self.net(x))

class TCNBlock(nn.Module):
 def __init__(self,d,dil,drop):super().__init__();self.net=nn.Sequential(nn.Conv1d(d,d,3,padding=dil,dilation=dil),nn.BatchNorm1d(d),nn.GELU(),nn.Dropout(drop),nn.Conv1d(d,d,3,padding=dil,dilation=dil),nn.BatchNorm1d(d),nn.GELU())
 def forward(self,x):return x+self.net(x)
class TCNEncoder(nn.Module):
 def __init__(self,d=64,drop=.3):super().__init__();self.stem=ConvBNAct(2,d,9,5);self.blocks=nn.Sequential(*[TCNBlock(d,q,drop) for q in [1,2,4,8,16,32]]);self.head=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.LayerNorm(d));self.output_dim=d
 def forward(self,x):return self.head(self.blocks(self.stem(x)))

class SepBlock(nn.Module):
 def __init__(self,ni,nf,k,stride=1):super().__init__();self.net=nn.Sequential(nn.Conv1d(ni,ni,k,stride,padding=k//2,groups=ni,bias=False),nn.Conv1d(ni,nf,1,bias=False),nn.BatchNorm1d(nf),nn.GELU())
 def forward(self,x):return self.net(x)
class XceptionTimeEncoder(nn.Module):
 def __init__(self,d=64,drop=.3):super().__init__();self.stem=ConvBNAct(2,32,9,5);self.branches=nn.ModuleList([nn.Sequential(SepBlock(32,32,k),SepBlock(32,32,k)) for k in [7,15,31]]);self.mix=ConvBNAct(96,128,1);self.head=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(128,d),nn.LayerNorm(d),nn.Dropout(drop));self.output_dim=d
 def forward(self,x):x=self.stem(x);return self.head(self.mix(torch.cat([b(x) for b in self.branches],1)))

class GRUFCNEncoder(nn.Module):
 def __init__(self,d=64,drop=.3):
  super().__init__();self.down=nn.Sequential(ConvBNAct(2,32,9,10),ConvBNAct(32,32,5,2));self.gru=nn.GRU(32,d//2,2,batch_first=True,bidirectional=True,dropout=drop);self.fcn=nn.Sequential(ConvBNAct(32,64,7),ConvBNAct(64,64,5),nn.AdaptiveAvgPool1d(1));self.head=nn.Sequential(nn.Linear(d+64,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop));self.output_dim=d
 def forward(self,x):h=self.down(x);_,s=self.gru(h.transpose(1,2));r=torch.cat([s[-2],s[-1]],-1);c=self.fcn(h).squeeze(-1);return self.head(torch.cat([r,c],-1))

class RawClassifier(nn.Module):
 def __init__(self,encoder,d=64):super().__init__();self.encoder=encoder;self.risk=nn.Linear(d,1);self.proj=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,32))
 def encode(self,x):return self.encoder(x)
 def forward(self,x):return self.risk(self.encode(x)).squeeze(-1)
 def project(self,x):return F.normalize(self.proj(self.encode(x)),dim=-1)

def build(name,cfg):
 if name=="fcn":e=FCNEncoder(cfg.d_model,cfg.dropout)
 elif name=="resnet1d":e=ResNet1DEncoder(cfg.d_model,cfg.dropout)
 elif name=="tcn":e=TCNEncoder(cfg.d_model,cfg.dropout)
 elif name=="xceptiontime":e=XceptionTimeEncoder(cfg.d_model,cfg.dropout)
 elif name=="grufcn":e=GRUFCNEncoder(cfg.d_model,cfg.dropout)
 elif name=="inception":e=cohort.InceptionEncoder(cfg.d_model,cfg.dropout)
 elif name=="patchtst":e=cohort.PatchTSTEncoder(cfg.d_model,cfg.dropout)
 elif name=="sameclock":e=cohort.SameClockEncoder(cfg.d_model,cfg.dropout)
 else:raise ValueError(name)
 return RawClassifier(e,cfg.d_model)

def make_loaders(cfg,d,y,idx,train,val):
 # Dummy tabular matrix is never supplied to a model; WeekDataset returns it
 # only to reuse the rigorously aligned cohort data loader.
 dummy=np.zeros((len(d),1),np.float32);loc,scale=cohort.robust_raw_scale(cfg.cache_dir,idx,train,cfg.seed)
 ssl_ds=cohort.WeekDataset(cfg.cache_dir,idx[train],y[train],dummy[train],loc,scale,True);tr_ds=cohort.WeekDataset(cfg.cache_dir,idx[train],y[train],dummy[train],loc,scale,False);va_ds=cohort.WeekDataset(cfg.cache_dir,idx[val],y[val],dummy[val],loc,scale,False)
 pos=max(y[train].sum(),1);neg=max((y[train]==0).sum(),1);q=cfg.sampled_positive_fraction;w=np.where(y[train]==1,q/pos,(1-q)/neg);sampler=torch.utils.data.WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),len(train),True)
 nw=0 if cfg.mode=="smoke" else cfg.num_workers;kw={"num_workers":nw,"pin_memory":torch.cuda.is_available(),"persistent_workers":nw>0}
 return torch.utils.data.DataLoader(ssl_ds,cfg.batch_size,shuffle=True,drop_last=True,**kw),torch.utils.data.DataLoader(tr_ds,cfg.batch_size,sampler=sampler,**kw),torch.utils.data.DataLoader(va_ds,cfg.eval_batch_size,shuffle=False,**kw),loc,scale

def predict(model,loader,device,amp):
 model.eval();ys=[];ss=[]
 with torch.no_grad():
  for x,y,_ in loader:
   with cohort.amp_ctx(device,amp):s=model(x.to(device,non_blocking=True))
   ys.append(y.numpy());ss.append(s.float().cpu().numpy())
 return np.concatenate(ys),np.concatenate(ss)

def train_model(name,cfg,d,y,idx,train,val,run_out):
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu");ssl,tr,va,loc,scale=make_loaders(cfg,d,y,idx,train,val);model=build(name,cfg).to(device);logging.info("%s parameters=%d",name,sum(p.numel() for p in model.parameters()))
 if cfg.ssl_epochs:
  opt=torch.optim.AdamW(model.parameters(),lr=cfg.ssl_lr,weight_decay=cfg.weight_decay);gs=cohort.grad_scaler(device,cfg.amp)
  for ep in range(1,cfg.ssl_epochs+1):
   model.train();losses=[]
   for a,b,_,_ in ssl:
    a=a.to(device);b=b.to(device);opt.zero_grad(set_to_none=True)
    with cohort.amp_ctx(device,cfg.amp):loss=cohort.nt_xent(model.project(a),model.project(b),cfg.temperature)
    gs.scale(loss).backward();gs.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);gs.step(opt);gs.update();losses.append(float(loss.detach().cpu()))
   logging.info("%s SSL epoch=%d loss=%.4f",name,ep,np.mean(losses))
 opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay);gs=cohort.grad_scaler(device,cfg.amp);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(cfg.epochs,1),eta_min=cfg.lr*.1);bce=nn.BCEWithLogitsLoss();best=-1;stale=0;hist=[];path=run_out/(name+"_best.pt")
 for ep in range(1,cfg.epochs+1):
  model.train();ls=[]
  for x,yy,_ in tr:
   x=x.to(device);yy=yy.to(device);opt.zero_grad(set_to_none=True)
   with cohort.amp_ctx(device,cfg.amp):loss=bce(model(x),yy)
   gs.scale(loss).backward();gs.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);gs.step(opt);gs.update();ls.append(float(loss.detach().cpu()))
  sch.step();vy,vs=predict(model,va,device,cfg.amp);ap=average_precision_score(vy,vs);auc=roc_auc_score(vy,vs);hist.append({"architecture":name,"epoch":ep,"loss":np.mean(ls),"validation_auprc":ap,"validation_auroc":auc});logging.info("%s epoch=%d val_AUPRC=%.4f val_AUROC=%.4f",name,ep,ap,auc)
  if ap>best+1e-5:best=ap;stale=0;torch.save({"model":model.state_dict(),"architecture":name,"epoch":ep,"loc":loc,"scale":scale,"config":asdict(cfg)},path)
  else:stale+=1
  if ep>=cfg.min_epochs and stale>=cfg.patience:break
 state=torch.load(path,map_location=device);model.load_state_dict(state["model"]);vy,vs=predict(model,va,device,cfg.amp);return {"y":vy,"score":vs,"best_epoch":state["epoch"],"history":hist,"parameters":sum(p.numel() for p in model.parameters())}


