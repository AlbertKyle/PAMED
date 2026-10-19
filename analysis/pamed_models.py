#!/usr/bin/env python3
"""GRU-FCN and selective state-space encoders for wearable activity."""
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
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
 import mamba_ssm
 from mamba_ssm import Mamba
except Exception as exc:
 raise SystemExit("The mamba-ssm package is required. Import error: "+repr(exc))

import cohort
import temporal_models as arena

BASE=DATA_ROOT.parent
DEM=DATA_ROOT;DEFAULT_OUT=RESULTS_ROOT/"02_core_model"
MODELS=["grufcn","mamba_fcn","bimambasl_fcn","convcontext_mamba","chronoview_mamba_fcn","gru_mamba_fcn"]

def atomic_json(path,obj):
 tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf-8");os.replace(tmp,path)
def setup(out):
 out.mkdir(parents=True,exist_ok=True);logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",handlers=[logging.FileHandler(out/"run.log",encoding="utf-8"),logging.StreamHandler(sys.stdout)],force=True)
def seed_all(seed):cohort.seed_all(seed);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


class MultiHeadPool(nn.Module):
 def __init__(self,d,heads=4):super().__init__();self.score=nn.Linear(d,heads);self.heads=heads;self.out=nn.Sequential(nn.Linear(d*heads,d),nn.LayerNorm(d),nn.GELU())
 def forward(self,x,valid=None):
  s=self.score(x).transpose(1,2)
  if valid is not None:s=s.masked_fill(~valid[:,None,:],-1e4)
  w=torch.softmax(s,-1);z=torch.einsum("bhl,bld->bhd",w,x);return self.out(z.flatten(1))

class BiMamba(nn.Module):
 def __init__(self,d,state=16,conv=4,expand=2):super().__init__();self.f=Mamba(d_model=d,d_state=state,d_conv=conv,expand=expand);self.r=Mamba(d_model=d,d_state=state,d_conv=conv,expand=expand);self.mix=nn.Sequential(nn.Linear(2*d,d),nn.LayerNorm(d),nn.GELU())
 def forward(self,x):return self.mix(torch.cat([self.f(x),torch.flip(self.r(torch.flip(x,[1])),[1])],-1))

class ExactLocal(nn.Module):
 """Downsampling and FCN branches shared with the GRU-FCN encoder."""
 def __init__(self,d=64,drop=.3):
  super().__init__();self.down=nn.Sequential(arena.ConvBNAct(2,32,9,10),arena.ConvBNAct(32,32,5,2));self.fcn=nn.Sequential(arena.ConvBNAct(32,64,7),arena.ConvBNAct(64,64,5),nn.AdaptiveAvgPool1d(1));self.local_head=nn.Sequential(nn.Linear(64,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop))
 def tokens(self,x):return self.down(x).transpose(1,2)
 def local(self,t):return self.local_head(self.fcn(t.transpose(1,2)).squeeze(-1))

class MambaFCN(nn.Module):
 def __init__(self,d=64,drop=.3):super().__init__();self.core=ExactLocal(d,drop);self.m=Mamba(d_model=32,d_state=16,d_conv=4,expand=2);self.pool=MultiHeadPool(32,4);self.head=nn.Sequential(nn.Linear(d+32,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop));self.output_dim=d
 def forward(self,x):t=self.core.tokens(x);return self.head(torch.cat([self.core.local(t),self.pool(self.m(t))],-1))

class BiMambaSLFCN(nn.Module):
 """Single-layer bidirectional Mamba with wide TSC input projection."""
 def __init__(self,d=64,drop=.3):
  super().__init__();self.patch=nn.Sequential(nn.Conv1d(2,d,61,stride=30,padding=30,bias=False),nn.BatchNorm1d(d),nn.GELU());self.m=BiMamba(d,16,4,2);self.pool=MultiHeadPool(d,4);self.local=ExactLocal(d,drop);self.head=nn.Sequential(nn.Linear(2*d,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop));self.output_dim=d
 def forward(self,x):p=self.patch(x).transpose(1,2);t=self.local.tokens(x);return self.head(torch.cat([self.pool(self.m(p)),self.local.local(t)],-1))

class ConvContextMamba(nn.Module):
 """DeepConvContext idea: convolution within 168 hourly windows, Mamba across them."""
 def __init__(self,d=64,drop=.3):
  super().__init__();self.intra=nn.Sequential(nn.Conv1d(2,32,9,padding=4,bias=False),nn.BatchNorm1d(32),nn.GELU(),nn.Conv1d(32,d,7,stride=3,padding=3,bias=False),nn.BatchNorm1d(d),nn.GELU(),nn.AdaptiveAvgPool1d(1));self.inter=BiMamba(d,16,4,2);self.pool=MultiHeadPool(d,4);self.out=nn.Sequential(nn.LayerNorm(d),nn.Dropout(drop));self.output_dim=d
 def forward(self,x):
  b=x.shape[0];hour=x.reshape(b,2,168,60).permute(0,2,1,3).reshape(b*168,2,60);token=self.intra(hour).squeeze(-1).reshape(b,168,-1);valid=hour[:,1].mean(-1).reshape(b,168).gt(.2);return self.out(self.pool(self.inter(token),valid))

def halfhour(x):
 b=x.shape[0];q=x.reshape(b,2,7,48,30);m=q[:,1].clamp(0,1);a=(q[:,0]*m).sum(-1)/m.sum(-1).clamp_min(1.);return torch.stack([a,m.mean(-1)],-1)

class ChronoViewMambaFCN(nn.Module):
 """Minute/local, aligned clock and seven-day Mamba views plus FCN."""
 def __init__(self,d=64,drop=.3):
  super().__init__();q=d//2;self.local_patch=nn.Sequential(nn.Conv1d(2,q,61,stride=30,padding=30),nn.GELU());self.local_m=BiMamba(q,12,4,2);self.local_pool=MultiHeadPool(q,2);self.clock_proj=nn.Sequential(nn.Linear(3,q),nn.LayerNorm(q),nn.GELU());self.clock_m=BiMamba(q,12,3,2);self.clock_pool=MultiHeadPool(q,2);self.day_proj=nn.Sequential(nn.Linear(2,q),nn.LayerNorm(q),nn.GELU());self.day_m=BiMamba(q,8,3,2);self.day_pool=MultiHeadPool(q,2);self.fcn=ExactLocal(d,drop);self.head=nn.Sequential(nn.Linear(d+3*q,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop));self.output_dim=d
 def forward(self,x):
  b=x.shape[0];local=self.local_pool(self.local_m(self.local_patch(x).transpose(1,2)));z=halfhour(x);support=z[...,1];den=support.sum(1).clamp_min(1.);clock_mean=(z[...,0]*support).sum(1)/den;clock_var=((z[...,0]-clock_mean[:,None])**2*support).sum(1)/den;clock=self.clock_pool(self.clock_m(self.clock_proj(torch.stack([clock_mean,torch.sqrt(clock_var.clamp_min(1e-4)),support.mean(1)],-1))),support.mean(1).gt(.2));day=self.day_pool(self.day_m(self.day_proj(torch.stack([(z[...,0]*support).sum(2)/support.sum(2).clamp_min(1.),support.mean(2)],-1))),support.mean(2).gt(.2));t=self.fcn.tokens(x);return self.head(torch.cat([self.fcn.local(t),local,clock,day],-1))

class GRUMambaFCN(nn.Module):
 def __init__(self,d=64,drop=.3):
  super().__init__();self.core=ExactLocal(d,drop);self.gru=nn.GRU(32,d//2,2,batch_first=True,bidirectional=True,dropout=drop);self.m=BiMamba(32,16,4,2);self.mp=MultiHeadPool(32,4);self.weight=nn.Sequential(nn.Linear(d+32+d,3),nn.Softmax(-1));self.gproj=nn.Linear(d,d);self.mproj=nn.Linear(32,d);self.output_dim=d;self.last_weights=None
 def forward(self,x):
  t=self.core.tokens(x);_,s=self.gru(t);g=torch.cat([s[-2],s[-1]],-1);m=self.mp(self.m(t));f=self.core.local(t);w=self.weight(torch.cat([g,m,f],-1));self.last_weights=w.detach();return w[:,0:1]*self.gproj(g)+w[:,1:2]*self.mproj(m)+w[:,2:3]*f

class Classifier(nn.Module):
 def __init__(self,encoder,d=64):super().__init__();self.encoder=encoder;self.risk=nn.Linear(d,1);self.proj=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,32))
 def forward(self,x):return self.risk(self.encoder(x)).squeeze(-1)
 def project(self,x):return F.normalize(self.proj(self.encoder(x)),dim=-1)

def build(name,cfg):
 if name=="grufcn":e=arena.GRUFCNEncoder(cfg.d_model,cfg.dropout)
 elif name=="mamba_fcn":e=MambaFCN(cfg.d_model,cfg.dropout)
 elif name=="bimambasl_fcn":e=BiMambaSLFCN(cfg.d_model,cfg.dropout)
 elif name=="convcontext_mamba":e=ConvContextMamba(cfg.d_model,cfg.dropout)
 elif name=="chronoview_mamba_fcn":e=ChronoViewMambaFCN(cfg.d_model,cfg.dropout)
 elif name=="gru_mamba_fcn":e=GRUMambaFCN(cfg.d_model,cfg.dropout)
 else:raise ValueError(name)
 return Classifier(e,cfg.d_model)

def loaders(cfg,d,y,train,val):
 idx=d.cache_index.to_numpy(int);dummy=np.zeros((len(d),1),np.float32);loc,scale=cohort.robust_raw_scale(cfg.cache_dir,idx,train,cfg.seed);ssl=cohort.WeekDataset(cfg.cache_dir,idx[train],y[train],dummy[train],loc,scale,True);tr=cohort.WeekDataset(cfg.cache_dir,idx[train],y[train],dummy[train],loc,scale,False);va=cohort.WeekDataset(cfg.cache_dir,idx[val],y[val],dummy[val],loc,scale,False);pos=max(y[train].sum(),1);neg=max((y[train]==0).sum(),1);q=cfg.sampled_positive_fraction;w=np.where(y[train]==1,q/pos,(1-q)/neg);sampler=WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),len(train),True);nw=0 if cfg.mode=="smoke" else cfg.num_workers;kw={"num_workers":nw,"pin_memory":torch.cuda.is_available(),"persistent_workers":nw>0};return DataLoader(ssl,cfg.batch_size,shuffle=True,drop_last=True,**kw),DataLoader(tr,cfg.batch_size,sampler=sampler,**kw),DataLoader(va,cfg.eval_batch_size,shuffle=False,**kw),loc,scale

def predict(model,loader,device,amp):
 model.eval();ys=[];ss=[]
 with torch.no_grad():
  for x,y,_ in loader:
   with cohort.amp_ctx(device,amp):s=model(x.to(device))
   ys.append(y.numpy());ss.append(s.float().cpu().numpy())
 y=np.concatenate(ys);s=np.concatenate(ss)
 if not np.isfinite(s).all():raise FloatingPointError("Non-finite scores")
 return y,s

def train_one(name,cfg,d,y,train,val,out,seed):
 seed_all(seed);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");ssl,tr,va,loc,scale=loaders(cfg,d,y,train,val);model=build(name,cfg).to(device);npar=sum(p.numel() for p in model.parameters());logging.info("Training %-23s seed=%d parameters=%d",name,seed,npar);torch.cuda.reset_peak_memory_stats() if device.type=="cuda" else None;t0=time.time()
 if cfg.ssl_epochs:
  opt=torch.optim.AdamW(model.parameters(),lr=cfg.ssl_lr,weight_decay=cfg.weight_decay);gs=cohort.grad_scaler(device,cfg.amp)
  for ep in range(1,cfg.ssl_epochs+1):
   model.train();ls=[]
   for a,b,_,_ in ssl:
    a=a.to(device);b=b.to(device);opt.zero_grad(set_to_none=True)
    with cohort.amp_ctx(device,cfg.amp):loss=cohort.nt_xent(model.project(a),model.project(b),cfg.temperature)
    if not torch.isfinite(loss):raise FloatingPointError(f"{name} SSL NaN")
    gs.scale(loss).backward();gs.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);gs.step(opt);gs.update();ls.append(float(loss.detach().cpu()))
   logging.info("%s SSL epoch=%d loss=%.4f",name,ep,np.mean(ls))
 opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay);gs=cohort.grad_scaler(device,cfg.amp);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(cfg.cosine_epochs,1),eta_min=cfg.lr*.1);bce=nn.BCEWithLogitsLoss();hist=[];best=-1.;stale=0;path=out/f"{name}_best.pt"
 for ep in range(1,cfg.epochs+1):
  model.train();ls=[];lr_used=opt.param_groups[0]["lr"]
  for x,yy,_ in tr:
   x=x.to(device);yy=yy.to(device);opt.zero_grad(set_to_none=True)
   with cohort.amp_ctx(device,cfg.amp):loss=bce(model(x),yy)
   if not torch.isfinite(loss):raise FloatingPointError(f"{name} training NaN")
   gs.scale(loss).backward();gs.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);gs.step(opt);gs.update();ls.append(float(loss.detach().cpu()))
  if ep<=cfg.cosine_epochs:sch.step()
  vy,vs=predict(model,va,device,cfg.amp);ap=average_precision_score(vy,vs);auc=roc_auc_score(vy,vs);hist.append({"model":name,"epoch":ep,"train_loss":np.mean(ls),"validation_auprc":ap,"validation_auroc":auc,"learning_rate":lr_used});logging.info("%s epoch=%d lr=%.2e loss=%.4f val_AUPRC=%.4f val_AUROC=%.4f",name,ep,lr_used,np.mean(ls),ap,auc)
  if ap>best+1e-5:best=ap;stale=0;torch.save({"model":model.state_dict(),"architecture":name,"best_epoch":ep,"loc":loc,"scale":scale,"seed":seed,"config":vars(cfg)},path)
  else:stale+=1
  if cfg.early_stopping and ep>=cfg.min_epochs and stale>=cfg.patience:break
 state=torch.load(path,map_location=device);model.load_state_dict(state["model"]);vy,score=predict(model,va,device,cfg.amp);elapsed=time.time()-t0;peak=torch.cuda.max_memory_allocated()/1024**3 if device.type=="cuda" else 0.;return {"score":score,"y":vy,"best_epoch":state["best_epoch"],"seed":seed,"history":hist,"parameters":npar,"seconds":elapsed,"peak_gpu_gb":peak}


