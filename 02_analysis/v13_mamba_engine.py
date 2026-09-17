#!/usr/bin/env python3
"""v11 fixed-Arena-repeat long-sequence backbone challenge for 7-day raw PA.

The selected development split is recovered exactly from Arena repeat 2, where
GRU-FCN was strongest.  The original Arena optimization recipe is retained:
identical normalization seed, weighted sampling, cosine learning-rate schedule,
and validation-AUPRC checkpoint selection.  The locked v6 internal test remains
excluded from all model development.  All models receive activity and mask.

Official mamba_ssm is mandatory; this script never substitutes a pseudo-Mamba.
"""
from __future__ import annotations
import argparse, json, logging, os, random, sys, time
from datetime import datetime
from pathlib import Path

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
 raise SystemExit("Official mamba_ssm is required. Run bash install_mamba_v11.sh first. Import error: "+repr(exc))

import v13_cohort_engine as v6
import v13_classic_engine as arena
import v13_rhythm_engine as v9

BASE=Path("/root/autodl-tmp/PA_CR_CNN/Project_new/PA_Foundation_Model")
DEM=BASE/"dementia";DEFAULT_ARENA=DEM/"raw_pa_algorithm_arena";DEFAULT_OUT=DEM/"supervised_v11_mamba_arena"
MODELS=["grufcn","mamba_fcn","bimambasl_fcn","convcontext_mamba","chronoview_mamba_fcn","gru_mamba_fcn"]

def atomic_json(path,obj):
 tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf-8");os.replace(tmp,path)
def setup(out):
 out.mkdir(parents=True,exist_ok=True);logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",handlers=[logging.FileHandler(out/"run.log",encoding="utf-8"),logging.StreamHandler(sys.stdout)],force=True)
def seed_all(seed):v6.seed_all(seed);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def load_split(cfg):
 d=v6.load_cohort(v6.Config(cfg.labels_csv,cfg.apoe_csv,cfg.cache_dir,cfg.output_dir,horizon_years=cfg.horizon_years));y=d.target.to_numpy(np.int8);ad=Path(cfg.arena_dir)
 p=pd.read_csv(ad/"validation_predictions.csv",dtype={"eid":"string"},usecols=["eid","repeat"]);p.eid=p.eid.str.strip();held=pd.read_csv(ad/"excluded_v6_test_eids.csv",dtype={"eid":"string"});held.eid=held.eid.str.strip();valset=set(p.loc[p.repeat.eq(cfg.arena_repeat),"eid"]);trainset=set(d.eid)-set(held.eid)-valset;train=np.where(d.eid.isin(trainset))[0];val=np.where(d.eid.isin(valset))[0]
 if (len(train),len(val))!=(42402,10601) or y[val].sum()!=80:raise ValueError(f"Arena split mismatch train={len(train)} validation={len(val)} cases={y[val].sum()}")
 if set(d.eid.iloc[train])&set(held.eid) or set(d.eid.iloc[val])&set(held.eid):raise ValueError("Locked v6 test leakage detected")
 logging.info("Locked Arena repeat=%d train N=%d cases=%d validation N=%d cases=%d",cfg.arena_repeat,len(train),y[train].sum(),len(val),y[val].sum());return d,y,train,val,held

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
 """The same downsampling and FCN branches used by Arena GRU-FCN."""
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
 idx=d.cache_index.to_numpy(int);dummy=np.zeros((len(d),1),np.float32);loc,scale=v6.robust_raw_scale(cfg.cache_dir,idx,train,cfg.seed);ssl=v6.WeekDataset(cfg.cache_dir,idx[train],y[train],dummy[train],loc,scale,True);tr=v6.WeekDataset(cfg.cache_dir,idx[train],y[train],dummy[train],loc,scale,False);va=v6.WeekDataset(cfg.cache_dir,idx[val],y[val],dummy[val],loc,scale,False);pos=max(y[train].sum(),1);neg=max((y[train]==0).sum(),1);q=cfg.sampled_positive_fraction;w=np.where(y[train]==1,q/pos,(1-q)/neg);sampler=WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),len(train),True);nw=0 if cfg.mode=="smoke" else cfg.num_workers;kw={"num_workers":nw,"pin_memory":torch.cuda.is_available(),"persistent_workers":nw>0};return DataLoader(ssl,cfg.batch_size,shuffle=True,drop_last=True,**kw),DataLoader(tr,cfg.batch_size,sampler=sampler,**kw),DataLoader(va,cfg.eval_batch_size,shuffle=False,**kw),loc,scale

def predict(model,loader,device,amp):
 model.eval();ys=[];ss=[]
 with torch.no_grad():
  for x,y,_ in loader:
   with v6.amp_ctx(device,amp):s=model(x.to(device))
   ys.append(y.numpy());ss.append(s.float().cpu().numpy())
 y=np.concatenate(ys);s=np.concatenate(ss)
 if not np.isfinite(s).all():raise FloatingPointError("Non-finite scores")
 return y,s

def train_one(name,cfg,d,y,train,val,out,seed):
 seed_all(seed);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");ssl,tr,va,loc,scale=loaders(cfg,d,y,train,val);model=build(name,cfg).to(device);npar=sum(p.numel() for p in model.parameters());logging.info("Training %-23s seed=%d parameters=%d",name,seed,npar);torch.cuda.reset_peak_memory_stats() if device.type=="cuda" else None;t0=time.time()
 if cfg.ssl_epochs:
  opt=torch.optim.AdamW(model.parameters(),lr=cfg.ssl_lr,weight_decay=cfg.weight_decay);gs=v6.grad_scaler(device,cfg.amp)
  for ep in range(1,cfg.ssl_epochs+1):
   model.train();ls=[]
   for a,b,_,_ in ssl:
    a=a.to(device);b=b.to(device);opt.zero_grad(set_to_none=True)
    with v6.amp_ctx(device,cfg.amp):loss=v6.nt_xent(model.project(a),model.project(b),cfg.temperature)
    if not torch.isfinite(loss):raise FloatingPointError(f"{name} SSL NaN")
    gs.scale(loss).backward();gs.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);gs.step(opt);gs.update();ls.append(float(loss.detach().cpu()))
   logging.info("%s SSL epoch=%d loss=%.4f",name,ep,np.mean(ls))
 opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay);gs=v6.grad_scaler(device,cfg.amp);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(cfg.cosine_epochs,1),eta_min=cfg.lr*.1);bce=nn.BCEWithLogitsLoss();hist=[];best=-1.;stale=0;path=out/f"{name}_best.pt"
 for ep in range(1,cfg.epochs+1):
  model.train();ls=[];lr_used=opt.param_groups[0]["lr"]
  for x,yy,_ in tr:
   x=x.to(device);yy=yy.to(device);opt.zero_grad(set_to_none=True)
   with v6.amp_ctx(device,cfg.amp):loss=bce(model(x),yy)
   if not torch.isfinite(loss):raise FloatingPointError(f"{name} training NaN")
   gs.scale(loss).backward();gs.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);gs.step(opt);gs.update();ls.append(float(loss.detach().cpu()))
  if ep<=cfg.cosine_epochs:sch.step()
  vy,vs=predict(model,va,device,cfg.amp);ap=average_precision_score(vy,vs);auc=roc_auc_score(vy,vs);hist.append({"model":name,"epoch":ep,"train_loss":np.mean(ls),"validation_auprc":ap,"validation_auroc":auc,"learning_rate":lr_used});logging.info("%s epoch=%d lr=%.2e loss=%.4f val_AUPRC=%.4f val_AUROC=%.4f",name,ep,lr_used,np.mean(ls),ap,auc)
  if ap>best+1e-5:best=ap;stale=0;torch.save({"model":model.state_dict(),"architecture":name,"best_epoch":ep,"loc":loc,"scale":scale,"seed":seed,"config":vars(cfg)},path)
  else:stale+=1
  if cfg.early_stopping and ep>=cfg.min_epochs and stale>=cfg.patience:break
 state=torch.load(path,map_location=device);model.load_state_dict(state["model"]);vy,score=predict(model,va,device,cfg.amp);elapsed=time.time()-t0;peak=torch.cuda.max_memory_allocated()/1024**3 if device.type=="cuda" else 0.;return {"score":score,"y":vy,"best_epoch":state["best_epoch"],"seed":seed,"history":hist,"parameters":npar,"seconds":elapsed,"peak_gpu_gb":peak}

def report(metrics,out):
 order=metrics.sort_values("auprc");fig,ax=plt.subplots(figsize=(8,5));ax.barh(order.model,order.auprc,color="#668DB7");ax.axvline(metrics.prevalence.iloc[0],color="k",ls="--",lw=.8);ax.set(xlabel="Locked Arena-repeat-2 validation AUPRC",title="v11 long-sequence backbone challenge");fig.tight_layout();fig.savefig(out/"01_mamba_backbone_ranking.png",dpi=300,bbox_inches="tight");plt.close(fig);pdf=out/"DEMENTIA_V11_MAMBA_REPORT.pdf"
 with PdfPages(pdf) as pages:
  fig=plt.figure(figsize=(11.69,8.27));fig.text(.06,.87,"Dementia v11",fontsize=24,weight="bold");fig.text(.06,.77,"Mamba and long-context raw-PA backbone challenge",fontsize=14);fig.text(.06,.59,"Exact Arena repeat 2 development split and original optimization recipe.\nThe v6 locked internal test remains excluded from all tuning.",fontsize=12,linespacing=1.7);fig.text(.06,.41,"GRU-FCN, Mamba-FCN, BiMambaSL-FCN, ConvContext-Mamba,\nChronoView-Mamba-FCN and hybrid GRU-Mamba-FCN.",fontsize=11,linespacing=1.6);pages.savefig(fig);plt.close(fig)
  fig,ax=plt.subplots(figsize=(11.69,8.27));ax.axis("off");show=metrics.round(4);tb=ax.table(cellText=show.values,colLabels=show.columns,loc="center",cellLoc="center");tb.auto_set_font_size(False);tb.set_fontsize(7);tb.scale(1,1.5);pages.savefig(fig,bbox_inches="tight");plt.close(fig)
  fig,ax=plt.subplots(figsize=(11.69,8.27));ax.imshow(plt.imread(out/"01_mamba_backbone_ranking.png"));ax.axis("off");pages.savefig(fig,bbox_inches="tight");plt.close(fig)
 return pdf

def self_test(cfg):
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 for name in MODELS:
  m=build(name,cfg).to(device);x=torch.randn(2,2,10080,device=device);o=m(x);assert o.shape==(2,) and torch.isfinite(o).all();o.sum().backward();logging.info("Self-test %-23s OK parameters=%d",name,sum(p.numel() for p in m.parameters()));del m,x,o
 if device.type=="cuda":torch.cuda.empty_cache()

def run(cfg):
 out=Path(cfg.output_dir);setup(out);atomic_json(out/"config.json",vars(cfg));self_test(cfg)
 if cfg.mode=="smoke":atomic_json(out/"SMOKE_COMPLETED.json",{"completed":True,"time":datetime.now().isoformat(),"official_mamba_module":Mamba.__module__});return
 d,y,train,val,held=load_split(cfg);held.to_csv(out/"excluded_v6_test_eids.csv",index=False);split=np.full(len(d),"excluded",dtype="U10");split[train]="train";split[val]="validation";pd.DataFrame({"eid":d.eid,"target":y,"split":split}).to_csv(out/"fixed_split_assignment.csv",index=False)
 models=[q.strip() for q in cfg.models.split(",") if q.strip()];rows=[];pred=pd.DataFrame({"eid":d.eid.iloc[val].to_numpy(),"target":y[val]});hist=[]
 for j,name in enumerate(models):
  # GRU-FCN keeps its exact original Arena architecture index (4).  New
  # backbones continue from index 8, after the eight original Arena models.
  model_seed=cfg.seed+cfg.arena_repeat*100+(4 if name=="grufcn" else 7+j)
  r=train_one(name,cfg,d,y,train,val,out,model_seed);p=v6.sigmoid(r["score"]);rows.append({"model":name,"validation_n":len(val),"cases":int(y[val].sum()),"prevalence":y[val].mean(),"best_epoch":r["best_epoch"],"seed":r["seed"],"auroc":roc_auc_score(y[val],p),"auprc":average_precision_score(y[val],p),"auprc_over_prevalence":average_precision_score(y[val],p)/y[val].mean(),"brier_uncalibrated":brier_score_loss(y[val],p),"parameters":r["parameters"],"training_seconds":r["seconds"],"peak_gpu_gb":r["peak_gpu_gb"]});pred[name+"_score"]=r["score"];hist.extend(r["history"]);logging.info("VALIDATION %-23s best_epoch=%d AUPRC=%.4f AUROC=%.4f",name,r["best_epoch"],rows[-1]["auprc"],rows[-1]["auroc"])
 metrics=pd.DataFrame(rows).sort_values("auprc",ascending=False);metrics.to_csv(out/"fixed_validation_metrics.csv",index=False);pred.to_csv(out/"fixed_validation_predictions.csv",index=False);pd.DataFrame(hist).to_csv(out/"training_history.csv",index=False);pdf=report(metrics,out);audit={"protocol":"exact selected Arena repeat 2 extended development validation protocol","arena_repeat":cfg.arena_repeat,"train_n":len(train),"train_cases":int(y[train].sum()),"validation_n":len(val),"validation_cases":int(y[val].sum()),"checkpoint_selection":"highest validation AUPRC across the complete extended run","maximum_epochs":cfg.epochs,"cosine_epochs":cfg.cosine_epochs,"low_lr_tail_epochs":max(cfg.epochs-cfg.cosine_epochs,0),"early_stopping":cfg.early_stopping,"locked_v6_test_excluded":True,"locked_v6_test_n":len(held),"input":["minute activity","observation mask"],"clinical_used":False,"apoe_used":False,"traditional_pa_used":False,"handcrafted_rhythm_used":False,"mamba_implementation":Mamba.__module__,"mamba_ssm_version":getattr(mamba_ssm,"__version__","unknown"),"mamba_generation":"Mamba-1 selective SSM block","external_validation":"NHATS after backbone selection"};atomic_json(out/"design_and_leakage_audit.json",audit);atomic_json(out/"COMPLETED.json",{"completed":True,"time":datetime.now().isoformat(),"best_model":metrics.iloc[0].model,"validation_auprc":metrics.iloc[0].auprc,"report":pdf.name});logging.info("v11 completed best=%s validation_AUPRC=%.4f",metrics.iloc[0].model,metrics.iloc[0].auprc)

def args():
 a=argparse.ArgumentParser();a.add_argument("--mode",choices=["smoke","full"],default="full");a.add_argument("--labels-csv",default=str(v6.DEFAULT_LABELS));a.add_argument("--apoe-csv",default=str(v6.DEFAULT_APOE));a.add_argument("--cache-dir",default=str(v6.DEFAULT_CACHE));a.add_argument("--arena-dir",default=str(DEFAULT_ARENA));a.add_argument("--output-dir",default=str(DEFAULT_OUT));a.add_argument("--arena-repeat",type=int,default=2,choices=[2],help="Locked selected Arena split; intentionally fixed to repeat 2");a.add_argument("--models",default=",".join(MODELS));a.add_argument("--horizon-years",type=float,default=7.);a.add_argument("--ssl-epochs",type=int,default=1);a.add_argument("--epochs",type=int,default=16);a.add_argument("--cosine-epochs",type=int,default=8,help="Preserve original Arena LR trajectory, then hold eta_min");a.add_argument("--min-epochs",type=int,default=8);a.add_argument("--patience",type=int,default=5);a.add_argument("--early-stopping",action=argparse.BooleanOptionalAction,default=False);a.add_argument("--batch-size",type=int,default=64);a.add_argument("--eval-batch-size",type=int,default=128);a.add_argument("--num-workers",type=int,default=6);a.add_argument("--lr",type=float,default=1e-4);a.add_argument("--ssl-lr",type=float,default=2e-4);a.add_argument("--weight-decay",type=float,default=1e-3);a.add_argument("--dropout",type=float,default=.30);a.add_argument("--d-model",type=int,default=64);a.add_argument("--sampled-positive-fraction",type=float,default=.08);a.add_argument("--temperature",type=float,default=.15);a.add_argument("--seed",type=int,default=20260817,help="Original Arena base seed; do not change for comparable runs");a.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True);return a.parse_args()
if __name__=="__main__":run(args())

