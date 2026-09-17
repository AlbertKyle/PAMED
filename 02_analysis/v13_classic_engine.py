#!/usr/bin/env python3
"""Raw-PA time-series classification arena for 7-day dementia prediction.

This is an architecture discovery port, not a final test. The previously viewed
v6 locked test set is excluded. Every model receives exactly two 10,080-minute
channels (activity and observation mask); no clinical, APOE, traditional PA or
handcrafted circadian variable is available to any classifier.

Repeated target-age-sex stratified development splits compare eight families:
FCN, ResNet1D, TCN, XceptionTime, GRU-FCN, InceptionTime, PatchTST and the
day-aligned SameClockFormer. All are native PyTorch implementations to preserve
the existing torch/CUDA environment; they correspond to major families exposed
by time-series libraries such as tsai without requiring fastai/tsai installation.
"""
from __future__ import annotations

import argparse, json, logging, random, sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

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

import v13_cohort_engine as v6

BASE=Path("/root/autodl-tmp/PA_CR_CNN/Project_new/PA_Foundation_Model")
DEM=BASE/"dementia"
DEFAULT_V6_SPLIT=DEM/"supervised_v6_tsc"/"v6_locked_split.csv"
DEFAULT_OUT=DEM/"raw_pa_algorithm_arena"
ARCHS=["fcn","resnet1d","tcn","xceptiontime","grufcn","inception","patchtst","sameclock"]

@dataclass
class Config:
 labels_csv:str=str(v6.DEFAULT_LABELS);apoe_csv:str=str(v6.DEFAULT_APOE)
 cache_dir:str=str(v6.DEFAULT_CACHE);v6_split_csv:str=str(DEFAULT_V6_SPLIT)
 output_dir:str=str(DEFAULT_OUT);architectures:str=",".join(ARCHS)
 horizon_years:float=7.;repeats:int=2;validation_fraction:float=.20
 ssl_epochs:int=1;epochs:int=8;min_epochs:int=2;patience:int=2
 batch_size:int=64;eval_batch_size:int=128;num_workers:int=6
 lr:float=1e-4;ssl_lr:float=2e-4;weight_decay:float=1e-3
 dropout:float=.30;d_model:int=64;sampled_positive_fraction:float=.08
 temperature:float=.15;seed:int=20260817;amp:bool=True
 mode:str="full";overwrite:bool=False

def seed_all(seed):v6.seed_all(seed)
def setup(out):
 out.mkdir(parents=True,exist_ok=True);logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",handlers=[logging.FileHandler(out/"run.log",encoding="utf-8"),logging.StreamHandler(sys.stdout)],force=True)

def load_development(cfg):
 vc=v6.Config(cfg.labels_csv,cfg.apoe_csv,cfg.cache_dir,cfg.output_dir,horizon_years=cfg.horizon_years);d=v6.load_cohort(vc)
 s=pd.read_csv(cfg.v6_split_csv,dtype={"eid":"string"});s["eid"]=s.eid.str.strip()
 if set(s.eid)!=set(d.eid):raise ValueError("v6 split/cohort mismatch")
 d=d.merge(s[["eid","split"]],on="eid",how="left",validate="one_to_one");held=d.split.eq("test")
 pd.DataFrame({"eid":d.loc[held,"eid"]}).to_csv(Path(cfg.output_dir)/"excluded_v6_test_eids.csv",index=False)
 audit={"development_n":int((~held).sum()),"development_cases":int(d.loc[~held,"target"].sum()),"excluded_v6_test_n":int(held.sum()),"excluded_v6_test_cases":int(d.loc[held,"target"].sum())}
 d=d.loc[~held].drop(columns="split").reset_index(drop=True);logging.info("Raw arena development N=%d cases=%d; v6 test excluded N=%d",len(d),d.target.sum(),held.sum());return d,audit

def strata(d):
 age=pd.to_numeric(d.enrollment_age,errors="coerce");ab=pd.cut(age,[-np.inf,60,70,np.inf],labels=["lt60","60_69","ge70"]).astype(str);key=d.target.astype(str)+"|"+ab+"|"+d.gender.astype(str)
 c=key.value_counts();key=key.where(~key.isin(set(c[c<2].index)),d.target.astype(str)+"|"+ab);return key.to_numpy()

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
 elif name=="inception":e=v6.InceptionEncoder(cfg.d_model,cfg.dropout)
 elif name=="patchtst":e=v6.PatchTSTEncoder(cfg.d_model,cfg.dropout)
 elif name=="sameclock":e=v6.SameClockEncoder(cfg.d_model,cfg.dropout)
 else:raise ValueError(name)
 return RawClassifier(e,cfg.d_model)

def make_loaders(cfg,d,y,idx,train,val):
 # Dummy tabular matrix is never supplied to a model; WeekDataset returns it
 # only to reuse the rigorously aligned v6 data loader.
 dummy=np.zeros((len(d),1),np.float32);loc,scale=v6.robust_raw_scale(cfg.cache_dir,idx,train,cfg.seed)
 ssl_ds=v6.WeekDataset(cfg.cache_dir,idx[train],y[train],dummy[train],loc,scale,True);tr_ds=v6.WeekDataset(cfg.cache_dir,idx[train],y[train],dummy[train],loc,scale,False);va_ds=v6.WeekDataset(cfg.cache_dir,idx[val],y[val],dummy[val],loc,scale,False)
 pos=max(y[train].sum(),1);neg=max((y[train]==0).sum(),1);q=cfg.sampled_positive_fraction;w=np.where(y[train]==1,q/pos,(1-q)/neg);sampler=torch.utils.data.WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),len(train),True)
 nw=0 if cfg.mode=="smoke" else cfg.num_workers;kw={"num_workers":nw,"pin_memory":torch.cuda.is_available(),"persistent_workers":nw>0}
 return torch.utils.data.DataLoader(ssl_ds,cfg.batch_size,shuffle=True,drop_last=True,**kw),torch.utils.data.DataLoader(tr_ds,cfg.batch_size,sampler=sampler,**kw),torch.utils.data.DataLoader(va_ds,cfg.eval_batch_size,shuffle=False,**kw),loc,scale

def predict(model,loader,device,amp):
 model.eval();ys=[];ss=[]
 with torch.no_grad():
  for x,y,_ in loader:
   with v6.amp_ctx(device,amp):s=model(x.to(device,non_blocking=True))
   ys.append(y.numpy());ss.append(s.float().cpu().numpy())
 return np.concatenate(ys),np.concatenate(ss)

def train_model(name,cfg,d,y,idx,train,val,run_out):
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu");ssl,tr,va,loc,scale=make_loaders(cfg,d,y,idx,train,val);model=build(name,cfg).to(device);logging.info("%s parameters=%d",name,sum(p.numel() for p in model.parameters()))
 if cfg.ssl_epochs:
  opt=torch.optim.AdamW(model.parameters(),lr=cfg.ssl_lr,weight_decay=cfg.weight_decay);gs=v6.grad_scaler(device,cfg.amp)
  for ep in range(1,cfg.ssl_epochs+1):
   model.train();losses=[]
   for a,b,_,_ in ssl:
    a=a.to(device);b=b.to(device);opt.zero_grad(set_to_none=True)
    with v6.amp_ctx(device,cfg.amp):loss=v6.nt_xent(model.project(a),model.project(b),cfg.temperature)
    gs.scale(loss).backward();gs.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);gs.step(opt);gs.update();losses.append(float(loss.detach().cpu()))
   logging.info("%s SSL epoch=%d loss=%.4f",name,ep,np.mean(losses))
 opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay);gs=v6.grad_scaler(device,cfg.amp);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(cfg.epochs,1),eta_min=cfg.lr*.1);bce=nn.BCEWithLogitsLoss();best=-1;stale=0;hist=[];path=run_out/(name+"_best.pt")
 for ep in range(1,cfg.epochs+1):
  model.train();ls=[]
  for x,yy,_ in tr:
   x=x.to(device);yy=yy.to(device);opt.zero_grad(set_to_none=True)
   with v6.amp_ctx(device,cfg.amp):loss=bce(model(x),yy)
   gs.scale(loss).backward();gs.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);gs.step(opt);gs.update();ls.append(float(loss.detach().cpu()))
  sch.step();vy,vs=predict(model,va,device,cfg.amp);ap=average_precision_score(vy,vs);auc=roc_auc_score(vy,vs);hist.append({"architecture":name,"epoch":ep,"loss":np.mean(ls),"validation_auprc":ap,"validation_auroc":auc});logging.info("%s epoch=%d val_AUPRC=%.4f val_AUROC=%.4f",name,ep,ap,auc)
  if ap>best+1e-5:best=ap;stale=0;torch.save({"model":model.state_dict(),"architecture":name,"epoch":ep,"loc":loc,"scale":scale,"config":asdict(cfg)},path)
  else:stale+=1
  if ep>=cfg.min_epochs and stale>=cfg.patience:break
 state=torch.load(path,map_location=device);model.load_state_dict(state["model"]);vy,vs=predict(model,va,device,cfg.amp);return {"y":vy,"score":vs,"best_epoch":state["epoch"],"history":hist,"parameters":sum(p.numel() for p in model.parameters())}

def report(summary,per_run,hist,out,audit):
 figs=[]
 def save(fig,n):fig.tight_layout();p=out/n;fig.savefig(p,dpi=300,bbox_inches="tight");plt.close(fig);figs.append(p)
 order=summary.sort_values("mean_validation_auprc");fig,ax=plt.subplots(figsize=(8,5));ax.barh(order.architecture,order.mean_validation_auprc,xerr=order.sd_validation_auprc,color="#5B8DB8",alpha=.85);ax.axvline(per_run.prevalence.mean(),color="k",ls="--",lw=.8);ax.set(xlabel="Mean validation AUPRC ± SD",title="Raw-PA algorithm arena");save(fig,"01_algorithm_ranking.png")
 fig,ax=plt.subplots(figsize=(8,5))
 for n,g in hist.groupby("architecture"):ax.plot(g.epoch,g.validation_auprc,"o-",label=n,alpha=.8)
 ax.set(xlabel="Epoch",ylabel="Validation AUPRC",title="Representative training trajectories");ax.legend(fontsize=7,ncol=2);save(fig,"02_training_trajectories.png")
 pdf=out/"RAW_PA_ALGORITHM_ARENA_REPORT.pdf"
 with PdfPages(pdf) as pages:
  fig=plt.figure(figsize=(11.69,8.27));fig.text(.06,.88,"Raw-PA Algorithm Arena",fontsize=24,weight="bold");fig.text(.06,.79,"Broad end-to-end 7-day time-series classifier screening",fontsize=14);fig.text(.06,.62,f"Development N={audit['development_n']:,}; cases={audit['development_cases']}; v6 locked test excluded.\nInput for every model: [activity, observation mask] × 10,080 minutes.",fontsize=12,linespacing=1.6);fig.text(.06,.43,"No clinical, APOE, traditional PA or handcrafted circadian feature is available to any model.\nThis is architecture discovery; winners require nested OOF validation before biological fusion.",fontsize=10.5,linespacing=1.6);pages.savefig(fig);plt.close(fig)
  show=summary.round(4);fig,ax=plt.subplots(figsize=(11.69,8.27));ax.axis("off");tb=ax.table(cellText=show.values,colLabels=show.columns,loc="center",cellLoc="center");tb.auto_set_font_size(False);tb.set_fontsize(7);tb.scale(1,1.5);ax.set_title("Repeated validation ranking",fontsize=16);pages.savefig(fig,bbox_inches="tight");plt.close(fig)
  for p in figs:fig,ax=plt.subplots(figsize=(11.69,8.27));ax.imshow(plt.imread(p));ax.axis("off");pages.savefig(fig,bbox_inches="tight");plt.close(fig)
 return pdf

def run(cfg):
 out=Path(cfg.output_dir);setup(out);seed_all(cfg.seed);v6.atomic_json(out/"config.json",asdict(cfg));archs=[x.strip().lower() for x in cfg.architectures.split(",") if x.strip()]
 if not archs or not set(archs)<=set(ARCHS):raise ValueError(f"architectures must be subset of {ARCHS}")
 if (out/"COMPLETED.json").exists() and not cfg.overwrite:raise RuntimeError(f"Completed output exists: {out}")
 d,audit=load_development(cfg);y=d.target.to_numpy(np.int8);idx=d.cache_index.to_numpy(np.int64);key=strata(d);runs=[];histories=[];predictions=[]
 if cfg.mode=="smoke":cfg.repeats=1;cfg.ssl_epochs=1;cfg.epochs=2;cfg.min_epochs=1;cfg.patience=1;cfg.batch_size=8;cfg.eval_batch_size=16;cfg.num_workers=0
 for repeat in range(1,cfg.repeats+1):
  split=StratifiedShuffleSplit(1,test_size=cfg.validation_fraction,random_state=cfg.seed+repeat);train,val=next(split.split(np.zeros(len(y)),key))
  if cfg.mode=="smoke":
   rng=np.random.default_rng(cfg.seed)
   def sub(ix,np_,nn_):z=np.r_[rng.choice(ix[y[ix]==1],min(np_,sum(y[ix]==1)),False),rng.choice(ix[y[ix]==0],min(nn_,sum(y[ix]==0)),False)];rng.shuffle(z);return z
   train=sub(train,32,128);val=sub(val,16,64)
  ro=out/f"repeat_{repeat}";ro.mkdir(parents=True,exist_ok=True)
  for name in archs:
   seed_all(cfg.seed+repeat*100+archs.index(name));r=train_model(name,cfg,d,y,idx,train,val,ro);p=v6.sigmoid(r["score"]);runs.append({"repeat":repeat,"architecture":name,"validation_n":len(val),"validation_cases":int(y[val].sum()),"prevalence":y[val].mean(),"best_epoch":r["best_epoch"],"parameters":r["parameters"],"validation_auroc":roc_auc_score(y[val],p),"validation_auprc":average_precision_score(y[val],p),"auprc_over_prevalence":average_precision_score(y[val],p)/y[val].mean(),"brier":brier_score_loss(y[val],p)});histories.extend([{"repeat":repeat,**z} for z in r["history"]]);predictions.append(pd.DataFrame({"eid":d.iloc[val].eid,"target":y[val],"repeat":repeat,"architecture":name,"score":r["score"],"probability_uncalibrated":p}))
 per=pd.DataFrame(runs);per.to_csv(out/"validation_metrics_by_repeat.csv",index=False);pd.concat(predictions,ignore_index=True).to_csv(out/"validation_predictions.csv",index=False);hist=pd.DataFrame(histories);hist.to_csv(out/"training_history.csv",index=False)
 summary=per.groupby("architecture").agg(mean_validation_auroc=("validation_auroc","mean"),sd_validation_auroc=("validation_auroc","std"),mean_validation_auprc=("validation_auprc","mean"),sd_validation_auprc=("validation_auprc","std"),mean_auprc_over_prevalence=("auprc_over_prevalence","mean"),mean_brier=("brier","mean"),median_best_epoch=("best_epoch","median"),parameters=("parameters","first")).reset_index();summary[["sd_validation_auroc","sd_validation_auprc"]]=summary[["sd_validation_auroc","sd_validation_auprc"]].fillna(0);summary=summary.sort_values(["mean_validation_auprc","mean_validation_auroc"],ascending=False);summary["rank"]=np.arange(1,len(summary)+1);summary.to_csv(out/"algorithm_ranking.csv",index=False)
 pdf=report(summary,per,hist[hist.repeat.eq(1)],out,audit);v6.atomic_json(out/"design_and_leakage_audit.json",{"design":"repeated development validation raw-PA architecture arena","input_shape":"[batch,2,10080] except SameClock internal reshape [batch,7,2,1440]","channels":["minute activity","observation mask"],"clinical_used":False,"apoe_used":False,"traditional_pa_used":False,"handcrafted_rhythm_used":False,"v6_test_scored":False,"v6_test_excluded":True,"architectures":archs,"repeats":cfg.repeats,**audit,"warning":"Architecture discovery only; confirm winners using nested OOF and independent validation."});v6.atomic_json(out/"COMPLETED.json",{"completed":True,"time":datetime.now().isoformat(),"best_raw_pa_architecture":summary.iloc[0].architecture,"mean_validation_auprc":summary.iloc[0].mean_validation_auprc,"report":pdf.name});logging.info("Arena completed. Best=%s mean AUPRC=%.4f",summary.iloc[0].architecture,summary.iloc[0].mean_validation_auprc)

def parse_args():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument("--mode",choices=["smoke","full"],default="full");a.add_argument("--labels-csv",type=Path,default=v6.DEFAULT_LABELS);a.add_argument("--apoe-csv",type=Path,default=v6.DEFAULT_APOE);a.add_argument("--cache-dir",type=Path,default=v6.DEFAULT_CACHE);a.add_argument("--v6-split-csv",type=Path,default=DEFAULT_V6_SPLIT);a.add_argument("--output-dir",type=Path,default=DEFAULT_OUT);a.add_argument("--architectures",default=",".join(ARCHS));a.add_argument("--repeats",type=int,default=2);a.add_argument("--ssl-epochs",type=int,default=1);a.add_argument("--epochs",type=int,default=8);a.add_argument("--batch-size",type=int,default=64);a.add_argument("--num-workers",type=int,default=6);a.add_argument("--seed",type=int,default=20260817);a.add_argument("--overwrite",action="store_true");z=a.parse_args()
 return Config(str(z.labels_csv),str(z.apoe_csv),str(z.cache_dir),str(z.v6_split_csv),str(z.output_dir),architectures=z.architectures,repeats=z.repeats,ssl_epochs=z.ssl_epochs,epochs=z.epochs,batch_size=z.batch_size,num_workers=z.num_workers,seed=z.seed,mode=z.mode,overwrite=z.overwrite)
if __name__=="__main__":run(parse_args())

