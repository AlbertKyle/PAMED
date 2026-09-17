#!/usr/bin/env python3
"""Dementia v9: progressive physiology-guided GRU-FCN family.

Exact Arena protocol: the same two internal evaluation EID sets are recovered
from raw_pa_algorithm_arena/validation_predictions.csv; the old v6 test EIDs
remain excluded for direct comparison. NHATS is reserved for external testing.

Progressive neural family:
  grufcn                 established baseline
  r2_grufcn              safe daily-rhythm residual
  chrono_r2_grufcn       + individual chronotype alignment
  multiscale_chrono      + short/intraday/circadian temporal scales
  stc_grufcn             + learnable spectral-temporal residual
  stc_clinical_apoe      + separate clinical and APOE residual branches

Every new PA component is added to the inherited risk logit through a sigmoid
gate initialized at sigmoid(-4)=0.018. Thus a new stage starts almost exactly
as its predecessor. Clinical variables exclude Townsend index. APOE dosage is
encoded and reported as an independent information set.
"""
from __future__ import annotations
import argparse, json, logging, os, sys
from datetime import datetime
from pathlib import Path

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
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

import v13_cohort_engine as v6
import v13_classic_engine as arena
import v13_rhythm_base as v8

BASE=Path("/root/autodl-tmp/PA_CR_CNN/Project_new/PA_Foundation_Model")
DEM=BASE/"dementia";DEFAULT_ARENA=DEM/"raw_pa_algorithm_arena";DEFAULT_OUT=DEM/"supervised_v9_progressive"
CLIN_CONT=["enrollment_age","BMI"]
CLIN_CAT=["gender","education_level","smoking","alcohol_use","ethic"]
CLINICAL=CLIN_CONT+CLIN_CAT
APOE=["apoe_e4"]
CLINICAL_APOE=CLINICAL+APOE
TRAD_PA=v6.TRAD_PA
BLOCKED=set(v6.BLOCKED)|{"Townsend_index","Followingyears_birthday","comorbidity_index","charlson_index"}
STAGES=["grufcn","r2_grufcn","chrono_r2_grufcn","multiscale_chrono","stc_grufcn","stc_clinical_apoe"]

def atomic_json(path,obj):
 tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf-8");os.replace(tmp,path)
def setup(out):
 out.mkdir(parents=True,exist_ok=True);logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",handlers=[logging.FileHandler(out/"run.log",encoding="utf-8"),logging.StreamHandler(sys.stdout)],force=True)

def load_cohort(cfg):
 d=v6.load_cohort(v6.Config(cfg.labels_csv,cfg.apoe_csv,cfg.cache_dir,cfg.output_dir,horizon_years=cfg.horizon_years));y=d.target.to_numpy(np.int8)
 if set(CLINICAL_APOE)&BLOCKED:raise RuntimeError("Blocked predictor entered v9")
 return d,y

def arena_splits(cfg,d):
 ad=Path(cfg.arena_dir);pred=pd.read_csv(ad/"validation_predictions.csv",dtype={"eid":"string"},usecols=["eid","repeat"]);pred.eid=pred.eid.str.strip();held=pd.read_csv(ad/"excluded_v6_test_eids.csv",dtype={"eid":"string"});held.eid=held.eid.str.strip();heldset=set(held.eid);allset=set(d.eid);out=[]
 for repeat in sorted(pred.repeat.unique()):
  vaset=set(pred.loc[pred.repeat.eq(repeat),"eid"]);trset=allset-heldset-vaset;tr=np.where(d.eid.isin(trset))[0];va=np.where(d.eid.isin(vaset))[0]
  if (len(tr),len(va))!=(42402,10601):raise ValueError(f"Arena repeat {repeat} mismatch: {len(tr)}/{len(va)}")
  out.append((int(repeat),tr,va));logging.info("Exact Arena repeat=%d train=%d/%d cases evaluation=%d/%d cases",repeat,len(tr),y_global(d,tr),len(va),y_global(d,va))
 return out,held
def y_global(d,idx):return int(d.target.iloc[idx].sum())

class AttnPool(nn.Module):
 def __init__(self,d):super().__init__();self.s=nn.Sequential(nn.Linear(d,max(8,d//2)),nn.Tanh(),nn.Linear(max(8,d//2),1))
 def forward(self,x,valid=None):
  z=self.s(x).squeeze(-1)
  if valid is not None:z=z.masked_fill(~valid,-1e4)
  w=torch.softmax(z,-1);return (x*w.unsqueeze(-1)).sum(1),w

def halfhour(x):
 """Mask-aware 30-min clock bins: [B,7,48,2], activity plus support."""
 b=x.shape[0];q=x.reshape(b,2,7,48,30);m=q[:,1].clamp(0,1);den=m.sum(-1).clamp_min(1.);a=(q[:,0]*m).sum(-1)/den;support=m.mean(-1);return torch.stack([a,support],-1)

class DailyRhythm(nn.Module):
 def __init__(self,d=64,drop=.30):
  super().__init__();q=d//2;self.clock=nn.Sequential(nn.Conv1d(2,q,5,padding=2,bias=False),nn.BatchNorm1d(q),nn.GELU(),nn.Conv1d(q,q,3,padding=1,bias=False),nn.BatchNorm1d(q),nn.GELU());self.pool=AttnPool(q);self.days=nn.GRU(q,d//2,1,batch_first=True,bidirectional=True);self.change=nn.Sequential(nn.Linear(q,d),nn.GELU(),nn.Dropout(drop));self.out=nn.Sequential(nn.Linear(2*d,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop))
 def forward(self,x):
  z=halfhour(x);b=z.shape[0];valid=z[...,1].gt(.2);h=self.clock(z.reshape(b*7,48,2).transpose(1,2)).transpose(1,2).reshape(b,7,48,-1);days=[]
  for j in range(7):days.append(self.pool(h[:,j],valid[:,j])[0])
  days=torch.stack(days,1);_,s=self.days(days);seq=torch.cat([s[-2],s[-1]],-1);change=self.change((days[:,1:]-days[:,:-1]).abs().mean(1));return self.out(torch.cat([seq,change],-1))

class ChronotypeAlign(nn.Module):
 """Differentiable alignment to an individual low-activity phase.

 Tests fixed circular shifts from -6 to +6 h and softly weights them by fit to
 a physiological reference with its activity nadir around 03:00. Both original
 and aligned information remain available through residual risk gating.
 """
 def __init__(self,d=64,drop=.30):
  super().__init__();self.shifts=list(range(-12,13,2));hour=torch.arange(48)/2.;ref=1.-torch.cos(2*torch.pi*(hour-3.)/24.);self.register_buffer("reference",(ref-ref.mean())/(ref.std()+1e-6));self.temperature=nn.Parameter(torch.tensor(1.));self.net=nn.Sequential(nn.Conv1d(2,32,5,padding=2),nn.GELU(),nn.Conv1d(32,d,3,padding=1),nn.GELU());self.pool=AttnPool(d);self.out=nn.Sequential(nn.Linear(d+2,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop));self.last_shift_hours=None
 def forward(self,x):
  z=halfhour(x);support=z[...,1];den=support.sum(1).clamp_min(1.);profile=(z[...,0]*support).sum(1)/den;profile=(profile-profile.mean(1,keepdim=True))/(profile.std(1,keepdim=True)+1e-4)
  candidates=torch.stack([torch.roll(profile,s,-1) for s in self.shifts],1);score=-((candidates-self.reference.view(1,1,-1))**2).mean(-1)/self.temperature.abs().clamp_min(.2);w=torch.softmax(score,1);aligned=(candidates*w.unsqueeze(-1)).sum(1);support_profile=support.mean(1);h=self.net(torch.stack([aligned,support_profile],1)).transpose(1,2);feat,_=self.pool(h,support_profile.gt(.2));sh=torch.as_tensor(self.shifts,device=x.device,dtype=x.dtype)/2.;sin=(w*torch.sin(2*torch.pi*sh/24)).sum(1);cos=(w*torch.cos(2*torch.pi*sh/24)).sum(1);self.last_shift_hours=(w*sh).sum(1);return self.out(torch.cat([feat,sin[:,None],cos[:,None]],1))

class MultiScale(nn.Module):
 def __init__(self,d=64,drop=.30):
  super().__init__();q=d//2;self.stem=nn.Sequential(nn.Conv1d(2,q,15,stride=10,padding=7,bias=False),nn.BatchNorm1d(q),nn.GELU());self.branches=nn.ModuleList([nn.Sequential(nn.Conv1d(q,q,5,padding=2*r,dilation=r,groups=q,bias=False),nn.Conv1d(q,q,1,bias=False),nn.BatchNorm1d(q),nn.GELU()) for r in [1,6,36]]);self.mix=nn.Sequential(nn.Linear(6*q,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop))
 def forward(self,x):
  h=self.stem(x);v=[]
  for branch in self.branches:
   q=branch(h);v.extend([q.mean(-1),q.amax(-1)])
  return self.mix(torch.cat(v,1))

class Spectral(nn.Module):
 def __init__(self,d=64,drop=.30):super().__init__();self.mlp=nn.Sequential(nn.Linear(10,32),nn.LayerNorm(32),nn.GELU(),nn.Dropout(drop),nn.Linear(32,d),nn.LayerNorm(d),nn.GELU())
 def forward(self,x):
  # A seven-day FFT has frequency-bin k = 7 days / period. Bands correspond to
  # circadian, harmonic, ultradian and fragmentation ranges; amplitudes remain
  # learned inputs rather than handcrafted phenotype labels.
  with torch.autocast(device_type=x.device.type,enabled=False):
   a=(x[:,0].float()*x[:,1].float());a=a-a.mean(-1,keepdim=True);sp=torch.log1p(torch.abs(torch.fft.rfft(a,dim=-1)))
   bands=[(5,10),(10,20),(20,45),(45,170),(170,600)];v=[]
   for lo,hi in bands:q=sp[:,lo:min(hi,sp.shape[1])];v.extend([q.mean(-1),q.amax(-1)])
   return self.mlp(torch.stack(v,1))

class ProgressiveRisk(nn.Module):
 def __init__(self,level,tab_dim=0,d=64,drop=.30):
  super().__init__();self.level=level;self.base=arena.GRUFCNEncoder(d,drop);self.base_risk=nn.Linear(d,1);self.proj=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,32));idx=STAGES.index(level);self.components=nn.ModuleDict();self.heads=nn.ModuleDict();self.gates=nn.ParameterDict()
  specs=[]
  if idx>=1:specs.append(("rhythm",DailyRhythm(d,drop)))
  if idx>=2:specs.append(("chrono",ChronotypeAlign(d,drop)))
  if idx>=3:specs.append(("multiscale",MultiScale(d,drop)))
  if idx>=4:specs.append(("spectral",Spectral(d,drop)))
  for name,module in specs:self.components[name]=module;self.heads[name]=nn.Linear(d,1);self.gates[name]=nn.Parameter(torch.tensor(-4.))
  if idx>=5:
   self.clinical_dim=tab_dim-2;self.clinical=nn.Sequential(nn.Linear(self.clinical_dim,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,d),nn.GELU());self.apoe=nn.Sequential(nn.Linear(2,16),nn.GELU(),nn.Linear(16,d),nn.GELU());self.clinical_head=nn.Linear(d,1);self.apoe_head=nn.Linear(d,1);self.clinical_gate=nn.Parameter(torch.tensor(-1.));self.apoe_gate=nn.Parameter(torch.tensor(-1.))
 def encode(self,x):return self.base(x)
 def project(self,x):return F.normalize(self.proj(self.encode(x)),dim=-1)
 def forward(self,x,tab=None):
  base=self.base_risk(self.base(x)).squeeze(-1);logit=base
  for name,module in self.components.items():logit=logit+torch.sigmoid(self.gates[name])*self.heads[name](module(x)).squeeze(-1)
  if self.level=="stc_clinical_apoe":
   if tab is None:raise ValueError("Clinical/APOE input required")
   c=self.clinical(tab[:,:self.clinical_dim]);a=self.apoe(tab[:,self.clinical_dim:]);logit=logit+torch.sigmoid(self.clinical_gate)*self.clinical_head(c).squeeze(-1)+torch.sigmoid(self.apoe_gate)*self.apoe_head(a).squeeze(-1)
  return logit
 def gate_values(self):
  z={k:float(torch.sigmoid(v).detach().cpu()) for k,v in self.gates.items()}
  if self.level=="stc_clinical_apoe":z.update({"clinical":float(torch.sigmoid(self.clinical_gate).detach().cpu()),"apoe":float(torch.sigmoid(self.apoe_gate).detach().cpu())})
  return z

def load_inherited(model,path):
 state=torch.load(path,map_location="cpu")["model"];missing,unexpected=model.load_state_dict(state,strict=False);logging.info("Inherited %s missing=%d unexpected=%d",path.name,len(missing),len(unexpected));return missing

def datasets(cfg,d,y,tr,va,tab):
 idx=d.cache_index.to_numpy(int);loc,scale=v6.robust_raw_scale(cfg.cache_dir,idx,tr,cfg.seed);ssl=v6.WeekDataset(cfg.cache_dir,idx[tr],y[tr],tab[tr],loc,scale,True);train=v6.WeekDataset(cfg.cache_dir,idx[tr],y[tr],tab[tr],loc,scale,False);valid=v6.WeekDataset(cfg.cache_dir,idx[va],y[va],tab[va],loc,scale,False);pos=max(y[tr].sum(),1);neg=max((y[tr]==0).sum(),1);q=cfg.sampled_positive_fraction;w=np.where(y[tr]==1,q/pos,(1-q)/neg);sampler=WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),len(tr),True);nw=0 if cfg.mode=="smoke" else cfg.num_workers;kw={"num_workers":nw,"pin_memory":torch.cuda.is_available(),"persistent_workers":nw>0};return DataLoader(ssl,cfg.batch_size,shuffle=True,drop_last=True,**kw),DataLoader(train,cfg.batch_size,sampler=sampler,**kw),DataLoader(valid,cfg.eval_batch_size,shuffle=False,**kw),loc,scale
def use_tab(name):return name=="stc_clinical_apoe"
def call(model,name,x,t):return model(x,t) if use_tab(name) else model(x)
def predict(model,name,loader,device,amp):
 model.eval();ys=[];ss=[]
 with torch.no_grad():
  for x,y,t in loader:
   with v6.amp_ctx(device,amp):s=call(model,name,x.to(device),t.to(device))
   ys.append(y.numpy());ss.append(s.float().cpu().numpy())
 y=np.concatenate(ys);s=np.concatenate(ss)
 if not np.isfinite(s).all():raise FloatingPointError(f"Non-finite predictions: {name}")
 return y,s

def train_stage(name,cfg,d,y,tr,va,tab,out,inherit=None,seed=1):
 v6.seed_all(seed);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");ssl,train,valid,loc,scale=datasets(cfg,d,y,tr,va,tab);model=ProgressiveRisk(name,tab.shape[1],cfg.d_model,cfg.dropout)
 if inherit:missing=load_inherited(model,inherit)
 model=model.to(device);npar=sum(p.numel() for p in model.parameters());logging.info("Training %-25s parameters=%d",name,npar)
 if name=="grufcn" and cfg.ssl_epochs:
  opt=torch.optim.AdamW(model.parameters(),lr=cfg.ssl_lr,weight_decay=cfg.weight_decay);gs=v6.grad_scaler(device,cfg.amp)
  for ep in range(1,cfg.ssl_epochs+1):
   model.train();ls=[]
   for a,b,_,_ in ssl:
    a=a.to(device);b=b.to(device);opt.zero_grad(set_to_none=True)
    with v6.amp_ctx(device,cfg.amp):loss=v6.nt_xent(model.project(a),model.project(b),cfg.temperature)
    if not torch.isfinite(loss):raise FloatingPointError(f"{name} SSL NaN")
    gs.scale(loss).backward();gs.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);gs.step(opt);gs.update();ls.append(float(loss.detach().cpu()))
   logging.info("%s SSL epoch=%d loss=%.4f",name,ep,np.mean(ls))
 # Inherited parameters receive one-fifth of the learning rate; new residuals
 # adapt faster without erasing the already validated risk representation.
 inherited_names=set(torch.load(inherit,map_location="cpu")["model"].keys()) if inherit else set();old=[];new=[]
 for n,p in model.named_parameters():(old if n in inherited_names else new).append(p)
 groups=[]
 if old:groups.append({"params":old,"lr":cfg.lr*.2})
 if new:groups.append({"params":new,"lr":cfg.lr})
 if not groups:groups=[{"params":model.parameters(),"lr":cfg.lr}]
 opt=torch.optim.AdamW(groups,weight_decay=cfg.weight_decay);gs=v6.grad_scaler(device,cfg.amp);bce=nn.BCEWithLogitsLoss();best=-1;stale=0;hist=[];path=out/f"{name}_best.pt"
 for ep in range(1,cfg.epochs+1):
  model.train();ls=[]
  for x,yy,t in train:
   x=x.to(device);t=t.to(device);yy=yy.to(device);opt.zero_grad(set_to_none=True)
   with v6.amp_ctx(device,cfg.amp):loss=bce(call(model,name,x,t),yy)
   if not torch.isfinite(loss):raise FloatingPointError(f"{name} supervised NaN epoch={ep}")
   gs.scale(loss).backward();gs.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);gs.step(opt);gs.update();ls.append(float(loss.detach().cpu()))
  vy,vs=predict(model,name,valid,device,cfg.amp);ap=average_precision_score(vy,vs);auc=roc_auc_score(vy,vs);row={"model":name,"epoch":ep,"loss":np.mean(ls),"auprc":ap,"auroc":auc,**{"gate_"+k:v for k,v in model.gate_values().items()}};hist.append(row);logging.info("%s epoch=%d val_AUPRC=%.4f val_AUROC=%.4f gates=%s",name,ep,ap,auc,model.gate_values())
  if ap>best+1e-5:best=ap;stale=0;torch.save({"model":model.state_dict(),"epoch":ep,"name":name,"loc":loc,"scale":scale,"gates":model.gate_values(),"config":vars(cfg)},path)
  else:stale+=1
  if ep>=cfg.min_epochs and stale>=cfg.patience:break
 state=torch.load(path,map_location=device);model.load_state_dict(state["model"]);vy,vs=predict(model,name,valid,device,cfg.amp);return vs,state,pd.DataFrame(hist),npar

def tabular(name,cols,d,y,tr,va):
 pipe=v6.Pipeline([("prep",v6.tab_preprocessor(cols)),("model",LogisticRegression(C=.25,max_iter=4000,solver="lbfgs"))]);pipe.fit(d.iloc[tr][cols],y[tr]);p=pipe.predict_proba(d.iloc[va][cols])[:,1];return p,pipe

def neural_tabular(d,tr):
 # APOE remains an explicit information set: dosage plus missingness. It is not
 # hidden among clinical one-hot columns, enabling a genuinely separate branch.
 pp=v6.tab_preprocessor(CLINICAL);pp.fit(d.iloc[tr][CLINICAL]);clinical=np.asarray(pp.transform(d[CLINICAL]),np.float32)
 dosage=pd.to_numeric(d.apoe_e4,errors="coerce").to_numpy(float);missing=np.isnan(dosage);median=float(np.nanmedian(dosage[tr]));dosage=np.where(missing,median,dosage);apoe=np.column_stack([(dosage-median)/max(np.nanstd(dosage[tr]),.5),missing.astype(float)]).astype(np.float32)
 return np.concatenate([clinical,apoe],1),pp,{"train_median":median,"columns":["standardized_e4_dosage","e4_missing"]}

def report(summary,per,hist,out):
 figs=[]
 def save(fig,n):fig.tight_layout();p=out/n;fig.savefig(p,dpi=300,bbox_inches="tight");plt.close(fig);figs.append(p)
 order=summary.sort_values("mean_auprc");fig,ax=plt.subplots(figsize=(8,5));ax.barh(order.model,order.mean_auprc,xerr=order.sd_auprc,color="#6A8FBF");ax.axvline(per.prevalence.mean(),color="k",ls="--",lw=.8);ax.set(xlabel="Mean internal-evaluation AUPRC ± SD",title="v9 progressive GRU-FCN family");save(fig,"01_progressive_auprc.png")
 fig,ax=plt.subplots(figsize=(8,5))
 for name,g in hist.groupby("model"):ax.plot(g.epoch,g.auprc,"o-",label=name)
 ax.set(xlabel="Epoch",ylabel="Evaluation AUPRC",title="Progressive training trajectories");ax.legend(fontsize=7,ncol=2);save(fig,"02_training.png")
 pdf=out/"DEMENTIA_GRUFCN_V9_REPORT.pdf"
 with PdfPages(pdf) as pages:
  fig=plt.figure(figsize=(11.69,8.27));fig.text(.06,.88,"Dementia GRU-FCN v9",fontsize=24,weight="bold");fig.text(.06,.79,"Progressive physiology-guided risk modelling",fontsize=15);fig.text(.06,.60,"Exact two-repeat Arena internal evaluation\nGRU-FCN → residual rhythm → chronotype alignment → multi-scale → spectral-temporal → clinical/APOE",fontsize=12,linespacing=1.7);fig.text(.06,.40,"Each stage inherits its predecessor and enters through a near-zero residual gate.\nClinical excludes Townsend; APOE is a separate information branch. NHATS remains external validation.",fontsize=10.5,linespacing=1.6);pages.savefig(fig);plt.close(fig)
  show=summary.round(4);fig,ax=plt.subplots(figsize=(11.69,8.27));ax.axis("off");tb=ax.table(cellText=show.values,colLabels=show.columns,loc="center",cellLoc="center");tb.auto_set_font_size(False);tb.set_fontsize(7);tb.scale(1,1.5);pages.savefig(fig,bbox_inches="tight");plt.close(fig)
  for p in figs:fig,ax=plt.subplots(figsize=(11.69,8.27));ax.imshow(plt.imread(p));ax.axis("off");pages.savefig(fig,bbox_inches="tight");plt.close(fig)
 return pdf

def smoke(cfg,d,y,tr,va,tab,out):
 # Real aligned input and two optimizer steps through the most complex model.
 small=np.r_[tr[y[tr]==1][:8],tr[y[tr]==0][:120]];sev=np.r_[va[y[va]==1][:4],va[y[va]==0][:60]];old=cfg.epochs;cfg.epochs=2;score,state,hist,npar=train_stage("stc_clinical_apoe",cfg,d,y,small,sev,tab,out,None,cfg.seed);cfg.epochs=old;logging.info("v9 smoke OK score finite=%s parameters=%d",np.isfinite(score).all(),npar)

def run(cfg):
 out=Path(cfg.output_dir);setup(out);atomic_json(out/"config.json",vars(cfg));d,y=load_cohort(cfg);splits,held=arena_splits(cfg,d);held.to_csv(out/"excluded_v6_test_eids.csv",index=False)
 if cfg.mode=="smoke":
  repeat,tr,va=splits[0];tab,pp,apinfo=neural_tabular(d,tr);smoke(cfg,d,y,tr,va,tab,out);atomic_json(out/"SMOKE_COMPLETED.json",{"completed":True,"time":datetime.now().isoformat(),"neural_tab_dim":tab.shape[1],"apoe":apinfo});return
 per=[];pred=[];histories=[];gate_rows=[]
 for repeat,tr,va in splits:
  ro=out/f"repeat_{repeat}";ro.mkdir(exist_ok=True);tab,pp,apinfo=neural_tabular(d,tr);joblib.dump(pp,ro/"clinical_preprocessor.joblib");atomic_json(ro/"apoe_preprocessor.json",apinfo)
  for name,cols in [("traditional_pa_only",TRAD_PA),("clinical_no_townsend",CLINICAL),("apoe_only",APOE),("clinical_plus_apoe",CLINICAL_APOE)]:
   p,pipe=tabular(name,cols,d,y,tr,va);joblib.dump(pipe,ro/f"{name}.joblib");per.append({"repeat":repeat,"model":name,"n":len(va),"cases":int(y[va].sum()),"prevalence":y[va].mean(),"auroc":roc_auc_score(y[va],p),"auprc":average_precision_score(y[va],p),"parameters":np.nan,"best_epoch":np.nan});pred.append(pd.DataFrame({"eid":d.eid.iloc[va],"target":y[va],"repeat":repeat,"model":name,"score":v6.logit(p)}))
  inherit=None
  for j,name in enumerate(STAGES):
   seed=cfg.seed+repeat*100+(4 if name=="grufcn" else 10+j);score,state,hist,npar=train_stage(name,cfg,d,y,tr,va,tab,ro,inherit,seed);inherit=ro/f"{name}_best.pt";hist["repeat"]=repeat;histories.append(hist);g={"repeat":repeat,"model":name,**state.get("gates",{})};gate_rows.append(g);per.append({"repeat":repeat,"model":name,"n":len(va),"cases":int(y[va].sum()),"prevalence":y[va].mean(),"auroc":roc_auc_score(y[va],score),"auprc":average_precision_score(y[va],score),"parameters":npar,"best_epoch":state["epoch"]});pred.append(pd.DataFrame({"eid":d.eid.iloc[va],"target":y[va],"repeat":repeat,"model":name,"score":score}))
 per=pd.DataFrame(per);per.to_csv(out/"metrics_by_repeat.csv",index=False);pd.concat(pred,ignore_index=True).to_csv(out/"predictions.csv",index=False);hist=pd.concat(histories,ignore_index=True);hist.to_csv(out/"training_history.csv",index=False);pd.DataFrame(gate_rows).to_csv(out/"learned_residual_gates.csv",index=False)
 increments=[]
 for repeat in sorted(per.repeat.unique()):
  q=per[per.repeat.eq(repeat)].set_index("model")
  for previous,current in zip(STAGES[:-1],STAGES[1:]):increments.append({"repeat":repeat,"previous":previous,"current":current,"delta_auprc":q.loc[current,"auprc"]-q.loc[previous,"auprc"],"delta_auroc":q.loc[current,"auroc"]-q.loc[previous,"auroc"]})
 pd.DataFrame(increments).to_csv(out/"progressive_increment_by_repeat.csv",index=False)
 summary=per.groupby("model").agg(mean_auroc=("auroc","mean"),sd_auroc=("auroc","std"),mean_auprc=("auprc","mean"),sd_auprc=("auprc","std"),parameters=("parameters","first"),median_best_epoch=("best_epoch","median")).reset_index().sort_values("mean_auprc",ascending=False);summary["auprc_over_prevalence"]=summary.mean_auprc/per.prevalence.mean();summary.to_csv(out/"model_summary.csv",index=False);pdf=report(summary,per,hist,out)
 audit={"protocol":"exact Arena evaluation EIDs, two repeats","development_n":53003,"development_cases":399,"per_repeat_train_n":42402,"per_repeat_evaluation_n":10601,"per_repeat_evaluation_cases":80,"old_v6_test_excluded":True,"role":"internal model development comparison","external_validation":"NHATS","clock_alignment":"each day 00:00-23:59","clinical_predictors":CLINICAL,"townsend_used":False,"apoe_information_set":APOE,"apoe_separate_branch":True,"traditional_pa_role":"comparator only","inheritance":"each neural stage inherits previous best checkpoint","residual_gate_initial":float(torch.sigmoid(torch.tensor(-4.))),"primary_metric":"AUPRC","prohibited_predictors":sorted(BLOCKED)};atomic_json(out/"design_and_leakage_audit.json",audit);atomic_json(out/"COMPLETED.json",{"completed":True,"time":datetime.now().isoformat(),"best_model":summary.iloc[0].model,"mean_auprc":summary.iloc[0].mean_auprc,"report":pdf.name});logging.info("v9 completed best=%s mean AUPRC=%.4f",summary.iloc[0].model,summary.iloc[0].mean_auprc)

def args():
 a=argparse.ArgumentParser();a.add_argument("--mode",choices=["smoke","full"],default="full");a.add_argument("--labels-csv",default=str(v6.DEFAULT_LABELS));a.add_argument("--apoe-csv",default=str(v6.DEFAULT_APOE));a.add_argument("--cache-dir",default=str(v6.DEFAULT_CACHE));a.add_argument("--arena-dir",default=str(DEFAULT_ARENA));a.add_argument("--output-dir",default=str(DEFAULT_OUT));a.add_argument("--horizon-years",type=float,default=7.);a.add_argument("--ssl-epochs",type=int,default=1);a.add_argument("--epochs",type=int,default=7);a.add_argument("--min-epochs",type=int,default=2);a.add_argument("--patience",type=int,default=2);a.add_argument("--batch-size",type=int,default=64);a.add_argument("--eval-batch-size",type=int,default=128);a.add_argument("--num-workers",type=int,default=6);a.add_argument("--lr",type=float,default=1e-4);a.add_argument("--ssl-lr",type=float,default=2e-4);a.add_argument("--weight-decay",type=float,default=1e-3);a.add_argument("--dropout",type=float,default=.30);a.add_argument("--d-model",type=int,default=64);a.add_argument("--sampled-positive-fraction",type=float,default=.08);a.add_argument("--temperature",type=float,default=.15);a.add_argument("--seed",type=int,default=20260817);a.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True);return a.parse_args()
if __name__=="__main__":run(args())


