#!/usr/bin/env python3
"""Dementia v12: fixed-repeat-2 fusion of locked raw-PA backbones.

GRU-FCN and ConvContext-Mamba are loaded from their v11 best checkpoints and
are never retrained here.  Their embeddings and logits are fused with balanced
feature, decision, and hybrid heads.  Age and sex are the only full-cohort
baseline variables. APOE is evaluated in a separate complete-genotype subset.
The v6 locked test remains excluded from all v12 development.
"""
from __future__ import annotations
import argparse, json, logging, os, random, sys, time
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
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import v13_cohort_engine as v6
import v13_mamba_engine as v11

BASE=Path("/root/autodl-tmp/PA_CR_CNN/Project_new/PA_Foundation_Model")
DEM=BASE/"dementia"
DEFAULT_V11=DEM/"supervised_v11_mamba_arena_repeat2_extended"
DEFAULT_OUT=DEM/"supervised_v12_fixed_repeat2_fusion"
AGE_SEX=["enrollment_age","gender"]

def atomic_json(path,obj):
 tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf-8");os.replace(tmp,path)

def setup(out):
 out.mkdir(parents=True,exist_ok=True);logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",handlers=[logging.FileHandler(out/"run.log",encoding="utf-8"),logging.StreamHandler(sys.stdout)],force=True)

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

def self_test(cfg):
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu");b=8;g=torch.randn(b,64,device=device);c=torch.randn(b,64,device=device);gl=torch.randn(b,device=device);cl=torch.randn(b,device=device);t=torch.randn(b,5,device=device);e=torch.randn(b,1,device=device)
 models=[DualPAFeature(),DualPADecision(),ClinicalFeature(5),ClinicalDecision(5),HybridFusion(5),APOEHybrid(5)]
 for m in models:
  m=m.to(device);o=m(g,c,gl,cl,t,e);assert o.shape==(b,) and torch.isfinite(o).all();o.sum().backward();logging.info("Self-test %-24s OK parameters=%d",m.__class__.__name__,sum(p.numel() for p in m.parameters()))

def load_locked(cfg,d,ids,out,label):
 cache=out/f"locked_features_{label}.npz"
 if cache.exists():
  try:
   # v12 caches are locally generated and trusted.  allow_pickle=True also
   # permits reuse of the first-run cache that stored an object-typed eid;
   # only the five numeric arrays below are ever returned to the model.
   with np.load(cache,allow_pickle=True) as z:
    need=("g","c","gl","cl","y");outz={k:z[k] for k in need}
   if any(len(outz[k])!=len(ids) for k in need):raise ValueError("cache length mismatch")
   logging.info("Reusing validated %s",cache);return outz
  except Exception as exc:
   logging.warning("Discarding invalid derived cache %s: %r",cache,exc);cache.unlink(missing_ok=True)
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu");gc=torch.load(Path(cfg.v11_dir)/"grufcn_best.pt",map_location=device);cc=torch.load(Path(cfg.v11_dir)/"convcontext_mamba_best.pt",map_location=device)
 if abs(float(gc["loc"])-float(cc["loc"]))>1e-6 or abs(float(gc["scale"])-float(cc["scale"]))>1e-6:raise ValueError("Locked backbone scaling mismatch")
 gm=v11.build("grufcn",cfg).to(device);cm=v11.build("convcontext_mamba",cfg).to(device);gm.load_state_dict(gc["model"]);cm.load_state_dict(cc["model"]);gm.eval();cm.eval()
 idx=d.cache_index.to_numpy(int);dummy=np.zeros((len(ids),1),np.float32);ds=v6.WeekDataset(cfg.cache_dir,idx[ids],d.target.to_numpy(np.int8)[ids],dummy,float(gc["loc"]),float(gc["scale"]),False);loader=DataLoader(ds,cfg.extract_batch_size,shuffle=False,num_workers=cfg.num_workers,pin_memory=torch.cuda.is_available(),persistent_workers=cfg.num_workers>0)
 ge=[];ce=[];gl=[];cl=[];yy=[]
 with torch.no_grad():
  for x,y,_ in loader:
   x=x.to(device,non_blocking=True)
   with v6.amp_ctx(device,cfg.amp):
    a=gm.encoder(x);b=cm.encoder(x);sa=gm.risk(a).squeeze(-1);sb=cm.risk(b).squeeze(-1)
   ge.append(a.float().cpu().numpy());ce.append(b.float().cpu().numpy());gl.append(sa.float().cpu().numpy());cl.append(sb.float().cpu().numpy());yy.append(y.numpy())
 z={"g":np.concatenate(ge).astype(np.float32),"c":np.concatenate(ce).astype(np.float32),"gl":np.concatenate(gl).astype(np.float32),"cl":np.concatenate(cl).astype(np.float32),"y":np.concatenate(yy).astype(np.float32)};tmp=cache.with_name(cache.stem+".tmp.npz");np.savez_compressed(tmp,**z);os.replace(tmp,cache);logging.info("Extracted locked features %s N=%d",label,len(ids));return z

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

def metric_row(name,y,s,epoch=None,npar=None,seconds=None,subset="full"):
 p=v6.sigmoid(s);prev=float(np.mean(y));return {"model":name,"information_set":subset,"n":len(y),"cases":int(np.sum(y)),"prevalence":prev,"best_epoch":epoch,"auprc":average_precision_score(y,p),"auroc":roc_auc_score(y,p),"auprc_over_prevalence":average_precision_score(y,p)/prev,"brier":brier_score_loss(y,p),"parameters":npar,"training_seconds":seconds}

def paired_bootstrap(y,pred,reference,out,repeats,seed):
 rng=np.random.default_rng(seed);case=np.where(y==1)[0];ctrl=np.where(y==0)[0];rows=[]
 for name,s in pred.items():
  if name==reference:continue
  da=[];dr=[]
  for _ in range(repeats):
   ix=np.r_[rng.choice(case,len(case),True),rng.choice(ctrl,len(ctrl),True)];rng.shuffle(ix);da.append(roc_auc_score(y[ix],s[ix])-roc_auc_score(y[ix],pred[reference][ix]));dr.append(average_precision_score(y[ix],s[ix])-average_precision_score(y[ix],pred[reference][ix]))
  rows.append({"model":name,"reference":reference,"delta_auprc":average_precision_score(y,s)-average_precision_score(y,pred[reference]),"delta_auprc_ci_low":np.quantile(dr,.025),"delta_auprc_ci_high":np.quantile(dr,.975),"delta_auroc":roc_auc_score(y,s)-roc_auc_score(y,pred[reference]),"delta_auroc_ci_low":np.quantile(da,.025),"delta_auroc_ci_high":np.quantile(da,.975)})
 pd.DataFrame(rows).to_csv(out,index=False)

def report(metrics,out):
 full=metrics[metrics.information_set.eq("full")].sort_values("auprc");fig,ax=plt.subplots(figsize=(9,6));ax.barh(full.model,full.auprc,color="#668DB7");ax.axvline(full.prevalence.iloc[0],color="k",ls="--",lw=.8);ax.set(xlabel="Validation AUPRC",title="Dementia v12 fixed-repeat-2 fusion");fig.tight_layout();fig.savefig(out/"01_v12_fusion_ranking.png",dpi=300,bbox_inches="tight");plt.close(fig);pdf=out/"DEMENTIA_V12_FUSION_REPORT.pdf"
 with PdfPages(pdf) as pages:
  fig=plt.figure(figsize=(11.69,8.27));fig.text(.06,.86,"Dementia v12",fontsize=24,weight="bold");fig.text(.06,.76,"Locked-backbone multimodal fusion",fontsize=15);fig.text(.06,.57,"Exact Arena repeat 2; locked GRU-FCN and ConvContext-Mamba embeddings.\nAge and sex only in the full cohort; APOE in a separate complete-genotype subset.",fontsize=11,linespacing=1.7);fig.text(.06,.38,"Balanced feature fusion, decision fusion and hybrid interaction-residual fusion.\nThe v6 locked test remains excluded.",fontsize=11,linespacing=1.7);pages.savefig(fig);plt.close(fig)
  for subset in ["full","apoe_complete"]:
   show=metrics[metrics.information_set.eq(subset)].round(4);fig,ax=plt.subplots(figsize=(11.69,8.27));ax.axis("off");tb=ax.table(cellText=show.values,colLabels=show.columns,loc="center",cellLoc="center");tb.auto_set_font_size(False);tb.set_fontsize(6.5);tb.scale(1,1.45);ax.set_title(subset.replace("_"," "),fontsize=15);pages.savefig(fig,bbox_inches="tight");plt.close(fig)
  fig,ax=plt.subplots(figsize=(11.69,8.27));ax.imshow(plt.imread(out/"01_v12_fusion_ranking.png"));ax.axis("off");pages.savefig(fig,bbox_inches="tight");plt.close(fig)
 return pdf

def run(cfg):
 out=Path(cfg.output_dir);setup(out);atomic_json(out/"config.json",vars(cfg));self_test(cfg)
 if cfg.mode=="smoke":atomic_json(out/"SMOKE_COMPLETED.json",{"completed":True,"time":datetime.now().isoformat()});return
 d,y,tr,va,held=v11.load_split(cfg);held.to_csv(out/"excluded_v6_test_eids.csv",index=False);logging.info("v12 locked repeat=2 train=%d/%d cases validation=%d/%d cases",len(tr),int(y[tr].sum()),len(va),int(y[va].sum()))
 pp=v6.tab_preprocessor(AGE_SEX);pp.fit(d.iloc[tr][AGE_SEX]);age=np.asarray(pp.transform(d[AGE_SEX]),np.float32);joblib.dump(pp,out/"age_sex_preprocessor.joblib")
 dosage=pd.to_numeric(d.apoe_e4,errors="coerce").to_numpy(float);apoe_ok=np.isfinite(dosage);med=float(np.nanmedian(dosage[tr]));sd=max(float(np.nanstd(dosage[tr])),.5);e4=((np.where(apoe_ok,dosage,med)-med)/sd).astype(np.float32)[:,None];atomic_json(out/"apoe_preprocessor.json",{"median":med,"scale":sd,"complete_case":True})
 tz=load_locked(cfg,d,tr,out,"train");vz=load_locked(cfg,d,va,out,"validation");trtab=age[tr];vatab=age[va];tre=e4[tr];vae=e4[va]
 trds=tensor_dataset(tz,trtab,tre);vads=tensor_dataset(vz,vatab,vae);rows=[];pred={};hist=[]
 pred["grufcn_pa"]=v6.sigmoid(vz["gl"]);rows.append(metric_row("grufcn_pa",vz["y"],vz["gl"],subset="full"));pred["convcontext_pa"]=v6.sigmoid(vz["cl"]);rows.append(metric_row("convcontext_pa",vz["y"],vz["cl"],subset="full"))
 agepipe=Pipeline([("prep",v6.tab_preprocessor(AGE_SEX)),("model",LogisticRegression(C=.25,max_iter=4000,solver="lbfgs"))]);agepipe.fit(d.iloc[tr][AGE_SEX],y[tr]);ages=v6.logit(agepipe.predict_proba(d.iloc[va][AGE_SEX])[:,1]);joblib.dump(agepipe,out/"age_sex_logistic.joblib");pred["age_sex"]=v6.sigmoid(ages);rows.append(metric_row("age_sex",y[va],ages,subset="full"))
 specs=[("dual_pa_feature",DualPAFeature()),("dual_pa_decision",DualPADecision()),("conv_age_sex_feature",ClinicalFeature(age.shape[1])),("conv_age_sex_decision",ClinicalDecision(age.shape[1])),("dual_pa_age_sex_hybrid",HybridFusion(age.shape[1]))]
 for j,(name,model) in enumerate(specs):
  s,ep,h,np_,sec=train_head(name,model,trds,vads,cfg,out,cfg.seed+100+j);pred[name]=v6.sigmoid(s);rows.append(metric_row(name,vz["y"],s,ep,np_,sec,"full"));hist.extend(h)
 trkeep=np.where(apoe_ok[tr])[0];vakeep=np.where(apoe_ok[va])[0];atr=tr[trkeep];ava=va[vakeep];logging.info("APOE complete subset train N=%d cases=%d validation N=%d cases=%d",len(atr),int(y[atr].sum()),len(ava),int(y[ava].sum()))
 apcols=AGE_SEX+["apoe_e4"];appipe=Pipeline([("prep",v6.tab_preprocessor(apcols)),("model",LogisticRegression(C=.25,max_iter=4000,solver="lbfgs"))]);appipe.fit(d.iloc[atr][apcols],y[atr]);aps=v6.logit(appipe.predict_proba(d.iloc[ava][apcols])[:,1]);joblib.dump(appipe,out/"age_sex_apoe_logistic.joblib");rows.append(metric_row("age_sex_apoe",y[ava],aps,subset="apoe_complete"))
 atrds=tensor_dataset(tz,trtab,tre,trkeep);avads=tensor_dataset(vz,vatab,vae,vakeep);aps2,ep,h,np_,sec=train_head("dual_pa_age_sex_apoe_hybrid",APOEHybrid(age.shape[1]),atrds,avads,cfg,out,cfg.seed+200);rows.append(metric_row("dual_pa_age_sex_apoe_hybrid",y[ava],aps2,ep,np_,sec,"apoe_complete"));hist.extend(h)
 metrics=pd.DataFrame(rows).sort_values(["information_set","auprc"],ascending=[True,False]);metrics.to_csv(out/"v12_validation_metrics.csv",index=False);pd.DataFrame(hist).to_csv(out/"v12_training_history.csv",index=False);pd.DataFrame({"eid":d.eid.iloc[va].to_numpy(),"target":y[va],**pred}).to_csv(out/"v12_validation_predictions.csv",index=False);paired_bootstrap(y[va],pred,"grufcn_pa",out/"paired_bootstrap_vs_grufcn.csv",cfg.bootstrap_repeats,cfg.seed);pdf=report(metrics,out)
 audit={"fixed_split":"Arena repeat 2","train_n":len(tr),"validation_n":len(va),"validation_cases":int(y[va].sum()),"locked_v6_test_excluded":True,"locked_backbones":["grufcn","convcontext_mamba"],"backbone_retraining":False,"full_cohort_baseline_variables":AGE_SEX,"feature_fusion":"separate 64-dimensional projections + LayerNorm + product and absolute-difference interactions","decision_fusion":"learned branch-logit weights","hybrid_fusion":"feature interaction plus residual branch logits","apoe_role":"separate complete-genotype subset","prohibited":["Followingyears_birthday","Townsend_index","comorbidity_index"]};atomic_json(out/"design_and_leakage_audit.json",audit);atomic_json(out/"COMPLETED.json",{"completed":True,"time":datetime.now().isoformat(),"best_full_model":metrics[metrics.information_set.eq("full")].iloc[0].model,"best_full_auprc":metrics[metrics.information_set.eq("full")].iloc[0].auprc,"report":pdf.name});logging.info("v12 completed best=%s AUPRC=%.4f",metrics[metrics.information_set.eq("full")].iloc[0].model,metrics[metrics.information_set.eq("full")].iloc[0].auprc)

def args():
 a=argparse.ArgumentParser();a.add_argument("--mode",choices=["smoke","full"],default="full");a.add_argument("--labels-csv",default=str(v6.DEFAULT_LABELS));a.add_argument("--apoe-csv",default=str(v6.DEFAULT_APOE));a.add_argument("--cache-dir",default=str(v6.DEFAULT_CACHE));a.add_argument("--arena-dir",default=str(v11.DEFAULT_ARENA));a.add_argument("--v11-dir",default=str(DEFAULT_V11));a.add_argument("--output-dir",default=str(DEFAULT_OUT));a.add_argument("--arena-repeat",type=int,default=2,choices=[2]);a.add_argument("--horizon-years",type=float,default=7.);a.add_argument("--d-model",type=int,default=64);a.add_argument("--dropout",type=float,default=.30);a.add_argument("--extract-batch-size",type=int,default=128);a.add_argument("--eval-batch-size",type=int,default=1024);a.add_argument("--fusion-batch-size",type=int,default=512);a.add_argument("--fusion-epochs",type=int,default=60);a.add_argument("--fusion-min-epochs",type=int,default=10);a.add_argument("--fusion-patience",type=int,default=12);a.add_argument("--fusion-lr",type=float,default=3e-4);a.add_argument("--fusion-weight-decay",type=float,default=1e-3);a.add_argument("--sampled-positive-fraction",type=float,default=.08);a.add_argument("--bootstrap-repeats",type=int,default=1000);a.add_argument("--num-workers",type=int,default=6);a.add_argument("--seed",type=int,default=20260817);a.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True);return a.parse_args()

if __name__=="__main__":run(args())

