#!/usr/bin/env python3
"""Clinical analyses of the frozen PAMED representations and predictions.

No stage in this file changes the primary model, preprocessing, split, or
threshold.  The script reuses frozen representations and predictions to study
subtypes, hierarchical temporal Shapley values, landmark survival, effect
modification, mortality/comorbidity associations, and prespecified paired
comparisons. Concept analyses include unadjusted-for-volume primary estimates
and separately identified activity-volume and common-scale sensitivities.
Optional follow-up and death columns contain days since accelerometry; no
death time is inferred from a binary mortality indicator.
"""
from __future__ import annotations

import argparse, itertools, json, logging, math, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA_ROOT, RESULTS_ROOT
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             mean_absolute_error, r2_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import cohort
import train_evaluate as core
import fusion
import pamed_models as backbones
import sensitivity_analysis as sup
from clinical_helpers import binomial_term, distribution_summary, full_sequence_iv

ROOT=DATA_ROOT
RESULTS=RESULTS_ROOT
HORIZON_DAYS=int(round(365.25*7));LANDMARK_DAYS=365
SUBTYPES={
 "alzheimers_disease":"days_accel_to_alzheimers_disease",
 "vascular_dementia":"days_accel_to_vascular_dementia",
 "frontotemporal_dementia":"days_accel_to_frontotemporal_dementia",
}
AUTO_COMORBIDITY=["comorbidity_index","charlson_index","Charlson_index",
                  "charlson_comorbidity_index","CCI","Elixhauser_index",
                  "elixhauser_index","ECI"]


def setup(out:Path):
 out.mkdir(parents=True,exist_ok=True)
 logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",handlers=[logging.FileHandler(out/"run.log",encoding="utf-8"),logging.StreamHandler(sys.stdout)],force=True)


def atomic_json(path:Path,obj):
 tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf-8");os.replace(tmp,path)


def bh_fdr(p):
 a=np.asarray(p,float);out=np.full(len(a),np.nan);ok=np.isfinite(a)
 if not ok.any():return out
 v=a[ok];order=np.argsort(v);ranked=v[order];q=ranked*len(v)/np.arange(1,len(v)+1);q=np.minimum.accumulate(q[::-1])[::-1];q=np.clip(q,0,1);rev=np.empty(len(v),int);rev[order]=np.arange(len(v));out[ok]=q[rev];return out


def safe_logit(p):
 p=np.clip(np.asarray(p,float),1e-6,1-1e-6);return np.log(p/(1-p))


def load_d(cfg):
 return cohort.load_cohort(cohort.Config(cfg.labels_csv,cfg.apoe_csv,cfg.cache_dir,cfg.output_dir,horizon_years=7.))


def split(d,cfg):return sup.split_indices(d,Path(cfg.protocol_dir))[:2]


def exp_cfg(cfg,out):
 return SimpleNamespace(labels_csv=cfg.labels_csv,apoe_csv=cfg.apoe_csv,cache_dir=cfg.cache_dir,output_dir=str(out),horizon_years=7.,mode="full",d_model=64,dropout=.30,extract_batch_size=cfg.extract_batch_size,eval_batch_size=cfg.eval_batch_size,num_workers=cfg.num_workers,amp=cfg.amp)


def load_rep(path:Path):
 p=path/"frozen_latent_vectors.npz"
 if not p.exists():raise FileNotFoundError(f"Run representations first: {p}")
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def predict_head(head,z,tab,batch,device):
 ds=core.fusion_dataset(z,tab);return fusion.infer(head,ds,device,batch)[1]


def run_representations(cfg):
 out=Path(cfg.output_dir);setup(out);ec=exp_cfg(cfg,out);d=load_d(ec);dev,test=split(d,cfg);core_dir=Path(cfg.core_dir)
 fg=core_dir/"grufcn_80pct_refit.pt";fc=core_dir/"convcontext_mamba_80pct_refit.pt"
 zd=core.extract_pair(ec,d,d.target.to_numpy(np.int8),dev,fg,fc);zt=core.extract_pair(ec,d,d.target.to_numpy(np.int8),test,fg,fc)
 pp=joblib.load(core_dir/"age_sex_preprocessor_80pct.joblib");tab=np.asarray(pp.transform(d[core.AGE_SEX]),np.float32)
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu");head=fusion.HybridFusion(tab.shape[1]).to(device);head.load_state_dict(torch.load(core_dir/"final_hybrid_80pct_refit.pt",map_location=device)["model"]);head.eval();raw_d=predict_head(head,zd,tab[dev],cfg.eval_batch_size,device);raw_t=predict_head(head,zt,tab[test],cfg.eval_batch_size,device)
 cal=joblib.load(core_dir/"development_only_calibrators.joblib")["final_hybrid"];pdv=cal.predict_proba(raw_d[:,None])[:,1];pt=cal.predict_proba(raw_t[:,None])[:,1]
 saved=pd.read_csv(core_dir/"ukb_20pct_final_test_predictions.csv",dtype={"eid":"string"});saved.eid=saved.eid.str.strip();expected=d.eid.iloc[test].astype(str).to_numpy()
 if saved.eid.to_numpy().tolist()!=expected.tolist():raise ValueError("Representation/test EID mismatch")
 maxdiff=float(np.max(np.abs(pt-saved.probability_final_hybrid.to_numpy(float))))
 Xd=np.c_[zd["g"],zd["c"]];Xt=np.c_[zt["g"],zt["c"]];sc=StandardScaler().fit(Xd);ncomp=min(cfg.pca_components,Xd.shape[1]);pca=PCA(n_components=ncomp,random_state=cfg.seed).fit(sc.transform(Xd));pcd=pca.transform(sc.transform(Xd)).astype(np.float32);pct=pca.transform(sc.transform(Xt)).astype(np.float32);joblib.dump({"scaler":sc,"pca":pca},out/"latent_pca_development_fitted.joblib")
 role=np.r_[np.repeat("development",len(dev)),np.repeat("test",len(test))];ids=np.r_[dev,test];g=np.r_[zd["g"],zt["g"]];c=np.r_[zd["c"],zt["c"]];gl=np.r_[zd["gl"],zt["gl"]];cl=np.r_[zd["cl"],zt["cl"]];raw=np.r_[raw_d,raw_t];prob=np.r_[pdv,pt];pcs=np.r_[pcd,pct]
 np.savez_compressed(out/"frozen_latent_vectors.npz",eid=d.eid.iloc[ids].astype(str).to_numpy(dtype="U"),cohort_index=ids.astype(np.int64),split=role.astype("U"),target=d.target.to_numpy(np.int8)[ids],g=g.astype(np.float32),c=c.astype(np.float32),gl=gl.astype(np.float32),cl=cl.astype(np.float32),tab=tab[ids].astype(np.float32),final_raw_logit=raw.astype(np.float32),final_probability=prob.astype(np.float32),pcs=pcs)
 q=pd.DataFrame({"eid":d.eid.iloc[ids].to_numpy(),"split":role,"target":d.target.to_numpy(int)[ids],"grufcn_logit":gl,"convcontext_logit":cl,"final_raw_logit":raw,"final_probability":prob})
 for j in range(ncomp):q[f"latent_pc{j+1:02d}"]=pcs[:,j]
 q.to_csv(out/"frozen_latent_pcs.csv",index=False);pd.DataFrame({"component":np.arange(1,ncomp+1),"explained_variance_ratio":pca.explained_variance_ratio_,"cumulative_explained_variance":np.cumsum(pca.explained_variance_ratio_)}).to_csv(out/"latent_pca_variance.csv",index=False)
 threshold=json.loads((core_dir/"locked_epochs_and_thresholds.json").read_text())["thresholds"]["final_hybrid"]
 atomic_json(out/"deployment_input_contract.json",{"model":"PAMED","seed":cfg.seed,"input_activity_shape":[7,1440],"neural_channels":["log1p-scaled activity","observation mask"],"tabular_inputs":["age at accelerometry (enrollment_age)","sex"],"forbidden_inputs":["follow-up age","death","comorbidity","post-baseline diagnoses"],"ukb_threshold":threshold,"warning":"Research risk stratification only; threshold and absolute probability are not transported to NHATS or consumer devices without prospective calibration.","minimum_device_qc":["exactly 7 local-midnight aligned days","minute timestamps and timezone retained","wear/missing mask retained","device-unit harmonization fixed","reject insufficient wear rather than silently zero-fill"]})
 atomic_json(out/"COMPLETED.json",{"completed":True,"n":len(ids),"development_n":len(dev),"test_n":len(test),"latent_dimension":128,"pca_components":ncomp,"test_probability_max_abs_difference":maxdiff,"weights_changed":False})
 logging.info("COMPLETED representations N=%d test_probability_maxdiff=%.3g",len(ids),maxdiff)


def strat_boot(y,p,n,seed):
 if y.sum()<3 or (y==0).sum()<3:return (np.nan,np.nan,np.nan,np.nan)
 rng=np.random.default_rng(seed);pos=np.where(y==1)[0];neg=np.where(y==0)[0];ap=[];au=[]
 for _ in range(n):
  ix=np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)];ap.append(average_precision_score(y[ix],p[ix]));au.append(roc_auc_score(y[ix],p[ix]))
 return *np.quantile(ap,[.025,.975]),*np.quantile(au,[.025,.975])


def subtype_labels(d,sub):
 s=pd.read_csv(sub,dtype={"eid":"string"},low_memory=False);s.eid=s.eid.str.strip();s=d[["eid"]].merge(s,on="eid",how="left",validate="one_to_one");out={}
 for name,col in SUBTYPES.items():
  days=pd.to_numeric(s[col],errors="coerce");out[name]=(days.gt(LANDMARK_DAYS)&days.le(HORIZON_DAYS)).to_numpy(np.int8)
 return s,out


def subtype_features(rep,d,indices):
 lookup={e:i for i,e in enumerate(rep["eid"].astype(str))};r=np.array([lookup[str(e)] for e in d.eid.iloc[indices]],int);age=pd.to_numeric(d.enrollment_age,errors="coerce").to_numpy()[indices];sex=pd.to_numeric(d.gender,errors="coerce").to_numpy()[indices]
 return r,np.c_[age,sex],np.c_[rep["pcs"][r],age,sex],rep["final_probability"][r]


def fit_subtype_probe(X,y,seed):
 nfold=max(2,min(5,int(y.sum())));cv=StratifiedKFold(nfold,shuffle=True,random_state=seed);best=None;rows=[]
 for C in [.003,.01,.03,.1,.3,1.]:
  m=Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),("sc",StandardScaler()),("lr",LogisticRegression(C=C,class_weight="balanced",max_iter=4000,solver="liblinear"))]);score=cross_val_score(m,X,y,cv=cv,scoring="average_precision").mean();rows.append((C,score))
  if best is None or score>best[1]:best=(C,score)
 model=Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),("sc",StandardScaler()),("lr",LogisticRegression(C=best[0],class_weight="balanced",max_iter=4000,solver="liblinear"))]).fit(X,y);return model,best,rows


def run_subtypes(cfg):
 out=Path(cfg.output_dir);setup(out);d=load_d(cfg);dev,test=split(d,cfg);rep=load_rep(Path(cfg.representation_dir));_,labs=subtype_labels(d,cfg.subtypes_csv);rd,age_d,latent_d,risk_d=subtype_features(rep,d,dev);rt,age_t,latent_t,risk_t=subtype_features(rep,d,test);yall=d.target.to_numpy(int);rows=[];predrows=[];cvrows=[];counts=[]
 any_sub=np.zeros(len(d),bool)
 for name,y in labs.items():any_sub|=y.astype(bool)
 for name,ysub in labs.items():
  counts.append({"subtype":name,"development_events":int(ysub[dev].sum()),"test_events":int(ysub[test].sum()),"allcause_test_events":int(yall[test].sum())})
  md=(yall[dev]==0)|(ysub[dev]==1);mt=(yall[test]==0)|(ysub[test]==1);yd=ysub[dev][md];yt=ysub[test][mt]
  direct=risk_t[mt];models_pred={"frozen_allcause_score":direct}
  if yd.sum()>=cfg.minimum_subtype_development_events and yt.sum()>=cfg.minimum_subtype_test_events:
   for label,XD,XT in [("age_sex_probe",age_d[md],age_t[mt]),("latent_pa_age_sex_probe",latent_d[md],latent_t[mt])]:
    model,best,cvv=fit_subtype_probe(XD,yd,cfg.seed+len(rows));models_pred[label]=model.predict_proba(XT)[:,1];joblib.dump(model,out/f"{name}_{label}.joblib")
    for C,score in cvv:cvrows.append({"subtype":name,"model":label,"C":C,"development_cv_auprc":score,"selected":C==best[0]})
  for model,p in models_pred.items():
   lo,hi,ulo,uhi=strat_boot(yt,p,cfg.bootstrap_repeats,cfg.seed+len(rows));prev=yt.mean();rows.append({"subtype":name,"model":model,"test_n":len(yt),"test_events":int(yt.sum()),"prevalence":prev,"auprc":average_precision_score(yt,p) if yt.sum() else np.nan,"auprc_ci_low":lo,"auprc_ci_high":hi,"auroc":roc_auc_score(yt,p) if len(np.unique(yt))==2 else np.nan,"auroc_ci_low":ulo,"auroc_ci_high":uhi,"auprc_over_prevalence":average_precision_score(yt,p)/prev if prev>0 else np.nan})
   for eid,yy,pp in zip(d.eid.iloc[test].to_numpy()[mt],yt,p):predrows.append({"eid":eid,"subtype":name,"target":yy,"model":model,"probability_or_score":pp})
 pd.DataFrame(counts).to_csv(out/"subtype_cohort_counts.csv",index=False);pd.DataFrame(rows).to_csv(out/"subtype_test_metrics.csv",index=False);pd.DataFrame(predrows).to_csv(out/"subtype_test_predictions_long.csv",index=False);pd.DataFrame(cvrows).to_csv(out/"subtype_probe_cv.csv",index=False)
 case=pd.DataFrame({"eid":d.eid.iloc[test].to_numpy(),"allcause_event":yall[test],"final_probability":risk_t})
 for name,y in labs.items():case[name]=y[test]
 case["unclassified_or_other_dementia"]=(case.allcause_event.eq(1)&case[list(labs)].sum(axis=1).eq(0)).astype(int);case[case.allcause_event.eq(1)].to_csv(out/"allcause_cases_subtype_membership.csv",index=False)
 atomic_json(out/"COMPLETED.json",{"completed":True,"role":"secondary subtype analysis with frozen all-cause representation","minimum_development_events_for_probe":cfg.minimum_subtype_development_events,"minimum_test_events_for_metrics":cfg.minimum_subtype_test_events,"warning":"Overlapping subtypes are retained; rare FTD is descriptive and no deep subtype model is trained."});logging.info("COMPLETED subtype analysis")


def load_models(cfg,d):
 ec=exp_cfg(cfg,Path(cfg.output_dir));device=torch.device("cuda" if torch.cuda.is_available() else "cpu");cd=Path(cfg.core_dir);gp=torch.load(cd/"grufcn_80pct_refit.pt",map_location=device);cp=torch.load(cd/"convcontext_mamba_80pct_refit.pt",map_location=device);gm=backbones.build("grufcn",ec).to(device);cm=backbones.build("convcontext_mamba",ec).to(device);gm.load_state_dict(gp["model"]);cm.load_state_dict(cp["model"]);gm.eval();cm.eval();pp=joblib.load(cd/"age_sex_preprocessor_80pct.joblib");tab=np.asarray(pp.transform(d[core.AGE_SEX]),np.float32);head=fusion.HybridFusion(tab.shape[1]).to(device);head.load_state_dict(torch.load(cd/"final_hybrid_80pct_refit.pt",map_location=device)["model"]);head.eval();cal=joblib.load(cd/"development_only_calibrators.joblib")["final_hybrid"];return ec,device,gp,gm,cm,tab,head,cal


def calibrated_probability(cal,raw):return cal.predict_proba(np.asarray(raw).reshape(-1,1))[:,1]


def forward_array(x,tab,gm,cm,head,cal,device,batch):
 out=[]
 for st in range(0,len(x),batch):
  xx=torch.from_numpy(x[st:st+batch]).to(device);tt=torch.from_numpy(tab[st:st+batch]).to(device);e=torch.zeros((len(xx),1),device=device)
  with torch.no_grad(),cohort.amp_ctx(device,True):g=gm.encoder(xx);c=cm.encoder(xx);gl=gm.risk(g).squeeze(-1);cl=cm.risk(c).squeeze(-1);s=head(g,c,gl,cl,tt,e)
  out.append(s.float().cpu().numpy())
 return calibrated_probability(cal,np.concatenate(out))


def matched_explanation_sample(d,test,ratio,seed):
 y=d.target.to_numpy(int);cases=test[y[test]==1];controls=list(test[y[test]==0]);rng=np.random.default_rng(seed);rng.shuffle(controls);chosen=[]
 age=pd.to_numeric(d.enrollment_age,errors="coerce").to_numpy();sex=pd.to_numeric(d.gender,errors="coerce").to_numpy()
 for i in cases:
  for _ in range(ratio):
   if not controls:break
   cost=np.array([abs(age[j]-age[i])+5*(sex[j]!=sex[i]) for j in controls]);k=int(np.nanargmin(cost));chosen.append(controls.pop(k))
 ids=np.r_[cases,np.asarray(chosen,int)];rng.shuffle(ids);return ids


def population_reference(cache,d,dev,loc,scale,n,seed):
 rng=np.random.default_rng(seed);sel=rng.choice(dev,min(n,len(dev)),False);act=np.load(Path(cache)/"activity_7d.npy",mmap_mode="r");obs=np.load(Path(cache)/"observed_7d.npy",mmap_mode="r");ci=d.cache_index.to_numpy(int)[sel];total=np.zeros((7,1440),np.float64);count=np.zeros((7,1440),np.float64)
 for st in range(0,len(ci),128):
  a=np.asarray(act[ci[st:st+128]],np.float32);m=np.asarray(obs[ci[st:st+128]],bool);z=(np.log1p(np.clip(np.nan_to_num(a,nan=0.),0,None))-loc)/scale;total+=(z*m).sum(0);count+=m.sum(0)
 return np.divide(total,count,out=np.zeros_like(total),where=count>0).astype(np.float32)


def bootstrap_mean(a,n,seed):
 a=np.asarray(a,float);rng=np.random.default_rng(seed);v=[np.mean(rng.choice(a,len(a),True)) for _ in range(n)];return np.quantile(v,[.025,.975])


def exact_modality_shapley(cfg,rep,d,dev,test,tab,head,cal,device,out):
 lookup={e:i for i,e in enumerate(rep["eid"].astype(str))};rd=np.array([lookup[str(e)] for e in d.eid.iloc[dev]],int);rt=np.array([lookup[str(e)] for e in d.eid.iloc[test]],int);sources=["gru","mamba","clinical"]
 refs={"g":rep["g"][rd].mean(0),"c":rep["c"][rd].mean(0),"gl":float(rep["gl"][rd].mean()),"cl":float(rep["cl"][rd].mean()),"tab":rep["tab"][rd].mean(0)};values={}
 for bits in itertools.product([0,1],repeat=3):
  n=len(rt);z={"g":np.repeat(refs["g"][None],n,0).astype(np.float32),"c":np.repeat(refs["c"][None],n,0).astype(np.float32),"gl":np.repeat(refs["gl"],n).astype(np.float32),"cl":np.repeat(refs["cl"],n).astype(np.float32),"y":d.target.to_numpy(np.float32)[test]};tb=np.repeat(refs["tab"][None],n,0).astype(np.float32)
  if bits[0]:z["g"]=rep["g"][rt];z["gl"]=rep["gl"][rt]
  if bits[1]:z["c"]=rep["c"][rt];z["cl"]=rep["cl"][rt]
  if bits[2]:tb=rep["tab"][rt]
  raw=predict_head(head,z,tb,cfg.eval_batch_size,device);values[bits]=calibrated_probability(cal,raw)
 phi={s:np.zeros(len(rt)) for s in sources};fact=math.factorial
 for j,s in enumerate(sources):
  others=[k for k in range(3) if k!=j]
  for r in range(3):
   for subset in itertools.combinations(others,r):
    b=[0,0,0]
    for k in subset:b[k]=1
    b1=b.copy();b1[j]=1;w=fact(r)*fact(2-r)/fact(3);phi[s]+=w*(values[tuple(b1)]-values[tuple(b)])
 q=pd.DataFrame({"eid":d.eid.iloc[test].to_numpy(),"target":d.target.to_numpy(int)[test],"baseline_reference_probability":values[(0,0,0)],"full_probability":values[(1,1,1)],**{f"shapley_{k}":v for k,v in phi.items()}});q["efficiency_error"]=q.full_probability-q.baseline_reference_probability-q[[f"shapley_{k}" for k in sources]].sum(axis=1);q.to_csv(out/"modality_shapley_individual.csv",index=False);rows=[]
 for group,z in q.groupby(q.target.map({0:"control",1:"incident"})):
  for s in sources:
   a=z[f"shapley_{s}"].to_numpy();lo,hi=bootstrap_mean(a,cfg.explanation_bootstrap_repeats,cfg.seed+len(rows));rows.append({"group":group,"source":s,"n":len(a),"mean_shapley_probability":a.mean(),"mean_absolute_shapley":np.abs(a).mean(),"median_shapley_probability":np.median(a),"ci_low":lo,"ci_high":hi})
 pd.DataFrame(rows).to_csv(out/"modality_shapley_summary.csv",index=False);return float(np.abs(q.efficiency_error).max())


def run_temporal_shapley(cfg):
 out=Path(cfg.output_dir);setup(out);d=load_d(cfg);dev,test=split(d,cfg);rep=load_rep(Path(cfg.representation_dir));ec,device,gp,gm,cm,tab,head,cal=load_models(cfg,d);sample=matched_explanation_sample(d,test,cfg.explanation_control_ratio,cfg.seed);loc=float(gp["loc"]);scale=float(gp["scale"]);ref=population_reference(cfg.cache_dir,d,dev,loc,scale,cfg.reference_participants,cfg.seed);np.savez_compressed(out/"population_temporal_reference.npz",standardized_activity_reference=ref,location=loc,scale=scale)
 act=np.load(Path(cfg.cache_dir)/"activity_7d.npy",mmap_mode="r");obs=np.load(Path(cfg.cache_dir)/"observed_7d.npy",mmap_mode="r");ci=d.cache_index.to_numpy(int)[sample];a=np.asarray(act[ci],np.float32);m=np.asarray(obs[ci],bool);z=(np.log1p(np.clip(np.nan_to_num(a,nan=0.),0,None))-loc)/scale;z=np.where(m,z,0.).astype(np.float32);base=np.where(m,ref[None],0.).astype(np.float32);full=np.stack([z.reshape(len(z),-1),m.astype(np.float32).reshape(len(z),-1)],1);xb=np.stack([base.reshape(len(z),-1),m.astype(np.float32).reshape(len(z),-1)],1);tabs=tab[sample]
 basep=forward_array(xb,tabs,gm,cm,head,cal,device,cfg.eval_batch_size);fullp=forward_array(full,tabs,gm,cm,head,cal,device,cfg.eval_batch_size);blocks=[(day,part,day*1440+part*360,day*1440+(part+1)*360) for day in range(7) for part in range(4)];rng=np.random.default_rng(cfg.seed);acc=np.zeros((len(sample),len(blocks)),np.float64)
 for r in range(cfg.shapley_permutations):
  perms=np.vstack([rng.permutation(len(blocks)) for _ in sample]);cur=xb.copy();prev=basep.copy()
  for step in range(len(blocks)):
   choice=perms[:,step]
   for b,(_,_,st,en) in enumerate(blocks):
    rows=np.where(choice==b)[0]
    if len(rows):cur[rows,0,st:en]=full[rows,0,st:en]
   now=forward_array(cur,tabs,gm,cm,head,cal,device,cfg.eval_batch_size);acc[np.arange(len(sample)),choice]+=now-prev;prev=now
  logging.info("Temporal Shapley permutation %d/%d",r+1,cfg.shapley_permutations)
 phi=acc/cfg.shapley_permutations;wide=pd.DataFrame({"eid":d.eid.iloc[sample].to_numpy(),"target":d.target.to_numpy(int)[sample],"baseline_reference_probability":basep,"full_probability":fullp})
 for j,(day,part,_,_) in enumerate(blocks):wide[f"shapley_day{day+1}_{part*6:02d}_{(part+1)*6:02d}"]=phi[:,j]
 wide["efficiency_error"]=fullp-basep-phi.sum(1);wide.to_csv(out/"temporal_shapley_individual.csv",index=False);rows=[]
 for group,mask in [("incident",d.target.to_numpy(int)[sample]==1),("matched_control",d.target.to_numpy(int)[sample]==0)]:
  for j,(day,part,_,_) in enumerate(blocks):
   v=phi[mask,j];lo,hi=bootstrap_mean(v,cfg.explanation_bootstrap_repeats,cfg.seed+j);rows.append({"group":group,"day":day+1,"clock_start":part*6,"clock_end":(part+1)*6,"n":len(v),"mean_shapley_probability":v.mean(),"mean_absolute_shapley":np.abs(v).mean(),"median_shapley_probability":np.median(v),"ci_low":lo,"ci_high":hi})
 cell=pd.DataFrame(rows);cell.to_csv(out/"temporal_shapley_7day_x_4period.csv",index=False);cell.groupby(["group","day"],as_index=False).agg(mean_shapley_probability=("mean_shapley_probability","sum"),mean_absolute_shapley=("mean_absolute_shapley","sum")).to_csv(out/"temporal_shapley_by_day.csv",index=False);cell.groupby(["group","clock_start","clock_end"],as_index=False).agg(mean_shapley_probability=("mean_shapley_probability","sum"),mean_absolute_shapley=("mean_absolute_shapley","sum")).to_csv(out/"temporal_shapley_by_clock.csv",index=False)
 mode_error=exact_modality_shapley(cfg,rep,d,dev,test,tab,head,cal,device,out);atomic_json(out/"COMPLETED.json",{"completed":True,"method":"Monte-Carlo permutation Shapley over 28 non-overlapping day-by-6-hour coalitions plus exact 3-source Shapley","sample_n":len(sample),"incident_n":int(d.target.to_numpy(int)[sample].sum()),"permutations":cfg.shapley_permutations,"reference":"development-sample same-minute standardized activity mean; observed mask retained","temporal_efficiency_max_abs_error":float(np.abs(wide.efficiency_error).max()),"modality_efficiency_max_abs_error":mode_error,"causal_interpretation":False});logging.info("COMPLETED hierarchical Shapley")


def circular_rolling_extrema(profile):
 # Thirty-minute bins are sufficient for stable M10/L5 discovery phenotypes.
 b=profile.reshape(len(profile),48,30);h=np.nanmean(b,2);hh=np.c_[h,h]
 m10=np.stack([np.nanmean(hh[:,j:j+20],1) for j in range(48)],1);l5=np.stack([np.nanmean(hh[:,j:j+10],1) for j in range(48)],1);imax=np.nanargmax(np.nan_to_num(m10,nan=-np.inf),1);imin=np.nanargmin(np.nan_to_num(l5,nan=np.inf),1);return np.nanmax(m10,1),np.nanmin(l5,1),(imax+10)%48/2.,(imin+5)%48/2.


def rhythm_batch(a,m,start_dates):
 """Compute concepts in raw activity units; IV uses ordered one-minute data."""
 m=np.asarray(m,bool)&np.isfinite(a)
 safe=np.where(m,a,np.nan);profile=np.nanmean(safe,1);daily=np.nanmean(safe,2);overall=np.nanmean(safe,(1,2));day_sd=np.nanstd(daily,1);sameclock=np.nanmean(np.nanstd(safe,axis=1),1);adj=np.nanmean(np.abs(safe[:,1:]-safe[:,:-1]),(1,2));night=np.nanmean(profile[:,:360],1);morning=np.nanmean(profile[:,360:720],1);afternoon=np.nanmean(profile[:,720:1080],1);evening=np.nanmean(profile[:,1080:],1);iv,_=full_sequence_iv(a,m);isd=np.nanvar(profile,axis=1)/np.maximum(np.nanvar(safe.reshape(len(a),-1),axis=1),1e-8)
 # Mean pairwise correlation of the seven aligned daily profiles.
 centered=safe-np.nanmean(safe,2,keepdims=True);centered=np.nan_to_num(centered);norm=np.sqrt(np.sum(centered**2,2));corr=np.einsum("bik,bjk->bij",centered,centered)/np.maximum(norm[:,:,None]*norm[:,None,:],1e-8);tri=np.triu_indices(7,1);regular=np.nanmean(corr[:,tri[0],tri[1]],1);m10,l5,m10time,l5time=circular_rolling_extrema(profile);ra=(m10-l5)/np.maximum(m10+l5,1e-8);weekday=[];weekend=[]
 for row,date in zip(daily,start_dates):
  dt=pd.to_datetime(date,errors="coerce");wd=np.array([(dt+pd.Timedelta(days=j)).dayofweek if pd.notna(dt) else -1 for j in range(7)]);weekday.append(np.nanmean(row[(wd>=0)&(wd<5)]) if np.any((wd>=0)&(wd<5)) else np.nan);weekend.append(np.nanmean(row[wd>=5]) if np.any(wd>=5) else np.nan)
 return {"overall_activity":overall,"day_mean_sd":day_sd,"sameclock_between_day_sd":sameclock,"adjacent_day_absolute_change":adj,"mean_pairwise_day_correlation":regular,"night_00_06":night,"morning_06_12":morning,"afternoon_12_18":afternoon,"evening_18_24":evening,"day_night_difference":(morning+afternoon)/2-night,"intradaily_variability":iv,"interdaily_stability":isd,"M10":m10,"L5":l5,"relative_amplitude":ra,"M10_midpoint_hour":m10time,"L5_midpoint_hour":l5time,"weekday_activity":np.asarray(weekday),"weekend_activity":np.asarray(weekend),"weekend_minus_weekday":np.asarray(weekend)-np.asarray(weekday)}


def run_rhythm_concepts(cfg):
 out=Path(cfg.output_dir);setup(out);d=load_d(cfg);dev,test=split(d,cfg);rep=load_rep(Path(cfg.representation_dir));act=np.load(Path(cfg.cache_dir)/"activity_7d.npy",mmap_mode="r");obs=np.load(Path(cfg.cache_dir)/"observed_7d.npy",mmap_mode="r");ci=d.cache_index.to_numpy(int);parts=[];quality=[]
 for st in range(0,len(d),cfg.rhythm_batch_size):
  en=min(st+cfg.rhythm_batch_size,len(d));a=np.asarray(act[ci[st:en]],np.float32);m=np.asarray(obs[ci[st:en]],bool);q=pd.DataFrame(rhythm_batch(a,m,d.accelerometer_start_date.iloc[st:en].to_numpy()));q.insert(0,"eid",d.eid.iloc[st:en].to_numpy());parts.append(q)
  _,counts=full_sequence_iv(a,m);qc=pd.DataFrame(counts);qc.insert(0,"eid",d.eid.iloc[st:en].to_numpy());qc["minimum_observed_minutes_per_day"]=(m&np.isfinite(a)).sum(axis=2).min(axis=1);quality.append(qc)
  if en%4096==0 or en==len(d):logging.info("Rhythm concepts %d/%d",en,len(d))
 concepts=pd.concat(parts,ignore_index=True);concepts.to_csv(out/"posthoc_rhythm_concepts.csv",index=False);pd.concat(quality,ignore_index=True).to_csv(out/"rhythm_record_quality.csv",index=False);q=d[["eid","target","enrollment_age","gender"]].merge(concepts,on="eid",validate="one_to_one");lookup={e:i for i,e in enumerate(rep["eid"].astype(str))};rows=[];corr=[];summaries=[];sensitivity=[];concept_cols=[c for c in concepts if c!="eid"]
 dev_stats={c:distribution_summary(pd.to_numeric(q.iloc[dev][c],errors="coerce")) for c in concept_cols}
 cross_day=["day_mean_sd","sameclock_between_day_sd","adjacent_day_absolute_change"]
 for split_name,ids in [("development",dev),("test",test)]:
  z=q.iloc[ids];rr=np.array([lookup[str(e)] for e in z.eid],int);risk=rep["final_probability"][rr]
  for c in concept_cols:
   v=pd.to_numeric(z[c],errors="coerce").to_numpy();age=pd.to_numeric(z.enrollment_age,errors="coerce").to_numpy();gender=pd.to_numeric(z.gender,errors="coerce").to_numpy();sex=(gender==2).astype(float);y=z.target.to_numpy(int);ok=np.isfinite(v)&np.isfinite(age)&np.isfinite(gender)
   summary=distribution_summary(v[ok]);summaries.append({"split":split_name,"concept":c,"events":int(y[ok].sum()),**summary})
   if ok.sum()<100 or np.nanstd(v[ok])==0:continue
   vz=(v[ok]-v[ok].mean())/v[ok].std();az=(age[ok]-age[ok].mean())/age[ok].std();r=glm_term(y[ok],np.c_[vz,az,sex[ok]],["concept","age","female"],"concept");rows.append({"split":split_name,"concept":c,"n":int(ok.sum()),"events":int(y[ok].sum()),"concept_mean":summary["mean"],"concept_sd":summary["sd"],"standardization":"within_partition_population_sd","control_median":np.nanmedian(v[ok&(y==0)]),"incident_median":np.nanmedian(v[ok&(y==1)]),"spearman_with_final_risk":stats.spearmanr(v[ok],risk[ok]).statistic,**r})
   ds=dev_stats[c]
   if np.isfinite(ds["sd"]) and ds["sd"]>0:
    common=(v[ok]-ds["mean"])/ds["sd"]
    sensitivity.append({"split":split_name,"concept":c,"analysis":"development_sd_age_sex","n":int(ok.sum()),"events":int(y[ok].sum()),"concept_reference_mean":ds["mean"],"concept_reference_scale":ds["sd"],**glm_term(y[ok],np.c_[common,az,sex[ok]],["concept","age","female"],"concept")})
   if c in cross_day:
    volume=pd.to_numeric(z.overall_activity,errors="coerce").to_numpy();both=ok&np.isfinite(volume);vs=dev_stats["overall_activity"]
    if both.sum()>=100 and ds["sd"]>0 and vs["sd"]>0:
     common=(v[both]-ds["mean"])/ds["sd"];az_b=(age[both]-age[both].mean())/age[both].std();volume_z=(volume[both]-vs["mean"])/vs["sd"]
     sensitivity.append({"split":split_name,"concept":c,"analysis":"development_sd_age_sex_overall_activity","n":int(both.sum()),"events":int(y[both].sum()),"concept_reference_mean":ds["mean"],"concept_reference_scale":ds["sd"],**glm_term(y[both],np.c_[common,az_b,sex[both],volume_z],["concept","age","female","overall_activity"],"concept")})
    ratio=np.divide(v,volume,out=np.full(len(v),np.nan),where=np.isfinite(volume)&(volume>0));dv=pd.to_numeric(q.iloc[dev][c],errors="coerce").to_numpy();dm=pd.to_numeric(q.iloc[dev].overall_activity,errors="coerce").to_numpy();dr=np.divide(dv,dm,out=np.full(len(dv),np.nan),where=np.isfinite(dm)&(dm>0));rs=distribution_summary(dr);rok=ok&np.isfinite(ratio)
    if rok.sum()>=100 and rs["sd"]>0:
     rz=(ratio[rok]-rs["mean"])/rs["sd"];az_r=(age[rok]-age[rok].mean())/age[rok].std()
     sensitivity.append({"split":split_name,"concept":c,"analysis":"ratio_to_overall_activity_development_sd_age_sex","n":int(rok.sum()),"events":int(y[rok].sum()),"concept_reference_mean":rs["mean"],"concept_reference_scale":rs["sd"],**glm_term(y[rok],np.c_[rz,az_r,sex[rok]],["concept","age","female"],"concept")})
   for j in range(min(cfg.association_pcs,rep["pcs"].shape[1])):corr.append({"split":split_name,"concept":c,"latent_component":f"PC{j+1:02d}","spearman_r":stats.spearmanr(v[ok],rep["pcs"][rr,j][ok]).statistic,"p_value":stats.spearmanr(v[ok],rep["pcs"][rr,j][ok]).pvalue,"n":int(ok.sum())})
 a=pd.DataFrame(rows)
 if len(a):a["fdr_q_value_within_split"]=a.groupby("split")["p_value"].transform(bh_fdr)
 a.to_csv(out/"rhythm_concept_dementia_associations.csv",index=False);c=pd.DataFrame(corr)
 if len(c):c["fdr_q_value_within_split"]=c.groupby("split")["p_value"].transform(bh_fdr)
 c.to_csv(out/"rhythm_concept_latent_correlations.csv",index=False);pd.DataFrame(summaries).to_csv(out/"rhythm_concept_distribution_diagnostics.csv",index=False);s=pd.DataFrame(sensitivity)
 if len(s):s["fdr_q_value_within_split_analysis"]=s.groupby(["split","analysis"])["p_value"].transform(bh_fdr)
 s.to_csv(out/"rhythm_concept_sensitivity_associations.csv",index=False)
 atomic_json(out/"COMPLETED.json",{"completed":True,"concepts":concept_cols,"role":"post-hoc behavioural analysis; no concept entered model training","discovery":"development","replication":"fixed test","causal_interpretation":False,"intradaily_variability":{"resolution_minutes":1,"record_minutes":10080,"numerator":"mean squared change across adjacent observed minutes, including day boundaries","denominator":"population variance of observed minutes","missingness":"no imputation or bridging of missing gaps"},"primary_concept_model":"age and sex adjusted; within-partition SD; no volume covariate","sensitivity_models":"separate output; development-fixed SD, age-sex-volume adjustment and range/overall-activity ratios","diagnostics":"partition distributions, convergence, design rank and influence summaries"});logging.info("COMPLETED rhythm concept analyses")


def cox_fit(duration,event,X,names):
 try:
  from statsmodels.duration.hazard_regression import PHReg
  fit=PHReg(duration,np.asarray(X,float),status=event,ties="efron").fit(disp=0);rows=[]
  for j,n in enumerate(names):rows.append({"term":n,"coefficient":fit.params[j],"se":fit.bse[j],"hazard_ratio":np.exp(fit.params[j]),"ci_low":np.exp(fit.params[j]-1.96*fit.bse[j]),"ci_high":np.exp(fit.params[j]+1.96*fit.bse[j]),"p_value":fit.pvalues[j]})
  return rows
 except Exception as e:
  logging.exception("Cox model failed");return [{"term":n,"coefficient":np.nan,"se":np.nan,"hazard_ratio":np.nan,"ci_low":np.nan,"ci_high":np.nan,"p_value":np.nan,"error":repr(e)} for n in names]


def km_curve(t,e,group):
 rows=[]
 for level in pd.unique(group):
  m=group==level;tt=t[m];ee=e[m];surv=1.;at=len(tt);rows.append({"group":level,"time_years":0.,"n_at_risk":at,"events":0,"survival":1.})
  for x in np.sort(np.unique(tt[ee==1])):
   ne=int(np.sum((tt==x)&(ee==1)));nr=int(np.sum(tt>=x));surv*=1-ne/nr;rows.append({"group":level,"time_years":x,"n_at_risk":nr,"events":ne,"survival":surv})
 return rows


def observed_followup_inputs(d,cfg):
 """Read explicitly named times measured in days after accelerometry.

 followup_days_col is the last observable/administrative day, not a duration
 since recruitment. death_days_col is the recorded death day; missing means
 no recorded death time. A binary is_dead flag never supplies a death time.
 """
 follow_col=getattr(cfg,"followup_days_col","")
 death_col=getattr(cfg,"death_days_col","")
 if not follow_col:
  if death_col:raise ValueError("--death-days-col requires --followup-days-col")
  return None
 required=[follow_col]+([death_col] if death_col else [])
 missing=[name for name in required if name not in d]
 if missing:raise ValueError(f"Configured follow-up columns missing: {missing}")
 follow=pd.to_numeric(d[follow_col],errors="coerce").to_numpy(float)
 death=pd.to_numeric(d[death_col],errors="coerce").to_numpy(float) if death_col else np.full(len(d),np.nan)
 valid=np.isfinite(follow)&(follow>0)&(~np.isfinite(death)|(death>=0))
 unknown_death=np.zeros(len(d),bool)
 if death_col and "is_dead" in d:
  unknown_death=pd.to_numeric(d.is_dead,errors="coerce").eq(1).to_numpy()&~np.isfinite(death)
  valid&=~unknown_death
 return {"followup_days":follow,"death_days":death,"valid":valid,
         "followup_column":follow_col,"death_column":death_col,
         "known_death_without_time":unknown_death}


def observed_survival_sensitivity(cfg,d,test,risk,agez,sex,out):
 """Additional cause-specific analyses; fixed-window outputs stay separate."""
 times=observed_followup_inputs(d,cfg)
 if times is None:return {"enabled":False}
 follow=times["followup_days"][test];death=times["death_days"][test]
 event=d.target.to_numpy(int)[test]
 days=pd.to_numeric(d.days_accel_to_dementia,errors="coerce").to_numpy(float)[test]
 stop=np.minimum(follow,HORIZON_DAYS)
 stop=np.minimum(stop,np.where(np.isfinite(death),death,np.inf))
 invalid_order=(event==1)&np.isfinite(death)&np.isfinite(days)&(days>death)
 eligible=times["valid"][test]&(stop>LANDMARK_DAYS)&~invalid_order&np.isfinite(risk)&np.isfinite(agez)&np.isfinite(sex)
 observed_event=(event==1)&np.isfinite(days)&(days<=stop)
 end=np.where(observed_event,days,stop);duration=(end-LANDMARK_DAYS)/365.25
 eligible&=np.isfinite(duration)&(duration>0)
 pd.DataFrame({"eid":d.eid.iloc[test].to_numpy(),"included":eligible,"fixed_window_event":event,"observed_event":observed_event.astype(int),"followup_days":follow,"death_days":death,"end_days":end,"diagnosis_after_death":invalid_order,"known_death_without_time":times["known_death_without_time"][test]}).to_csv(out/"observed_followup_eligibility.csv",index=False)
 rows=[]
 if observed_event[eligible].sum()>0:
  for model,X,names in [("risk_only",risk[:,None],["final_risk_per_SD"]),("risk_age_sex",np.c_[risk,agez,sex],["final_risk_per_SD","age_per_SD","female"])]:
   for result in cox_fit(duration[eligible],observed_event[eligible].astype(int),X[eligible],names):rows.append({"model":model,"n":int(eligible.sum()),"events":int(observed_event[eligible].sum()),**result})
 pd.DataFrame(rows).to_csv(out/"observed_followup_cox_models.csv",index=False)
 horizon_rows=[]
 for years in [2.,3.,5.,7.]:
  cutoff=365.25*years
  case=eligible&observed_event&(days<=cutoff)
  control=eligible&(end>=cutoff)&~case
  use=case|control
  horizon_rows.append({"horizon_years_from_accelerometer":years,"n_evaluable":int(use.sum()),"events_by_horizon":int(case.sum()),"complete_case_auroc":roc_auc_score(case[use].astype(int),risk[use]) if case.any() and control.any() else np.nan,"method":"observed by horizon; no IPCW"})
 pd.DataFrame(horizon_rows).to_csv(out/"observed_followup_horizon_discrimination.csv",index=False)
 return {"enabled":True,"followup_days_column":times["followup_column"],"death_days_column":times["death_column"],"time_origin":"accelerometry","n_included":int(eligible.sum()),"events":int(observed_event[eligible].sum()),"diagnosis_after_death_excluded":int(invalid_order.sum()),"known_death_without_time_excluded":int(times["known_death_without_time"][test].sum()),"death_handling":"cause-specific censoring when a recorded death time is supplied; no competing-risk cumulative-incidence estimator","cohort_scope":"the original fixed-window analytical cohort; excluded source participants are not recovered"}


def run_survival(cfg):
 out=Path(cfg.output_dir);setup(out);d=load_d(cfg);d,_=merge_optional_clinical(d,cfg.clinical_outcomes_csv);_,test=split(d,cfg);rep=load_rep(Path(cfg.representation_dir));lookup={e:i for i,e in enumerate(rep["eid"].astype(str))};r=np.array([lookup[str(e)] for e in d.eid.iloc[test]],int);event=d.target.to_numpy(int)[test];days=pd.to_numeric(d.days_accel_to_dementia,errors="coerce").to_numpy()[test];duration=np.where(event==1,days-LANDMARK_DAYS,HORIZON_DAYS-LANDMARK_DAYS)/365.25;age=pd.to_numeric(d.enrollment_age,errors="coerce").to_numpy()[test];sex=(pd.to_numeric(d.gender,errors="coerce").to_numpy()[test]==2).astype(float);risk=safe_logit(rep["final_probability"][r]);risk=(risk-risk.mean())/risk.std();agez=(age-np.nanmean(age))/np.nanstd(age);rows=[]
 for model,X,names in [("risk_only",risk[:,None],["final_risk_per_SD"]),("risk_age_sex",np.c_[risk,agez,sex],["final_risk_per_SD","age_per_SD","female"] )]:
  for q in cox_fit(duration,event,X,names):rows.append({"model":model,"n":len(event),"events":int(event.sum()),**q})
 pd.DataFrame(rows).to_csv(out/"landmark_cox_models.csv",index=False);quart=np.asarray(pd.qcut(risk,4,labels=["Q1","Q2","Q3","Q4"],duplicates="drop").astype(str));pd.DataFrame(km_curve(duration,event,quart)).to_csv(out/"landmark_km_by_risk_quartile.csv",index=False)
 auc=[]
 years_from_accel=duration+1
 for horizon in [2.,3.,5.,7.]:
  case=(event==1)&(years_from_accel<=horizon);control=((event==0)&(years_from_accel>=horizon))|((event==1)&(years_from_accel>horizon));use=case|control
  auc.append({"horizon_years_from_accelerometer":horizon,"n_evaluable":int(use.sum()),"events_by_horizon":int(case.sum()),"dynamic_auroc":roc_auc_score(case[use].astype(int),risk[use]) if case.sum() and control.sum() else np.nan})
 pd.DataFrame(auc).to_csv(out/"time_dependent_auc.csv",index=False)
 # Cause-specific subtype survival: another dementia type is censored at its first all-cause event.
 subframe,labs=subtype_labels(d,cfg.subtypes_csv);subrows=[]
 for name,ys in labs.items():
  ev=ys[test];subdays=pd.to_numeric(subframe[SUBTYPES[name]],errors="coerce").to_numpy()[test];dur=np.where(ev==1,subdays-LANDMARK_DAYS,np.where(event==1,days-LANDMARK_DAYS,HORIZON_DAYS-LANDMARK_DAYS))/365.25
  ok=np.isfinite(dur)&(dur>0)
  if ev[ok].sum()>=5:
   for q in cox_fit(dur[ok],ev[ok],np.c_[risk[ok],agez[ok],sex[ok]],["final_risk_per_SD","age_per_SD","female"]):subrows.append({"subtype":name,"n":int(ok.sum()),"events":int(ev[ok].sum()),**q})
 pd.DataFrame(subrows).to_csv(out/"cause_specific_subtype_cox.csv",index=False)
 observed=observed_survival_sensitivity(cfg,d,test,risk,agez,sex,out)
 atomic_json(out/"COMPLETED.json",{"completed":True,"design":"365-day landmark; follow from year 1 through year 7","control_censoring":"administrative at 7 years in the fixed-horizon eligible cohort","death_competing_risk_performed":False,"horizon_auc_method":"binary horizon labels; no IPCW","observed_followup_sensitivity":observed,"time_input_contract":"optional day fields start at accelerometry; no death time inferred from is_dead"});logging.info("COMPLETED survival analyses")


def glm_term(y,X,names,target):
 return binomial_term(y,X,names,target)


def merge_optional_clinical(d,path):
 p=Path(path)
 if not p.exists():return d.copy(),[]
 x=pd.read_csv(p,dtype={"eid":"string"},low_memory=False);x.eid=x.eid.str.strip();extra=[c for c in x if c!="eid" and c not in d]
 return d.merge(x[["eid",*extra]],on="eid",how="left",validate="one_to_one"),extra


def comorbidity_columns(d,spec):
 requested=[x.strip() for x in spec.split(",") if x.strip()]
 candidates=requested or list(dict.fromkeys(AUTO_COMORBIDITY+[c for c in d.columns if any(k in c.lower() for k in ["comorbid","charlson","elixhauser"]) ]))
 return [c for c in candidates if c in d and pd.to_numeric(d[c],errors="coerce").notna().sum()>0]


def run_interactions(cfg):
 out=Path(cfg.output_dir);setup(out);d=load_d(cfg);d,extra=merge_optional_clinical(d,cfg.clinical_outcomes_csv);_,test=split(d,cfg);rep=load_rep(Path(cfg.representation_dir));lookup={e:i for i,e in enumerate(rep["eid"].astype(str))};r=np.array([lookup[str(e)] for e in d.eid.iloc[test]],int);y=d.target.to_numpy(int)[test];risk=safe_logit(rep["final_probability"][r]);risk=(risk-risk.mean())/risk.std();age=pd.to_numeric(d.enrollment_age,errors="coerce").to_numpy()[test];age=(age-np.nanmean(age))/np.nanstd(age);sex=(pd.to_numeric(d.gender,errors="coerce").to_numpy()[test]==2).astype(float);e4=pd.to_numeric(d.apoe_e4,errors="coerce").to_numpy()[test];mods={"age":age,"sex_female":sex,"APOE_e4_dosage":e4};mods.update({c:pd.to_numeric(d[c],errors="coerce").to_numpy()[test] for c in comorbidity_columns(d,cfg.comorbidity_cols)});rows=[]
 for name,m in mods.items():
  ok=np.isfinite(risk)&np.isfinite(age)&np.isfinite(sex)&np.isfinite(m);mz=(m[ok]-np.mean(m[ok]))/np.std(m[ok]) if name not in ["sex_female"] else m[ok];X=np.c_[risk[ok],mz,age[ok],sex[ok],risk[ok]*mz];names=["risk","modifier","age","female","risk_x_modifier"];rows.append({"modifier":name,"n":int(ok.sum()),"events":int(y[ok].sum()),**glm_term(y[ok],X,names,"risk_x_modifier")})
 z=pd.DataFrame(rows);z["fdr_q_value"]=bh_fdr(pd.to_numeric(z.get("p_value"),errors="coerce"));z.to_csv(out/"risk_score_interaction_tests.csv",index=False);atomic_json(out/"COMPLETED.json",{"completed":True,"interpretation":"Exploratory interaction tests on the fixed test set; subgroup-specific performance alone is not an interaction test.","comorbidity_columns_found":comorbidity_columns(d,cfg.comorbidity_cols)});logging.info("COMPLETED interaction tests")


def fit_binary_probe(Xd,yd,Xt):
 m=Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),("sc",StandardScaler()),("lr",LogisticRegression(C=.03,class_weight="balanced",solver="liblinear",max_iter=4000))]).fit(Xd,yd);return m,m.predict_proba(Xt)[:,1]


def mortality_time_inputs(d,cfg):
 """Use recorded death/censor times, or the supplied endpoint-age field.

 The endpoint-age field must contain age at death for deceased participants
 and age at censoring for survivors. enrollment_age is age at accelerometry.
 An explicit mortality-time column follows that same death/censor contract.
 """
 duration_col=getattr(cfg,"mortality_time_days_col","")
 if duration_col:
  if duration_col not in d or "is_dead" not in d:raise ValueError("--mortality-time-days-col requires the named column and is_dead")
  follow=pd.to_numeric(d[duration_col],errors="coerce").to_numpy(float)/365.25
  event=pd.to_numeric(d.is_dead,errors="coerce").to_numpy(float)
  valid=np.isfinite(follow)&(follow>0)&np.isin(event,[0,1])
  return follow,event,valid,{"source":duration_col,"unit":"days from accelerometry to death or censoring"}
 if getattr(cfg,"followup_days_col","") and getattr(cfg,"death_days_col",""):
  times=observed_followup_inputs(d,cfg);end=times["followup_days"];death=times["death_days"]
  event=(np.isfinite(death)&(death<=end)).astype(int)
  follow=np.minimum(end,np.where(np.isfinite(death),death,np.inf))/365.25
  valid=times["valid"]&np.isfinite(follow)&(follow>0)
  return follow,event,valid,{"source":"explicit observation and death day columns","followup_column":times["followup_column"],"death_column":times["death_column"],"time_origin":"accelerometry"}
 if {"is_dead","Followingyears_birthday","enrollment_age"}.issubset(d.columns):
  follow=pd.to_numeric(d.Followingyears_birthday,errors="coerce").to_numpy()-pd.to_numeric(d.enrollment_age,errors="coerce").to_numpy();event=pd.to_numeric(d.is_dead,errors="coerce").to_numpy()
  valid=np.isfinite(follow)&(follow>0)&np.isin(event,[0,1])
  return follow,event,valid,{"source":"Followingyears_birthday minus enrollment_age","unit":"years","field_contract":"endpoint age is age at death for deceased participants and age at censoring for survivors; enrollment_age is age at accelerometry"}
 return None


def run_clinical_associations(cfg):
 out=Path(cfg.output_dir);setup(out);d=load_d(cfg);d,extra=merge_optional_clinical(d,cfg.clinical_outcomes_csv);dev,test=split(d,cfg);rep=load_rep(Path(cfg.representation_dir));lookup={e:i for i,e in enumerate(rep["eid"].astype(str))};rd=np.array([lookup[str(e)] for e in d.eid.iloc[dev]],int);rt=np.array([lookup[str(e)] for e in d.eid.iloc[test]],int);age=pd.to_numeric(d.enrollment_age,errors="coerce").to_numpy();sex=pd.to_numeric(d.gender,errors="coerce").to_numpy();Xd=np.c_[rep["pcs"][rd],age[dev],sex[dev]];Xt=np.c_[rep["pcs"][rt],age[test],sex[test]];rows=[];assoc=[]
 outcomes=[]
 if "is_dead" in d:outcomes.append(("is_dead","binary"))
 outcomes += [(c,"continuous") for c in comorbidity_columns(d,cfg.comorbidity_cols)]
 for name,kind in outcomes:
  yd=pd.to_numeric(d[name],errors="coerce").to_numpy()[dev];yt=pd.to_numeric(d[name],errors="coerce").to_numpy()[test];od=np.isfinite(yd);ot=np.isfinite(yt)
  feature_sets={"age_sex":(np.c_[age[dev],sex[dev]],np.c_[age[test],sex[test]]),"final_risk_plus_age_sex":(np.c_[safe_logit(rep["final_probability"][rd]),age[dev],sex[dev]],np.c_[safe_logit(rep["final_probability"][rt]),age[test],sex[test]]),"latent_PCs_plus_age_sex":(Xd,Xt)}
  if kind=="binary" and len(np.unique(yd[od]))==2:
   for label,(xdev,xtest) in feature_sets.items():
    _,p=fit_binary_probe(xdev[od],yd[od].astype(int),xtest[ot]);rows.append({"outcome":name,"analysis":label+"_probe","test_n":int(ot.sum()),"test_events":int(yt[ot].sum()),"auroc":roc_auc_score(yt[ot],p),"auprc":average_precision_score(yt[ot],p),"prevalence":yt[ot].mean(),"brier":brier_score_loss(yt[ot],p)})
  elif kind=="continuous" and od.sum()>100 and ot.sum()>20:
   for label,(xdev,xtest) in feature_sets.items():
    m=Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),("sc",StandardScaler()),("ridge",RidgeCV(alphas=[.1,1.,10.,100.]))]).fit(xdev[od],yd[od]);p=m.predict(xtest[ot]);rows.append({"outcome":name,"analysis":label+"_probe","test_n":int(ot.sum()),"r2":r2_score(yt[ot],p),"mae":mean_absolute_error(yt[ot],p),"spearman_r":stats.spearmanr(yt[ot],p).statistic})
  # Adjusted risk and PC associations are discovery/replication, not predictors.
  for split_name,ids,rr in [("development",dev,rd),("test",test,rt)]:
   yy=pd.to_numeric(d[name],errors="coerce").to_numpy()[ids];aa=age[ids];ss=sex[ids];risk=safe_logit(rep["final_probability"][rr]);features={"final_risk":risk,**{f"PC{j+1:02d}":rep["pcs"][rr,j] for j in range(min(cfg.association_pcs,rep["pcs"].shape[1]))}}
   for fn,v in features.items():
    ok=np.isfinite(yy)&np.isfinite(aa)&np.isfinite(ss)&np.isfinite(v)
    if ok.sum()<30:continue
    vz=(v[ok]-v[ok].mean())/v[ok].std();az=(aa[ok]-aa[ok].mean())/aa[ok].std();sx=(ss[ok]==2).astype(float)
    if kind=="binary" and len(np.unique(yy[ok]))==2:r=glm_term(yy[ok].astype(int),np.c_[vz,az,sx],["feature","age","female"],"feature")
    else:
     X=np.c_[np.ones(ok.sum()),vz,az,sx];coef=np.linalg.lstsq(X,yy[ok],rcond=None)[0];res=yy[ok]-X@coef;df=max(ok.sum()-X.shape[1],1);s2=np.sum(res**2)/df;cov=s2*np.linalg.pinv(X.T@X);se=np.sqrt(cov[1,1]);r={"coefficient":coef[1],"se":se,"ci_low":coef[1]-1.96*se,"ci_high":coef[1]+1.96*se,"p_value":2*stats.t.sf(abs(coef[1]/se),df)}
    assoc.append({"outcome":name,"outcome_type":kind,"split":split_name,"feature":fn,"n":int(ok.sum()),**r})
 pd.DataFrame(rows).to_csv(out/"clinical_linear_probe_test_metrics.csv",index=False);a=pd.DataFrame(assoc);a["fdr_q_value_within_split_outcome"]=a.groupby(["outcome","split"])["p_value"].transform(bh_fdr) if len(a) else [];a.to_csv(out/"latent_clinical_associations.csv",index=False)
 mortality=[];mortality_inputs=mortality_time_inputs(d,cfg);mortality_contract={"available":False}
 if mortality_inputs is not None:
  follow,death,time_valid,mortality_contract=mortality_inputs
  for split_name,ids,rr in [("development",dev,rd),("test",test,rt)]:
   risk=safe_logit(rep["final_probability"][rr]);aa=age[ids];sx=(sex[ids]==2).astype(float);ok=time_valid[ids]&np.isfinite(aa)&np.isfinite(sex[ids])&np.isfinite(risk)
   if ok.sum()<5 or death[ids][ok].sum()==0:continue
   rz=(risk[ok]-risk[ok].mean())/risk[ok].std();az=(aa[ok]-aa[ok].mean())/aa[ok].std()
   for q in cox_fit(follow[ids][ok],death[ids][ok].astype(int),np.c_[rz,az,sx[ok]],["final_risk_per_SD","age_per_SD","female"]):mortality.append({"split":split_name,"model":"final_risk_age_sex","n":int(ok.sum()),"deaths":int(death[ids][ok].sum()),**q})
   for j in range(min(4,rep["pcs"].shape[1])):
    pc=rep["pcs"][rr,j][ok];pc=(pc-pc.mean())/pc.std()
    for q in cox_fit(follow[ids][ok],death[ids][ok].astype(int),np.c_[pc,az,sx[ok]],[f"PC{j+1:02d}_per_SD","age_per_SD","female"]):mortality.append({"split":split_name,"model":f"PC{j+1:02d}_age_sex","n":int(ok.sum()),"deaths":int(death[ids][ok].sum()),**q})
 pd.DataFrame(mortality).to_csv(out/"mortality_cox_associations.csv",index=False);atomic_json(out/"COMPLETED.json",{"completed":True,"outcomes_analyzed":[x[0] for x in outcomes],"optional_columns_merged":extra,"comorbidity_columns_found":comorbidity_columns(d,cfg.comorbidity_cols),"mortality_time":mortality_contract,"warning":"Post-hoc representation associations do not establish causality and outcome variables never enter the dementia predictor."});logging.info("COMPLETED clinical representation associations")


def paired_delta(y,a,b,n,seed):
 rng=np.random.default_rng(seed);pos=np.where(y==1)[0];neg=np.where(y==0)[0];da=[];du=[];db=[]
 for _ in range(n):
  ix=np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)];da.append(average_precision_score(y[ix],a[ix])-average_precision_score(y[ix],b[ix]));du.append(roc_auc_score(y[ix],a[ix])-roc_auc_score(y[ix],b[ix]));db.append(brier_score_loss(y[ix],a[ix])-brier_score_loss(y[ix],b[ix]))
 return {"delta_auprc":average_precision_score(y,a)-average_precision_score(y,b),"delta_auprc_ci_low":np.quantile(da,.025),"delta_auprc_ci_high":np.quantile(da,.975),"delta_auroc":roc_auc_score(y,a)-roc_auc_score(y,b),"delta_auroc_ci_low":np.quantile(du,.025),"delta_auroc_ci_high":np.quantile(du,.975),"delta_brier":brier_score_loss(y,a)-brier_score_loss(y,b),"delta_brier_ci_low":np.quantile(db,.025),"delta_brier_ci_high":np.quantile(db,.975)}


def run_final_paired(cfg):
 out=Path(cfg.output_dir);setup(out);rows=[];ab=pd.read_csv(Path(cfg.ablation_dir)/"ablation_test_predictions.csv",dtype={"eid":"string"});rows.append({"comparison":"decision_only minus prespecified final_hybrid","n":len(ab),"events":int(ab.target.sum()),**paired_delta(ab.target.to_numpy(int),ab.probability_decision_only_no_feature_interaction.to_numpy(float),ab.probability_final_hybrid.to_numpy(float),cfg.bootstrap_repeats,cfg.seed)})
 ap=pd.read_csv(Path(cfg.apoe_dir)/"apoe_complete_test_predictions.csv",dtype={"eid":"string"});cols=list(ap.columns);new=next((c for c in cols if c in ["probability_final_hybrid_apoe","final_hybrid_apoe"]),None);ref=next((c for c in cols if c in ["probability_final_hybrid_fullcohort_restricted","final_hybrid_fullcohort_restricted"]),None)
 if new and ref:rows.append({"comparison":"APOE hybrid minus full-cohort hybrid on identical complete cases","n":len(ap),"events":int(ap.target.sum()),**paired_delta(ap.target.to_numpy(int),ap[new].to_numpy(float),ap[ref].to_numpy(float),cfg.bootstrap_repeats,cfg.seed+1)})
 else:logging.warning("APOE paired columns not found; columns=%s",cols)
 pd.DataFrame(rows).to_csv(out/"final_paired_increment_tests.csv",index=False);atomic_json(out/"COMPLETED.json",{"completed":True,"comparisons":[r["comparison"] for r in rows],"method":"stratified participant-level paired bootstrap"});logging.info("COMPLETED final paired comparisons")


def smoke(cfg):
 out=Path(cfg.output_dir);setup(out);required=[Path(cfg.core_dir)/"COMPLETED.json",Path(cfg.protocol_dir)/"PROTOCOL_LOCK.json",Path(cfg.subtypes_csv),Path(cfg.cache_dir)/"activity_7d.npy",Path(cfg.cache_dir)/"observed_7d.npy"];missing=[str(p) for p in required if not p.exists()]
 if missing:raise FileNotFoundError("Missing clinical-extension prerequisites:\n"+"\n".join(missing))
 d=pd.read_csv(cfg.labels_csv,nrows=2);s=pd.read_csv(cfg.subtypes_csv,nrows=2);cp=Path(cfg.clinical_outcomes_csv);clinical=pd.read_csv(cp,low_memory=False) if cp.exists() else pd.DataFrame();detected=comorbidity_columns(clinical,cfg.comorbidity_cols) if len(clinical.columns) else [];audit={"completed":True,"labels_columns":list(d.columns),"subtype_columns":list(s.columns),"clinical_outcomes_csv":str(cp),"clinical_outcomes_csv_exists":cp.exists(),"clinical_outcome_columns":list(clinical.columns),"requested_comorbidity_columns":cfg.comorbidity_cols or "auto-detect","detected_comorbidity_columns":detected,"warning":None if detected else "No comorbidity index detected; specify --comorbidity-cols for the clinical association stages.","weights_changed":False};atomic_json(out/"SMOKE_COMPLETED.json",audit);logging.info("Clinical extension smoke OK comorbidity=%s",detected)


def parse():
 a=argparse.ArgumentParser(description=__doc__)
 a.add_argument("stage",choices=["clinical_smoke","representations","subtypes","temporal_shapley","rhythm_concepts","survival","interactions","clinical_associations","final_paired"])
 a.add_argument("--labels-csv",default=str(ROOT/"ukb_dementia_labels_covariates_N80990.csv"))
 a.add_argument("--apoe-csv",default=str(ROOT/"APOE_phenotype.csv"))
 a.add_argument("--subtypes-csv",default=str(ROOT/"ukb_dementia_subtypes_N80990.csv"))
 a.add_argument("--clinical-outcomes-csv",default=str(ROOT/"UKB_Demographic_clinic_PA_baseline.csv"),help="Additional EID-aligned clinical fields, including explicitly named time columns if supplied.")
 a.add_argument("--comorbidity-cols",default="")
 a.add_argument("--followup-days-col",default="",help="Optional last observable/administrative day since accelerometry; produces separate observed-follow-up sensitivities within the fixed analytical cohort.")
 a.add_argument("--death-days-col",default="",help="Optional recorded day of death since accelerometry; requires --followup-days-col. No time is inferred from is_dead.")
 a.add_argument("--mortality-time-days-col",default="",help="Optional days from accelerometry to death (is_dead=1) or censoring (is_dead=0); takes precedence for mortality Cox models.")
 a.add_argument("--cache-dir",default=str(ROOT/"cache"))
 a.add_argument("--core-dir",default=str(RESULTS/"02_core_model"))
 a.add_argument("--protocol-dir",default=str(RESULTS/"00_protocol"))
 a.add_argument("--representation-dir",default=str(RESULTS/"11_representations"))
 a.add_argument("--ablation-dir",default=str(RESULTS/"08_ablation"))
 a.add_argument("--apoe-dir",default=str(RESULTS/"03_apoe_secondary"))
 a.add_argument("--output-dir",required=True)
 a.add_argument("--bootstrap-repeats",type=int,default=2000)
 a.add_argument("--explanation-bootstrap-repeats",type=int,default=1000)
 a.add_argument("--shapley-permutations",type=int,default=16)
 a.add_argument("--explanation-control-ratio",type=int,default=1)
 a.add_argument("--reference-participants",type=int,default=1024)
 a.add_argument("--pca-components",type=int,default=16)
 a.add_argument("--association-pcs",type=int,default=16)
 a.add_argument("--rhythm-batch-size",type=int,default=256)
 a.add_argument("--minimum-subtype-development-events",type=int,default=20)
 a.add_argument("--minimum-subtype-test-events",type=int,default=5)
 a.add_argument("--extract-batch-size",type=int,default=128)
 a.add_argument("--eval-batch-size",type=int,default=256)
 a.add_argument("--num-workers",type=int,default=6)
 a.add_argument("--seed",type=int,default=20260820)
 a.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True)
 return a.parse_args()


if __name__=="__main__":
 cfg=parse();dispatch={"clinical_smoke":smoke,"representations":run_representations,"subtypes":run_subtypes,"temporal_shapley":run_temporal_shapley,"rhythm_concepts":run_rhythm_concepts,"survival":run_survival,"interactions":run_interactions,"clinical_associations":run_clinical_associations,"final_paired":run_final_paired};dispatch[cfg.stage](cfg)
