#!/usr/bin/env python3
"""Frozen UKB-wide 80:20 retraining protocol for the final dementia system.

Model/epoch selection occurs only inside the 80% development partition. After
epochs are locked, both raw-PA backbones and fusion heads are reinitialized and
refit on the complete 80%. The 20% test is inferred once and is the only UKB
performance result intended for the main manuscript.
"""
from __future__ import annotations
import argparse, hashlib, json, logging, os, random, sys, time
from datetime import datetime
from pathlib import Path

for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    if not os.environ.get(_v, "").strip().isdigit() or int(os.environ.get(_v, "0")) < 1: os.environ[_v] = "4"

import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import v13_cohort_engine as v6
import v13_mamba_engine as v11
import v13_fusion_engine as v12

BASE=Path("/root/autodl-tmp/PA_CR_CNN/Project_new/PA_Foundation_Model"); DEM=BASE/"dementia"
DEFAULT_OUT=DEM/"Results"/"v13_final"/"02_core_model"
AGE_SEX=["enrollment_age","gender"]; TRAD=v6.TRAD_PA
MODEL_NAMES=["age_sex","traditional_pa","age_sex_traditional_pa","grufcn_pa","convcontext_pa","dual_pa_feature","final_hybrid"]

def atomic_json(path,obj):
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf-8");os.replace(tmp,path)
def setup(out):
    out.mkdir(parents=True,exist_ok=True);logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",handlers=[logging.FileHandler(out/"run.log",encoding="utf-8"),logging.StreamHandler(sys.stdout)],force=True)
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def sigmoid(x):return v6.sigmoid(x)

def locked_outer_split(d,y,cfg,out):
    path=out/"ukb_fixed_80_20_split.csv";lock=out/"ukb_fixed_80_20_lock.json"
    if path.exists():
        z=pd.read_csv(path,dtype={"eid":"string"});z.eid=z.eid.str.strip()
        if z.eid.tolist()!=d.eid.tolist():raise ValueError("Existing 80:20 split no longer matches cohort")
        logging.info("Reusing immutable UKB 80:20 split")
    else:
        dev,test=next(StratifiedShuffleSplit(1,test_size=.20,random_state=cfg.split_seed).split(np.zeros(len(y)),y));s=np.full(len(y),"development",dtype="U12");s[test]="test"
        z=pd.DataFrame({"eid":d.eid,"split":s});z.to_csv(path,index=False)
        digest=hashlib.sha256("\n".join(z.loc[z.split.eq("test"),"eid"]).encode()).hexdigest();atomic_json(lock,{"created":datetime.now().isoformat(),"split_seed":cfg.split_seed,"development_fraction":.80,"test_fraction":.20,"test_eid_sha256":digest,"rule":"Test is inaccessible until all epochs, weights, calibration and thresholds are locked."})
    return np.where(z.split.eq("development"))[0],np.where(z.split.eq("test"))[0]

def inner_split(dev,y,cfg,out,d):
    a,b=next(StratifiedShuffleSplit(1,test_size=cfg.inner_validation_fraction,random_state=cfg.split_seed+1).split(np.zeros(len(dev)),y[dev]));tr,va=dev[a],dev[b]
    pd.DataFrame({"eid":d.eid.iloc[dev].to_numpy(),"target":y[dev],"role":np.where(np.isin(dev,va),"epoch_selection","inner_training")}).to_csv(out/"development_internal_epoch_selection.csv",index=False)
    return tr,va

def make_week_loader(cfg,d,y,ids,loc,scale,batch,shuffle=False,sampler=None,augment=False):
    idx=d.cache_index.to_numpy(int);dummy=np.zeros((len(ids),1),np.float32);ds=v6.WeekDataset(cfg.cache_dir,idx[ids],y[ids],dummy,loc,scale,augment)
    nw=0 if cfg.mode=="smoke" else cfg.num_workers;return DataLoader(ds,batch_size=batch,shuffle=shuffle if sampler is None else False,sampler=sampler,num_workers=nw,pin_memory=torch.cuda.is_available(),persistent_workers=nw>0,drop_last=augment)

def weighted_sampler(y,ids,q,seed):
    yy=y[ids];pos=max(yy.sum(),1);neg=max((yy==0).sum(),1);w=np.where(yy==1,q/pos,(1-q)/neg);g=torch.Generator().manual_seed(seed);return WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),len(ids),True,generator=g)

def train_backbone_selection(name,cfg,d,y,tr,va,out,seed):
    r=v11.train_one(name,cfg,d,y,tr,va,out,seed);return r["best_epoch"],r["score"],r["history"]

def refit_backbone(name,epochs,cfg,d,y,dev,out,seed):
    seed_all(seed);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");idx=d.cache_index.to_numpy(int);loc,scale=v6.robust_raw_scale(cfg.cache_dir,idx,dev,cfg.seed)
    ssl=make_week_loader(cfg,d,y,dev,loc,scale,cfg.batch_size,shuffle=True,augment=True);tr=make_week_loader(cfg,d,y,dev,loc,scale,cfg.batch_size,sampler=weighted_sampler(y,dev,cfg.sampled_positive_fraction,seed))
    model=v11.build(name,cfg).to(device);logging.info("REFIT %-20s full development N=%d cases=%d fixed_epochs=%d",name,len(dev),int(y[dev].sum()),epochs)
    if cfg.ssl_epochs:
        opt=torch.optim.AdamW(model.parameters(),lr=cfg.ssl_lr,weight_decay=cfg.weight_decay);gs=v6.grad_scaler(device,cfg.amp)
        for ep in range(1,cfg.ssl_epochs+1):
            model.train();losses=[]
            for a,b,_,_ in ssl:
                a=a.to(device);b=b.to(device);opt.zero_grad(set_to_none=True)
                with v6.amp_ctx(device,cfg.amp):loss=v6.nt_xent(model.project(a),model.project(b),cfg.temperature)
                gs.scale(loss).backward();gs.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);gs.step(opt);gs.update();losses.append(float(loss.detach().cpu()))
            logging.info("REFIT %s SSL epoch=%d loss=%.4f",name,ep,np.mean(losses))
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(min(cfg.cosine_epochs,epochs),1),eta_min=cfg.lr*.1);gs=v6.grad_scaler(device,cfg.amp);lossfn=nn.BCEWithLogitsLoss()
    for ep in range(1,epochs+1):
        model.train();losses=[]
        for x,yy,_ in tr:
            x=x.to(device);yy=yy.to(device);opt.zero_grad(set_to_none=True)
            with v6.amp_ctx(device,cfg.amp):loss=lossfn(model(x),yy)
            gs.scale(loss).backward();gs.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);gs.step(opt);gs.update();losses.append(float(loss.detach().cpu()))
        if ep<=cfg.cosine_epochs:sch.step()
        logging.info("REFIT %s epoch=%d/%d loss=%.4f",name,ep,epochs,np.mean(losses))
    path=out/f"{name}_80pct_refit.pt";torch.save({"model":model.state_dict(),"architecture":name,"fixed_epochs":epochs,"loc":loc,"scale":scale,"seed":seed,"config":vars(cfg)},path);return path

def extract_pair(cfg,d,y,ids,gpath,cpath):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");gs=torch.load(gpath,map_location=device);cs=torch.load(cpath,map_location=device)
    gm=v11.build("grufcn",cfg).to(device);cm=v11.build("convcontext_mamba",cfg).to(device);gm.load_state_dict(gs["model"]);cm.load_state_dict(cs["model"]);gm.eval();cm.eval()
    if abs(float(gs["loc"])-float(cs["loc"]))>1e-6 or abs(float(gs["scale"])-float(cs["scale"]))>1e-6:raise ValueError("Backbone raw scaling mismatch")
    loader=make_week_loader(cfg,d,y,ids,float(gs["loc"]),float(gs["scale"]),cfg.extract_batch_size);ge=[];ce=[];gl=[];cl=[];yy=[]
    with torch.no_grad():
        for x,t,_ in loader:
            x=x.to(device)
            with v6.amp_ctx(device,cfg.amp):g=gm.encoder(x);c=cm.encoder(x);a=gm.risk(g).squeeze(-1);b=cm.risk(c).squeeze(-1)
            ge.append(g.float().cpu().numpy());ce.append(c.float().cpu().numpy());gl.append(a.float().cpu().numpy());cl.append(b.float().cpu().numpy());yy.append(t.numpy())
    return {"g":np.concatenate(ge).astype(np.float32),"c":np.concatenate(ce).astype(np.float32),"gl":np.concatenate(gl).astype(np.float32),"cl":np.concatenate(cl).astype(np.float32),"y":np.concatenate(yy).astype(np.float32)}

def fusion_dataset(z,tab):
    n=len(z["y"]);e=np.zeros((n,1),np.float32);return v12.tensor_dataset(z,tab,e)

def refit_head(name,model,ds,epochs,cfg,out,seed):
    seed_all(seed);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model=model.to(device);y=ds.tensors[-1].numpy();ids=np.arange(len(y));sampler=weighted_sampler(y,ids,cfg.sampled_positive_fraction,seed);loader=DataLoader(ds,cfg.fusion_batch_size,sampler=sampler,num_workers=0);opt=torch.optim.AdamW(model.parameters(),lr=cfg.fusion_lr,weight_decay=cfg.fusion_weight_decay);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(epochs,1),eta_min=cfg.fusion_lr*.05);lossfn=nn.BCEWithLogitsLoss()
    for ep in range(1,epochs+1):
        model.train();losses=[]
        for q in loader:
            *x,yy=q;x=[a.to(device) for a in x];yy=yy.to(device);opt.zero_grad(set_to_none=True);loss=lossfn(model(*x),yy);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach().cpu()))
        sch.step();logging.info("REFIT %s epoch=%d/%d loss=%.4f",name,ep,epochs,np.mean(losses))
    path=out/f"{name}_80pct_refit.pt";torch.save({"model":model.state_dict(),"fixed_epochs":epochs,"seed":seed},path);return model,path

def fit_platt_threshold(y,score,target=.80):
    cal=LogisticRegression(C=1e6,solver="lbfgs",max_iter=4000).fit(np.asarray(score).reshape(-1,1),y);p=cal.predict_proba(np.asarray(score).reshape(-1,1))[:,1];pr,re,th=precision_recall_curve(y,p);ok=np.where(re[:-1]>=target)[0];j=ok[np.argmax(pr[:-1][ok])] if len(ok) else np.argmin(np.abs(re[:-1]-target));return cal,float(th[j])

def tab_model(cols,d,y,tr,pred):
    pipe=Pipeline([("prep",v6.tab_preprocessor(cols)),("model",LogisticRegression(C=.25,max_iter=4000,solver="lbfgs"))]);pipe.fit(d.iloc[tr][cols],y[tr]);return pipe,pipe.predict_proba(d.iloc[pred][cols])[:,1]

def metrics(name,y,p,threshold):
    z=p>=threshold;tp=int(np.sum(z&(y==1)));fp=int(np.sum(z&(y==0)));fn=int(np.sum(~z&(y==1)));tn=int(np.sum(~z&(y==0)));recal=LogisticRegression(C=1e6,solver="lbfgs",max_iter=4000).fit(v6.logit(p).reshape(-1,1),y)
    return {"model":name,"n":len(y),"events":int(y.sum()),"prevalence":y.mean(),"auprc":average_precision_score(y,p),"auroc":roc_auc_score(y,p),"auprc_over_prevalence":average_precision_score(y,p)/y.mean(),"brier":brier_score_loss(y,p),"log_loss":log_loss(y,p,labels=[0,1]),"calibration_intercept":recal.intercept_[0],"calibration_slope":recal.coef_[0,0],"threshold":threshold,"sensitivity":tp/max(tp+fn,1),"specificity":tn/max(tn+fp,1),"ppv":tp/max(tp+fp,1),"npv":tn/max(tn+fn,1),"tp":tp,"fp":fp,"fn":fn,"tn":tn}

def bootstrap(rows,y,preds,n,seed):
    rng=np.random.default_rng(seed);case=np.where(y==1)[0];ctrl=np.where(y==0)[0];vals={k:{"auprc":[],"auroc":[],"brier":[]} for k in preds}
    for _ in range(n):
        ix=np.r_[rng.choice(case,len(case),True),rng.choice(ctrl,len(ctrl),True)];rng.shuffle(ix);yy=y[ix]
        for k,p in preds.items():vals[k]["auprc"].append(average_precision_score(yy,p[ix]));vals[k]["auroc"].append(roc_auc_score(yy,p[ix]));vals[k]["brier"].append(brier_score_loss(yy,p[ix]))
    for row in rows:
        for m,a in vals[row["model"]].items():row[m+"_ci_low"],row[m+"_ci_high"]=np.quantile(a,[.025,.975])
    return rows

def top_risk(y,preds):
    rows=[]
    for name,p in preds.items():
        for f in (.01,.02,.05,.10,.20):
            n=max(1,int(np.ceil(len(y)*f)));ix=np.argsort(p)[-n:];e=int(y[ix].sum());ppv=y[ix].mean();rows.append({"model":name,"top_fraction":f,"selected_n":n,"events_captured":e,"event_capture_fraction":e/y.sum(),"ppv":ppv,"enrichment":ppv/y.mean()})
    return pd.DataFrame(rows)

def figures(out,y,preds,metrics_df):
    colors=["#777777","#A0A0A0","#9C755F","#4C78A8","#59A14F","#F28E2B","#C44E52"];paths=[]
    fig,ax=plt.subplots(figsize=(7,6))
    for (name,p),c in zip(preds.items(),colors):f,t,_=roc_curve(y,p);ax.plot(f,t,color=c,label=f"{name} ({roc_auc_score(y,p):.3f})")
    ax.plot([0,1],[0,1],"k--",lw=.7);ax.set(xlabel="1 − specificity",ylabel="Sensitivity",title="UKB fixed 20% test ROC");ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/"01_test_roc.pdf");fig.savefig(out/"01_test_roc.png",dpi=350);plt.close(fig);paths.append(out/"01_test_roc.png")
    fig,ax=plt.subplots(figsize=(7,6))
    for (name,p),c in zip(preds.items(),colors):pr,re,_=precision_recall_curve(y,p);ax.plot(re,pr,color=c,label=f"{name} ({average_precision_score(y,p):.3f})")
    ax.axhline(y.mean(),color="k",ls="--",lw=.7);ax.set(xlabel="Recall",ylabel="Precision",title="UKB fixed 20% test precision–recall");ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/"02_test_pr.pdf");fig.savefig(out/"02_test_pr.png",dpi=350);plt.close(fig);paths.append(out/"02_test_pr.png")
    fig,ax=plt.subplots(figsize=(7,6))
    for name in ["age_sex","final_hybrid"]:
        o,m=calibration_curve(y,preds[name],n_bins=10,strategy="quantile");ax.plot(m,o,"o-",label=name)
    lim=max(.05,ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([0,lim],[0,lim],"k--",lw=.7);ax.set(xlabel="Predicted probability",ylabel="Observed risk",title="UKB fixed 20% test calibration");ax.legend();fig.tight_layout();fig.savefig(out/"03_test_calibration.pdf");fig.savefig(out/"03_test_calibration.png",dpi=350);plt.close(fig);paths.append(out/"03_test_calibration.png")
    pdf=out/"DEMENTIA_V13_UKB_80_20_FINAL_REPORT.pdf"
    with PdfPages(pdf) as pages:
        fig=plt.figure(figsize=(11.69,8.27));fig.text(.06,.87,"Dementia v13 final",fontsize=25,weight="bold");fig.text(.06,.77,"UKB-wide fixed 80:20 retraining",fontsize=16);fig.text(.06,.57,f"Final test N={len(y):,}; events={int(y.sum())}; prevalence={y.mean():.3%}\nOnly this fixed 20% test is intended as the manuscript internal result.",fontsize=12,linespacing=1.7);fig.text(.06,.36,"Epochs were selected inside the 80% development partition.\nAll neural components were then reinitialized and refit on the complete 80% before test inference.",fontsize=11,linespacing=1.6);pages.savefig(fig);plt.close(fig)
        show=metrics_df[["model","n","events","auprc","auprc_ci_low","auprc_ci_high","auroc","auroc_ci_low","auroc_ci_high","brier","calibration_slope"]].round(4);fig,ax=plt.subplots(figsize=(11.69,8.27));ax.axis("off");tb=ax.table(cellText=show.values,colLabels=show.columns,loc="center",cellLoc="center");tb.auto_set_font_size(False);tb.set_fontsize(7);tb.scale(1,1.5);pages.savefig(fig,bbox_inches="tight");plt.close(fig)
        for p in paths:fig,ax=plt.subplots(figsize=(11.69,8.27));ax.imshow(plt.imread(p));ax.axis("off");pages.savefig(fig,bbox_inches="tight");plt.close(fig)
    return pdf

def self_test(cfg):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");x=torch.randn(2,2,10080,device=device)
    for name in ["grufcn","convcontext_mamba"]:
        m=v11.build(name,cfg).to(device);o=m(x);assert o.shape==(2,) and torch.isfinite(o).all();o.sum().backward();logging.info("Self-test %s OK parameters=%d",name,sum(p.numel() for p in m.parameters()))
    g=torch.randn(2,64,device=device);c=torch.randn(2,64,device=device);l=torch.randn(2,device=device);t=torch.randn(2,3,device=device);e=torch.zeros(2,1,device=device)
    for m in [v12.DualPAFeature(),v12.HybridFusion(3)]:m=m.to(device);o=m(g,c,l,l,t,e);assert o.shape==(2,) and torch.isfinite(o).all();o.sum().backward()

def run(cfg):
    out=Path(cfg.output_dir);setup(out);atomic_json(out/"config.json",vars(cfg));self_test(cfg)
    if cfg.mode=="smoke":atomic_json(out/"SMOKE_COMPLETED.json",{"completed":True,"time":datetime.now().isoformat(),"test_accessed":False});return
    if (out/"COMPLETED.json").exists():logging.info("80:20 final run already completed; refusing to overwrite");return
    d=v6.load_cohort(v6.Config(cfg.labels_csv,cfg.apoe_csv,cfg.cache_dir,cfg.output_dir,horizon_years=cfg.horizon_years));y=d.target.to_numpy(np.int8);dev,test=locked_outer_split(d,y,cfg,out);itr,iva=inner_split(dev,y,cfg,out,d)
    logging.info("OUTER development N=%d cases=%d | LOCKED TEST N=%d (events remain unreported until unlock)",len(dev),int(y[dev].sum()),len(test));sel=out/"epoch_selection";sel.mkdir(exist_ok=True);hist=[];epochs={};selection_scores={}
    for j,name in enumerate(["grufcn","convcontext_mamba"]):
        ep,score,h=train_backbone_selection(name,cfg,d,y,itr,iva,sel,cfg.seed+10+j);epochs[name]=ep;selection_scores["grufcn_pa" if name=="grufcn" else "convcontext_pa"]=score;hist.extend(h);logging.info("LOCKED EPOCH %s=%d",name,ep)
    sg=sel/"grufcn_best.pt";sc=sel/"convcontext_mamba_best.pt";sztr=extract_pair(cfg,d,y,itr,sg,sc);szva=extract_pair(cfg,d,y,iva,sg,sc)
    pp_inner=v6.tab_preprocessor(AGE_SEX);pp_inner.fit(d.iloc[itr][AGE_SEX]);tab_inner=np.asarray(pp_inner.transform(d[AGE_SEX]),np.float32);trds=fusion_dataset(sztr,tab_inner[itr]);vads=fusion_dataset(szva,tab_inner[iva])
    s,ep,h,_,_=v12.train_head("dual_pa_feature",v12.DualPAFeature(),trds,vads,cfg,sel,cfg.seed+30);epochs["dual_pa_feature"]=ep;selection_scores["dual_pa_feature"]=s;hist.extend(h)
    s,ep,h,_,_=v12.train_head("final_hybrid",v12.HybridFusion(tab_inner.shape[1]),trds,vads,cfg,sel,cfg.seed+31);epochs["final_hybrid"]=ep;selection_scores["final_hybrid"]=s;hist.extend(h)
    # Lock validation-derived calibrators/thresholds before test inference.
    selection_pipes={};selection_scores["age_sex"]=v6.logit(tab_model(AGE_SEX,d,y,itr,iva)[1]);selection_scores["traditional_pa"]=v6.logit(tab_model(TRAD,d,y,itr,iva)[1]);selection_scores["age_sex_traditional_pa"]=v6.logit(tab_model(AGE_SEX+TRAD,d,y,itr,iva)[1])
    calibrators={};thresholds={}
    for name,score in selection_scores.items():calibrators[name],thresholds[name]=fit_platt_threshold(y[iva],score,cfg.target_sensitivity)
    joblib.dump(calibrators,out/"development_only_calibrators.joblib");atomic_json(out/"locked_epochs_and_thresholds.json",{"epochs":epochs,"thresholds":thresholds,"source":"80% development internal epoch-selection subset only"});pd.DataFrame(hist).to_csv(out/"development_training_history.csv",index=False)
    # Reinitialize and refit every neural component on all 80%.
    fg=refit_backbone("grufcn",epochs["grufcn"],cfg,d,y,dev,out,cfg.seed+110);fc=refit_backbone("convcontext_mamba",epochs["convcontext_mamba"],cfg,d,y,dev,out,cfg.seed+111)
    zdev=extract_pair(cfg,d,y,dev,fg,fc);pp=v6.tab_preprocessor(AGE_SEX);pp.fit(d.iloc[dev][AGE_SEX]);tab=np.asarray(pp.transform(d[AGE_SEX]),np.float32);joblib.dump(pp,out/"age_sex_preprocessor_80pct.joblib")
    devds=fusion_dataset(zdev,tab[dev]);dual,_=refit_head("dual_pa_feature",v12.DualPAFeature(),devds,epochs["dual_pa_feature"],cfg,out,cfg.seed+130);hybrid,_=refit_head("final_hybrid",v12.HybridFusion(tab.shape[1]),devds,epochs["final_hybrid"],cfg,out,cfg.seed+131)
    tab_models={}
    for name,cols in [("age_sex",AGE_SEX),("traditional_pa",TRAD),("age_sex_traditional_pa",AGE_SEX+TRAD)]:tab_models[name],_=tab_model(cols,d,y,dev,dev);joblib.dump(tab_models[name],out/f"{name}_80pct.joblib")
    # This is the first model inference on the outer 20% test.
    atomic_json(out/"TEST_UNLOCKED.json",{"time":datetime.now().isoformat(),"reason":"All models, epochs, calibration and thresholds locked"});ztest=extract_pair(cfg,d,y,test,fg,fc);testds=fusion_dataset(ztest,tab[test]);raw_test={name:pipe.predict_proba(d.iloc[test][cols])[:,1] for (name,cols),pipe in zip([("age_sex",AGE_SEX),("traditional_pa",TRAD),("age_sex_traditional_pa",AGE_SEX+TRAD)],tab_models.values())};device=torch.device("cuda" if torch.cuda.is_available() else "cpu");ty,ds=v12.infer(dual,testds,device,cfg.eval_batch_size);_,hs=v12.infer(hybrid,testds,device,cfg.eval_batch_size)
    raw_scores={"age_sex":v6.logit(raw_test["age_sex"]),"traditional_pa":v6.logit(raw_test["traditional_pa"]),"age_sex_traditional_pa":v6.logit(raw_test["age_sex_traditional_pa"]),"grufcn_pa":ztest["gl"],"convcontext_pa":ztest["cl"],"dual_pa_feature":ds,"final_hybrid":hs}
    preds={name:calibrators[name].predict_proba(np.asarray(score).reshape(-1,1))[:,1] for name,score in raw_scores.items()};rows=[metrics(name,ty.astype(int),p,thresholds[name]) for name,p in preds.items()];rows=bootstrap(rows,ty.astype(int),preds,cfg.bootstrap_repeats,cfg.seed);md=pd.DataFrame(rows).sort_values("auprc",ascending=False);md["prespecified_primary"]=md.model.eq("final_hybrid");md.to_csv(out/"ukb_20pct_final_test_metrics.csv",index=False)
    pd.DataFrame({"eid":d.eid.iloc[test].to_numpy(),"target":ty.astype(int),**{f"probability_{k}":p for k,p in preds.items()}}).to_csv(out/"ukb_20pct_final_test_predictions.csv",index=False);top_risk(ty.astype(int),preds).to_csv(out/"ukb_20pct_top_risk_enrichment.csv",index=False);pdf=figures(out,ty.astype(int),preds,md)
    audit={"outer_protocol":"UKB-wide fixed stratified 80:20","development_n":len(dev),"development_cases":int(y[dev].sum()),"test_n":len(test),"test_cases":int(y[test].sum()),"inner_validation_role":"epoch/calibration/threshold selection only; not reported as performance","full_80pct_refit":True,"test_inference_count":1,"prespecified_primary":"final_hybrid","models":MODEL_NAMES,"clinical_variables":AGE_SEX,"traditional_pa_role":"separate baselines","apoe_role":"not in primary full-cohort model","blocked":["Followingyears_birthday","Townsend_index","comorbidity_index"],"external_validation":"NHATS"};atomic_json(out/"design_and_leakage_audit.json",audit);atomic_json(out/"COMPLETED.json",{"completed":True,"time":datetime.now().isoformat(),"primary":"final_hybrid","report":pdf.name});r=md.loc[md.model.eq("final_hybrid")].iloc[0];logging.info("COMPLETED 80:20 final: test N=%d events=%d AUPRC=%.4f AUROC=%.4f",len(test),int(ty.sum()),r.auprc,r.auroc)

def args():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument("--mode",choices=["smoke","full"],default="full");a.add_argument("--labels-csv",default=str(v6.DEFAULT_LABELS));a.add_argument("--apoe-csv",default=str(v6.DEFAULT_APOE));a.add_argument("--cache-dir",default=str(v6.DEFAULT_CACHE));a.add_argument("--output-dir",default=str(DEFAULT_OUT));a.add_argument("--horizon-years",type=float,default=7.);a.add_argument("--split-seed",type=int,default=20260820);a.add_argument("--inner-validation-fraction",type=float,default=.15);a.add_argument("--ssl-epochs",type=int,default=1);a.add_argument("--epochs",type=int,default=24);a.add_argument("--cosine-epochs",type=int,default=12);a.add_argument("--min-epochs",type=int,default=8);a.add_argument("--patience",type=int,default=6);a.add_argument("--early-stopping",action=argparse.BooleanOptionalAction,default=True);a.add_argument("--batch-size",type=int,default=64);a.add_argument("--eval-batch-size",type=int,default=256);a.add_argument("--extract-batch-size",type=int,default=128);a.add_argument("--num-workers",type=int,default=6);a.add_argument("--lr",type=float,default=1e-4);a.add_argument("--ssl-lr",type=float,default=2e-4);a.add_argument("--weight-decay",type=float,default=1e-3);a.add_argument("--dropout",type=float,default=.30);a.add_argument("--d-model",type=int,default=64);a.add_argument("--sampled-positive-fraction",type=float,default=.08);a.add_argument("--temperature",type=float,default=.15);a.add_argument("--fusion-batch-size",type=int,default=512);a.add_argument("--fusion-epochs",type=int,default=60);a.add_argument("--fusion-min-epochs",type=int,default=10);a.add_argument("--fusion-patience",type=int,default=12);a.add_argument("--fusion-lr",type=float,default=3e-4);a.add_argument("--fusion-weight-decay",type=float,default=1e-3);a.add_argument("--target-sensitivity",type=float,default=.80);a.add_argument("--bootstrap-repeats",type=int,default=2000);a.add_argument("--seed",type=int,default=20260820);a.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True);return a.parse_args()
if __name__=="__main__":run(args())

