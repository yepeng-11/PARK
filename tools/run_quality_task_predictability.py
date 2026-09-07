"""Nested OOF experiment: can deployable signals predict errors, losses and action gains?"""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits
import nested_agent_pipeline as base


TIERS=("confidence","uncertainty","quality")
ERROR_TARGETS=("finger","speech","smile","ufnet","available_mean")
ACTION_TARGETS=("drop_speech","drop_smile")
PRIMARY_TASKS=("error_available_mean","benefit_drop_speech","benefit_drop_smile")


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj): Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf-8")
def logit(p):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6); return np.log(p/(1-p))


def feature_matrix(frame,tier):
    d=base.sanitized(frame)
    p=d[list(base.EXPERTS)+["ufnet"]].to_numpy(float)
    a=d[[n+"_available" for n in base.EXPERTS]].to_numpy(float)
    blocks=[logit(p),np.abs(p-.5),a]
    names=[f"logit_{n}" for n in (*base.EXPERTS,"ufnet")]+[f"confidence_{n}" for n in (*base.EXPERTS,"ufnet")]+[f"available_{n}" for n in base.EXPERTS]
    if tier in {"uncertainty","quality"}:
        masked=np.where(a>0,p[:,:3],np.nan)
        spread=np.nanmax(masked,axis=1)-np.nanmin(masked,axis=1)
        pairs=np.column_stack([np.where((a[:,i]>0)&(a[:,j]>0),np.abs(p[:,i]-p[:,j]),0.) for i,j in ((0,1),(0,2),(1,2))])
        blocks += [d[[n+"_mc_std" for n in base.EXPERTS]].to_numpy(float),spread[:,None],pairs]
        names += [f"mc_std_{n}" for n in base.EXPERTS]+["available_spread","gap_finger_speech","gap_finger_smile","gap_speech_smile"]
    if tier=="quality":
        cols=[n+"_feature_"+v for n in base.EXPERTS for v in base.DESCRIPTORS]
        blocks.append(d[cols].to_numpy(float)); names += cols
    x=np.column_stack(blocks)
    if not np.isfinite(x).all(): raise ValueError("Nonfinite feature")
    return x,names


def targets(frame):
    y=frame.label.to_numpy(int); actions=base.actions(frame)
    scores={n:frame[n].to_numpy(float) for n in (*base.EXPERTS,"ufnet")}
    scores["available_mean"]=actions[:,0]
    result={}
    for name,p in scores.items():
        eligible=np.ones(len(frame),bool) if name not in base.EXPERTS else frame[name+"_available"].to_numpy(bool)
        loss=base.loss(y,p)
        result["error_"+name]=(eligible,((p>=.5)!=y).astype(int),loss)
    for action,index in (("drop_speech",1),("drop_smile",2)):
        modality="speech" if action=="drop_speech" else "smile"
        eligible=frame[modality+"_available"].to_numpy(bool) & (np.abs(actions[:,0]-actions[:,index])>1e-12)
        gain=base.loss(y,actions[:,0])-base.loss(y,actions[:,index])
        result["benefit_"+action]=(eligible,(gain>0).astype(int),gain)
    return result


def weights(frame,eligible):
    value=np.where(frame.scenario.eq("clean"),.5,.5/14).astype(float)
    value[~eligible]=0
    return value


def subsets(frame):
    return {"clean":frame.scenario.eq("clean").to_numpy(),
        "fit_stress":(frame.fit_allowed.astype(bool)&frame.scenario.ne("clean")).to_numpy(),
        "speech_gaussian":frame.scenario.str.startswith("speech_gaussian").to_numpy(),
        "speech_mask":frame.scenario.str.startswith("speech_mask").to_numpy(),
        "smile_feature":frame.scenario.isin(["smile_gaussian_0.25","smile_gaussian_0.5","smile_feature_permutation"]).to_numpy(),
        "heldout_smile_conflict":frame.scenario.eq("smile_opposite_consensus").to_numpy()}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args(); run=args.run_dir.resolve(); out=args.output_dir.resolve()
    if out.exists(): raise FileExistsError("Use a new output directory")
    audit=json.loads((run/"postrun_audit.json").read_text()); assert audit["status"]=="PASS"
    sources={}; folds=[]; seen=set()
    for k in range(5):
        folder=run/f"outer_{k}/agent"; summary=json.loads((folder/"agent_summary.json").read_text()); pair=[]
        for name in ("final_train","outer_validation"):
            p=folder/f"private_{name}.csv"; assert sha(p)==summary["private_artifact_sha256"][p.name]
            sources[str(p.relative_to(run))]=sha(p)
            f=pd.read_csv(p,dtype={"id":str},float_precision="round_trip")
            assert not f.duplicated(["id","scenario"]).any() and f.groupby("id").label.nunique().max()==1
            assert f.groupby("id").size().eq(16).all(); pair.append(f)
        train,test=pair; assert not set(train.id)&set(test.id)
        assert not seen&set(test.id); seen|=set(test.id); folds.append(pair)
    assert len(seen)==audit["outer_participants"]==632
    out.mkdir(parents=True)
    protocol={"mode":"FIXED_EXPLORATORY_DEVELOPMENT_NO_PROMOTION","parent_protocol_sha256":audit["protocol_sha256"],
        "source_sha256":sources,"script_sha256":sha(Path(__file__)),"pipeline_sha256":sha(Path(base.__file__)),
        "tiers":{"confidence":"4 calibrated probabilities/logits, confidence magnitudes, 3 availability flags",
            "uncertainty":"confidence + 3 MC std, available spread, 3 pair gaps",
            "quality":"uncertainty + all 3-modality feature descriptors"},
        "targets":{"binary_error":list(ERROR_TARGETS),"continuous_log_loss":list(ERROR_TARGETS),
            "binary_action_benefit":list(ACTION_TARGETS),"continuous_action_gain":list(ACTION_TARGETS)},
        "model":{"type":"HistGradientBoosting","max_iter":100,"max_depth":2,"max_leaf_nodes":4,"min_samples_leaf":450,
            "learning_rate":.05,"l2_regularization":2.,"early_stopping":False},
        "fit":"15 fit-allowed final-train OOF scenarios; per participant clean weight .5 and 14 stress scenarios share .5",
        "primary":"Mean fold AUROC macro over error_available_mean, benefit_drop_speech, benefit_drop_smile on fit_stress",
        "quality_increment_support":"quality-confidence-via-uncertainty delta >= .01, bootstrap lower>0, positive in >=4/5 folds",
        "absolute_predictability_flag":"quality primary macro AUROC >= .65 (descriptive, not promotion)",
        "bootstrap":"1000 resamples of participant IDs within each outer fold and disease class, preserving all scenarios/tasks; no refit",
        "limitations":["Previously exposed development data; synthetic scenarios are not real acquisition failures",
            "Targets use labels only for supervised fitting/evaluation; inference features exclude id,label,scenario,fold and corruption targets",
            "Action benefit excludes missing target modality and exact score ties",
            "Repeated scenarios are weighted but not independent people; bootstrap preserves participant clusters",
            "Fixed capacity can miss nonlinear signal; negative result does not prove information absence",
            "No multiplicity-adjusted confirmatory inference or locked test prediction"]}
    write(out/"protocol.json",protocol)
    start=time.perf_counter(); class_predictions=[]; regression_predictions=[]; metrics=[]; checks=[]
    with threadpool_limits(limits=2):
        for k,(train,test) in enumerate(folds):
            train=train.loc[train.fit_allowed.astype(bool)].reset_index(drop=True); assert train.groupby("id").size().eq(15).all()
            train_targets=targets(train); test_targets=targets(test); models={}
            for tier_i,tier in enumerate(TIERS):
                xtr,names=feature_matrix(train,tier); xte,names2=feature_matrix(test.drop(columns=["label"]),tier); assert names==names2
                changed=test.copy(); changed["label"]=1-changed.label
                np.testing.assert_array_equal(xte,feature_matrix(changed,tier)[0])
                for task,(eligible,target,continuous) in train_targets.items():
                    mask=eligible; w=weights(train,mask)[mask]
                    clf=HistGradientBoostingClassifier(max_iter=100,max_depth=2,max_leaf_nodes=4,min_samples_leaf=450,
                        learning_rate=.05,l2_regularization=2.,early_stopping=False,random_state=3100+100*k+10*tier_i+len(models))
                    reg=HistGradientBoostingRegressor(max_iter=100,max_depth=2,max_leaf_nodes=4,min_samples_leaf=450,
                        learning_rate=.05,l2_regularization=2.,early_stopping=False,random_state=4100+100*k+10*tier_i+len(models))
                    if len(np.unique(target[mask]))<2: raise ValueError("Degenerate target "+task)
                    clf.fit(xtr[mask],target[mask],sample_weight=w); reg.fit(xtr[mask],continuous[mask],sample_weight=w)
                    te,actual,amount=test_targets[task]; cp=clf.predict_proba(xte)[:,1]; rp=reg.predict(xte)
                    for i in np.flatnonzero(te):
                        class_predictions.append((k,test.id.iloc[i],int(test.label.iloc[i]),test.scenario.iloc[i],bool(test.fit_allowed.iloc[i]),task,tier,int(actual[i]),float(cp[i])))
                        regression_predictions.append((k,test.id.iloc[i],int(test.label.iloc[i]),test.scenario.iloc[i],bool(test.fit_allowed.iloc[i]),task,tier,float(amount[i]),float(rp[i])))
                    models[(tier,task)]=(clf,reg,names)
            folder=out/f"outer_{k}"; folder.mkdir()
            with (folder/"models.pkl").open("wb") as h: pickle.dump(models,h)
            with (folder/"models.pkl").open("rb") as h: restored=pickle.load(h)
            key=("quality","error_available_mean"); tx=feature_matrix(test.drop(columns=["label"]),"quality")[0]
            np.testing.assert_array_equal(models[key][0].predict_proba(tx),restored[key][0].predict_proba(tx))
            checks.append({"fold":k,"train_participants":train.id.nunique(),"test_participants":test.id.nunique(),
                "tasks":len(train_targets),"models":len(models)*2,"label_free_features":True,"roundtrip":True})
            print(f"Fold {k}: {len(models)*2} fixed predictors fitted; checks passed",flush=True)
    cols=["fold","id","label","scenario","fit_allowed","task","tier","target","prediction"]
    cp=pd.DataFrame(class_predictions,columns=cols); rp=pd.DataFrame(regression_predictions,columns=cols)
    for kind,frame in (("classification",cp),("regression",rp)):
        for fold,g0 in [("pooled",frame)]+[(str(k),g) for k,g in frame.groupby("fold")]:
            source=folds[int(fold)][1] if fold!="pooled" else None
            for subset in ("clean","fit_stress","speech_gaussian","speech_mask","smile_feature","heldout_smile_conflict"):
                def choose(g):
                    s=g.scenario
                    return {"clean":s.eq("clean"),"fit_stress":g.fit_allowed.astype(bool)&s.ne("clean"),
                        "speech_gaussian":s.str.startswith("speech_gaussian"),"speech_mask":s.str.startswith("speech_mask"),
                        "smile_feature":s.isin(["smile_gaussian_0.25","smile_gaussian_0.5","smile_feature_permutation"]),
                        "heldout_smile_conflict":s.eq("smile_opposite_consensus")}[subset]
                sg=g0.loc[choose(g0)]
                for (task,tier),g in sg.groupby(["task","tier"]):
                    if kind=="classification":
                        if g.target.nunique()<2: continue
                        metrics.append(dict(kind=kind,fold=fold,subset=subset,task=task,tier=tier,n=len(g),prevalence=g.target.mean(),
                            auroc=roc_auc_score(g.target,g.prediction),auprc=average_precision_score(g.target,g.prediction),
                            brier=brier_score_loss(g.target,g.prediction),spearman=None,mae=None,constant_mae=None))
                    else:
                        rho=spearmanr(g.target,g.prediction).statistic
                        metrics.append(dict(kind=kind,fold=fold,subset=subset,task=task,tier=tier,n=len(g),prevalence=None,
                            auroc=None,auprc=None,brier=None,spearman=None if np.isnan(rho) else rho,
                            mae=np.abs(g.target-g.prediction).mean(),constant_mae=np.abs(g.target-g.target.median()).mean()))
    metric=pd.DataFrame(metrics); metric.to_csv(out/"metrics.csv",index=False)
    # Primary macro is computed within fold to prevent cross-fold score-scale artifacts.
    primary=metric.loc[(metric.kind=="classification")&(metric.subset=="fit_stress")&metric.task.isin(PRIMARY_TASKS)&(metric.fold!="pooled")]
    foldmacro=primary.groupby(["fold","tier"]).auroc.mean().unstack(); foldmacro.to_csv(out/"primary_fold_macro.csv")
    observed={t:float(foldmacro[t].mean()) for t in TIERS}; delta=foldmacro.quality-foldmacro.uncertainty
    # Cluster bootstrap the already-generated outer predictions; no refitting.
    relevant=cp.loc[cp.fit_allowed.astype(bool)&cp.scenario.ne("clean")&cp.task.isin(PRIMARY_TASKS)]
    structures={}
    for (k,task,tier),g in relevant.groupby(["fold","task","tier"]):
        g=g.sort_values(["id","scenario"]); counts=g.groupby("id").size()
        if counts.nunique()!=1: raise ValueError("Unequal scenario copies in bootstrap")
        people=counts.index.to_numpy(); copies=int(counts.iloc[0])
        disease=g.groupby("id").label.first().loc[people].to_numpy(int)
        structures[(k,task,tier)]=(g.target.to_numpy().reshape(len(people),copies),
            g.prediction.to_numpy().reshape(len(people),copies),disease)
    rng=np.random.default_rng(20260909); draws=[]
    for _ in range(1000):
        values=[]
        for k in range(5):
            for task in PRIMARY_TASKS:
                y,q,disease=structures[(k,task,"quality")]
                yu,u,du=structures[(k,task,"uncertainty")]
                np.testing.assert_array_equal(y,yu); np.testing.assert_array_equal(disease,du)
                picked=np.concatenate([rng.choice(np.flatnonzero(disease==label),int((disease==label).sum()),replace=True) for label in (0,1)])
                target=y[picked].ravel()
                values.append(roc_auc_score(target,q[picked].ravel())-roc_auc_score(target,u[picked].ravel()))
        draws.append(float(np.mean(values)))
    ci=[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]
    quality_delta=float(delta.mean()); positive=int((delta>0).sum())
    decision={"primary_macro_auroc":observed,"quality_minus_uncertainty":quality_delta,"bootstrap_95":ci,
        "positive_folds":positive,"quality_increment_supported":bool(quality_delta>=.01 and ci[0]>0 and positive>=4),
        "quality_absolute_predictability_flag":bool(observed["quality"]>=.65),"promotion_authorized":False}
    write(out/"decision.json",decision)
    cp.to_csv(out/"private_classification_predictions.csv",index=False); rp.to_csv(out/"private_regression_predictions.csv",index=False)
    for p,h in sources.items(): assert sha(run/p)==h
    assert sha(Path(__file__))==protocol["script_sha256"]
    public=("metrics.csv","primary_fold_macro.csv","decision.json")
    write(out/"summary.json",{"status":"COMPLETE_DEVELOPMENT_ONLY","participants":632,"binary_tasks":7,"continuous_tasks":7,
        "trained_models":210,"wall_seconds":time.perf_counter()-start,"checks":checks,"inputs_unchanged":True,
        "protocol_sha256":sha(out/"protocol.json"),"output_sha256":{n:sha(out/n) for n in public},
        "private_sha256":{n:sha(out/n) for n in ("private_classification_predictions.csv","private_regression_predictions.csv")},
        "model_sha256":{str(p.relative_to(out)):sha(p) for p in out.glob("outer_*/*.pkl")}})
    print(json.dumps(decision,indent=2),flush=True)


if __name__=="__main__": main()
