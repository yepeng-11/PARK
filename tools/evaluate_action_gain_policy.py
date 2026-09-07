"""Execute frozen action-benefit predictors under a train-clean route budget."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import run_quality_task_predictability as prior


TIERS=prior.TIERS
HEADS=("classifier","regressor")
POLICIES=tuple(f"{tier}_{head}" for tier in TIERS for head in HEADS)


def signals(frame,models,tier,head):
    x,_=prior.feature_matrix(frame,tier); column=0 if head=="classifier" else 1
    values=[]
    for action in prior.ACTION_TARGETS:
        model=models[(tier,"benefit_"+action)][column]
        values.append(model.predict_proba(x)[:,1] if head=="classifier" else model.predict(x))
    out=np.column_stack(values)
    if not np.isfinite(out).all(): raise ValueError("Nonfinite action signal")
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir",type=Path,required=True); ap.add_argument("--predictability-dir",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True); args=ap.parse_args()
    run=args.run_dir.resolve(); parent=args.predictability_dir.resolve(); out=args.output_dir.resolve()
    if out.exists(): raise FileExistsError("Use a new output directory")
    ps=json.loads((parent/"summary.json").read_text()); pp=json.loads((parent/"protocol.json").read_text())
    assert ps["status"]=="COMPLETE_DEVELOPMENT_ONLY" and prior.sha(parent/"protocol.json")==ps["protocol_sha256"]
    assert prior.sha(Path(prior.__file__))==pp["script_sha256"]
    sources={}; folds=[]; seen=set()
    for k in range(5):
        pair=[]
        for name in ("final_train","outer_validation"):
            p=run/f"outer_{k}/agent/private_{name}.csv"; rel=str(p.relative_to(run))
            assert prior.sha(p)==pp["source_sha256"][rel]; sources[str(p)]=prior.sha(p)
            pair.append(pd.read_csv(p,dtype={"id":str},float_precision="round_trip"))
        mp=parent/f"outer_{k}/models.pkl"; assert prior.sha(mp)==ps["model_sha256"][str(mp.relative_to(parent))]
        sources[str(mp)]=prior.sha(mp)
        with mp.open("rb") as h: models=pickle.load(h)
        train,test=pair; assert not set(train.id)&set(test.id) and not seen&set(test.id); seen|=set(test.id)
        folds.append((train,test,models))
    assert len(seen)==632
    out.mkdir(parents=True)
    protocol={"mode":"FIXED_ACTION_POLICY_DEVELOPMENT_NO_PROMOTION","parent_protocol_sha256":ps["protocol_sha256"],
        "source_sha256":sources,"script_sha256":prior.sha(Path(__file__)),"actions":["available_mean","drop_speech","drop_smile"],
        "policies":list(POLICIES),"primary_policy":"quality_regressor",
        "route_budget":"For each fold/policy, threshold=max(natural threshold, train-clean 80th percentile of maximum predicted action signal); strict > threshold",
        "natural_threshold":{"classifier":.5,"regressor":0.},
        "selection":"Choose higher eligible speech/smile action signal; unavailable modality signal=-infinity; otherwise keep mean",
        "primary":"Participant-average log-loss gain over 14 fit stress scenarios versus available mean",
        "success":{"stress":"gain>0, bootstrap lower>0, >=4/5 positive folds",
            "clean":"mean-fold AUROC delta>=-.005, log-loss gain>=0, action rate<=.20",
            "speech_gaussian":"each of 0.25/0.5/1.0 pooled log-loss gain>=0",
            "heldout_smile_conflict":"pooled log-loss gain>=0"},
        "bootstrap":"1000 participant resamples within outer fold and disease class; all scenarios kept; no refit",
        "limitations":["Primary choice is motivated by exposed prior development diagnostics; not independent confirmation",
            "Threshold uses in-sample train-clean signal quantile without labels; outer evaluation remains participant-disjoint",
            "No abstention, new fitting, hyperparameter search, or locked test prediction",
            "Synthetic feature/score faults do not establish real acquisition robustness"]}
    prior.write(out/"protocol.json",protocol)
    thresholds=[]; records=[]; checks=[]
    for k,(train,test,models) in enumerate(folds):
        train_clean=train.loc[train.scenario.eq("clean")].reset_index(drop=True)
        action_scores=prior.base.actions(test); y=test.label.to_numpy(int); baseline=action_scores[:,0]
        frame=test[["id","label","scenario","fit_allowed"]].copy(); frame["fold"]=k
        frame["available_mean"]=baseline; frame["fixed_drop_speech"]=action_scores[:,1]; frame["fixed_drop_smile"]=action_scores[:,2]; frame["ufnet"]=action_scores[:,4]
        train_no_label=train_clean.drop(columns=["label"]); test_no_label=test.drop(columns=["label"])
        changed=test.copy(); changed["label"]=1-changed.label
        action_audit=[]
        for tier in TIERS:
            for head in HEADS:
                name=f"{tier}_{head}"; tr=signals(train_no_label,models,tier,head); te=signals(test_no_label,models,tier,head)
                np.testing.assert_array_equal(te,signals(changed,models,tier,head))
                threshold=max(.5 if head=="classifier" else 0.,float(np.quantile(tr.max(1),.8,method="higher")))
                thresholds.append({"fold":k,"policy":name,"threshold":threshold,"train_clean_action_rate":float((tr.max(1)>threshold).mean())})
                available=test[["speech_available","smile_available"]].to_numpy(bool); eligible=te.copy(); eligible[~available]=-np.inf
                choice=eligible.argmax(1); maximum=eligible[np.arange(len(test)),choice]; route=maximum>threshold
                score=baseline.copy(); score[route]=action_scores[np.arange(len(test))[route],choice[route]+1]
                frame[name]=score; frame[name+"_action"]=np.where(route,choice+1,0)
                action_audit.append(int(route.sum()))
        frame["baseline_loss"]=prior.base.loss(y,baseline)
        for name in ("fixed_drop_speech","fixed_drop_smile","ufnet",*POLICIES): frame[name+"_loss"]=prior.base.loss(y,frame[name].to_numpy())
        records.append(frame); checks.append({"fold":k,"train_participants":train.id.nunique(),"test_participants":test.id.nunique(),
            "label_free_actions":True,"policies":len(POLICIES),"routed_counts":action_audit})
        print(f"Fold {k}: six frozen policies executed; label-swap invariance passed",flush=True)
    allrows=pd.concat(records,ignore_index=True); allrows.to_csv(out/"private_policy_predictions.csv",index=False)
    model_names=("available_mean","fixed_drop_speech","fixed_drop_smile","ufnet",*POLICIES); metrics=[]
    for fold,g0 in [("pooled",allrows)]+[(str(k),g) for k,g in allrows.groupby("fold")]:
        for scenario,g in g0.groupby("scenario"):
            y=g.label.to_numpy(int)
            for name in model_names:
                p=g[name].to_numpy(float); action_rate=0. if name not in POLICIES else float((g[name+"_action"]!=0).mean())
                metrics.append({"fold":fold,"scenario":scenario,"model":name,"n":len(g),"auroc":float(roc_auc_score(y,p)),
                    "log_loss":float(prior.base.loss(y,p).mean()),"error_rate":float(((p>=.5)!=y).mean()),"action_rate":action_rate,
                    "drop_speech_rate":0. if name not in POLICIES else float((g[name+"_action"]==1).mean()),
                    "drop_smile_rate":0. if name not in POLICIES else float((g[name+"_action"]==2).mean())})
    pd.DataFrame(metrics).to_csv(out/"metrics.csv",index=False); pd.DataFrame(thresholds).to_csv(out/"thresholds.csv",index=False)
    comparisons=[]
    masks={"clean":allrows.scenario.eq("clean"),"fit_stress":allrows.fit_allowed.astype(bool)&allrows.scenario.ne("clean"),
        "speech_gaussian":allrows.scenario.str.startswith("speech_gaussian"),"speech_mask":allrows.scenario.str.startswith("speech_mask"),
        "smile_feature":allrows.scenario.isin(["smile_gaussian_0.25","smile_gaussian_0.5","smile_feature_permutation"]),
        "heldout_smile_conflict":allrows.scenario.eq("smile_opposite_consensus")}
    for subset,mask in masks.items():
        g=allrows.loc[mask]
        for name in (*POLICIES,"fixed_drop_speech","fixed_drop_smile","ufnet"):
            gain=g.baseline_loss-g[name+"_loss"]
            byfold=g.assign(gain=gain).groupby("fold").gain.mean()
            comparisons.append({"subset":subset,"model":name,"gain_vs_mean":float(gain.mean()),"positive_folds":int((byfold>0).sum()),
                "action_rate":0. if name not in POLICIES else float((g[name+"_action"]!=0).mean())})
    pd.DataFrame(comparisons).to_csv(out/"comparisons.csv",index=False)
    primary="quality_regressor"; stress=allrows.loc[masks["fit_stress"]].copy(); stress["gain"]=stress.baseline_loss-stress[primary+"_loss"]
    people=stress.groupby(["fold","id","label"]).gain.mean().reset_index(); rng=np.random.default_rng(20260911)
    groups=[g.gain.to_numpy() for _,g in people.groupby(["fold","label"])]; draws=[]
    for _ in range(1000): draws.append(float(np.mean(np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]))))
    foldgain=people.groupby("fold").gain.mean(); stress_gain=float(people.gain.mean()); ci=[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]
    clean_metrics=pd.read_csv(out/"metrics.csv"); clean_metrics=clean_metrics.loc[(clean_metrics.fold.astype(str)!="pooled")&clean_metrics.scenario.eq("clean")]
    auc_delta=float(clean_metrics.loc[clean_metrics.model.eq(primary),"auroc"].mean()-clean_metrics.loc[clean_metrics.model.eq("available_mean"),"auroc"].mean())
    clean=allrows.loc[masks["clean"]]; clean_gain=float((clean.baseline_loss-clean[primary+"_loss"]).mean()); clean_rate=float((clean[primary+"_action"]!=0).mean())
    gaussian={s:float((g.baseline_loss-g[primary+"_loss"]).mean()) for s,g in allrows.loc[allrows.scenario.str.startswith("speech_gaussian")].groupby("scenario")}
    held=allrows.loc[masks["heldout_smile_conflict"]]; held_gain=float((held.baseline_loss-held[primary+"_loss"]).mean())
    result={"primary_policy":primary,"stress_gain":stress_gain,"bootstrap_95":ci,"positive_folds":int((foldgain>0).sum()),
        "clean_mean_fold_auc_delta":auc_delta,"clean_log_loss_gain":clean_gain,"clean_action_rate":clean_rate,
        "speech_gaussian_gain":gaussian,"heldout_smile_conflict_gain":held_gain}
    result["action_policy_supported"]=bool(stress_gain>0 and ci[0]>0 and (foldgain>0).sum()>=4 and auc_delta>=-.005 and clean_gain>=0 and clean_rate<=.2 and min(gaussian.values())>=0 and held_gain>=0)
    result["promotion_authorized"]=False; prior.write(out/"decision.json",result)
    for p,h in sources.items(): assert prior.sha(Path(p))==h
    assert prior.sha(Path(__file__))==protocol["script_sha256"]
    public=("thresholds.csv","metrics.csv","comparisons.csv","decision.json")
    prior.write(out/"summary.json",{"status":"COMPLETE_DEVELOPMENT_ONLY","participants":632,"trained_models":0,"policies":len(POLICIES),
        "checks":checks,"inputs_unchanged":True,"protocol_sha256":prior.sha(out/"protocol.json"),
        "output_sha256":{n:prior.sha(out/n) for n in public},"private_sha256":prior.sha(out/"private_policy_predictions.csv")})
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
