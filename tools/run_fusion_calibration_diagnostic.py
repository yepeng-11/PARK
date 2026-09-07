"""Experiment 2b: fixed scalar calibration controls and frozen fusion failure audit."""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import run_fusion_experiment2 as exp


MODELS = ("available_mean", "mean_platt", "ufnet", "ufnet_platt", "logistic")


def scalar(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
    return np.log(p/(1-p))[:, None]


def stats(y, p):
    loss = exp.base.loss(y, p)
    error = (p >= .5) != y
    confident_error = error & (np.maximum(p, 1-p) >= .9)
    ece = 0.
    for b in range(10):
        mask = (p >= b/10) & ((p < (b+1)/10) if b < 9 else (p <= 1))
        if mask.any(): ece += mask.mean()*abs(y[mask].mean()-p[mask].mean())
    return dict(auroc=float(roc_auc_score(y, p)), log_loss=float(loss.mean()),
        brier=float(np.mean((y-p)**2)), ece10=float(ece), error_rate=float(error.mean()),
        confident_error_n=int(confident_error.sum()), confident_error_rate=float(confident_error.mean()),
        confident_error_loss_per_person=float(np.where(confident_error, loss, 0).mean()),
        wrong_case_mean_loss=float(loss[error].mean()) if error.any() else None,
        mean_positive_score=float(p[y==1].mean()), mean_negative_score=float(p[y==0].mean()),
        false_negative_rate=float((p[y==1]<.5).mean()), false_positive_rate=float((p[y==0]>=.5).mean()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--experiment2-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(); run=args.run_dir.resolve(); parent=args.experiment2_dir.resolve(); out=args.output_dir.resolve()
    if out.exists(): raise FileExistsError("Refuse existing output")
    original=json.loads((parent/"protocol.json").read_text())
    summary=json.loads((parent/"summary.json").read_text())
    assert exp.sha(parent/"protocol.json")==summary["protocol_sha256"]
    assert exp.sha(Path(exp.__file__))==original["script_sha256"]
    assert exp.sha(Path(exp.base.__file__))==original["pipeline_sha256"]
    sources={}; data=[]; seen=set()
    for k in range(5):
        frames=[]
        for name in ("final_train", "outer_validation"):
            p=run/f"outer_{k}/agent/private_{name}.csv"
            assert exp.sha(p)==original["source_sha256"][str(p.relative_to(run))]
            sources[str(p)]=exp.sha(p)
            frames.append(pd.read_csv(p,dtype={"id":str},float_precision="round_trip"))
        train,target=frames
        assert not set(train.id)&set(target.id)
        assert not seen&set(target.id); seen|=set(target.id)
        for name in ("models.pkl","private_predictions.csv"):
            p=parent/f"outer_{k}"/name
            assert exp.sha(p)==summary["private_artifact_sha256"][str(p.relative_to(parent))]
            sources[str(p)]=exp.sha(p)
        with (parent/f"outer_{k}/models.pkl").open("rb") as h: frozen=pickle.load(h)["logistic"][1]
        prior=pd.read_csv(parent/f"outer_{k}/private_predictions.csv",dtype={"id":str},float_precision="round_trip")
        assert target[["id","scenario","label"]].equals(prior[["id","scenario","label"]])
        np.testing.assert_allclose(frozen.predict_proba(exp.features(target.drop(columns="label"),0))[:,1],prior.logistic,rtol=0,atol=1e-12)
        data.append((train,target,frozen,prior))
    assert len(seen)==632
    out.mkdir(parents=True)
    protocol=dict(mode="POSTHOC_MOTIVATED_FIXED_DEVELOPMENT_EXPERIMENT",source_sha256=sources,
        script_sha256=exp.sha(Path(__file__)),parent_protocol_sha256=summary["protocol_sha256"],
        new_models="Two scalar Platt controls per fold: logit(mean), logit(UFNet); StandardScaler + LogisticRegression C=1 max_iter=2000",
        fitting="Same final training OOF, labels and per-person clean .5 + 14 stress .5 weights as experiment2; no search",
        comparisons="mean_platt vs mean; logistic vs mean_platt; ufnet_platt vs ufnet; logistic vs ufnet_platt",
        diagnostics="0.5 class threshold; confident error >=0.9 predicted-class probability; fixed 10 probability bins; class-conditional score drift; frozen logit contribution decomposition",
        bootstrap="1000 participant resamples within outer fold and class, complete paired scenarios; descriptive intervals only",
        limitations=["Exposed development data; no promotion or causal attribution from this ablation",
            "Scalar Platt changes intercept and slope; cannot separate prior correction from calibration",
            "No recalibration of frozen logistic and no abstention; no external test evaluation",
            "Per-fold monotonic Platt preserves within-fold AUC, not pooled cross-fold AUC",
            "Coefficients and additive logit changes are algebraic explanations, not causal feature importance"])
    exp.write(out/"protocol.json",protocol)
    start=time.perf_counter(); predictions=[]; coefficients=[]; shifts=[]; checks=[]
    with threadpool_limits(limits=2):
        for k,(train,target,frozen,prior) in enumerate(data):
            train=train.loc[train.fit_allowed.astype(bool)]
            assert train.groupby("id").size().eq(15).all()
            weights=np.where(train.scenario.eq("clean"),.5,.5/14)
            r=prior[["id","label","scenario","fit_allowed","outer_fold","available_mean","ufnet","logistic"]].copy()
            saved={}
            for source,name in (("available_mean","mean_platt"),("ufnet","ufnet_platt")):
                p=exp.base.actions(train)[:,0] if source=="available_mean" else train.ufnet.to_numpy()
                model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000))
                model.fit(scalar(p),train.label,logisticregression__sample_weight=weights)
                assert model[-1].n_iter_.max()<2000
                slope=float(model[-1].coef_[0,0]/model[0].scale_[0]); assert slope>0
                intercept=float(model[-1].intercept_[0]-slope*model[0].mean_[0])
                r[name]=model.predict_proba(scalar(r[source]))[:,1]
                restored=pickle.loads(pickle.dumps(model))
                np.testing.assert_array_equal(r[name],restored.predict_proba(scalar(r[source]))[:,1])
                for scenario,g in r.groupby("scenario"):
                    assert abs(roc_auc_score(g.label,g[name])-roc_auc_score(g.label,g[source]))<1e-12
                coefficients.append(dict(fold=k,model=name,feature="intercept",coefficient=intercept))
                coefficients.append(dict(fold=k,model=name,feature="source_logit",coefficient=slope)); saved[name]=model
            coef=frozen[-1].coef_[0]/frozen[0].scale_
            intercept=float(frozen[-1].intercept_[0]-np.dot(coef,frozen[0].mean_))
            names=list(exp.base.EXPERTS)+["ufnet"]+[n+"_available" for n in exp.base.EXPERTS]
            x=exp.features(target.drop(columns="label"),0)
            np.testing.assert_allclose(x@coef+intercept,frozen.decision_function(x),atol=1e-12)
            coefficients.extend(dict(fold=k,model="logistic",feature=n,coefficient=float(v)) for n,v in zip(names,coef))
            coefficients.append(dict(fold=k,model="logistic",feature="intercept",coefficient=intercept))
            contributions=pd.DataFrame(x*coef,columns=names); contributions["id"]=target.id.to_numpy(); contributions["scenario"]=target.scenario.to_numpy()
            clean=contributions.loc[contributions.scenario.eq("clean")].set_index("id")
            for scenario,g in contributions.groupby("scenario"):
                delta=g[names].to_numpy()-clean.loc[g.id,names].to_numpy()
                for j,n in enumerate(names): shifts.append(dict(fold=k,scenario=scenario,feature=n,mean_logit_change=float(delta[:,j].mean()),mean_abs_logit_change=float(np.abs(delta[:,j]).mean())))
            folder=out/f"outer_{k}"; folder.mkdir()
            with (folder/"calibrators.pkl").open("wb") as h: pickle.dump(saved,h)
            r.to_csv(folder/"private_predictions.csv",index=False); predictions.append(r)
            checks.append(dict(fold=k,monotonic_auc_preserved=True,frozen_logistic_reproduced=True,logit_decomposition=True,roundtrip=True))
            print(f"Fold {k}: scalar controls fitted; AUC invariance and frozen logit checks passed",flush=True)
    f=pd.concat(predictions,ignore_index=True); rows=[]
    for k,g in [("pooled",f)]+[(str(k),g) for k,g in enumerate(predictions)]:
        for scenario,s in g.groupby("scenario"):
            for n in MODELS: rows.append(dict(fold=k,scenario=scenario,model=n,n=len(s),**stats(s.label.to_numpy(),s[n].to_numpy())))
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(coefficients).to_csv(out/"coefficients.csv",index=False)
    pd.DataFrame(shifts).to_csv(out/"logit_shifts.csv",index=False)
    for n in MODELS: f[n+"_loss"]=exp.base.loss(f.label.to_numpy(),f[n].to_numpy())
    comparisons=[]
    subsets={"clean":f.scenario.eq("clean"),"stress":f.fit_allowed.astype(bool)&f.scenario.ne("clean"),"speech_gaussian":f.scenario.str.startswith("speech_gaussian"),"speech_mask":f.scenario.str.startswith("speech_mask")}
    for subset,mask in subsets.items():
        people=f.loc[mask].groupby(["outer_fold","id","label"])[[n+"_loss" for n in MODELS]].mean().reset_index()
        groups=[g.index.to_numpy() for _,g in people.groupby(["outer_fold","label"])]; rng=np.random.default_rng(20260907)
        indices=[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]) for _ in range(1000)]
        for model,reference in (("mean_platt","available_mean"),("logistic","mean_platt"),("ufnet_platt","ufnet"),("logistic","ufnet_platt")):
            delta=people[reference+"_loss"].to_numpy()-people[model+"_loss"].to_numpy(); draws=[delta[i].mean() for i in indices]
            comparisons.append(dict(subset=subset,model=model,reference=reference,gain=float(delta.mean()),p025=float(np.quantile(draws,.025)),p975=float(np.quantile(draws,.975)),positive_folds=int((people.assign(delta=delta).groupby("outer_fold").delta.mean()>0).sum())))
    pd.DataFrame(comparisons).to_csv(out/"comparisons.csv",index=False)
    for p,h in sources.items(): assert exp.sha(Path(p))==h
    assert exp.sha(Path(__file__))==protocol["script_sha256"]
    outputs=("metrics.csv","coefficients.csv","logit_shifts.csv","comparisons.csv")
    exp.write(out/"summary.json",dict(status="COMPLETE_DEVELOPMENT_ONLY",participants=632,new_models=10,
        wall_seconds=time.perf_counter()-start,checks=checks,inputs_unchanged=True,
        protocol_sha256=exp.sha(out/"protocol.json"),output_sha256={n:exp.sha(out/n) for n in outputs},
        private_sha256={str(p.relative_to(out)):exp.sha(p) for p in out.glob("outer_*/*")}))
    print(pd.DataFrame(comparisons).to_string(index=False),flush=True)


if __name__=="__main__": main()
