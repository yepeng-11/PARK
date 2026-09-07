"""Fixed full-coverage fusion ablation on audited, nested base OOF caches."""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import nested_agent_pipeline as base


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, obj): p.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def features(f, level):
    d = base.sanitized(f)
    p = d[list(base.EXPERTS)+["ufnet"]].to_numpy(float)
    a = d[[n+"_available" for n in base.EXPERTS]].to_numpy(float)
    x = [np.log(np.clip(p, 1e-6, 1-1e-6)/np.clip(1-p, 1e-6, 1)), a]
    if level >= 1:
        spread = np.where(a>0,p[:,:3],-np.inf).max(1)-np.where(a>0,p[:,:3],np.inf).min(1)
        x += [d[[n+"_mc_std" for n in base.EXPERTS]].to_numpy(float), spread[:,None], np.abs(p-.5)]
    if level >= 2:
        x += [d[[n+"_feature_"+v for n in base.EXPERTS for v in base.DESCRIPTORS]].to_numpy(float)]
    out = np.column_stack(x)
    assert np.isfinite(out).all()
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(); run=args.run_dir.resolve(); out=args.output_dir.resolve()
    if out.exists(): raise FileExistsError("Use a new output directory")
    audit=json.loads((run/"postrun_audit.json").read_text()); assert audit["status"]=="PASS"
    sources={}; caches=[]; seen=set()
    for k in range(5):
        folder=run/f"outer_{k}/agent"
        summary=json.loads((folder/"agent_summary.json").read_text())
        pair=[]
        for name in ("final_train","outer_validation"):
            p=folder/f"private_{name}.csv"
            assert sha(p)==summary["private_artifact_sha256"][p.name]
            sources[str(p.relative_to(run))]=sha(p)
            f=pd.read_csv(p,dtype={"id":str},float_precision="round_trip")
            assert not f.duplicated(["id","scenario"]).any()
            assert f.groupby("id").label.nunique().max()==1
            assert set(f.scenario)=={s["name"] for s in base.scenario_registry()}
            assert f.groupby("id").size().eq(16).all()
            pair.append(f)
        train, target=pair
        assert not set(train.id)&set(target.id)
        assert len(set(train.id)|set(target.id))==632
        assert not seen&set(target.id)
        seen|=set(target.id)
        caches.append(pair)
    assert len(seen)==632
    out.mkdir(parents=True)
    protocol={"mode":"FIXED_EXPLORATORY_DEVELOPMENT_NO_PROMOTION", "parent_protocol":audit["protocol_sha256"],
        "source_sha256":sources,"script_sha256":sha(Path(__file__)),
        "pipeline_sha256":sha(Path(base.__file__)),"coverage":1.0,"parameter_search":False,
        "models":["available_mean","ufnet","logistic","tree_prob","tree_uncertainty","tree_quality"],
        "logistic":{"C":1.0,"max_iter":2000,"class_weight":None},
        "tree":{"max_iter":100,"max_depth":2,"max_leaf_nodes":4,"learning_rate":0.05,
            "l2_regularization":2.0,"early_stopping":False,"min_leaf_participants_lower_bound":30},
        "fit_weights":"Each participant total=1: clean=.5; remaining 14 fit-allowed scenarios share .5",
        "primary":"Participant-averaged 14 stress scenario log-loss gain versus mean; quality increment versus uncertainty-only tree",
        "guards":"Clean mean-fold AUROC delta >= -0.005; report each scenario and every fold; no abstention",
        "bootstrap":"1000 participant resamples stratified by outer fold and class, shared across scenarios; descriptive 95% intervals, no refitting",
        "limitations":["Previously exposed development data, not independent confirmation",
            "Fixed models fit final training OOF only, no meta parameter selection; existing audited base calibration retained",
            "No new output calibration, all learned models use same sample weights",
            "Trees directly predict disease probability: adaptive nonlinear fusion, not explicit action routing",
            "Quality means raw feature descriptors, not a learned detector or corruption label",
            "Pooled AUROC and mean-fold AUROC differ; superiority cannot be inferred from log-loss alone"]}
    write(out/"protocol.json",protocol)  # Freeze before any new fit or prediction.
    start=time.perf_counter(); records=[]; checks=[]
    with threadpool_limits(limits=2):
        for k,(train,target) in enumerate(caches):
            train=train.loc[train.fit_allowed.astype(bool)].copy()
            assert train.groupby("id").size().eq(15).all()
            weights=np.where(train.scenario.eq("clean"),.5,.5/14)
            y=train.label.to_numpy(int); states={}
            pred={"available_mean":base.actions(target)[:,0],"ufnet":target.ufnet.to_numpy(float)}
            for name,level in (("logistic",0),("tree_prob",0),("tree_uncertainty",1),("tree_quality",2)):
                if name=="logistic":
                    model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000))
                    model.fit(features(train,level),y,logisticregression__sample_weight=weights)
                    assert model[-1].n_iter_.max()<2000
                else:
                    model=HistGradientBoostingClassifier(max_iter=100,max_depth=2,max_leaf_nodes=4,
                        learning_rate=.05,l2_regularization=2.,min_samples_leaf=450,
                        early_stopping=False,random_state=2201+k)
                    model.fit(features(train,level),y,sample_weight=weights)
                x=features(target.drop(columns=["label"]),level)
                prediction=model.predict_proba(x)[:,1]
                changed=target.copy(); changed["label"]=1-changed.label
                np.testing.assert_array_equal(x,features(changed,level))
                restored=pickle.loads(pickle.dumps(model))
                np.testing.assert_array_equal(prediction,restored.predict_proba(x)[:,1])
                pred[name]=prediction; states[name]=(level,model)
            folder=out/f"outer_{k}"; folder.mkdir()
            with (folder/"models.pkl").open("wb") as h: pickle.dump(states,h)
            r=target[["id","label","scenario","fit_allowed"]].copy(); r["outer_fold"]=k
            for name,p in pred.items(): r[name]=p
            r.to_csv(folder/"private_predictions.csv",index=False)
            records.append(r); checks.append({"fold":k,"fit_participants":train.id.nunique(),
                "outer_participants":target.id.nunique(),"label_free":True,"pickle_roundtrip":True,"no_overlap":True})
            print(f"Fold {k}: four fixed models fitted; label-free and roundtrip checks passed",flush=True)
    allrows=pd.concat(records,ignore_index=True); models=protocol["models"]; metrics=[]
    for fold,f in [("pooled",allrows)]+[(str(k),f) for k,f in enumerate(records)]:
        for scenario,g in f.groupby("scenario"):
            y=g.label.to_numpy(int)
            for name in models:
                p=g[name].to_numpy(float)
                metrics.append(dict(fold=fold,scenario=scenario,model=name,n=len(g),coverage=1.,
                    auroc=roc_auc_score(y,p),log_loss=base.loss(y,p).mean(),brier=np.mean((y-p)**2)))
    table=pd.DataFrame(metrics); table.to_csv(out/"metrics.csv",index=False)
    for n in models: allrows[n+"_loss"]=base.loss(allrows.label.to_numpy(),allrows[n].to_numpy())
    comparisons=[]
    for subset in ("clean","stress"):
        part=allrows.loc[allrows.scenario.eq("clean") if subset=="clean" else (allrows.fit_allowed.astype(bool)&allrows.scenario.ne("clean"))]
        people=part.groupby(["outer_fold","id","label"])[[n+"_loss" for n in models]].mean().reset_index()
        groups=[g.index.to_numpy() for _,g in people.groupby(["outer_fold","label"])]
        random=np.random.default_rng(20260906)
        indices=[np.concatenate([random.choice(g,len(g),replace=True) for g in groups]) for _ in range(1000)]
        for model,ref in [(n,"available_mean") for n in models if n!="available_mean"]+[("tree_quality","tree_uncertainty"),("tree_uncertainty","tree_prob")]:
            gain=people[ref+"_loss"].to_numpy()-people[model+"_loss"].to_numpy()
            draws=np.array([gain[ix].mean() for ix in indices])
            foldgain=people.assign(gain=gain).groupby("outer_fold").gain.mean()
            clean=table.loc[(table.fold!="pooled")&table.scenario.eq("clean")]
            aucdelta=clean.loc[clean.model.eq(model),"auroc"].mean()-clean.loc[clean.model.eq(ref),"auroc"].mean()
            comparisons.append(dict(subset=subset,model=model,reference=ref,log_loss_gain=gain.mean(),
                p025=np.quantile(draws,.025),p975=np.quantile(draws,.975),positive_folds=int((foldgain>0).sum()),
                clean_mean_fold_auc_delta=aucdelta,clean_guard=bool(aucdelta>=-.005)))
    pd.DataFrame(comparisons).to_csv(out/"comparisons.csv",index=False)
    for path,h in sources.items(): assert sha(run/path)==h
    assert sha(Path(__file__))==protocol["script_sha256"]
    write(out/"summary.json",{"status":"COMPLETE_DEVELOPMENT_ONLY","wall_seconds":time.perf_counter()-start,
        "participants":632,"trained_models":20,"fold_checks":checks,"input_hashes_unchanged":True,
        "protocol_sha256":sha(out/"protocol.json"),"output_sha256":{n:sha(out/n) for n in ("metrics.csv","comparisons.csv")},
        "private_artifact_sha256":{str(p.relative_to(out)):sha(p) for p in out.glob("outer_*/*")}})
    print(pd.DataFrame(comparisons).to_string(index=False),flush=True)


if __name__=="__main__": main()
