"""Fixed-score selective prediction audit at equal coverage and train-set thresholds."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import run_quality_task_predictability as prior


METHODS=("confidence","disagreement","mc_uncertainty","error_confidence","error_uncertainty","error_quality")
COVERAGES=(.9,.8,.7)


def risks(frame,models):
    d=prior.base.sanitized(frame)
    actions=prior.base.actions(d); mean=actions[:,0]
    a=d[[n+"_available" for n in prior.base.EXPERTS]].to_numpy(float)
    p=d[list(prior.base.EXPERTS)].to_numpy(float)
    spread=np.where(a>0,p,-np.inf).max(1)-np.where(a>0,p,np.inf).min(1)
    mc=d[[n+"_mc_std" for n in prior.base.EXPERTS]].to_numpy(float)
    out={"confidence":1-2*np.abs(mean-.5),"disagreement":spread,
         "mc_uncertainty":(mc*a).sum(1)/a.sum(1)}
    for tier in prior.TIERS:
        x,_=prior.feature_matrix(d,tier)
        out["error_"+tier]=models[(tier,"error_available_mean")][0].predict_proba(x)[:,1]
    if any(not np.isfinite(v).all() for v in out.values()): raise ValueError("Nonfinite risk")
    return out,mean


def tie_value(fold,scenario,identity):
    return int.from_bytes(hashlib.sha256(f"{fold}:{scenario}:{identity}".encode()).digest()[:8],"big")


def metrics(group):
    accepted=group.accepted.to_numpy(bool); y=group.label.to_numpy(int); n=len(group)
    result={"n":n,"accepted_n":int(accepted.sum()),"coverage":float(accepted.mean()),
        "positive_coverage":float(accepted[y==1].mean()),"negative_coverage":float(accepted[y==0].mean())}
    if accepted.any():
        result["selective_error"]=float(group.loc[accepted,"error"].mean())
        result["selective_log_loss"]=float(group.loc[accepted,"loss"].mean())
    else: result.update(selective_error=None,selective_log_loss=None)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir",type=Path,required=True); ap.add_argument("--predictability-dir",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True); args=ap.parse_args()
    run=args.run_dir.resolve(); parent=args.predictability_dir.resolve(); out=args.output_dir.resolve()
    if out.exists(): raise FileExistsError("Use a new output directory")
    ps=json.loads((parent/"summary.json").read_text()); pp=json.loads((parent/"protocol.json").read_text())
    assert ps["status"]=="COMPLETE_DEVELOPMENT_ONLY" and prior.sha(parent/"protocol.json")==ps["protocol_sha256"]
    assert prior.sha(Path(prior.__file__))==pp["script_sha256"]
    sources={}; data=[]; seen=set()
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
        data.append((train,test,models))
    assert len(seen)==632
    out.mkdir(parents=True)
    protocol={"mode":"FIXED_SELECTIVE_DEVELOPMENT_AUDIT_NO_PROMOTION","parent_protocol_sha256":ps["protocol_sha256"],
        "source_sha256":sources,"script_sha256":prior.sha(Path(__file__)),"methods":list(METHODS),
        "fixed_predictor":"available_mean; prediction probability never changes","coverages":list(COVERAGES),
        "risk_direction":"lower risk accepted","simple_scores":{"confidence":"1-2*abs(mean-.5)",
            "disagreement":"max-min of available expert probabilities","mc_uncertainty":"mean available expert MC std"},
        "learned_scores":"Frozen error_available_mean classifiers from prior confidence/uncertainty/quality tiers",
        "selection":{"equal_count":"Within each outer fold/scenario keep floor(c*n), ties by same label-free SHA(id) order for every method",
            "fixed_threshold":"Per fold/method threshold is clean final-train risk quantile c; apply unchanged to every outer scenario"},
        "primary":"At equal-count 80% coverage over 14 fit stress scenarios: uncertainty-tier error score versus simple confidence",
        "success":{"stress_error_reduction":"bootstrap lower >0 and positive >=4/5 folds",
            "clean_guard":"candidate selective error - confidence <= .005",
            "operational_coverage":"fixed-threshold minimum across every fold/scenario >= .20",
            "class_coverage_gap":"fixed-threshold clean pooled abs(positive-negative coverage) <= .20"},
        "bootstrap":"1000 participant resamples within fold and disease class; all stress scenarios kept together; no refit",
        "limitations":["Exposed development data; equal-count mode assumes a batch coverage budget and is not a standalone fixed threshold",
            "Fixed thresholds target coverage only on training clean scores; actual shifted-scenario coverage may differ",
            "Selective error is undefined at zero accepted samples; never encoded as zero",
            "No protected test prediction, threshold tuning, model fitting, or multiplicity-adjusted inference"]}
    prior.write(out/"protocol.json",protocol)
    thresholds=[]; long=[]; risk_metrics=[]
    for k,(train,test,models) in enumerate(data):
        train_r,_=risks(train.drop(columns=["label"]),models); test_r,score=risks(test.drop(columns=["label"]),models)
        changed=test.copy(); changed["label"]=1-changed.label; swapped,_=risks(changed,models)
        for method in METHODS: np.testing.assert_array_equal(test_r[method],swapped[method])
        y=test.label.to_numpy(int); error=((score>=.5)!=y).astype(int); loss=prior.base.loss(y,score)
        tie=np.array([tie_value(k,s,i) for s,i in zip(test.scenario,test.id)],dtype=np.uint64)
        for method in METHODS:
            for scenario,ix in test.groupby("scenario").groups.items():
                idx=np.asarray(list(ix),int); actual=error[idx]
                if len(np.unique(actual))==2:
                    risk_metrics.append({"fold":k,"scenario":scenario,"method":method,"n":len(idx),
                        "error_prevalence":float(actual.mean()),"error_detection_auroc":float(roc_auc_score(actual,test_r[method][idx]))})
            clean=train.scenario.eq("clean").to_numpy()
            for coverage in COVERAGES:
                threshold=float(np.quantile(train_r[method][clean],coverage,method="higher"))
                thresholds.append({"fold":k,"method":method,"target_coverage":coverage,"threshold":threshold})
                fixed=test_r[method]<=threshold
                for scenario,idx0 in test.groupby("scenario").groups.items():
                    idx=np.asarray(list(idx0),int); keep=int(np.floor(coverage*len(idx)))
                    order=np.lexsort((tie[idx],test_r[method][idx])); exact=np.zeros(len(idx),bool); exact[order[:keep]]=True
                    for mode,accepted in (("equal_count",exact),("fixed_threshold",fixed[idx])):
                        for pos,j in enumerate(idx):
                            long.append({"fold":k,"id":test.id.iloc[j],"label":int(y[j]),"scenario":scenario,
                                "fit_allowed":bool(test.fit_allowed.iloc[j]),"method":method,"selection":mode,
                                "target_coverage":coverage,"risk":float(test_r[method][j]),"accepted":bool(accepted[pos] if mode=="equal_count" else accepted[pos]),
                                "error":int(error[j]),"loss":float(loss[j])})
        print(f"Fold {k}: six risk scores and train-clean thresholds evaluated",flush=True)
    decisions=pd.DataFrame(long); decisions.to_csv(out/"private_selection_decisions.csv",index=False)
    pd.DataFrame(thresholds).to_csv(out/"thresholds.csv",index=False); pd.DataFrame(risk_metrics).to_csv(out/"risk_discrimination.csv",index=False)
    rows=[]
    for keys,g in decisions.groupby(["selection","target_coverage","method","scenario"]):
        rows.append(dict(zip(("selection","target_coverage","method","scenario"),keys),fold="pooled",**metrics(g)))
    for keys,g in decisions.groupby(["selection","target_coverage","method","scenario","fold"]):
        rows.append(dict(zip(("selection","target_coverage","method","scenario","fold"),keys),**metrics(g)))
    table=pd.DataFrame(rows); table.to_csv(out/"selection_metrics.csv",index=False)
    # Primary paired clustered bootstrap.
    primary=decisions.loc[(decisions.selection=="equal_count")&(decisions.target_coverage==.8)&decisions.fit_allowed.astype(bool)&decisions.scenario.ne("clean")&decisions.method.isin(["confidence","error_uncertainty"])]
    foldgain={}
    for k,g in primary.groupby("fold"):
        rates={m:g.loc[g.method.eq(m)&g.accepted,"error"].mean() for m in ("confidence","error_uncertainty")}
        foldgain[str(k)]=float(rates["confidence"]-rates["error_uncertainty"])
    rng=np.random.default_rng(20260910); ids={(k,y):g.id.unique() for (k,y),g in primary.groupby(["fold","label"])}; draws=[]
    indexed={(k,y):g.set_index("id") for (k,y),g in primary.groupby(["fold","label"])}
    for _ in range(1000):
        sums={m:[0,0] for m in ("confidence","error_uncertainty")}
        for key,values in ids.items():
            chosen=rng.choice(values,len(values),replace=True); sample=indexed[key].loc[list(chosen)]
            for method in sums:
                z=sample.loc[sample.method==method]; sums[method][0]+=int(z.loc[z.accepted,"error"].sum()); sums[method][1]+=int(z.accepted.sum())
        draws.append(sums["confidence"][0]/sums["confidence"][1]-sums["error_uncertainty"][0]/sums["error_uncertainty"][1])
    def pooled_error(method,subset):
        g=decisions.loc[(decisions.selection=="equal_count")&(decisions.target_coverage==.8)&decisions.method.eq(method)&subset(decisions)]
        return float(g.loc[g.accepted,"error"].mean())
    stress=lambda d:d.fit_allowed.astype(bool)&d.scenario.ne("clean"); clean=lambda d:d.scenario.eq("clean")
    stress_gain=pooled_error("confidence",stress)-pooled_error("error_uncertainty",stress)
    clean_delta=pooled_error("error_uncertainty",clean)-pooled_error("confidence",clean)
    fixed=table.loc[(table.selection=="fixed_threshold")&(table.target_coverage==.8)&table.method.eq("error_uncertainty")&(table.fold!="pooled")]
    min_coverage=float(fixed.coverage.min())
    fixed_clean=table.loc[(table.selection=="fixed_threshold")&(table.target_coverage==.8)&table.method.eq("error_uncertainty")&table.scenario.eq("clean")&table.fold.eq("pooled")].iloc[0]
    class_gap=abs(float(fixed_clean.positive_coverage)-float(fixed_clean.negative_coverage))
    ci=[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]; positive=sum(v>0 for v in foldgain.values())
    checks={"stress_selective_error_reduction":stress_gain,"bootstrap_95":ci,"positive_folds":positive,
        "clean_selective_error_delta_candidate_minus_confidence":clean_delta,"fixed_threshold_min_fold_scenario_coverage":min_coverage,
        "fixed_threshold_clean_class_coverage_gap":class_gap}
    checks["rejection_score_supported"]=bool(stress_gain>0 and ci[0]>0 and positive>=4 and clean_delta<=.005 and min_coverage>=.2 and class_gap<=.2)
    checks["promotion_authorized"]=False; prior.write(out/"decision.json",checks)
    for p,h in sources.items(): assert prior.sha(Path(p))==h
    assert prior.sha(Path(__file__))==protocol["script_sha256"]
    public=("thresholds.csv","risk_discrimination.csv","selection_metrics.csv","decision.json")
    prior.write(out/"summary.json",{"status":"COMPLETE_DEVELOPMENT_ONLY","participants":632,"trained_models":0,
        "input_hashes_unchanged":True,"label_free_risk_scores":True,"protocol_sha256":prior.sha(out/"protocol.json"),
        "output_sha256":{n:prior.sha(out/n) for n in public},"private_sha256":prior.sha(out/"private_selection_decisions.csv")})
    print(json.dumps(checks,indent=2),flush=True)


if __name__=="__main__": main()
