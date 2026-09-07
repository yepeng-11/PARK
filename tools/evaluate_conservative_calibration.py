"""Fixed logit-interpolation ablation; no fitting, rejection, or policy selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import run_fusion_calibration_diagnostic as previous


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(); parent=args.parent_dir.resolve(); out=args.output_dir.resolve()
    if out.exists(): raise FileExistsError("Use a new output directory")
    sha=previous.exp.sha; write=previous.exp.write
    summary=json.loads((parent/"summary.json").read_text())
    protocol=json.loads((parent/"protocol.json").read_text())
    assert summary["status"]=="COMPLETE_DEVELOPMENT_ONLY"
    assert sha(parent/"protocol.json")==summary["protocol_sha256"]
    assert sha(Path(previous.__file__))==protocol["script_sha256"]
    frames=[]; sources={}; seen=set()
    for k in range(5):
        p=parent/f"outer_{k}/private_predictions.csv"
        assert sha(p)==summary["private_sha256"][str(p.relative_to(parent))]
        sources[str(p)]=sha(p)
        f=pd.read_csv(p,dtype={"id":str},float_precision="round_trip")
        assert not f.duplicated(["id","scenario"]).any()
        assert f.groupby("id").size().eq(16).all()
        assert f.groupby("id").label.nunique().eq(1).all()
        assert f.outer_fold.eq(k).all()
        assert not set(f.id)&seen; seen|=set(f.id); frames.append(f)
    assert len(seen)==632
    alphas={"raw":0.,"blend25":.25,"blend50":.5,"blend75":.75,"full":1.}
    out.mkdir(parents=True)
    frozen=dict(mode="POSTHOC_MOTIVATED_FIXED_ABLATION_NO_SELECTION",alphas=alphas,
        transform="sigmoid((1-alpha)*logit(raw_mean)+alpha*logit(calibrated_mean))",
        primary="Clean and three-scenario speech-Gaussian log-loss gains vs raw; report each severity",
        success_description="Nonzero alpha with positive clean gain and nonnegative Gaussian gain point estimates; descriptive only, not promotion",
        coverage=1.,fit=False,parameter_search=False,source_sha256=sources,
        script_sha256=sha(Path(__file__)),parent_protocol_sha256=summary["protocol_sha256"],
        bootstrap="1000 paired participant resamples stratified by fold and label; scenarios kept together",
        limitations=["Already exposed development data; no independent confirmation or selection of best alpha",
            "Positive monotone per-fold calibration preserves AUROC; this is not a discrimination improvement",
            "Intervals descriptive, not multiplicity corrected; no model refitting"])
    write(out/"protocol.json",frozen)
    f=pd.concat(frames,ignore_index=True)
    raw=previous.scalar(f.available_mean).ravel(); full=previous.scalar(f.mean_platt).ravel()
    for n,a in alphas.items(): f[n]=expit((1-a)*raw+a*full)
    np.testing.assert_allclose(f.raw,f.available_mean,atol=1e-12)
    np.testing.assert_allclose(f.full,f.mean_platt,atol=1e-12)
    metrics=[]
    for fold,g in [("pooled",f)]+[(str(k),g) for k,g in f.groupby("outer_fold")]:
        for scenario,s in g.groupby("scenario"):
            for n in alphas:
                if fold!="pooled":
                    assert abs(roc_auc_score(s.label,s[n])-roc_auc_score(s.label,s.raw))<1e-12
                metrics.append(dict(fold=fold,scenario=scenario,model=n,n=len(s),coverage=1.,**previous.stats(s.label.to_numpy(),s[n].to_numpy())))
    pd.DataFrame(metrics).to_csv(out/"metrics.csv",index=False)
    for n in alphas: f[n+"_loss"]=previous.exp.base.loss(f.label.to_numpy(),f[n].to_numpy())
    comparisons=[]
    masks={"clean":f.scenario.eq("clean"),"stress":f.fit_allowed.astype(bool)&f.scenario.ne("clean"),
        "speech_gaussian":f.scenario.str.startswith("speech_gaussian"),"speech_mask":f.scenario.str.startswith("speech_mask")}
    for subset,mask in masks.items():
        people=f.loc[mask].groupby(["outer_fold","id","label"])[[n+"_loss" for n in alphas]].mean().reset_index()
        groups=[g.index.to_numpy() for _,g in people.groupby(["outer_fold","label"])]; rng=np.random.default_rng(20260908)
        indices=[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]) for _ in range(1000)]
        for n in list(alphas)[1:]:
            gain=people.raw_loss.to_numpy()-people[n+"_loss"].to_numpy()
            draws=[gain[ix].mean() for ix in indices]
            comparisons.append(dict(subset=subset,model=n,gain=float(gain.mean()),
                p025=float(np.quantile(draws,.025)),p975=float(np.quantile(draws,.975)),
                positive_folds=int((people.assign(gain=gain).groupby("outer_fold").gain.mean()>0).sum())))
    table=pd.DataFrame(comparisons); table.to_csv(out/"comparisons.csv",index=False)
    checks=[]
    for n in list(alphas)[1:]:
        clean=float(table.loc[table.model.eq(n)&table.subset.eq("clean"),"gain"].iloc[0])
        noise=float(table.loc[table.model.eq(n)&table.subset.eq("speech_gaussian"),"gain"].iloc[0])
        checks.append(dict(model=n,clean_gain=clean,gaussian_gain=noise,point_tradeoff_satisfied=clean>0 and noise>=0))
    for p,h in sources.items(): assert sha(Path(p))==h
    assert sha(Path(__file__))==frozen["script_sha256"]
    write(out/"summary.json",dict(status="COMPLETE_DEVELOPMENT_ONLY",participants=632,
        trained_models=0,endpoint_checks=True,fold_auc_invariance=True,inputs_unchanged=True,
        checks=checks,selected_model=None,protocol_sha256=sha(out/"protocol.json"),
        output_sha256={n:sha(out/n) for n in ("metrics.csv","comparisons.csv")}))
    print(table.to_string(index=False),flush=True)


if __name__=="__main__": main()
