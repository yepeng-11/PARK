"""Independent post-run membership, configuration and saved-policy audit."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import nested_agent_pipeline as agent
import run_nested_agent_development as driver
import run_nested_oof_pilot as oof
import train_fusion_agent_v3_router as v3


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--protocol-dir",type=Path,required=True)
    args=parser.parse_args(); run=args.run_dir.resolve()
    root=Path(__file__).resolve().parents[1]
    protocol=driver.validate(root,args.protocol_dir.resolve())
    _,folds=v3.load_protocol(root/"results/fusion_agent_v3_protocol")
    expected_people=set(folds.participant_id)
    summary=json.loads((run/"run_summary.json").read_text())
    for n,h in summary["aggregate_sha256"].items():
        if oof.digest(run/n)!=h: raise ValueError("Aggregate hash mismatch")
    config_sets={n:set() for n in ("finger","speech","smile","fusion")}
    seen=set(); rows=[]; inventory={}
    for k in range(5):
        folder=run/f"outer_{k}"; base_dir=folder/"base"
        plan=json.loads((base_dir/"execution_plan.json").read_text())
        base_summary=json.loads((base_dir/"nested_oof_summary.json").read_text())
        for n,h in base_summary["aggregate_sha256"].items():
            if oof.digest(base_dir/n)!=h: raise ValueError("Base aggregate mismatch")
        jobs=[]
        for job in plan["suites"]:
            suite=base_dir/job["name"]
            scope=base_dir/"private_scopes"/f"{job['name']}.csv"
            s=json.loads((suite/"benchmark_summary.json").read_text())
            if oof.digest(scope)!=job["scope_sha256"] or s["scope_sha256"]!=job["scope_sha256"]:
                raise ValueError("Scope hash mismatch")
            for n,h in s["artifacts_sha256"].items():
                if oof.digest(suite/n)!=h: raise ValueError("Suite artifact mismatch")
            actual=pd.read_csv(suite/"private_members.csv",dtype={"participant_id":str})
            planned=pd.read_csv(scope,dtype={"participant_id":str})
            if set(map(tuple,actual.to_numpy()))!=set(map(tuple,planned.to_numpy())): raise ValueError("Member drift")
            jobs.append({**job,"roles":{r:set(g.participant_id) for r,g in actual.groupby("role")}})
            for n in config_sets:
                meta=json.loads((suite/n/"training.json").read_text())
                if len(meta["history"])!=meta["config"]["num_epochs"]: raise ValueError("Truncated training")
                config_sets[n].add(driver.canonical(meta["config"]))
        fold=folds.loc[folds.outer_fold==k]
        t=set(fold.loc[fold.outer_role=="train","participant_id"])
        v=set(fold.loc[fold.outer_role=="validation","participant_id"])
        h={j:set(fold.loc[(fold.outer_role=="train")&(fold.inner_validation_fold==j),"participant_id"]) for j in range(4)}
        oof.validate_plan(jobs,t,v,h)
        if seen&v: raise ValueError("Repeated outer participant")
        seen |= v
        a=folder/"agent"
        state=json.loads((a/"agent_summary.json").read_text())
        for n,sha in state["private_artifact_sha256"].items():
            if oof.digest(a/n)!=sha: raise ValueError("Private table changed")
        for j in range(4):
            train=pd.read_csv(a/f"private_meta_{j}_train.csv",dtype={"id":str})
            val=pd.read_csv(a/f"private_meta_{j}_validation.csv",dtype={"id":str})
            if set(train.id)!=t-h[j] or set(val.id)!=h[j]: raise ValueError("Meta input membership")
            c=set(val.loc[val.block=="utility_calibration","id"])
            s=set(val.loc[val.block=="strategy_selection","id"])
            if c&s or c|s!=h[j]: raise ValueError("Utility calibration/selection membership")
        f=pd.read_csv(a/"private_outer_validation.csv",dtype={"id":str},float_precision="round_trip")
        old=pd.read_csv(a/"private_outer_decisions.csv",dtype={"id":str},float_precision="round_trip")
        if set(f.id)!=v: raise ValueError("Wrong outer target")
        if not np.array_equal(f[["id","scenario"]].to_numpy(),old[["id","scenario"]].to_numpy()): raise ValueError("Prediction alignment")
        with (a/"agent.pkl").open("rb") as handle: saved=pickle.load(handle)
        new=agent.predict(f.drop(columns=["label"]),saved["state"],saved["calibrators"],saved["policy"]["margin"])
        for col in ("score","risk","action"):
            np.testing.assert_allclose(new[col],old[col].to_numpy(),atol=1e-12,rtol=1e-12)
        for p in a.glob("*.pkl"): inventory[str(p.relative_to(run))]=oof.digest(p)
        rows.append({"outer_fold":k,"members_verified":True,"separate_process_saved_inference_verified":True,
                     "outer_participants":len(v),"base_suites":len(jobs)})
    if seen!=expected_people: raise ValueError("Incomplete outer coverage")
    if any(len(v)!=1 for v in config_sets.values()): raise ValueError("Architecture/config drift between suites")
    result={"status":"PASS","protocol_sha256":protocol["protocol_sha256"],"suites":105,"networks":420,
            "outer_participants":len(seen),"fold_checks":rows,"config_sha256":{k:next(iter(v)) for k,v in config_sets.items()},
            "saved_agent_sha256":inventory,"note":"Read-only post-run verification; no model fitting or policy changes"}
    target=run/"postrun_audit.json"
    if target.exists(): raise FileExistsError("Audit already exists")
    oof.write_json(target,result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
