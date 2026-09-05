"""Read-only diagnostic decomposition of a completed nested Agent run."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import nested_agent_pipeline as agent
import run_nested_oof_pilot as oof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    args=parser.parse_args(); run=args.run_dir.resolve()
    target=run/"risk_component_audit.csv"
    if target.exists(): raise FileExistsError("Diagnostic already exists")
    rows=[]
    for k in range(5):
        folder=run/f"outer_{k}/agent"
        with (folder/"agent.pkl").open("rb") as h: saved=pickle.load(h)
        frame=pd.read_csv(folder/"private_outer_validation.csv",dtype={"id":str},float_precision="round_trip")
        pred=agent.predict(frame.drop(columns=["label"]),saved["state"],saved["calibrators"],saved["policy"]["margin"])
        data=agent.sanitized(frame)
        _,quality=agent.matrix(data,saved["state"]["detectors"])
        p=data[list(agent.EXPERTS)].to_numpy(float)
        a=data[[f"{n}_available" for n in agent.EXPERTS]].to_numpy(float)
        spread=np.where(a>0,p,-np.inf).max(axis=1)-np.where(a>0,p,np.inf).min(axis=1)
        parts=np.column_stack([.35*quality[:,1],.25*quality[:,2],.25*spread,.15*(1-2*np.abs(pred["score"]-.5))])
        np.testing.assert_allclose(parts.sum(axis=1),pred["risk"],atol=1e-12)
        threshold=saved["policy"]["risk_threshold"]
        for scenario in sorted(frame.scenario.unique()):
            mask=(frame.scenario==scenario).to_numpy()
            y=frame.loc[mask,"label"].to_numpy(int)
            components=parts[mask]
            rows.append({"outer_fold":k,"scenario":scenario,"n":int(mask.sum()),"threshold":threshold,
               "min_risk":float(pred["risk"][mask].min()),"mean_risk":float(pred["risk"][mask].mean()),
               "mean_speech_component":float(components[:,0].mean()),"mean_smile_component":float(components[:,1].mean()),
               "mean_disagreement_component":float(components[:,2].mean()),"mean_uncertainty_component":float(components[:,3].mean()),
               "speech_component_alone_exceeds_threshold_rate":float((components[:,0]>threshold).mean()),
               "accepted_n":int((pred["risk"][mask]<=threshold).sum()),
               "agent_full_error_rate_at_0_5":float(((pred["score"][mask]>=.5)!=y).mean()),
               "baseline_full_error_rate_at_0_5":float(((pred["action_scores"][mask,0]>=.5)!=y).mean())})
    table=pd.DataFrame(rows); table.to_csv(target,index=False)
    speech=table.loc[table.scenario.str.startswith("speech_")]
    summary={"mode":"POSTHOC_DIAGNOSIS_ONLY_NO_POLICY_CHANGE","speech_scenarios":int(speech.scenario.nunique()),
             "speech_fold_scenario_pairs":len(speech),"speech_all_rejected_pairs":int((speech.accepted_n==0).sum()),
             "speech_component_alone_exceeds_threshold_mean_rate":float(speech.speech_component_alone_exceeds_threshold_rate.mean()),
             "risk_threshold_min":float(table.threshold.min()),"risk_threshold_max":float(table.threshold.max()),
             "risk_reconstruction_verified":True,"output_sha256":oof.digest(target)}
    oof.write_json(run/"risk_diagnostic_summary.json",summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__": main()
