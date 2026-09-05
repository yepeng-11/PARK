"""Freeze then execute five fully nested development folds; no final-test claims."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import nested_agent_pipeline as agent
import run_nested_oof_pilot as oof
import train_fusion_agent_v3_router as v3


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def freeze(root, folder, engineering):
    if folder.exists(): raise FileExistsError("Never overwrite a frozen protocol")
    report=json.loads((engineering/"agent_summary.json").read_text())
    if report["status"]!="ENGINEERING_COMPLETE" or not all(report["regression_tests"].values()):
        raise ValueError("Engineering verification required")
    if report["source_sha256"]["nested_agent_pipeline.py"]!=oof.digest(root/"tools/nested_agent_pipeline.py"):
        raise ValueError("Engineering run did not verify current agent source")
    old,_=v3.load_protocol(root/"results/fusion_agent_v3_protocol")
    criteria=[
        {"id":"clean_noninferiority","operator":">=","value":-.005},
        {"id":"stress_log_loss_regret","operator":"<=","value":0.},
        {"id":"smile_feature_gain_vs_best_fixed","operator":">","value":0.},
        {"id":"clean_coverage","operator":">=","value":.8},
        {"id":"clean_action_rate","operator":"<=","value":.2},
        {"id":"clean_ece_delta","operator":"<=","value":.02},
        {"id":"fold_clean_stability","operator":">=","value":4},
        {"id":"stress_gain_bootstrap_lower","operator":">","value":0.},
        {"id":"stress_coverage","operator":">=","value":.5},
    ]
    protocol={"version":"nested-agent-v4-development-1.0","status":"FROZEN_DEVELOPMENT_ONLY",
        "source_sha256":{p.name:oof.digest(p) for p in (root/"tools").glob("*.py")},
        "parent_split_protocol_sha256":old["protocol_sha256"],"dataset_sha256":old["source_dataset_sha256"],
        "split_manifest_sha256":old["artifact_sha256"]["participant_fold_manifest.csv"],
        "prior_development_results_exposed":True,"outer_folds":[0,1,2,3,4],
        "base_seed_by_fold":{str(k):101+1000*k for k in range(5)},
        "base_suites_per_fold":21,"networks_per_suite":4,"mc_trials":30,
        "scenarios":agent.scenario_registry(),"actions":list(agent.ACTIONS),
        "baseline":"equal weights over explicitly available calibrated expert probabilities; UFNet also fixed comparator",
        "selection":{"quantile_candidates":list(agent.QUANTILES),"clean_auc_margin":-.005,"clean_action_cap":.2,
          "regret_lower_bound":0.,"bootstrap_replicates":1000,"calibration_selection_split":"deterministic label-blind halves of each meta holdout",
          "risk_coverage_target":.9,"final_margin":"median selected per-fold threshold; recheck and recompute policy and risk; fail closed if invalid"},
        "calibration":"participant-level base Platt; heldout utility Ridge per meta fold; OOF utility calibration then final refit",
        "quality_model":"fixed depth<=3 HGB; synthetic targets only; fit on lower OOF features",
        "utility_model":"fixed depth<=3 HGB; leaf row minimum = 30 times maximum scene copies per participant; early stopping disabled",
        "criteria":criteria,"criteria_aggregation":"mean across outer folds; stress = nonclean fit-allowed scenarios; bootstrap resamples participants after averaging stress gains",
        "smile_gain_scenarios":["smile_gaussian_0.25","smile_gaussian_0.5","smile_feature_permutation"],
        "heldout_stress_only":"smile_opposite_consensus; never used for loss fitting, calibration or policy selection",
        "engineering_report_sha256":oof.digest(engineering/"agent_summary.json"),
        "promotion_authorized":False,"final_validation":"requires a separate unseen participant-disjoint cohort and separately frozen confirmatory analysis",
        "privacy":"all participant data and pickles remain server-side; only aggregate reports may be downloaded"}
    protocol["protocol_sha256"]=canonical(protocol)
    folder.mkdir(parents=True)
    oof.write_json(folder/"protocol.json",protocol)
    print(f"Frozen development protocol: {protocol['protocol_sha256']}",flush=True)


def validate(root, protocol_dir):
    p=json.loads((protocol_dir/"protocol.json").read_text())
    expected=p.pop("protocol_sha256")
    if canonical(p)!=expected: raise ValueError("Canonical protocol mismatch")
    p["protocol_sha256"]=expected
    for n,h in p["source_sha256"].items():
        if oof.digest(root/"tools"/n)!=h: raise ValueError(f"Source changed since freeze: {n}")
    old,_=v3.load_protocol(root/"results/fusion_agent_v3_protocol")
    if old["protocol_sha256"]!=p["parent_split_protocol_sha256"]: raise ValueError("Split protocol changed")
    if oof.digest(root/"results/protocol_alignment_audit/cleaned_aligned.csv")!=p["dataset_sha256"]: raise ValueError("Dataset changed")
    return p


def execute(command,log):
    with log.open("w",encoding="utf-8") as handle:
        p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in p.stdout:
            handle.write(line); handle.flush()
            if line.startswith(("Starting ","Completed ","Meta state ","Agent fold ","Scenario inputs ")):
                print(f"{log.parent.name}: {line.strip()}",flush=True)
        if p.wait(): raise RuntimeError(f"Stage failed; inspect {log}")


def one_fold(root,folder,k,protocol):
    start=time.perf_counter(); folder.mkdir()
    execute([sys.executable,"-u",str(root/"tools/run_nested_oof_pilot.py"),"--fold",str(k),
        "--seed",str(protocol["base_seed_by_fold"][str(k)]),"--output-dir",str(folder/"base")],folder/"base.log")
    execute([sys.executable,"-u",str(root/"tools/nested_agent_pipeline.py"),"--base-dir",str(folder/"base"),
        "--output-dir",str(folder/"agent")],folder/"agent.log")
    state=json.loads((folder/"agent/agent_summary.json").read_text())
    if not all(state["regression_tests"].values()) or state["locked_predictions_generated"]:
        raise RuntimeError("Fold regression or isolation failed")
    return {"outer_fold":k,"wall_seconds":time.perf_counter()-start,"policy_enabled":state["policy"]["enabled"]}


def acceptance(metrics,decisions,protocol):
    piv=metrics.pivot(index=["outer_fold","scenario"],columns="model",values=["auroc","log_loss","ece","coverage","action_rate"])
    clean=piv.xs("clean",level="scenario")
    delta=clean[("auroc","fusion_agent")]-clean[("auroc","available_mean")]
    scenarios=[s["name"] for s in protocol["scenarios"] if s["name"]!="clean" and s["fit"]]
    stress=piv.loc[piv.index.get_level_values("scenario").isin(scenarios)].groupby(level="outer_fold").mean()
    smile=piv.loc[piv.index.get_level_values("scenario").isin(protocol["smile_gain_scenarios"])].groupby(level="outer_fold").mean()
    obs={"clean_noninferiority":float(delta.mean()),
         "stress_log_loss_regret":float((stress[("log_loss","fusion_agent")]-stress[("log_loss","available_mean")]).mean()),
         "smile_feature_gain_vs_best_fixed":float((smile[("auroc","fusion_agent")]-np.maximum(smile[("auroc","available_mean")],smile[("auroc","ufnet")])).mean()),
         "clean_coverage":float(clean[("coverage","fusion_agent")].mean()),
         "clean_action_rate":float(clean[("action_rate","fusion_agent")].mean()),
         "clean_ece_delta":float((clean[("ece","fusion_agent")]-clean[("ece","available_mean")]).mean()),
         "fold_clean_stability":int((delta>=-.005).sum()),
         "stress_gain_bootstrap_lower":agent.lower_bound(decisions.loc[decisions.scenario.isin(scenarios)]),
         "stress_coverage":float(stress[("coverage","fusion_agent")].mean())}
    rows=[]
    for c in protocol["criteria"]:
        value=obs[c["id"]]
        passed={">=":value>=c["value"],"<=":value<=c["value"],">":value>c["value"]}[c["operator"]]
        rows.append({"criterion":c["id"],"observed":value,"operator":c["operator"],"required":c["value"],"passed":bool(passed)})
    return pd.DataFrame(rows)


def report(output,metrics,checks,decision):
    lines=["# 完整嵌套 Fusion Agent 开发评估", "",f"协议哈希：`{decision['protocol_sha256']}`。",
       "", f"内部开发检查：{decision['passed_criteria']}/{decision['total_criteria']}；启用路由折数：{decision['enabled_folds']}/5。",
       "", "所有结果仅属于已暴露开发数据；不构成未见外部数据或临床有效性的证明。", "",
       "| 检查 | 观测 | 条件 | 门槛 | 通过 |", "| --- | ---: | --- | ---: | --- |"]
    for r in checks.itertuples(): lines.append(f"| {r.criterion} | {r.observed:.5f} | {r.operator} | {r.required:.5f} | {r.passed} |")
    lines += ["", "| 场景 | Agent AUC | Available mean AUC | UFNet AUC | Agent 覆盖率 |", "| --- | ---: | ---: | ---: | ---: |"]
    for s,f in metrics.groupby("scenario"):
        a=f.groupby("model").auroc.mean(); cov=f.loc[f.model=="fusion_agent","coverage"].mean()
        lines.append(f"| {s} | {a['fusion_agent']:.4f} | {a['available_mean']:.4f} | {a['ufnet']:.4f} | {cov:.4f} |")
    lines += ["", "原始特征扰动对专家与 UFNet 均重新推理；score_shift 与 opposite_consensus 则明确是分数通道故障。",
       "成员隔离覆盖基础优化训练、checkpoint 选择、概率校准和策略验证边界。收益校准使用独立 meta 留出半集，策略选择使用另一半；最终模型采用 cross-fitted 校准后重拟合。",
       "Bootstrap 下界是开发选择与评估量，不是搜索后仍无偏的确认性置信声明。上游架构选择与恢复特征来源、训练数据历史暴露仍是局限。",
       "选择后 AUC 仅在保留样本包含两个类别时报告；零覆盖不能解读为有效诊断。"]
    (output/"NESTED_AGENT_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir",type=Path,required=True)
    parser.add_argument("--engineering-dir",type=Path)
    parser.add_argument("--freeze",action="store_true")
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--workers",type=int,choices=(1,2),default=2)
    args=parser.parse_args(); root=Path(__file__).resolve().parents[1]
    if args.freeze:
        freeze(root,args.protocol_dir.resolve(),args.engineering_dir.resolve()); return
    protocol=validate(root,args.protocol_dir.resolve())
    output=args.output_dir.resolve()
    if output.exists(): raise FileExistsError("Never overwrite a run")
    output.mkdir(parents=True); start=time.perf_counter()
    outcomes=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(one_fold,root,output/f"outer_{k}",k,protocol) for k in protocol["outer_folds"]]
        for f in as_completed(futures):
            outcomes.append(f.result())
            oof.write_json(output/"progress.json",{"completed_folds":outcomes,"total":5})
    validate(root,args.protocol_dir.resolve())
    metrics=pd.concat([pd.read_csv(output/f"outer_{k}/agent/outer_metrics.csv") for k in range(5)],ignore_index=True)
    decisions=pd.concat([pd.read_csv(output/f"outer_{k}/agent/private_outer_decisions.csv",dtype={"id":str}).assign(outer_fold=k) for k in range(5)],ignore_index=True)
    if decisions.loc[decisions.scenario=="clean","id"].duplicated().any(): raise ValueError("Outer participant duplicated")
    checks=acceptance(metrics,decisions,protocol)
    decision={"status":"INTERNAL_CRITERIA_PASS" if checks.passed.all() else "INTERNAL_CRITERIA_NOT_MET",
              "passed_criteria":int(checks.passed.sum()),"total_criteria":len(checks),
              "enabled_folds":sum(int(x["policy_enabled"]) for x in outcomes),
              "protocol_sha256":protocol["protocol_sha256"],"promotion_authorized":False,
              "locked_predictions_generated":False,"outer_participants":int(decisions.id.nunique())}
    metrics.to_csv(output/"outer_metrics.csv",index=False); checks.to_csv(output/"acceptance_checks.csv",index=False)
    pd.DataFrame(outcomes).sort_values("outer_fold").to_csv(output/"fold_timings.csv",index=False)
    oof.write_json(output/"selection_decision.json",decision)
    report(output,metrics,checks,decision)
    summary={"protocol_sha256":protocol["protocol_sha256"],"wall_seconds":time.perf_counter()-start,"workers":args.workers,
             "base_suites":105,"neural_networks":420,"full_folds":5,"decision":decision,
             "output_bytes_before_summary":sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
             "aggregate_sha256":{n:oof.digest(output/n) for n in ("outer_metrics.csv","acceptance_checks.csv","fold_timings.csv","selection_decision.json","NESTED_AGENT_REPORT.md")}}
    oof.write_json(output/"run_summary.json",summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__": main()
