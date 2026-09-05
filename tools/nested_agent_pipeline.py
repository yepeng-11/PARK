"""Isolated nested Fusion Agent engineering and development evaluation.

Uses the 21-suite clean-base plan. Quality detectors and loss models are fixed
before validation. Each meta holdout is split label-blindly into utility
calibration and strategy selection; neither subset enters lower-model fitting.
"""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

import evaluate_calibrated_fusion as cal
import evaluate_learned_quality_gate as quality
import evaluate_pretrained as ev
import run_nested_oof_pilot as oof
import train_paired_baselines as paired
import train_fusion_agent_v3_router as v3


EXPERTS = ("finger", "speech", "smile")
ACTIONS = ("available_mean", "drop_speech", "drop_smile", "shrink_smile", "ufnet")
DESCRIPTORS = quality.DESCRIPTORS
QUANTILES = (0.80, 0.85, 0.90, 0.95, 0.975)


def scenario_registry():
    out = [{"name": "clean", "kind": "clean", "modality": -1, "severity": 0., "fit": True}]
    out += [{"name": f"missing_{n}", "kind": "missing", "modality": i, "severity": 1., "fit": True} for i, n in enumerate(EXPERTS)]
    for kind, values in (("gaussian", (0.25, 0.5, 1.)), ("mask", (0.1, 0.25, 0.5))):
        out += [{"name": f"speech_{kind}_{v}", "kind": kind, "modality": 1, "severity": v, "fit": True} for v in values]
    out += [{"name": f"smile_gaussian_{v}", "kind": "gaussian", "modality": 2, "severity": v, "fit": True} for v in (0.25, 0.5)]
    out += [{"name": "smile_feature_permutation", "kind": "feature_permutation", "modality": 2, "severity": 1., "fit": True}]
    out += [{"name": f"smile_score_shift_{v}", "kind": "score_shift", "modality": 2, "severity": v, "fit": True} for v in (-1., 1.)]
    out += [{"name": "smile_opposite_consensus", "kind": "score_stress", "modality": 2, "severity": 1., "fit": False}]
    return out


def rng(*parts):
    seed = int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little")
    return np.random.default_rng(seed)


def partition_ids(ids, salt):
    ordered = sorted(set(ids), key=lambda x: hashlib.sha256(f"{salt}:{x}".encode()).hexdigest())
    return set(ordered[:len(ordered)//2]), set(ordered[len(ordered)//2:])


def perturb(partition, scenario, seed):
    out = partition.copy(deep=True)
    index = scenario["modality"]
    if index < 0 or scenario["kind"].startswith("score"):
        return out
    col = f"features_{index}"
    matrix = np.stack(out[col]).copy()
    random = rng(seed, scenario["name"])
    if scenario["kind"] == "gaussian":
        matrix += random.normal(0, scenario["severity"], matrix.shape)
    elif scenario["kind"] == "mask":
        matrix *= random.random(matrix.shape) >= scenario["severity"]
    elif scenario["kind"] == "missing":
        matrix[:] = 0
    elif scenario["kind"] == "feature_permutation":
        matrix = matrix[random.permutation(len(matrix))]
    else:
        raise ValueError("Unknown perturbation")
    out[col] = list(matrix.astype(np.float32))
    return out


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)
    return np.log(p/(1-p)).reshape(-1, 1)


def loss(y, p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return -(y*np.log(p)+(1-y)*np.log(1-p))


def generate_inputs(base_dir, output):
    """Read target features only; never pass real target labels into inference."""
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((base_dir / "execution_plan.json").read_text())
    complete = json.loads((base_dir / "nested_oof_summary.json").read_text())
    if complete["suites"] != 21 or not complete["isolated_meta_ancestors"]:
        raise ValueError("Incomplete nested base plan")
    for n, h in complete["aggregate_sha256"].items():
        if oof.digest(base_dir/n) != h:
            raise ValueError("Base plan changed")
    protocol, manifest = v3.load_protocol(root/"results/fusion_agent_v3_protocol")
    data = root/"results/protocol_alignment_audit/cleaned_aligned.csv"
    if oof.digest(data) != protocol["source_dataset_sha256"]:
        raise ValueError("Dataset hash mismatch")
    frame = paired.load_vector_csv(data)
    module = ev.load_upstream_module(root)
    masks = paired.split_masks(module, frame)
    locked = set(frame.loc[masks["internal_test"] | masks["validation_1"] | masks["validation_2"] | masks["global"], "id"].astype(str))
    fusion_config = ev.read_json(Path(module.MODEL_CONFIG_PATH))
    choices = module.MODEL_SUBSETS[int(fusion_config["model_subset_choice"])]
    module.NUM_MODELS = len(choices)
    upstream = ev.checkpoint_paths(module, choices)
    configs = [ev.read_json(p["config"]) for p in upstream]
    frame = frame.loc[frame.id.astype(str).isin(set(manifest.participant_id))].reset_index(drop=True)
    labels = frame.groupby("id").label.first().to_dict()
    raw = paired.inverse_original_scaling(frame, configs, upstream)
    raw["label"] = 0  # inference loader may inspect labels only for row order
    device = ev.resolve_device("cuda")
    records, input_hashes = {}, {}
    for idx, suite_plan in enumerate(plan["suites"]):
        name = suite_plan["name"]
        suite = base_dir/name
        summary = json.loads((suite/"benchmark_summary.json").read_text())
        for path, h in summary["artifacts_sha256"].items():
            if oof.digest(suite/path) != h:
                raise ValueError(f"Changed base artifact: {name}/{path}")
            input_hashes[str((suite/path).resolve())] = h
        members = pd.read_csv(suite/"private_members.csv", dtype={"participant_id": str})
        roles = {r: set(g.participant_id) for r, g in members.groupby("role")}
        import benchmark_fold_training as b
        b.audit_members(roles, {"calibrated_prediction": list(oof.FIT_ROLES)}, roles["prediction"])
        if set().union(*roles.values()) & locked:
            raise ValueError("Locked participant in input")
        target = raw.loc[raw.id.astype(str).isin(roles["prediction"])].copy()
        if suite_plan["kind"] == "meta_validation":
            a, b_ids = partition_ids(roles["prediction"], f"utility-{name}")
            blocks = {"utility_calibration": a, "strategy_selection": b_ids}
        else:
            blocks = {"model_training" if suite_plan["kind"] in {"meta_oof", "final_oof"} else "outer_evaluation": roles["prediction"]}
        for i in range(3):
            with (suite/f"scaler_{i}.pkl").open("rb") as h:
                scaler = pickle.load(h)
            if scaler is not None:
                target[f"features_{i}"] = list(scaler.transform(np.stack(target[f"features_{i}"])).astype(np.float32))
        predictors, fusion = cal.load_models(module, suite, configs, fusion_config, [len(target.iloc[0][f"features_{i}"]) for i in range(3)], device)
        with (suite/"calibrators.pkl").open("rb") as h:
            calibrators = pickle.load(h)
        tables = []
        for block, ids in blocks.items():
            part = target.loc[target.id.astype(str).isin(ids)].reset_index(drop=True)
            clean = None
            for scenario in scenario_registry():
                seed = int(rng(name, block, scenario["name"]).integers(0, 2**30))
                if scenario["kind"].startswith("score"):
                    prediction = clean.copy()
                    if scenario["kind"] == "score_shift":
                        prediction["smile"] = 1/(1+np.exp(-(logit(prediction.smile).ravel()+scenario["severity"])))
                    else:
                        peer = prediction[["finger", "speech"]].mean(axis=1)
                        prediction["smile"] = np.where(peer >= .5, .02, .98)
                else:
                    changed = perturb(part, scenario, seed)
                    missing = [scenario["modality"]] if scenario["kind"] == "missing" else []
                    pred = quality.predict_quality_inputs(module, changed, predictors, fusion, device, 30, seed, missing, [])
                    numeric = [c for c in pred if c not in {"id", "label", "row_id"}]
                    prediction = pred.groupby("id", as_index=False)[numeric].mean()
                    for expert in (*EXPERTS, "ufnet"):
                        prediction[expert] = calibrators[expert].predict_proba(logit(prediction[expert]))[:, 1]
                    if scenario["kind"] == "clean":
                        clean = prediction.copy()
                for i, expert in enumerate(EXPERTS):
                    prediction[f"{expert}_available"] = float(not (scenario["kind"] == "missing" and scenario["modality"] == i))
                    prediction[f"{expert}_corrupt_target"] = int(scenario["modality"] == i and scenario["kind"] not in {"missing", "clean"})
                prediction["scenario"] = scenario["name"]
                prediction["scenario_kind"] = scenario["kind"]
                prediction["fit_allowed"] = scenario["fit"]
                prediction["block"] = block
                prediction["source_suite"] = name
                tables.append(prediction)
        records[name] = pd.concat(tables, ignore_index=True)
        print(f"Scenario inputs {idx+1}/21: {name}", flush=True)
    # Recheck every consumed model/scaler/calibrator/member artifact after inference.
    if any(oof.digest(p) != h for p, h in input_hashes.items()):
        raise ValueError("Base artifact changed during scenario inference")
    keys = {f"meta_{k}_train": [x["name"] for x in plan["suites"] if x["kind"] == "meta_oof" and x["meta_fold"] == k] for k in range(4)}
    keys.update({f"meta_{k}_validation": [f"meta_{k}_validation"] for k in range(4)})
    keys["final_train"] = [x["name"] for x in plan["suites"] if x["kind"] == "final_oof"]
    keys["outer_validation"] = ["outer_prediction"]
    caches = {}
    for name, names in keys.items():
        table = pd.concat([records[n] for n in names], ignore_index=True)
        if table.duplicated(["id", "scenario"]).any():
            raise ValueError("Duplicate participant scenario")
        if not np.isfinite(table[list(EXPERTS)+["ufnet"]].to_numpy()).all():
            raise ValueError("Nonfinite input")
        table["label"] = table.id.map(labels).astype(int)
        table.to_csv(output/f"private_{name}.csv", index=False)
        caches[name] = table
    return caches, complete["outer_fold"], input_hashes


def sanitized(frame):
    out = frame.copy()
    for expert in EXPERTS:
        missing = out[f"{expert}_available"].to_numpy() == 0
        out.loc[missing, expert] = .5
        for c in [f"{expert}_mc_std", *(f"{expert}_feature_{d}" for d in DESCRIPTORS)]:
            out.loc[missing, c] = 0.
    return out


def detector_matrix(frame, expert):
    columns = [expert, f"{expert}_mc_std", *(f"{expert}_feature_{d}" for d in DESCRIPTORS)]
    return sanitized(frame)[columns].to_numpy(float)


def actions(frame):
    data = sanitized(frame)
    probs = data[list(EXPERTS)].to_numpy(float)
    avail = data[[f"{n}_available" for n in EXPERTS]].to_numpy(float)
    if (avail.sum(axis=1) == 0).any():
        raise ValueError("No available modality")
    baseline = (probs*avail).sum(axis=1)/avail.sum(axis=1)
    def without(index):
        a = avail.copy(); a[:, index] = 0
        return np.divide((probs*a).sum(axis=1), a.sum(axis=1), out=baseline.copy(), where=a.sum(axis=1)>0)
    no_smile = without(2)
    shrink = probs.copy(); shrink[:, 2] = .5*probs[:, 2]+.5*no_smile
    return np.column_stack([baseline, without(1), no_smile, (shrink*avail).sum(axis=1)/avail.sum(axis=1), data.ufnet])


def matrix(frame, detectors):
    data = sanitized(frame)
    probs = data[list(EXPERTS)].to_numpy(float)
    avail = data[[f"{n}_available" for n in EXPERTS]].to_numpy(float)
    quality_scores = np.column_stack([(detectors[n].predict_proba(detector_matrix(data, n))[:, 1]
                                      if detectors[n] is not None else np.zeros(len(data))) * avail[:, i]
                                     for i, n in enumerate(EXPERTS)])
    gaps = np.column_stack([np.abs(probs[:, i]-probs[:, j])*avail[:, i]*avail[:, j] for i,j in ((0,1),(0,2),(1,2))])
    x = np.column_stack([probs, data.ufnet, avail, quality_scores, gaps,
                         data[[f"{n}_mc_std" for n in EXPERTS]].to_numpy(float), actions(data)])
    return x, quality_scores


def fit_state(train, seed):
    train = train.loc[train.fit_allowed].copy()
    copies = int(train.groupby("id").size().max())
    min_leaf = 30 * copies  # each participant contributes <= copies rows
    detectors = {}
    for i, expert in enumerate(EXPERTS):
        d = sanitized(train)
        y = d[f"{expert}_corrupt_target"].to_numpy(int)
        # Finger has only missingness, not a synthetic corrupt class: use a zero predictor.
        if len(np.unique(y)) < 2:
            detectors[expert] = None
        else:
            weights = np.where(y==1, len(y)/(2*y.sum()), len(y)/(2*(y==0).sum()))
            detectors[expert] = HistGradientBoostingClassifier(max_iter=100, max_depth=3, max_leaf_nodes=7,
                min_samples_leaf=min_leaf, l2_regularization=2., learning_rate=.05,
                early_stopping=False, random_state=seed+i).fit(detector_matrix(d, expert), y, sample_weight=weights)
    x, _ = matrix(train, detectors)
    y = train.label.to_numpy(int)
    scores = actions(train)
    models = [HistGradientBoostingRegressor(max_iter=100, max_depth=3, max_leaf_nodes=7,
                min_samples_leaf=min_leaf, l2_regularization=2., learning_rate=.05,
                early_stopping=False, random_state=seed+10+i).fit(x, loss(y, scores[:,i])) for i in range(len(ACTIONS))]
    return {"detectors": detectors, "models": models, "minimum_leaf_rows": min_leaf,
            "maximum_rows_per_person": copies, "minimum_leaf_participants_lower_bound": 30}


def raw_utilities(frame, state):
    x, _ = matrix(frame, state["detectors"])
    return np.column_stack([m.predict(x) for m in state["models"]])


def fit_utility_calibrators(predictions, frame):
    mask = frame.fit_allowed.to_numpy(bool)
    y = frame.label.to_numpy(int)[mask]
    scores = actions(frame)[mask]
    return [Ridge(alpha=1.).fit(predictions[mask, i:i+1], loss(y, scores[:,i])) for i in range(len(ACTIONS))]


def predict(frame, state, calibrators, margin=None):
    """Pure policy inference: never read label, identifier, scenario or fold."""
    utilities = raw_utilities(frame, state)
    utilities = np.column_stack([np.maximum(c.predict(utilities[:, i:i+1]), 0) for i,c in enumerate(calibrators)])
    best = utilities.argmin(axis=1)
    improvement = utilities[:,0] - utilities[np.arange(len(frame)), best]
    chosen = np.zeros(len(frame), dtype=int) if margin is None else np.where(improvement>margin, best, 0)
    score_matrix = actions(frame)
    score = score_matrix[np.arange(len(frame)), chosen]
    data = sanitized(frame)
    p = data[list(EXPERTS)].to_numpy(float)
    available = data[[f"{n}_available" for n in EXPERTS]].to_numpy(float)
    spread = np.where(available>0, p, -np.inf).max(axis=1) - np.where(available>0, p, np.inf).min(axis=1)
    _, q = matrix(data, state["detectors"])
    risk = np.clip(.35*q[:,1]+.25*q[:,2]+.25*spread+.15*(1-2*np.abs(score-.5)), 0, 1)
    return {"score": score, "action": chosen, "improvement": improvement, "risk": risk, "action_scores": score_matrix}


def attach(frame, prediction):
    out = frame[["id", "label", "scenario", "scenario_kind", "fit_allowed"]].copy()
    for name in ("score", "action", "improvement", "risk"):
        out[name] = prediction[name]
    for i, a in enumerate(ACTIONS):
        out[a] = prediction["action_scores"][:, i]
    out["gain"] = loss(out.label.to_numpy(int), out.available_mean.to_numpy(float))-loss(out.label.to_numpy(int), out.score.to_numpy(float))
    return out


def lower_bound(frame, seed=101):
    values = frame.groupby("id").gain.mean().to_numpy(float)
    samples = rng(seed, "bootstrap").choice(values, size=(1000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples,.05))


def choose_policy(selections):
    candidates, tables = [], []
    # Quantile policy is selected, final absolute margin estimated below from inner predictions.
    for q in QUANTILES:
        parts = []
        thresholds = []
        for frame, state, calibrators in selections:
            unrestricted = predict(frame, state, calibrators, 0.)
            margin = float(np.quantile(unrestricted["improvement"][frame.scenario.to_numpy()=="clean"], q))
            thresholds.append(margin)
            parts.append(attach(frame, predict(frame, state, calibrators, margin)))
        table = pd.concat(parts, ignore_index=True)
        clean = table.loc[table.scenario=="clean"]
        stress = table.loc[(table.scenario!="clean") & table.fit_allowed]
        delta = cal.safe_auc(clean.label, clean.score)-cal.safe_auc(clean.label, clean.available_mean)
        lo = lower_bound(stress)
        rate = float((clean.action!=0).mean())
        candidates.append({"quantile": q, "clean_auc_delta": delta, "clean_action_rate": rate,
            "gain_lower_95_exploratory": lo, "stress_gain": float(stress.gain.mean()),
            "feasible": bool(delta>=-.005 and rate<=.2 and lo>0), "absolute_margin": float(np.median(thresholds))})
        tables.append(table)
    tuning = pd.DataFrame(candidates)
    feasible = tuning.loc[tuning.feasible]
    if feasible.empty:
        policy = {"enabled": False, "margin": None, "reason": "no_inner_candidate_satisfies_all_constraints"}
    else:
        row = feasible.sort_values(["stress_gain","quantile"],ascending=[False,False]).iloc[0]
        policy = {"enabled": True, "margin": float(row.absolute_margin), "quantile": float(row["quantile"]), "reason": "inner_selection"}
    # Recompute risk using the final absolute margin, including the fail-closed case.
    recomputed = pd.concat([attach(f,predict(f,s,c,policy["margin"])) for f,s,c in selections],ignore_index=True)
    clean = recomputed.loc[recomputed.scenario=="clean"]
    stress = recomputed.loc[(recomputed.scenario!="clean") & recomputed.fit_allowed]
    # A median absolute margin can violate constraints despite fold-specific quantiles.
    feasible_final = (cal.safe_auc(clean.label,clean.score)-cal.safe_auc(clean.label,clean.available_mean)>=-.005
                      and (clean.action!=0).mean()<=.2 and lower_bound(stress)>0)
    if policy["enabled"] and not feasible_final:
        policy = {"enabled":False,"margin":None,"reason":"absolute_margin_guard"}
        recomputed = pd.concat([attach(f,predict(f,s,c,None)) for f,s,c in selections],ignore_index=True)
        clean = recomputed.loc[recomputed.scenario=="clean"]
    policy["risk_threshold"] = float(np.quantile(clean.risk, .90, method="higher"))
    return policy, tuning, recomputed


def regression_tests(frame, state, calibrators, margin):
    no_labels = frame.drop(columns=["label"], errors="ignore")
    a = predict(no_labels, state, calibrators, margin)
    altered = frame.copy(); altered["label"] = 1-altered.label
    b = predict(altered, state, calibrators, margin)
    for c in ("score", "action", "risk"):
        np.testing.assert_allclose(a[c],b[c],rtol=0,atol=0)
    absent = no_labels.copy(); absent["smile_available"]=0
    x = predict(absent,state,calibrators,margin)
    absent["smile"]=.999
    absent["smile_mc_std"]=999
    for d in DESCRIPTORS:
        absent[f"smile_feature_{d}"]=999
    z = predict(absent,state,calibrators,margin)
    for c in ("score","action","risk"):
        np.testing.assert_allclose(x[c],z[c],rtol=0,atol=0)
    fallback = predict(no_labels,state,calibrators,None)
    np.testing.assert_allclose(fallback["score"],actions(no_labels)[:,0])
    assert not fallback["action"].any()
    # Changing an out-of-partition feature cannot affect partition-local shuffle/noise.
    part = pd.DataFrame({"features_2":[np.array([1.,2.]),np.array([3.,4.])]})
    spec = next(s for s in scenario_registry() if s["kind"]=="feature_permutation")
    combined = pd.concat([part, pd.DataFrame({"features_2":[np.array([90.,99.])]})],ignore_index=True)
    baseline = perturb(combined.iloc[:2],spec,101)
    combined.at[2,"features_2"] = np.array([-999.,999.])
    np.testing.assert_array_equal(np.stack(baseline.features_2),np.stack(perturb(combined.iloc[:2],spec,101).features_2))
    return {"label_free_inference":True,"missing_modality_invariance":True,"fallback_scores":True,
            "partition_external_feature_invariance":True,"risk_recomputed_after_selection":True}


def evaluate(table, threshold, fold):
    metrics=[]
    for scenario, f in table.groupby("scenario"):
        y=f.label.to_numpy(int); accepted=f.risk.to_numpy(float)<=threshold
        for model in ("score","available_mean","ufnet"):
            p=f[model].to_numpy(float)
            metrics.append({"outer_fold":fold,"scenario":scenario,"scenario_kind":f.scenario_kind.iloc[0],
                "model":"fusion_agent" if model=="score" else model,"n":len(f),
                "auroc":cal.safe_auc(y,p),"log_loss":float(loss(y,p).mean()),"ece":float(ev.calibration_error(y,p)),
                "coverage":float(accepted.mean()),"accepted_n":int(accepted.sum()),
                "selective_auroc":float(roc_auc_score(y[accepted],p[accepted])) if len(np.unique(y[accepted]))==2 else None,
                "action_rate":float((f.action!=0).mean())})
    return pd.DataFrame(metrics)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--engineering-only",action="store_true")
    args=parser.parse_args()
    start=time.perf_counter(); output=args.output_dir.resolve()
    if output.exists(): raise FileExistsError("Use a new result directory")
    output.mkdir(parents=True)
    caches,fold,inputs=generate_inputs(args.base_dir.resolve(),output)
    selections=[]; calibration_frames=[]; calibration_predictions=[]; audits=[]
    for k in range(4):
        train=caches[f"meta_{k}_train"]; held=caches[f"meta_{k}_validation"]
        utility=held.loc[held.block=="utility_calibration"].copy()
        select=held.loc[held.block=="strategy_selection"].copy()
        if set(train.id)&set(held.id) or set(utility.id)&set(select.id):
            raise ValueError("Meta fitting/calibration/selection overlap")
        state=fit_state(train,101+100*k)
        predictions=raw_utilities(utility,state)
        calibrators=fit_utility_calibrators(predictions,utility)
        selections.append((select,state,calibrators))
        calibration_frames.append(utility); calibration_predictions.append(predictions)
        audits.append({"meta_fold":k,"train":train.id.nunique(),"utility_calibration":utility.id.nunique(),
                       "strategy_selection":select.id.nunique(),"disjoint":True})
        print(f"Meta state {k} fitted and utility calibrated",flush=True)
    policy,tuning,selected=choose_policy(selections)
    tuning.to_csv(output/"inner_tuning.csv",index=False)
    selected.to_csv(output/"private_inner_decisions.csv",index=False)
    final_state=fit_state(caches["final_train"],5101)
    final_cal=fit_utility_calibrators(np.concatenate(calibration_predictions),pd.concat(calibration_frames,ignore_index=True))
    prediction_frame=caches["outer_validation"]
    tests=regression_tests(prediction_frame,final_state,final_cal,policy["margin"])
    result=predict(prediction_frame.drop(columns=["label"]),final_state,final_cal,policy["margin"])
    decisions=attach(prediction_frame,result)
    decisions.to_csv(output/"private_outer_decisions.csv",index=False)
    with (output/"agent.pkl").open("wb") as h: pickle.dump({"state":final_state,"calibrators":final_cal,"policy":policy},h)
    with (output/"agent.pkl").open("rb") as h: restored = pickle.load(h)
    reloaded = predict(prediction_frame.drop(columns=["label"]),restored["state"],restored["calibrators"],restored["policy"]["margin"])
    for column in ("score","action","risk"):
        np.testing.assert_allclose(result[column],reloaded[column],rtol=0,atol=0)
    tests["saved_agent_roundtrip"] = True
    if not args.engineering_only:
        evaluate(decisions,policy["risk_threshold"],fold).to_csv(output/"outer_metrics.csv",index=False)
    summary={"status":"ENGINEERING_COMPLETE" if args.engineering_only else "DEVELOPMENT_FOLD_COMPLETE",
             "outer_fold":int(fold),"wall_seconds":time.perf_counter()-start,"policy":policy,"regression_tests":tests,
             "meta_membership_audit":audits,"scenario_count":len(scenario_registry()),
             "locked_predictions_generated":False,"base_artifacts_unchanged":True,
             "outer_performance_reported":not args.engineering_only,"minimum_leaf_participants_lower_bound":30,
             "calibration_method":"meta-holdout split calibration; final cross-fitted calibration followed by refit",
             "limitations":["development data previously exposed","fixed upstream architectures and recovered features",
                            "quality detectors fitted on synthetic training inputs; not a clinical quality benchmark",
                            "bootstrap used for exploratory selection, not multiplicity-adjusted confirmation"],
             "source_sha256":{p.name:oof.digest(p) for p in Path(__file__).parent.glob("*.py")},
             "private_artifact_sha256":{p.name:oof.digest(p) for p in output.glob("private_*.csv")},
             "aggregate_sha256":{n:oof.digest(output/n) for n in ("inner_tuning.csv","outer_metrics.csv") if (output/n).exists()}}
    oof.write_json(output/"agent_summary.json",summary)
    print(f"Agent fold {fold} complete; policy enabled={policy['enabled']}; elapsed={summary['wall_seconds']:.1f}s",flush=True)


if __name__=="__main__": main()
