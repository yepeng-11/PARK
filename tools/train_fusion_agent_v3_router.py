#!/usr/bin/env python3
"""Train the Fusion Agent v3 cross-fitted expected-regret router.

All fitting and evaluation is confined to the frozen v3 Train+Dev folds. Since
v2 outcomes on these participants are already exposed, the resulting metrics
are development/rejection evidence only and can never authorize promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import log_loss, roc_auc_score

import evaluate_calibrated_fusion as calibrated
import evaluate_pretrained as ev
import freeze_fusion_agent_v2_protocol as freeze_v2
import train_fusion_agent_v2_router as v2_router
import train_fusion_agent_v2_specialists as specialists
import train_paired_baselines as paired


EXPECTED_PROTOCOL_HASH = "263dd02d41ad5157092d9770e919bd0674cc7f5012eecb2dfbfe944a69031d73"
IMPLEMENTATION_REVISION = "v3.1-action-budget-quantiles"
EXPERTS = tuple(calibrated.EXPERT_COLUMNS)
ACTIONS = ("available_weighted", "drop_smile", "ufnet", "shrink_smile")
MODELS = ("fusion_agent_v3", "available_weighted", "ufnet")
FIXED_MARGINS = (0.0, 0.0025, 0.005, 0.01, 0.02)
ACTION_BUDGET_QUANTILES = (0.80, 0.85, 0.90, 0.95, 0.975)
FEATURES = (
    "finger", "speech", "smile", "ufnet", "available_weighted",
    "finger_speech_gap", "finger_smile_gap", "speech_smile_gap",
    "finger_entropy", "speech_entropy", "smile_entropy", "ufnet_entropy",
    "finger_available", "speech_available", "smile_available",
    "speech_corruption_probability", "smile_conflict_probability",
    "finger_mc_std", "speech_mc_std", "smile_mc_std", "ufnet_mc_std",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--protocol-dir", type=Path, default=None)
    parser.add_argument("--training-dir", type=Path, default=None)
    parser.add_argument("--specialist-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--paired-seeds", default="101,202,303,404,505")
    parser.add_argument("--outer-folds", default="", help="Optional fold subset for smoke tests")
    parser.add_argument("--mc-trials", type=int, default=30)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--target-clean-coverage", type=float, default=0.90)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def stable_rng(*parts: Any) -> np.random.Generator:
    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def load_protocol(protocol_dir: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    protocol = json.loads((protocol_dir / "protocol.json").read_text(encoding="utf-8"))
    recorded = protocol.pop("protocol_sha256")
    observed = freeze_v2.canonical_hash(protocol)
    protocol["protocol_sha256"] = recorded
    if recorded != EXPECTED_PROTOCOL_HASH or observed != recorded:
        raise ValueError(f"Frozen v3 protocol mismatch: {recorded=} {observed=}")
    manifest_path = protocol_dir / "participant_fold_manifest.csv"
    if ev.sha256_file(manifest_path) != protocol["artifact_sha256"][manifest_path.name]:
        raise ValueError("Participant manifest hash mismatch")
    return protocol, pd.read_csv(manifest_path, dtype={"participant_id": str})


def logit_transform(score: np.ndarray, temperature: float = 1.0, shift: float = 0.0) -> np.ndarray:
    clipped = np.clip(score, 1e-5, 1.0 - 1e-5)
    logits = np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-(temperature * logits + shift)))


def predict_scenarios(module, scaled, predictors, fusion_model, device, mc_trials, seed, fold):
    output = v2_router.predict_scenarios(
        module, scaled, predictors, fusion_model, device, mc_trials, seed, fold
    )
    clean = output["clean"]["prediction"]
    for temperature in (0.5, 2.0, 4.0):
        frame = clean.copy()
        frame["smile"] = logit_transform(frame.smile.to_numpy(float), temperature=temperature)
        output[f"smile_logit_temperature_{temperature:.1f}"] = {
            "prediction": frame, "kind": "smile_reliability", "missing": [], "trainable": True
        }
    for shift in (-1.0, -0.5, 0.5, 1.0):
        frame = clean.copy()
        frame["smile"] = logit_transform(frame.smile.to_numpy(float), shift=shift)
        output[f"smile_logit_shift_{shift:+.1f}"] = {
            "prediction": frame, "kind": "smile_reliability", "missing": [], "trainable": True
        }
    rng = stable_rng("v3-smile-permutation", fold)
    frame = clean.copy()
    permuted = frame.smile.to_numpy(float).copy()
    rng.shuffle(permuted)
    frame["smile"] = permuted
    output["smile_rank_permutation"] = {
        "prediction": frame, "kind": "smile_reliability", "missing": [], "trainable": True
    }
    frame = clean.copy()
    peer = frame[["finger", "speech"]].mean(axis=1).to_numpy(float)
    frame["smile"] = np.where(peer >= 0.5, 0.02, 0.98)
    output["smile_opposite_consensus"] = {
        "prediction": frame, "kind": "smile_stress_only", "missing": [], "trainable": False
    }
    output.pop("smile_conflict", None)
    for item in output.values():
        item.setdefault("trainable", True)
    return output


def entropy(score: np.ndarray) -> np.ndarray:
    value = np.clip(score, 1e-6, 1.0 - 1e-6)
    return -(value * np.log(value) + (1.0 - value) * np.log(1.0 - value))


def action_and_feature_frame(
    prediction: pd.DataFrame,
    base_weights: np.ndarray,
    speech_probability: np.ndarray,
    smile_probability: np.ndarray,
    missing: Sequence[int],
) -> pd.DataFrame:
    probabilities = np.clip(prediction[list(EXPERTS)].to_numpy(float), 1e-6, 1.0 - 1e-6)
    available = np.ones_like(probabilities)
    if missing:
        available[:, list(missing)] = 0.0
    weights = available * np.asarray(base_weights).reshape(1, -1)
    weights /= weights.sum(axis=1, keepdims=True)
    baseline = (weights * probabilities).sum(axis=1)
    drop_weights = weights.copy()
    drop_weights[:, 2] = 0.0
    denominator = drop_weights.sum(axis=1)
    can_drop = denominator > 0
    drop_weights[can_drop] /= denominator[can_drop, None]
    drop_weights[~can_drop] = weights[~can_drop]
    drop_score = (drop_weights * probabilities).sum(axis=1)
    peer_weight = weights[:, :2]
    peer_denominator = peer_weight.sum(axis=1)
    peer_score = np.divide(
        (peer_weight * probabilities[:, :2]).sum(axis=1),
        peer_denominator,
        out=baseline.copy(),
        where=peer_denominator > 0,
    )
    shrunk_probabilities = probabilities.copy()
    shrunk_probabilities[:, 2] = 0.5 * probabilities[:, 2] + 0.5 * peer_score
    shrink_score = (weights * shrunk_probabilities).sum(axis=1)
    output = prediction[["id", "label"]].copy()
    for index, name in enumerate(EXPERTS):
        output[name] = probabilities[:, index]
        output[f"{name}_available"] = available[:, index]
        std_name = f"{name}_mc_std"
        output[std_name] = prediction[std_name].to_numpy(float) if std_name in prediction else 0.0
        output[f"{name}_entropy"] = entropy(probabilities[:, index])
    ufnet = np.clip(prediction.ufnet.to_numpy(float), 1e-6, 1.0 - 1e-6)
    output["ufnet"] = ufnet
    output["ufnet_entropy"] = entropy(ufnet)
    output["ufnet_mc_std"] = prediction.ufnet_mc_std.to_numpy(float) if "ufnet_mc_std" in prediction else 0.0
    output["available_weighted"] = baseline
    output["finger_speech_gap"] = np.abs(probabilities[:, 0] - probabilities[:, 1])
    output["finger_smile_gap"] = np.abs(probabilities[:, 0] - probabilities[:, 2])
    output["speech_smile_gap"] = np.abs(probabilities[:, 1] - probabilities[:, 2])
    output["speech_corruption_probability"] = speech_probability * available[:, 1]
    output["smile_conflict_probability"] = smile_probability * available[:, 2]
    output["action_available_weighted"] = baseline
    output["action_drop_smile"] = drop_score
    output["action_ufnet"] = ufnet
    output["action_shrink_smile"] = shrink_score
    aggregation = {column: "mean" for column in output.columns if column not in {"id", "label"}}
    aggregation["label"] = "first"
    participant = output.groupby("id", as_index=False).agg(aggregation)
    label_counts = output.groupby("id").label.nunique()
    if (label_counts > 1).any():
        raise ValueError("Inconsistent participant label")
    return participant


def binary_losses(label: np.ndarray, score: np.ndarray) -> np.ndarray:
    score = np.clip(score, 1e-6, 1.0 - 1e-6)
    return -(label * np.log(score) + (1 - label) * np.log(1 - score))


def fit_loss_models(frame: pd.DataFrame, seed: int) -> Dict[str, HistGradientBoostingRegressor]:
    selected = frame.loc[frame.trainable].reset_index(drop=True)
    matrix = selected[list(FEATURES)].to_numpy(float)
    labels = selected.label.to_numpy(int)
    models = {}
    for index, action in enumerate(ACTIONS):
        target = binary_losses(labels, selected[f"action_{action}"].to_numpy(float))
        models[action] = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=150,
            max_depth=3,
            max_leaf_nodes=7,
            min_samples_leaf=30,
            l2_regularization=2.0,
            random_state=seed + index,
        ).fit(matrix, target)
    return models


def apply_loss_router(frame: pd.DataFrame, models, margin: float, enabled: bool) -> pd.DataFrame:
    output = frame.copy()
    matrix = output[list(FEATURES)].to_numpy(float)
    predicted = np.column_stack([models[name].predict(matrix) for name in ACTIONS])
    best_index = predicted.argmin(axis=1)
    baseline_index = ACTIONS.index("available_weighted")
    improvement = predicted[:, baseline_index] - predicted[np.arange(len(output)), best_index]
    choose = enabled & (best_index != baseline_index) & (improvement > margin)
    chosen_index = np.where(choose, best_index, baseline_index)
    chosen_action = np.asarray(ACTIONS, dtype=object)[chosen_index]
    scores = np.asarray(
        [output.iloc[i][f"action_{action}"] for i, action in enumerate(chosen_action)], dtype=float
    )
    labels = output.label.to_numpy(int)
    baseline_loss = binary_losses(labels, output.action_available_weighted.to_numpy(float))
    chosen_loss = binary_losses(labels, scores)
    output["fusion_agent_v3"] = scores
    output["chosen_action"] = chosen_action
    output["non_default_action"] = choose
    output["predicted_regret_improvement"] = np.where(choose, improvement, 0.0)
    output["actual_regret_improvement"] = baseline_loss - chosen_loss
    spread = output[list(EXPERTS)].max(axis=1) - output[list(EXPERTS)].min(axis=1)
    confidence_risk = 1.0 - 2.0 * np.abs(scores - 0.5)
    output["risk"] = np.clip(
        0.35 * output.speech_corruption_probability
        + 0.25 * output.smile_conflict_probability
        + 0.25 * spread
        + 0.15 * confidence_risk,
        0.0,
        1.0,
    )
    return output


def bootstrap_lower_bound(frame: pd.DataFrame, column: str, replicates: int, *seed_parts) -> float:
    by_id = frame.groupby("id", as_index=False)[column].mean()
    values = by_id[column].to_numpy(float)
    if len(values) == 0:
        return float("-inf")
    rng = stable_rng("bootstrap", *seed_parts)
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        estimates[index] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(estimates, 0.05))


def make_records(scenarios, participant_ids, base_weights, speech_detector, smile_detector):
    records = []
    for scenario, item in scenarios.items():
        prediction = item["prediction"].loc[
            item["prediction"].id.astype(str).isin(participant_ids)
        ].reset_index(drop=True)
        speech_p, smile_p = v2_router.detector_probabilities(prediction, speech_detector, smile_detector)
        participant = action_and_feature_frame(
            prediction, base_weights, speech_p, smile_p, item["missing"]
        )
        participant["scenario"] = scenario
        participant["scenario_type"] = item["kind"]
        participant["trainable"] = bool(item["trainable"])
        records.append(participant)
    return pd.concat(records, ignore_index=True)


def build_outer_oof(scenarios, outer_rows, params, seed, fold):
    records = []
    for inner_fold in sorted(outer_rows.inner_validation_fold.unique()):
        speech_detector, smile_detector, train_ids = v2_router.fit_inner_detectors(
            scenarios, outer_rows, int(inner_fold), params, seed, fold
        )
        validation_ids = set(
            outer_rows.loc[outer_rows.inner_validation_fold == inner_fold, "participant_id"].astype(str)
        )
        base_weights = v2_router.fit_base_weights(scenarios["clean"]["prediction"], train_ids)
        frame = make_records(
            scenarios, validation_ids, base_weights, speech_detector, smile_detector
        )
        frame["inner_fold"] = int(inner_fold)
        records.append(frame)
    return pd.concat(records, ignore_index=True)


def tune_margin(oof: pd.DataFrame, seed: int, fold: int, bootstrap_replicates: int):
    predictions = []
    for held_fold in sorted(oof.inner_fold.unique()):
        train = oof.loc[oof.inner_fold != held_fold]
        validation = oof.loc[oof.inner_fold == held_fold]
        models = fit_loss_models(train, seed * 100 + int(held_fold))
        base = apply_loss_router(validation, models, 0.0, True)
        base["held_inner_fold"] = int(held_fold)
        predictions.append(base)
    cross_fitted = pd.concat(predictions, ignore_index=True)
    # A fixed grid can fail to represent the frozen 20% clean action budget when
    # expected-loss models are strongly separated. Add label-free thresholds
    # derived only from the inner-OOF clean predicted-improvement distribution.
    clean_improvement = cross_fitted.loc[
        (cross_fitted.scenario == "clean")
        & (cross_fitted.chosen_action != "available_weighted"),
        "predicted_regret_improvement",
    ].to_numpy(float)
    adaptive_margins = (
        [float(np.quantile(clean_improvement, q)) for q in ACTION_BUDGET_QUANTILES]
        if len(clean_improvement)
        else []
    )
    candidate_margins = sorted(set([*FIXED_MARGINS, *adaptive_margins]))
    rows = []
    routed_by_margin = {}
    for margin in candidate_margins:
        frame = cross_fitted.copy()
        choose = (
            (frame.chosen_action != "available_weighted")
            & (frame.predicted_regret_improvement > margin)
        )
        frame.loc[~choose, "chosen_action"] = "available_weighted"
        frame["non_default_action"] = choose
        frame["fusion_agent_v3"] = [
            row[f"action_{row.chosen_action}"] for _, row in frame.iterrows()
        ]
        labels = frame.label.to_numpy(int)
        frame["actual_regret_improvement"] = binary_losses(
            labels, frame.action_available_weighted.to_numpy(float)
        ) - binary_losses(labels, frame.fusion_agent_v3.to_numpy(float))
        clean = frame.loc[frame.scenario == "clean"]
        clean_delta = calibrated.safe_auc(clean.label, clean.fusion_agent_v3) - calibrated.safe_auc(
            clean.label, clean.available_weighted
        )
        stress = frame.loc[frame.scenario != "clean"]
        lower = bootstrap_lower_bound(
            stress, "actual_regret_improvement", bootstrap_replicates, fold, margin
        )
        rows.append(
            {
                "margin": margin,
                "mean_stress_regret_improvement": float(stress.actual_regret_improvement.mean()),
                "bootstrap_lower_95": lower,
                "clean_auc_delta": float(clean_delta),
                "clean_action_rate": float(clean.non_default_action.mean()),
                "feasible": bool(lower > 0 and clean_delta >= -0.005 and clean.non_default_action.mean() <= 0.20),
            }
        )
        routed_by_margin[margin] = frame
    tuning = pd.DataFrame(rows)
    feasible = tuning.loc[tuning.feasible]
    if feasible.empty:
        return {"enabled": False, "margin": float("inf"), "reason": "fail_closed_no_positive_regret_lower_bound"}, tuning, cross_fitted
    selected = feasible.sort_values(
        ["mean_stress_regret_improvement", "bootstrap_lower_95"], ascending=False
    ).iloc[0]
    tuning["selected"] = tuning.margin == selected.margin
    return {
        "enabled": True,
        "margin": float(selected.margin),
        "reason": "inner_oof_positive_regret_lower_bound",
        "bootstrap_lower_95": float(selected.bootstrap_lower_95),
    }, tuning, routed_by_margin[float(selected.margin)]


def metric_record(labels, scores, accepted):
    result = calibrated.compute_metrics(labels, scores, 0.5)
    result["log_loss"] = float(log_loss(labels, np.clip(scores, 1e-6, 1 - 1e-6), labels=[0, 1]))
    result["coverage"] = float(accepted.mean())
    return result


def evaluate_outer(fold, records, models, config, risk_threshold):
    routed = apply_loss_router(records, models, config["margin"], config["enabled"])
    metric_rows, route_rows = [], []
    for scenario, frame in routed.groupby("scenario"):
        labels = frame.label.to_numpy(int)
        accepted = frame.risk.to_numpy(float) <= risk_threshold
        for model in MODELS:
            metric_rows.append({
                "outer_fold": fold,
                "scenario": scenario,
                "scenario_type": frame.scenario_type.iloc[0],
                "model": model,
                **metric_record(labels, frame[model].to_numpy(float), accepted),
            })
        route_rows.append({
            "outer_fold": fold,
            "scenario": scenario,
            "scenario_type": frame.scenario_type.iloc[0],
            "participants": len(frame),
            "coverage": float(accepted.mean()),
            "non_default_action_rate": float(frame.non_default_action.mean()),
            "mean_actual_regret_improvement": float(frame.actual_regret_improvement.mean()),
            **{f"action_rate_{name}": float((frame.chosen_action == name).mean()) for name in ACTIONS},
        })
    return pd.DataFrame(metric_rows), pd.DataFrame(route_rows), routed


def acceptance_decision(metrics, routes, decisions, selected_configs, bootstrap_replicates):
    pivot = metrics.pivot_table(
        index=["outer_fold", "scenario", "scenario_type"], columns="model", values=["auroc", "ece", "log_loss"]
    )
    clean = pivot.xs("clean", level="scenario")
    clean_auc_delta = clean[("auroc", "fusion_agent_v3")] - clean[("auroc", "available_weighted")]
    clean_ece_delta = clean[("ece", "fusion_agent_v3")] - clean[("ece", "available_weighted")]
    smile = pivot.loc[pivot.index.get_level_values("scenario_type") == "smile_reliability"]
    smile_fold = smile.groupby(level="outer_fold").mean()
    smile_best = np.maximum(
        smile_fold[("auroc", "available_weighted")], smile_fold[("auroc", "ufnet")]
    )
    smile_gain = smile_fold[("auroc", "fusion_agent_v3")] - smile_best
    stress = pivot.loc[pivot.index.get_level_values("scenario") != "clean"]
    stress_fold = stress.groupby(level="outer_fold").mean()
    stress_regret = stress_fold[("log_loss", "fusion_agent_v3")] - stress_fold[("log_loss", "available_weighted")]
    clean_routes = routes.loc[routes.scenario == "clean"]
    stress_decisions = decisions.loc[decisions.scenario != "clean"]
    lower = bootstrap_lower_bound(
        stress_decisions, "actual_regret_improvement", bootstrap_replicates, "outer-all"
    )
    checks = [
        ("clean_noninferiority", float(clean_auc_delta.mean()), ">=", -0.005, clean_auc_delta.mean() >= -0.005),
        ("smile_reliability_gain", float(smile_gain.mean()), ">", 0.0, smile_gain.mean() > 0),
        ("stress_regret_guard", float(stress_regret.mean()), "<=", 0.0, stress_regret.mean() <= 0),
        ("clean_coverage", float(clean_routes.coverage.mean()), ">=", 0.80, clean_routes.coverage.mean() >= 0.80),
        ("clean_action_rate", float(clean_routes.non_default_action_rate.mean()), "<=", 0.20, clean_routes.non_default_action_rate.mean() <= 0.20),
        ("calibration_guard", float(clean_ece_delta.mean()), "<=", 0.02, clean_ece_delta.mean() <= 0.02),
        ("fold_stability", int((clean_auc_delta >= -0.005).sum()), ">=", 4, int((clean_auc_delta >= -0.005).sum()) >= 4),
        ("bootstrap_route_evidence", lower, ">", 0.0, lower > 0),
    ]
    table = pd.DataFrame(checks, columns=["criterion", "observed", "operator", "required", "passed"])
    all_internal = bool(table.passed.all())
    decision = {
        "internal_development_criteria_passed": all_internal,
        "passed_criteria": int(table.passed.sum()),
        "total_criteria": len(table),
        "promotion_authorized": False,
        "promotion_blocker": "genuinely unseen eligible external cohort required even if all internal criteria pass",
        "protocol_sha256": EXPECTED_PROTOCOL_HASH,
        "prior_outer_results_exposed": True,
        "locked_test_predictions_generated": False,
        "enabled_outer_folds": int(sum(bool(item["enabled"]) for item in selected_configs)),
    }
    return decision, table


def write_report(path, metrics, routes, decision, checks):
    lines = [
        "# Fusion Agent v3 expected-regret router",
        "",
        f"Frozen protocol: `{EXPECTED_PROTOCOL_HASH}`.",
        "",
        f"Internal development criteria: **{decision['passed_criteria']}/{decision['total_criteria']}**.",
        f"Outer folds enabling learned routing: **{decision['enabled_outer_folds']}/5**.",
        "Promotion authorized: **False** (an eligible genuinely unseen cohort is mandatory).",
        "",
        "## Internal rejection checks",
        "",
        "| Criterion | Observed | Rule | Required | Pass |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for row in checks.itertuples(index=False):
        lines.append(f"| {row.criterion} | {row.observed:.4f} | {row.operator} | {row.required:.4f} | {row.passed} |")
    lines.extend(["", "## Mean development AUROC", "", "| Scenario | V3 | Available weighted | UFNet | Coverage |", "| --- | ---: | ---: | ---: | ---: |"])
    for scenario in sorted(metrics.scenario.unique()):
        frame = metrics.loc[metrics.scenario == scenario]
        auc = frame.groupby("model").auroc.mean()
        coverage = routes.loc[routes.scenario == scenario, "coverage"].mean()
        lines.append(f"| {scenario} | {auc['fusion_agent_v3']:.4f} | {auc['available_weighted']:.4f} | {auc['ufnet']:.4f} | {coverage:.4f} |")
    lines.extend([
        "",
        "These reused Train+Dev folds are development/rejection evidence only. V2 outcomes "
        "were previously exposed, and the released paired disease models were not retrained "
        "inside these folds. Absolute AUROCs are optimistic and cannot support a final claim.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.mc_trials < 2 or args.bootstrap_replicates < 100:
        raise ValueError("mc-trials must be >=2 and bootstrap-replicates >=100")
    repo_root = args.repo_root.resolve()
    data_path = (args.data or repo_root / "results/protocol_alignment_audit/cleaned_aligned.csv").resolve()
    protocol_dir = (args.protocol_dir or repo_root / "results/fusion_agent_v3_protocol").resolve()
    training_dir = (args.training_dir or repo_root / "results/paired_retraining").resolve()
    specialist_dir = (args.specialist_dir or repo_root / "results/fusion_agent_v2_specialists").resolve()
    output = (args.output_dir or repo_root / "results/fusion_agent_v3_router").resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol, manifest = load_protocol(protocol_dir)
    if ev.sha256_file(data_path) != protocol["source_dataset_sha256"]:
        raise ValueError("Dataset hash differs from frozen v3 protocol")
    all_folds = sorted(manifest.outer_fold.unique())
    requested = paired.parse_seeds(args.outer_folds) if args.outer_folds else all_folds
    if not set(requested) <= set(all_folds):
        raise ValueError("Unknown outer fold")
    seed_list = paired.parse_seeds(args.paired_seeds)
    if len(seed_list) != len(all_folds):
        raise ValueError("One paired seed is required per outer fold")
    seed_by_fold = dict(zip(all_folds, seed_list))
    frame = paired.load_vector_csv(data_path)
    allowed_ids = set(manifest.participant_id.astype(str))
    module = ev.load_upstream_module(repo_root)
    masks = paired.split_masks(module, frame)
    locked = masks["internal_test"] | masks["validation_1"] | masks["validation_2"] | masks["global"]
    if allowed_ids & set(frame.loc[locked, "id"].astype(str)):
        raise RuntimeError("Development/test participant leakage")
    development = frame.loc[frame.id.astype(str).isin(allowed_ids)].reset_index(drop=True)
    fusion_config = ev.read_json(Path(module.MODEL_CONFIG_PATH))
    selected_models = module.MODEL_SUBSETS[int(fusion_config["model_subset_choice"])]
    module.NUM_MODELS = len(selected_models)
    upstream_paths = ev.checkpoint_paths(module, selected_models)
    configs = [ev.read_json(item["config"]) for item in upstream_paths]
    raw = paired.inverse_original_scaling(development, configs, upstream_paths)
    device = ev.resolve_device(args.device)
    protected_paths = [path for item in upstream_paths for path in item.values() if path.suffix in {".pth", ".pkl"}]
    for fold in requested:
        seed = seed_by_fold[fold]
        run = training_dir / "cleaned" / f"seed_{seed}"
        protected_paths.extend(run.glob("*/model.pth"))
        protected_paths.extend(run.glob("scaler_*.pkl"))
        protected_paths.extend((specialist_dir / f"outer_{fold}").glob("*.pkl"))
    protected_before = {str(p.resolve().relative_to(repo_root)): ev.sha256_file(p.resolve()) for p in sorted(set(protected_paths))}
    all_metrics, all_routes, all_tuning, all_decisions, selected_configs = [], [], [], [], []
    for fold in requested:
        seed = seed_by_fold[fold]
        fold_output = output / f"outer_{fold}"
        fold_output.mkdir(parents=True, exist_ok=True)
        params = v2_router.load_specialist_parameters(specialist_dir, int(fold))
        _, scaled, predictors, fusion_model = specialists.load_outer_models(
            module, raw, training_dir, seed, configs, fusion_config, device
        )
        print(f"V3 outer fold {fold}; paired seed {seed}")
        scenarios = predict_scenarios(module, scaled, predictors, fusion_model, device, args.mc_trials, seed, int(fold))
        outer_rows = manifest.loc[(manifest.outer_fold == fold) & (manifest.outer_role == "train")].copy()
        oof = build_outer_oof(scenarios, outer_rows, params, seed, int(fold))
        router_config, tuning, routed_oof = tune_margin(
            oof, seed, int(fold), args.bootstrap_replicates
        )
        tuning.insert(0, "outer_fold", fold)
        all_tuning.append(tuning)
        selected_configs.append(router_config)
        final_models = fit_loss_models(oof, seed * 1000)
        outer_train_ids = set(outer_rows.participant_id.astype(str))
        outer_validation_ids = set(manifest.loc[(manifest.outer_fold == fold) & (manifest.outer_role == "validation"), "participant_id"].astype(str))
        speech_predictions = {name: item["prediction"] for name, item in scenarios.items() if item["kind"] in {"clean", "speech_noise"}}
        x_speech, y_speech, _ = specialists.speech_dataset(speech_predictions, outer_train_ids)
        x_smile, y_smile, _ = specialists.smile_dataset(scenarios["clean"]["prediction"], outer_train_ids, (fold, "v3-final"))
        speech_detector = specialists.fit_detector(x_speech, y_speech, params["speech_noise_detector"], seed * 10000)
        smile_detector = specialists.fit_detector(x_smile, y_smile, params["smile_conflict_detector"], seed * 10000 + 100)
        base_weights = v2_router.fit_base_weights(scenarios["clean"]["prediction"], outer_train_ids)
        outer_records = make_records(scenarios, outer_validation_ids, base_weights, speech_detector, smile_detector)
        clean_oof = routed_oof.loc[routed_oof.scenario == "clean"]
        risk_threshold = float(np.quantile(clean_oof.risk.to_numpy(float), args.target_clean_coverage, method="higher"))
        metrics, routes, decisions = evaluate_outer(
            int(fold), outer_records, final_models, router_config, risk_threshold
        )
        decisions["outer_fold"] = int(fold)
        metrics.to_csv(fold_output / "metrics.csv", index=False)
        routes.to_csv(fold_output / "routes.csv", index=False)
        decisions.to_csv(fold_output / "participant_decisions.csv", index=False)
        tuning.to_csv(fold_output / "inner_tuning.csv", index=False)
        with (fold_output / "fusion_agent_v3.pkl").open("wb") as handle:
            pickle.dump({
                "protocol_sha256": EXPECTED_PROTOCOL_HASH,
                "outer_fold": int(fold), "paired_seed": seed,
                "base_weights": base_weights, "router_config": router_config,
                "risk_threshold": risk_threshold, "loss_models": final_models,
                "speech_detector": speech_detector, "smile_detector": smile_detector,
            }, handle)
        (fold_output / "run_complete.json").write_text(json.dumps(ev.json_ready({
            "outer_fold": int(fold), "paired_seed": seed, "router_config": router_config,
            "risk_threshold": risk_threshold, "outer_train_participants": len(outer_train_ids),
            "outer_validation_participants": len(outer_validation_ids),
        }), indent=2), encoding="utf-8")
        all_metrics.append(metrics); all_routes.append(routes); all_decisions.append(decisions)
    metrics = pd.concat(all_metrics, ignore_index=True)
    routes = pd.concat(all_routes, ignore_index=True)
    tuning = pd.concat(all_tuning, ignore_index=True)
    decisions = pd.concat(all_decisions, ignore_index=True)
    metrics.to_csv(output / "outer_metrics.csv", index=False)
    routes.to_csv(output / "outer_routes.csv", index=False)
    tuning.to_csv(output / "inner_router_tuning.csv", index=False)
    if len(requested) == len(all_folds):
        decision, checks = acceptance_decision(
            metrics, routes, decisions, selected_configs, args.bootstrap_replicates
        )
        checks.to_csv(output / "acceptance_checks.csv", index=False)
        (output / "selection_decision.json").write_text(json.dumps(ev.json_ready(decision), indent=2), encoding="utf-8")
        write_report(output / "FUSION_AGENT_V3_ROUTER_REPORT.md", metrics, routes, decision, checks)
    else:
        decision = {"smoke_test": True, "folds": requested, "promotion_authorized": False}
    protected_after = {str(p.resolve().relative_to(repo_root)): ev.sha256_file(p.resolve()) for p in sorted(set(protected_paths))}
    if protected_before != protected_after:
        raise RuntimeError("A protected artifact changed")
    aggregate_names = ["outer_metrics.csv", "outer_routes.csv", "inner_router_tuning.csv"]
    aggregate_names += [name for name in ("acceptance_checks.csv", "selection_decision.json", "FUSION_AGENT_V3_ROUTER_REPORT.md") if (output / name).exists()]
    manifest_out = {
        "stage": "fusion-agent-v3-router", "protocol_sha256": EXPECTED_PROTOCOL_HASH,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "prior_outer_results_exposed": True, "internal_role": "development_and_rejection_only",
        "paired_seed_by_outer_fold": {str(f): seed_by_fold[f] for f in requested},
        "mc_trials": args.mc_trials, "bootstrap_replicates": args.bootstrap_replicates,
        "device": str(device), "locked_test_predictions_generated": False,
        "protected_artifacts_unchanged": True, "protected_artifact_sha256": protected_after,
        "decision": decision,
        "output_sha256": {name: ev.sha256_file(output / name) for name in aggregate_names},
    }
    (output / "run_manifest.json").write_text(json.dumps(ev.json_ready(manifest_out), indent=2), encoding="utf-8")
    print(f"V3 router results written to: {output}")
    print(json.dumps(ev.json_ready(decision), indent=2))


if __name__ == "__main__":
    main()
