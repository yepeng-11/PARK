#!/usr/bin/env python3
"""Train and evaluate the Fusion Agent v2 router under frozen nested CV.

Specialist hyperparameters and router actions are selected only in inner folds.
Each outer fold is evaluation-only. Released Test/global/validation cohorts are
never predicted. The final decision is computed from the eight criteria frozen
in the Fusion Agent v2 protocol.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import evaluate_calibrated_fusion as calibrated
import evaluate_learned_quality_gate as learned
import evaluate_pretrained as ev
import train_paired_baselines as paired
import train_fusion_agent_v2_specialists as specialists


PROTOCOL_HASH = specialists.EXPECTED_PROTOCOL_HASH
EXPERTS = tuple(calibrated.EXPERT_COLUMNS)
MODELS = ("fusion_agent_v2", "available_weighted", "ufnet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--protocol-dir", type=Path, default=None)
    parser.add_argument("--training-dir", type=Path, default=None)
    parser.add_argument("--specialist-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--paired-seeds", default="101,202,303,404,505")
    parser.add_argument("--mc-trials", type=int, default=30)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--target-clean-coverage", type=float, default=0.90)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def predict_scenarios(
    module,
    scaled: pd.DataFrame,
    predictors,
    fusion_model,
    device: torch.device,
    mc_trials: int,
    model_seed: int,
    outer_fold: int,
) -> Dict[str, Dict[str, Any]]:
    speech = specialists.predict_speech_scenarios(
        module,
        scaled,
        predictors,
        fusion_model,
        device,
        mc_trials,
        model_seed,
        outer_fold,
    )
    outputs = {
        name: {
            "prediction": frame,
            "kind": "clean" if name == "clean" else "speech_noise",
            "missing": [],
        }
        for name, frame in speech.items()
    }
    outputs["smile_conflict"] = {
        "prediction": learned.predict_quality_inputs(
            module,
            scaled,
            predictors,
            fusion_model,
            device,
            mc_trials,
            model_seed * 100000 + outer_fold * 1000 + 100,
            [],
            [2],
        ),
        "kind": "smile_conflict",
        "missing": [],
    }
    for index, name in enumerate(EXPERTS):
        missing_frame = scaled.copy()
        column = f"features_{index}"
        missing_frame[column] = [np.zeros_like(value) for value in missing_frame[column]]
        outputs[f"missing_{name}"] = {
            "prediction": learned.predict_quality_inputs(
                module,
                missing_frame,
                predictors,
                fusion_model,
                device,
                mc_trials,
                model_seed * 100000 + outer_fold * 1000 + 200 + index,
                [index],
                [],
            ),
            "kind": "missing",
            "missing": [index],
        }
    return outputs


def participant_view(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    selected = frame[["id", "label", *columns]].copy()
    counts = selected.groupby("id").label.nunique()
    if (counts > 1).any():
        raise ValueError("Participant has inconsistent labels inside frozen folds")
    return selected.groupby("id", as_index=False).agg(
        {"label": "first", **{column: "mean" for column in columns}}
    )


def fit_base_weights(clean: pd.DataFrame, participant_ids: set) -> np.ndarray:
    selected = clean.loc[clean.id.astype(str).isin(participant_ids)]
    participant = participant_view(selected, EXPERTS)
    labels = participant.label.to_numpy(dtype=int)
    aucs = np.asarray(
        [
            calibrated.safe_auc(labels, participant[name].to_numpy(dtype=float))
            for name in EXPERTS
        ]
    )
    weights = np.maximum(aucs - 0.5, 1e-6)
    return weights / weights.sum()


def load_specialist_parameters(specialist_dir: Path, outer_fold: int) -> Dict[str, Dict[str, Any]]:
    state = json.loads(
        (specialist_dir / f"outer_{outer_fold}" / "run_complete.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        name: item["parameters"] for name, item in state["specialists"].items()
    }


def fit_inner_detectors(
    scenarios: Dict[str, Dict[str, Any]],
    outer_rows: pd.DataFrame,
    inner_fold: int,
    params: Dict[str, Dict[str, Any]],
    model_seed: int,
    outer_fold: int,
):
    train_ids = set(
        outer_rows.loc[
            outer_rows.inner_validation_fold != inner_fold, "participant_id"
        ].astype(str)
    )
    clean = scenarios["clean"]["prediction"]
    speech_predictions = {
        name: item["prediction"]
        for name, item in scenarios.items()
        if item["kind"] in {"clean", "speech_noise"}
    }
    x_speech, y_speech, _ = specialists.speech_dataset(speech_predictions, train_ids)
    x_smile, y_smile, _ = specialists.smile_dataset(
        clean, train_ids, (outer_fold, inner_fold, "router-train")
    )
    speech_detector = specialists.fit_detector(
        x_speech,
        y_speech,
        params["speech_noise_detector"],
        model_seed * 10000 + inner_fold,
    )
    smile_detector = specialists.fit_detector(
        x_smile,
        y_smile,
        params["smile_conflict_detector"],
        model_seed * 10000 + 100 + inner_fold,
    )
    return speech_detector, smile_detector, train_ids


def detector_probabilities(
    prediction: pd.DataFrame, speech_detector, smile_detector
) -> Tuple[np.ndarray, np.ndarray]:
    speech = speech_detector.predict_proba(
        learned.quality_feature_matrix(prediction, specialists.SPEECH_INDEX)
    )[:, 1]
    smile = smile_detector.predict_proba(
        specialists.smile_feature_matrix(
            prediction, prediction.smile.to_numpy(dtype=float)
        )
    )[:, 1]
    return speech, smile


def apply_router(
    prediction: pd.DataFrame,
    base_weights: np.ndarray,
    speech_corruption: np.ndarray,
    smile_conflict: np.ndarray,
    missing: Sequence[int],
    config: Dict[str, Any],
) -> pd.DataFrame:
    probabilities = np.clip(prediction[list(EXPERTS)].to_numpy(dtype=float), 0.0, 1.0)
    available = np.ones_like(probabilities)
    if missing:
        available[:, list(missing)] = 0.0
    baseline_weights = available * np.asarray(base_weights).reshape(1, -1)
    baseline_weights /= baseline_weights.sum(axis=1, keepdims=True)
    available_score = (baseline_weights * probabilities).sum(axis=1)
    weights = baseline_weights.copy()
    speech_trigger = (
        (available[:, 1] > 0)
        & (speech_corruption >= float(config["speech_threshold"]))
        & (float(config["speech_factor"]) < 1.0)
    )
    weights[speech_trigger, 1] *= float(config["speech_factor"])
    smile_trigger = (
        (available[:, 2] > 0)
        & (smile_conflict >= float(config["smile_threshold"]))
        & (config["smile_action"] != "none")
    )
    if config["smile_action"] == "drop":
        weights[smile_trigger, 2] = 0.0
    denominator = weights.sum(axis=1, keepdims=True)
    failed = denominator[:, 0] <= 0
    weights[failed] = baseline_weights[failed]
    speech_trigger[failed] = False
    smile_trigger[failed] = False
    weights /= weights.sum(axis=1, keepdims=True)
    agent = (weights * probabilities).sum(axis=1)
    if config["smile_action"] == "ufnet":
        agent[smile_trigger] = prediction.loc[smile_trigger, "ufnet"].to_numpy(dtype=float)
    spread = (
        np.where(available > 0, probabilities, -np.inf).max(axis=1)
        - np.where(available > 0, probabilities, np.inf).min(axis=1)
    )
    confidence_risk = 1.0 - 2.0 * np.abs(agent - 0.5)
    # A missing modality is unavailable, not corrupted. Its detector score must
    # not force abstention after the router has already removed that modality.
    speech_risk = speech_corruption * available[:, 1]
    smile_risk = smile_conflict * available[:, 2]
    risk = np.clip(
        0.40 * speech_risk
        + 0.30 * smile_risk
        + 0.20 * spread
        + 0.10 * confidence_risk,
        0.0,
        1.0,
    )
    output = prediction[["id", "row_id", "label", *EXPERTS, "ufnet"]].copy()
    output["fusion_agent_v2"] = agent
    output["available_weighted"] = available_score
    output["risk"] = risk
    output["speech_trigger"] = speech_trigger
    output["smile_trigger"] = smile_trigger
    output["speech_corruption_probability"] = speech_corruption
    output["smile_conflict_probability"] = smile_conflict
    return output


def router_grid() -> List[Dict[str, Any]]:
    return [
        {
            "speech_threshold": speech_threshold,
            "speech_factor": speech_factor,
            "smile_threshold": smile_threshold,
            "smile_action": smile_action,
        }
        for speech_threshold, speech_factor, smile_threshold, smile_action in itertools.product(
            (0.5, 0.7, 0.9, 1.01),
            (0.0, 0.25, 0.5, 1.0),
            (0.5, 0.7, 0.9, 1.01),
            ("none", "drop", "ufnet"),
        )
    ]


def scenario_aucs(
    cache: Dict[str, Dict[str, Any]], config: Dict[str, Any]
) -> Tuple[Dict[str, Dict[str, float]], float]:
    values: Dict[str, Dict[str, float]] = {}
    clean_route_rate = 0.0
    for name, item in cache.items():
        routed = apply_router(
            item["prediction"],
            item["base_weights"],
            item["speech_probability"],
            item["smile_probability"],
            item["missing"],
            config,
        )
        participant = participant_view(
            routed, [*MODELS, "speech_trigger", "smile_trigger"]
        )
        labels = participant.label.to_numpy(dtype=int)
        values[name] = {
            model: float(roc_auc_score(labels, participant[model])) for model in MODELS
        }
        if name == "clean":
            clean_route_rate = float(
                ((participant.speech_trigger > 0) | (participant.smile_trigger > 0)).mean()
            )
    return values, clean_route_rate


def tune_router(inner_caches: Sequence[Dict[str, Dict[str, Any]]]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for candidate, config in enumerate(router_grid()):
        fold_values = []
        for inner_fold, cache in enumerate(inner_caches):
            aucs, clean_route_rate = scenario_aucs(cache, config)
            stress_names = [name for name in aucs if name != "clean"]
            agent_stress = float(np.mean([aucs[name]["fusion_agent_v2"] for name in stress_names]))
            available_stress = float(np.mean([aucs[name]["available_weighted"] for name in stress_names]))
            ufnet_stress = float(np.mean([aucs[name]["ufnet"] for name in stress_names]))
            clean_delta = aucs["clean"]["fusion_agent_v2"] - aucs["clean"]["available_weighted"]
            smile_delta = aucs["smile_conflict"]["fusion_agent_v2"] - aucs["smile_conflict"]["ufnet"]
            fold_values.append(
                {
                    "candidate": candidate,
                    "inner_fold": inner_fold,
                    **config,
                    "clean_delta": clean_delta,
                    "clean_route_rate": clean_route_rate,
                    "stress_agent": agent_stress,
                    "stress_delta_available": agent_stress - available_stress,
                    "stress_delta_ufnet": agent_stress - ufnet_stress,
                    "smile_conflict_delta": smile_delta,
                }
            )
        rows.extend(fold_values)
    tuning = pd.DataFrame(rows)
    keys = [
        "candidate", "speech_threshold", "speech_factor", "smile_threshold", "smile_action"
    ]
    aggregate = tuning.groupby(keys, as_index=False).agg(
        clean_delta=("clean_delta", "mean"),
        clean_pass_folds=("clean_delta", lambda value: int((value >= -0.005).sum())),
        clean_route_rate=("clean_route_rate", "mean"),
        stress_agent=("stress_agent", "mean"),
        stress_delta_available=("stress_delta_available", "mean"),
        stress_delta_ufnet=("stress_delta_ufnet", "mean"),
        smile_conflict_delta=("smile_conflict_delta", "mean"),
    )
    feasible = aggregate.loc[
        (aggregate.clean_delta >= -0.005)
        & (aggregate.clean_pass_folds >= 3)
        & (aggregate.clean_route_rate <= 0.05)
        & (aggregate.stress_delta_available > 0)
        & (aggregate.stress_delta_ufnet > 0)
        & (aggregate.smile_conflict_delta >= -0.01)
    ].copy()
    selection_reason = "inner_cv_all_router_constraints"
    if feasible.empty:
        feasible = aggregate.loc[
            (aggregate.speech_factor == 1.0)
            & (aggregate.smile_action == "none")
        ].copy()
        selection_reason = "fail_closed_available_weighted"
    selected = feasible.sort_values(
        ["stress_delta_available", "stress_delta_ufnet", "clean_delta"],
        ascending=False,
    ).iloc[0]
    selected_candidate = int(selected.candidate)
    tuning["selected"] = tuning.candidate == selected_candidate
    tuning["selection_reason"] = ""
    tuning.loc[tuning.selected, "selection_reason"] = selection_reason
    config = {
        "speech_threshold": float(selected.speech_threshold),
        "speech_factor": float(selected.speech_factor),
        "smile_threshold": float(selected.smile_threshold),
        "smile_action": str(selected.smile_action),
    }
    return config, tuning


def metric_record(labels: np.ndarray, scores: np.ndarray, accepted: np.ndarray) -> Dict[str, Any]:
    result = calibrated.compute_metrics(labels, scores, 0.5)
    retained_labels = labels[accepted]
    retained_scores = scores[accepted]
    result.update(
        {
            "coverage": float(accepted.mean()),
            "selective_n": int(accepted.sum()),
            "selective_auroc": (
                float(roc_auc_score(retained_labels, retained_scores))
                if len(np.unique(retained_labels)) == 2
                else float("nan")
            ),
        }
    )
    return result


def evaluate_outer(
    outer_fold: int,
    scenarios: Dict[str, Dict[str, Any]],
    validation_ids: set,
    speech_detector,
    smile_detector,
    base_weights: np.ndarray,
    config: Dict[str, Any],
    risk_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics: List[Dict[str, Any]] = []
    routes: List[Dict[str, Any]] = []
    for scenario, item in scenarios.items():
        prediction = item["prediction"].loc[
            item["prediction"].id.astype(str).isin(validation_ids)
        ].reset_index(drop=True)
        speech_probability, smile_probability = detector_probabilities(
            prediction, speech_detector, smile_detector
        )
        routed = apply_router(
            prediction,
            base_weights,
            speech_probability,
            smile_probability,
            item["missing"],
            config,
        )
        participant = participant_view(
            routed, [*MODELS, "risk", "speech_trigger", "smile_trigger"]
        )
        labels = participant.label.to_numpy(dtype=int)
        accepted = participant.risk.to_numpy(dtype=float) <= risk_threshold
        for model in MODELS:
            metrics.append(
                {
                    "outer_fold": outer_fold,
                    "scenario": scenario,
                    "scenario_type": item["kind"],
                    "model": model,
                    **metric_record(labels, participant[model].to_numpy(dtype=float), accepted),
                }
            )
        routes.append(
            {
                "outer_fold": outer_fold,
                "scenario": scenario,
                "participants": len(participant),
                "coverage": float(accepted.mean()),
                "speech_trigger_rate": float((participant.speech_trigger > 0).mean()),
                "smile_trigger_rate": float((participant.smile_trigger > 0).mean()),
                "mean_risk": float(participant.risk.mean()),
            }
        )
    return pd.DataFrame(metrics), pd.DataFrame(routes)


def acceptance_decision(metrics: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    pivot = metrics.pivot_table(
        index=["outer_fold", "scenario", "scenario_type"], columns="model", values=["auroc", "ece"]
    )
    clean = pivot.xs("clean", level="scenario")
    clean_delta = clean[("auroc", "fusion_agent_v2")] - clean[("auroc", "available_weighted")]
    clean_ece_delta = clean[("ece", "fusion_agent_v2")] - clean[("ece", "available_weighted")]
    stress = pivot.loc[pivot.index.get_level_values("scenario") != "clean"]
    by_fold_stress = stress.groupby(level="outer_fold").mean()
    stress_delta_available = (
        by_fold_stress[("auroc", "fusion_agent_v2")]
        - by_fold_stress[("auroc", "available_weighted")]
    )
    stress_delta_ufnet = (
        by_fold_stress[("auroc", "fusion_agent_v2")]
        - by_fold_stress[("auroc", "ufnet")]
    )
    speech = pivot.loc[pivot.index.get_level_values("scenario_type") == "speech_noise"]
    speech_by_fold = speech.groupby(level="outer_fold").mean()
    speech_best = np.maximum(
        speech_by_fold[("auroc", "available_weighted")],
        speech_by_fold[("auroc", "ufnet")],
    )
    speech_delta = speech_by_fold[("auroc", "fusion_agent_v2")] - speech_best
    smile = pivot.xs("smile_conflict", level="scenario")
    smile_delta = smile[("auroc", "fusion_agent_v2")] - smile[("auroc", "ufnet")]
    clean_coverage = metrics.loc[
        (metrics.scenario == "clean") & (metrics.model == "fusion_agent_v2"), "coverage"
    ]
    checks = [
        ("clean_noninferiority", float(clean_delta.mean()), ">=", -0.005, clean_delta.mean() >= -0.005),
        ("stress_superiority_available", float(stress_delta_available.mean()), ">", 0.0, stress_delta_available.mean() > 0),
        ("stress_superiority_ufnet", float(stress_delta_ufnet.mean()), ">", 0.0, stress_delta_ufnet.mean() > 0),
        ("speech_noise_gain", float(speech_delta.mean()), ">", 0.0, speech_delta.mean() > 0),
        ("smile_conflict_noninferiority", float(smile_delta.mean()), ">=", -0.01, smile_delta.mean() >= -0.01),
        ("clean_coverage", float(clean_coverage.mean()), ">=", 0.80, clean_coverage.mean() >= 0.80),
        ("calibration_guard", float(clean_ece_delta.mean()), "<=", 0.02, clean_ece_delta.mean() <= 0.02),
        ("fold_stability", int((clean_delta >= -0.005).sum()), ">=", 4, int((clean_delta >= -0.005).sum()) >= 4),
    ]
    table = pd.DataFrame(
        checks, columns=["criterion", "observed", "operator", "required", "passed"]
    )
    decision = {
        "promote_to_unseen_external_evaluation": bool(table.passed.all()),
        "passed_criteria": int(table.passed.sum()),
        "total_criteria": len(table),
        "protocol_sha256": PROTOCOL_HASH,
        "locked_test_predictions_generated": False,
    }
    return decision, table


def write_report(
    path: Path, metrics: pd.DataFrame, routes: pd.DataFrame, decision: Dict[str, Any], checks: pd.DataFrame
) -> None:
    lines = [
        "# Fusion Agent v2 nested-CV router",
        "",
        f"Frozen protocol: `{PROTOCOL_HASH}`.",
        "",
        f"Promotion to one genuinely unseen external evaluation: **{decision['promote_to_unseen_external_evaluation']}**.",
        f"Frozen criteria passed: **{decision['passed_criteria']}/{decision['total_criteria']}**.",
        "",
        "## Acceptance checks",
        "",
        "| Criterion | Observed | Rule | Required | Pass |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for row in checks.itertuples(index=False):
        lines.append(
            f"| {row.criterion} | {row.observed:.4f} | {row.operator} | {row.required:.4f} | {row.passed} |"
        )
    lines.extend(
        [
            "",
            "## Mean outer-fold AUROC",
            "",
            "| Scenario | Fusion Agent v2 | Available weighted | UFNet | Coverage |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for scenario in sorted(metrics.scenario.unique()):
        selected = metrics.loc[metrics.scenario == scenario]
        auc = selected.groupby("model").auroc.mean()
        coverage = routes.loc[routes.scenario == scenario, "coverage"].mean()
        lines.append(
            f"| {scenario} | {auc['fusion_agent_v2']:.4f} | {auc['available_weighted']:.4f} | "
            f"{auc['ufnet']:.4f} | {coverage:.4f} |"
        )
    lines.extend(
        [
            "",
            "All values above are nested Train+Dev outer-fold results. No released Test "
            "or external cohort was predicted. A passing result authorizes one new-cohort "
            "evaluation; it does not authorize reopening the existing Test for tuning.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.mc_trials < 2 or not 0 < args.target_clean_coverage <= 1:
        raise ValueError("Invalid MC trials or target coverage")
    repo_root = args.repo_root.resolve()
    data_path = (args.data or repo_root / "results/protocol_alignment_audit/cleaned_aligned.csv").resolve()
    protocol_dir = (args.protocol_dir or repo_root / "results/fusion_agent_v2_protocol").resolve()
    training_dir = (args.training_dir or repo_root / "results/paired_retraining").resolve()
    specialist_dir = (args.specialist_dir or repo_root / "results/fusion_agent_v2_specialists").resolve()
    output = (args.output_dir or repo_root / "results/fusion_agent_v2_router").resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol, manifest = specialists.load_and_validate_protocol(protocol_dir)
    if ev.sha256_file(data_path) != protocol["source_dataset_sha256"]:
        raise ValueError("Dataset hash differs from frozen protocol")
    seeds = paired.parse_seeds(args.paired_seeds)
    outer_folds = sorted(manifest.outer_fold.unique())
    if len(seeds) != len(outer_folds):
        raise ValueError("One paired seed is required per outer fold")

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
    protected_paths: List[Path] = [
        path
        for item in upstream_paths
        for path in item.values()
        if path.suffix in {".pth", ".pkl"}
    ]
    for fold, seed in zip(outer_folds, seeds):
        paired_run = training_dir / "cleaned" / f"seed_{seed}"
        protected_paths.extend(paired_run.glob("*/model.pth"))
        protected_paths.extend(paired_run.glob("scaler_*.pkl"))
        protected_paths.extend((specialist_dir / f"outer_{fold}").glob("*.pkl"))
    protected_before = {
        str(path.resolve().relative_to(repo_root)): ev.sha256_file(path.resolve())
        for path in sorted(set(protected_paths))
    }
    all_metrics: List[pd.DataFrame] = []
    all_routes: List[pd.DataFrame] = []
    all_tuning: List[pd.DataFrame] = []

    for outer_fold, model_seed in zip(outer_folds, seeds):
        fold_output = output / f"outer_{outer_fold}"
        fold_output.mkdir(parents=True, exist_ok=True)
        params = load_specialist_parameters(specialist_dir, int(outer_fold))
        run, scaled, predictors, fusion_model = specialists.load_outer_models(
            module, raw, training_dir, model_seed, configs, fusion_config, device
        )
        print(f"Router outer fold {outer_fold}; paired seed {model_seed}")
        scenarios = predict_scenarios(
            module, scaled, predictors, fusion_model, device, args.mc_trials, model_seed, int(outer_fold)
        )
        outer_rows = manifest.loc[
            (manifest.outer_fold == outer_fold) & (manifest.outer_role == "train")
        ].copy()
        inner_caches = []
        for inner_fold in sorted(outer_rows.inner_validation_fold.unique()):
            speech_detector, smile_detector, train_ids = fit_inner_detectors(
                scenarios, outer_rows, int(inner_fold), params, model_seed, int(outer_fold)
            )
            validation_ids = set(
                outer_rows.loc[
                    outer_rows.inner_validation_fold == inner_fold, "participant_id"
                ].astype(str)
            )
            base_weights = fit_base_weights(scenarios["clean"]["prediction"], train_ids)
            cache: Dict[str, Dict[str, Any]] = {}
            for name, item in scenarios.items():
                prediction = item["prediction"].loc[
                    item["prediction"].id.astype(str).isin(validation_ids)
                ].reset_index(drop=True)
                speech_probability, smile_probability = detector_probabilities(
                    prediction, speech_detector, smile_detector
                )
                cache[name] = {
                    "prediction": prediction,
                    "base_weights": base_weights,
                    "speech_probability": speech_probability,
                    "smile_probability": smile_probability,
                    "missing": item["missing"],
                }
            inner_caches.append(cache)
        config, tuning = tune_router(inner_caches)
        tuning.insert(0, "outer_fold", outer_fold)
        all_tuning.append(tuning)

        outer_train_ids = set(outer_rows.participant_id.astype(str))
        outer_validation_ids = set(
            manifest.loc[
                (manifest.outer_fold == outer_fold) & (manifest.outer_role == "validation"),
                "participant_id",
            ].astype(str)
        )
        speech_predictions = {
            name: item["prediction"]
            for name, item in scenarios.items()
            if item["kind"] in {"clean", "speech_noise"}
        }
        x_speech, y_speech, _ = specialists.speech_dataset(speech_predictions, outer_train_ids)
        x_smile, y_smile, _ = specialists.smile_dataset(
            scenarios["clean"]["prediction"], outer_train_ids, (outer_fold, "router-outer-train")
        )
        speech_detector = specialists.fit_detector(
            x_speech, y_speech, params["speech_noise_detector"], model_seed * 10000
        )
        smile_detector = specialists.fit_detector(
            x_smile, y_smile, params["smile_conflict_detector"], model_seed * 10000 + 100
        )
        base_weights = fit_base_weights(scenarios["clean"]["prediction"], outer_train_ids)
        clean_train = scenarios["clean"]["prediction"].loc[
            scenarios["clean"]["prediction"].id.astype(str).isin(outer_train_ids)
        ].reset_index(drop=True)
        speech_p, smile_p = detector_probabilities(clean_train, speech_detector, smile_detector)
        routed_train = apply_router(clean_train, base_weights, speech_p, smile_p, [], config)
        participant_train = participant_view(routed_train, ["risk"])
        risk_threshold = float(
            np.quantile(
                participant_train.risk.to_numpy(dtype=float),
                args.target_clean_coverage,
                method="higher",
            )
        )
        metrics, routes = evaluate_outer(
            int(outer_fold),
            scenarios,
            outer_validation_ids,
            speech_detector,
            smile_detector,
            base_weights,
            config,
            risk_threshold,
        )
        metrics.to_csv(fold_output / "metrics.csv", index=False)
        routes.to_csv(fold_output / "routes.csv", index=False)
        tuning.to_csv(fold_output / "inner_router_tuning.csv", index=False)
        with (fold_output / "fusion_agent_v2.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "protocol_sha256": PROTOCOL_HASH,
                    "outer_fold": int(outer_fold),
                    "paired_seed": model_seed,
                    "base_weights": base_weights,
                    "router_config": config,
                    "risk_threshold": risk_threshold,
                    "speech_detector": speech_detector,
                    "smile_detector": smile_detector,
                },
                handle,
            )
        (fold_output / "run_complete.json").write_text(
            json.dumps(
                ev.json_ready(
                    {
                        "outer_fold": int(outer_fold),
                        "paired_seed": model_seed,
                        "router_config": config,
                        "selection_reason": tuning.loc[tuning.selected, "selection_reason"].iloc[0],
                        "risk_threshold": risk_threshold,
                        "outer_train_participants": len(outer_train_ids),
                        "outer_validation_participants": len(outer_validation_ids),
                    }
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        all_metrics.append(metrics)
        all_routes.append(routes)

    metrics = pd.concat(all_metrics, ignore_index=True)
    routes = pd.concat(all_routes, ignore_index=True)
    tuning = pd.concat(all_tuning, ignore_index=True)
    metrics.to_csv(output / "outer_metrics.csv", index=False)
    routes.to_csv(output / "outer_routes.csv", index=False)
    tuning.to_csv(output / "inner_router_tuning.csv", index=False)
    decision, checks = acceptance_decision(metrics)
    checks.to_csv(output / "acceptance_checks.csv", index=False)
    (output / "selection_decision.json").write_text(
        json.dumps(ev.json_ready(decision), indent=2), encoding="utf-8"
    )
    write_report(output / "FUSION_AGENT_V2_ROUTER_REPORT.md", metrics, routes, decision, checks)
    protected_after = {
        str(path.resolve().relative_to(repo_root)): ev.sha256_file(path.resolve())
        for path in sorted(set(protected_paths))
    }
    if protected_before != protected_after:
        raise RuntimeError("A protected base or specialist artifact changed")
    run_manifest = {
        "stage": "fusion-agent-v2-router",
        "protocol_sha256": PROTOCOL_HASH,
        "paired_seed_by_outer_fold": {str(fold): seed for fold, seed in zip(outer_folds, seeds)},
        "mc_trials": args.mc_trials,
        "device": str(device),
        "nested_outer_folds": len(outer_folds),
        "nested_inner_folds": int(protocol["inner_folds"]),
        "locked_test_predictions_generated": False,
        "protected_artifact_sha256": protected_after,
        "protected_artifacts_unchanged": True,
        "decision": decision,
        "output_sha256": {
            name: ev.sha256_file(output / name)
            for name in (
                "outer_metrics.csv",
                "outer_routes.csv",
                "inner_router_tuning.csv",
                "acceptance_checks.csv",
                "selection_decision.json",
                "FUSION_AGENT_V2_ROUTER_REPORT.md",
            )
        },
    }
    (output / "run_manifest.json").write_text(
        json.dumps(ev.json_ready(run_manifest), indent=2), encoding="utf-8"
    )
    print(f"Criteria passed: {decision['passed_criteria']}/{decision['total_criteria']}")
    print(f"Promote: {decision['promote_to_unseen_external_evaluation']}")
    print(f"Router results written to: {output}")


if __name__ == "__main__":
    main()
