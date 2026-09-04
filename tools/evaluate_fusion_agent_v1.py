#!/usr/bin/env python3
"""Train and evaluate the conservative PARK Fusion Agent v1.

Fusion Agent v1 is a small, auditable router rather than another end-to-end
fusion network.  The default prediction is the clean-Dev AUROC-weighted expert
average.  It can (1) renormalize around missing modalities, (2) downweight a
speech expert that a Train-only synthetic quality detector marks as degraded,
and (3) fall back to UFNet when smile is both low-quality and strongly
inconsistent with the other experts.  A label-free risk score supports an
optional abstention path.

Quality detectors are fitted on Train synthetic corruptions.  Router parameters
and the abstention threshold are selected on Dev only.  Internal-test labels are
used only after the complete policy has been frozen.
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
import evaluate_modality_robustness as robustness
import evaluate_pretrained as ev
import evaluate_quality_gated_fusion as heuristic
import train_paired_baselines as paired


EXPERTS = tuple(calibrated.EXPERT_COLUMNS)
MODELS = ("fusion_agent_v1", "available_weighted", "ufnet")
KEY_SCENARIOS = (
    "clean",
    "noise_speech_1.0",
    "conflict_speech",
    "conflict_smile",
    "missing_speech",
    "missing_speech_smile",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--training-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--datasets", default="cleaned")
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--mc-trials", type=int, default=30)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quality-threshold-grid", default="0,0.2,0.35,0.5,0.65")
    parser.add_argument("--speech-factor-grid", default="0,0.25,0.5,1")
    parser.add_argument("--smile-disagreement-grid", default="0.2,0.3,0.4")
    parser.add_argument("--clean-margin", type=float, default=0.005)
    parser.add_argument("--smile-conflict-margin", type=float, default=0.01)
    parser.add_argument("--max-clean-route-rate", type=float, default=0.05)
    parser.add_argument("--target-clean-coverage", type=float, default=0.90)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_grid(value: str, lower: float = 0.0, upper: float = 1.0) -> List[float]:
    grid = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not grid or any(item < lower or item > upper for item in grid):
        raise ValueError(f"Grid values must be in [{lower}, {upper}]")
    return grid


def safe_auc(labels: Iterable[int], scores: Iterable[float]) -> float:
    labels_array = np.asarray(list(labels), dtype=int)
    scores_array = np.asarray(list(scores), dtype=float)
    if len(labels_array) == 0 or len(np.unique(labels_array)) < 2:
        return float("nan")
    return float(roc_auc_score(labels_array, scores_array))


def available_matrix(rows: int, missing: Sequence[int]) -> np.ndarray:
    available = np.ones((rows, len(EXPERTS)), dtype=float)
    if missing:
        available[:, list(missing)] = 0.0
    if np.any(available.sum(axis=1) == 0):
        raise ValueError("At least one expert must remain available")
    return available


def peer_mean(
    probabilities: np.ndarray, available: np.ndarray, modality_index: int
) -> np.ndarray:
    peers = available.copy()
    peers[:, modality_index] = 0.0
    denominator = peers.sum(axis=1)
    fallback = denominator <= 0
    denominator[fallback] = 1.0
    result = (probabilities * peers).sum(axis=1) / denominator
    result[fallback] = probabilities[fallback, modality_index]
    return result


def route_scores(
    frame: pd.DataFrame,
    detectors,
    base_weights: np.ndarray,
    missing: Sequence[int],
    config: Dict[str, float],
) -> pd.DataFrame:
    """Apply a frozen router without consulting outcome labels."""
    probabilities = np.clip(frame[list(EXPERTS)].to_numpy(dtype=float), 0.0, 1.0)
    qualities = np.column_stack(
        [
            detector.predict_proba(learned.quality_feature_matrix(frame, index))[:, 1]
            for index, detector in enumerate(detectors)
        ]
    )
    available = available_matrix(len(frame), missing)
    baseline_weights = available * np.asarray(base_weights, dtype=float).reshape(1, -1)
    weights = baseline_weights.copy()

    speech_index = EXPERTS.index("speech")
    smile_index = EXPERTS.index("smile")
    speech_trigger = (
        (available[:, speech_index] > 0)
        & (qualities[:, speech_index] < config["speech_quality_threshold"])
    )
    weights[speech_trigger, speech_index] *= config["speech_weight_factor"]
    denominator = weights.sum(axis=1, keepdims=True)
    failed = denominator[:, 0] <= 0
    if failed.any():
        # A quality rule must never discard the only available expert.
        weights[failed] = baseline_weights[failed]
        speech_trigger[failed] = False
        denominator = weights.sum(axis=1, keepdims=True)
    weights /= denominator
    baseline_weights /= baseline_weights.sum(axis=1, keepdims=True)
    available_weighted = (baseline_weights * probabilities).sum(axis=1)
    score = (weights * probabilities).sum(axis=1)

    smile_disagreement = np.abs(
        probabilities[:, smile_index] - peer_mean(probabilities, available, smile_index)
    )
    smile_fallback = (
        (available[:, smile_index] > 0)
        & (qualities[:, smile_index] < config["smile_quality_threshold"])
        & (smile_disagreement >= config["smile_disagreement_threshold"])
    )
    score[smile_fallback] = frame.loc[smile_fallback, "ufnet"].to_numpy(dtype=float)

    visible_quality = np.where(available > 0, qualities, 1.0)
    quality_risk = 1.0 - visible_quality.min(axis=1)
    prediction_range = (
        np.where(available > 0, probabilities, -np.inf).max(axis=1)
        - np.where(available > 0, probabilities, np.inf).min(axis=1)
    )
    confidence_risk = 1.0 - 2.0 * np.abs(score - 0.5)
    risk = np.clip(
        0.50 * quality_risk + 0.30 * prediction_range + 0.20 * confidence_risk,
        0.0,
        1.0,
    )

    output = frame[["id", "row_id", "label", *EXPERTS, "ufnet"]].copy()
    output["fusion_agent_v1"] = np.clip(score, 0.0, 1.0)
    output["available_weighted"] = available_weighted
    output["risk"] = risk
    output["speech_downweight"] = speech_trigger
    output["smile_ufnet_fallback"] = smile_fallback
    for index, name in enumerate(EXPERTS):
        output[f"quality_{name}"] = qualities[:, index]
        output[f"weight_{name}"] = weights[:, index]
        output[f"available_{name}"] = available[:, index].astype(bool)
    return output


def participant_view(frame: pd.DataFrame) -> pd.DataFrame:
    label_counts = frame.groupby("id").label.nunique()
    if (label_counts > 1).any():
        raise ValueError("A participant has inconsistent labels")
    numeric = [
        *MODELS,
        "risk",
        "speech_downweight",
        "smile_ufnet_fallback",
        *(f"quality_{name}" for name in EXPERTS),
        *(f"weight_{name}" for name in EXPERTS),
    ]
    aggregations: Dict[str, Any] = {
        "label": "first",
        **{column: "mean" for column in numeric},
    }
    return frame.groupby("id", as_index=False).agg(aggregations)


def score_candidate(
    dev_predictions: Dict[str, Dict[str, Any]],
    detectors,
    base_weights: np.ndarray,
    config: Dict[str, float],
    clean_margin: float,
    smile_conflict_margin: float,
    max_clean_route_rate: float,
) -> Dict[str, Any]:
    aucs: Dict[str, Dict[str, float]] = {}
    route_rates: Dict[str, float] = {}
    for scenario, item in dev_predictions.items():
        routed = participant_view(
            route_scores(item["prediction"], detectors, base_weights, item["missing"], config)
        )
        labels = routed.label.to_numpy(dtype=int)
        aucs[scenario] = {
            model: safe_auc(labels, routed[model].to_numpy(dtype=float)) for model in MODELS
        }
        route_rates[scenario] = float(
            (routed.speech_downweight > 0).mean()
            + (routed.smile_ufnet_fallback > 0).mean()
        )
    clean = aucs["clean"]
    stress = [values["fusion_agent_v1"] for name, values in aucs.items() if name != "clean"]
    stress_baseline = [values["available_weighted"] for name, values in aucs.items() if name != "clean"]
    stress_ufnet = [values["ufnet"] for name, values in aucs.items() if name != "clean"]
    smile = aucs["conflict_smile"]
    feasible_clean = clean["fusion_agent_v1"] >= clean["available_weighted"] - clean_margin
    feasible_smile = (
        smile["fusion_agent_v1"] >= smile["ufnet"] - smile_conflict_margin
    )
    feasible_route_rate = route_rates["clean"] <= max_clean_route_rate
    mean_stress = float(np.nanmean(stress))
    worst_stress = float(np.nanmin(stress))
    return {
        **config,
        "clean_agent_auroc": clean["fusion_agent_v1"],
        "clean_baseline_auroc": clean["available_weighted"],
        "clean_delta": clean["fusion_agent_v1"] - clean["available_weighted"],
        "stress_agent_mean_auroc": mean_stress,
        "stress_delta_vs_available": mean_stress - float(np.nanmean(stress_baseline)),
        "stress_delta_vs_ufnet": mean_stress - float(np.nanmean(stress_ufnet)),
        "worst_stress_auroc": worst_stress,
        "smile_conflict_agent_auroc": smile["fusion_agent_v1"],
        "smile_conflict_ufnet_auroc": smile["ufnet"],
        "smile_conflict_delta": smile["fusion_agent_v1"] - smile["ufnet"],
        "clean_route_rate": route_rates["clean"],
        "feasible_clean": feasible_clean,
        "feasible_smile": feasible_smile,
        "feasible_route_rate": feasible_route_rate,
        "feasible": feasible_clean and feasible_smile and feasible_route_rate,
        "objective": 0.20 * clean["fusion_agent_v1"] + 0.55 * mean_stress + 0.25 * worst_stress,
    }


def tune_router(
    dev_predictions: Dict[str, Dict[str, Any]],
    detectors,
    base_weights: np.ndarray,
    quality_grid: Sequence[float],
    speech_factor_grid: Sequence[float],
    smile_disagreement_grid: Sequence[float],
    clean_margin: float,
    smile_conflict_margin: float,
    max_clean_route_rate: float,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for speech_quality, speech_factor, smile_quality, smile_disagreement in itertools.product(
        quality_grid, speech_factor_grid, quality_grid, smile_disagreement_grid
    ):
        config = {
            "speech_quality_threshold": speech_quality,
            "speech_weight_factor": speech_factor,
            "smile_quality_threshold": smile_quality,
            "smile_disagreement_threshold": smile_disagreement,
        }
        rows.append(
            score_candidate(
                dev_predictions,
                detectors,
                base_weights,
                config,
                clean_margin,
                smile_conflict_margin,
                max_clean_route_rate,
            )
        )
    tuning = pd.DataFrame(rows)
    candidates = tuning.loc[
        tuning.feasible & (tuning.stress_delta_vs_available > 0)
    ].copy()
    selection_reason = "dev_safe_and_stress_superior"
    if candidates.empty:
        # Fail closed when Dev provides no evidence that active routing adds
        # value. The risk/abstention path remains available but accepted scores
        # stay identical to the availability-aware baseline.
        fallback = tuning.loc[
            (tuning.speech_quality_threshold == 0)
            & (tuning.speech_weight_factor == max(speech_factor_grid))
            & (tuning.smile_quality_threshold == 0)
        ]
        if fallback.empty:
            raise RuntimeError("Router grid does not contain the disabled fallback")
        selected_index = fallback.index[0]
        selection_reason = "fail_closed_available_weighted"
    else:
        selected_index = candidates.sort_values(
            ["stress_delta_vs_available", "objective", "clean_delta", "clean_route_rate"],
            ascending=[False, False, False, True],
        ).index[0]
    tuning["selected"] = tuning.index == selected_index
    tuning["selection_reason"] = ""
    tuning.loc[selected_index, "selection_reason"] = selection_reason
    columns = (
        "speech_quality_threshold",
        "speech_weight_factor",
        "smile_quality_threshold",
        "smile_disagreement_threshold",
    )
    return {column: float(tuning.loc[selected_index, column]) for column in columns}, tuning


def choose_risk_threshold(
    clean_dev: pd.DataFrame, target_clean_coverage: float
) -> Tuple[float, pd.DataFrame]:
    participant = participant_view(clean_dev)
    risks = participant.risk.to_numpy(dtype=float)
    threshold = float(np.quantile(risks, target_clean_coverage, method="higher"))
    audit = participant[["id", "risk"]].copy()
    audit["accepted"] = audit.risk <= threshold
    return threshold, audit


def metric_record(
    labels: np.ndarray, scores: np.ndarray, accepted: np.ndarray
) -> Dict[str, Any]:
    full = calibrated.compute_metrics(labels, scores, 0.5)
    retained_labels = labels[accepted]
    retained_scores = scores[accepted]
    full.update(
        {
            "coverage": float(accepted.mean()),
            "abstained": int((~accepted).sum()),
            "selective_n": int(accepted.sum()),
            "selective_auroc": safe_auc(retained_labels, retained_scores),
        }
    )
    return full


def evaluate_scenarios(
    dataset: str,
    seed: int,
    predictions: Dict[str, Dict[str, Any]],
    detectors,
    base_weights: np.ndarray,
    config: Dict[str, float],
    risk_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, Any]] = []
    route_rows: List[Dict[str, Any]] = []
    prediction_rows: List[pd.DataFrame] = []
    for scenario, item in predictions.items():
        definition = item["definition"]
        session = route_scores(
            item["prediction"], detectors, base_weights, item["missing"], config
        )
        session.insert(0, "scenario", scenario)
        prediction_rows.append(session)
        for level, viewed in (("session", session), ("participant", participant_view(session))):
            labels = viewed.label.to_numpy(dtype=int)
            accepted = viewed.risk.to_numpy(dtype=float) <= risk_threshold
            for model in MODELS:
                metric_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "scenario": scenario,
                        "scenario_type": definition["scenario_type"],
                        "modalities": definition["modalities"],
                        "severity": definition["severity"],
                        "level": level,
                        "model": model,
                        **metric_record(
                            labels,
                            viewed[model].to_numpy(dtype=float),
                            accepted,
                        ),
                    }
                )
            route_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "scenario": scenario,
                    "level": level,
                    "rows": len(viewed),
                    "speech_downweight_rate": float((viewed.speech_downweight > 0).mean()),
                    "smile_ufnet_fallback_rate": float((viewed.smile_ufnet_fallback > 0).mean()),
                    "abstention_rate": float((~accepted).mean()),
                    "mean_risk": float(viewed.risk.mean()),
                }
            )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(route_rows),
        pd.concat(prediction_rows, ignore_index=True),
    )


def run_one(
    module,
    dataset: str,
    seed: int,
    scaled: pd.DataFrame,
    masks: Dict[str, np.ndarray],
    predictors,
    fusion_model,
    device: torch.device,
    mc_trials: int,
    scenarios: Sequence[Dict[str, Any]],
    quality_grid: Sequence[float],
    speech_factor_grid: Sequence[float],
    smile_disagreement_grid: Sequence[float],
    clean_margin: float,
    smile_conflict_margin: float,
    max_clean_route_rate: float,
    target_clean_coverage: float,
    output: Path,
) -> None:
    train_frame = scaled.loc[masks["train"]].reset_index(drop=True)
    dev_frame = scaled.loc[masks["dev"]].reset_index(drop=True)
    test_frame = scaled.loc[masks["internal_test"]].reset_index(drop=True)
    detector_scenarios = learned.detector_training_scenarios()
    train_predictions = learned.generate_scenario_predictions(
        module, train_frame, predictors, fusion_model, device, mc_trials,
        seed, detector_scenarios, 41000,
    )
    detectors, detector_train = learned.fit_quality_detectors(train_predictions, seed)
    dev_predictions = learned.generate_scenario_predictions(
        module, dev_frame, predictors, fusion_model, device, mc_trials,
        seed, scenarios, 42000,
    )
    detector_dev = learned.quality_detector_validation(
        detectors,
        {name: item for name, item in dev_predictions.items() if name in train_predictions},
    )
    clean_dev = heuristic.aggregate_quality_level(
        dev_predictions["clean"]["prediction"], "participant"
    )
    base_weights = calibrated.fit_combiners(clean_dev, seed)["weights"]
    config, tuning = tune_router(
        dev_predictions,
        detectors,
        base_weights,
        quality_grid,
        speech_factor_grid,
        smile_disagreement_grid,
        clean_margin,
        smile_conflict_margin,
        max_clean_route_rate,
    )
    routed_clean_dev = route_scores(
        dev_predictions["clean"]["prediction"], detectors, base_weights, [], config
    )
    risk_threshold, risk_audit = choose_risk_threshold(
        routed_clean_dev, target_clean_coverage
    )
    test_predictions = learned.generate_scenario_predictions(
        module, test_frame, predictors, fusion_model, device, mc_trials,
        seed, scenarios, 43000,
    )
    metrics, routes, predictions = evaluate_scenarios(
        dataset,
        seed,
        test_predictions,
        detectors,
        base_weights,
        config,
        risk_threshold,
    )
    metrics.to_csv(output / "metrics.csv", index=False)
    routes.to_csv(output / "route_audit.csv", index=False)
    predictions.to_csv(output / "scenario_predictions.csv", index=False)
    detector_train.to_csv(output / "quality_detector_train.csv", index=False)
    detector_dev.to_csv(output / "quality_detector_dev.csv", index=False)
    tuning.to_csv(output / "dev_router_tuning.csv", index=False)
    risk_audit.to_csv(output / "dev_risk_threshold_audit.csv", index=False)
    with (output / "fusion_agent_v1.pkl").open("wb") as handle:
        pickle.dump(
            {
                "version": "fusion-agent-v1",
                "quality_detectors": detectors,
                "base_weights": base_weights,
                "router_config": config,
                "risk_threshold": risk_threshold,
                "target_clean_coverage": target_clean_coverage,
            },
            handle,
        )
    with (output / "run_complete.json").open("w", encoding="utf-8") as handle:
        json.dump(
            ev.json_ready(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "mc_trials": mc_trials,
                    "train_rows": len(train_frame),
                    "dev_rows": len(dev_frame),
                    "test_rows": len(test_frame),
                    "base_weights": base_weights,
                    "router_config": config,
                    "router_selection_reason": str(
                        tuning.loc[tuning.selected, "selection_reason"].iloc[0]
                    ),
                    "risk_threshold": risk_threshold,
                    "target_clean_coverage": target_clean_coverage,
                    "max_clean_route_rate": max_clean_route_rate,
                    "selection_partition": "dev",
                    "evaluation_partition": "internal_test",
                    "metric_rows": len(metrics),
                }
            ),
            handle,
            indent=2,
        )


def aggregate_outputs(output_dir: Path, clean_margin: float) -> None:
    metric_files = sorted(output_dir.glob("*/seed_*/metrics.csv"))
    route_files = sorted(output_dir.glob("*/seed_*/route_audit.csv"))
    if not metric_files:
        return
    metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    routes = pd.concat([pd.read_csv(path) for path in route_files], ignore_index=True)
    metrics.to_csv(output_dir / "metrics_per_run.csv", index=False)
    routes.to_csv(output_dir / "route_audit_per_run.csv", index=False)
    numeric = [
        "auroc", "average_precision", "accuracy", "balanced_accuracy", "f1",
        "sensitivity", "specificity", "brier", "ece", "coverage",
        "abstained", "selective_n", "selective_auroc",
    ]
    summary = metrics.groupby(
        ["dataset", "scenario", "scenario_type", "modalities", "severity", "level", "model"],
        as_index=False,
        dropna=False,
    )[numeric].agg(["mean", "std"])
    calibrated.flatten_aggregated_columns(summary).to_csv(
        output_dir / "fusion_agent_v1_summary.csv", index=False
    )

    participant = metrics.loc[metrics.level == "participant"]
    mean_auc = participant.groupby(["scenario", "model"]).auroc.mean().unstack()
    clean_delta = float(
        mean_auc.loc["clean", "fusion_agent_v1"]
        - mean_auc.loc["clean", "available_weighted"]
    )
    stress = participant.loc[participant.scenario != "clean"]
    stress_seed = stress.groupby(["seed", "model"]).auroc.mean().unstack()
    stress_agent = float(stress_seed.fusion_agent_v1.mean())
    stress_delta_available = float(
        (stress_seed.fusion_agent_v1 - stress_seed.available_weighted).mean()
    )
    stress_delta_ufnet = float((stress_seed.fusion_agent_v1 - stress_seed.ufnet).mean())
    speech_noise_delta = float(
        mean_auc.loc["noise_speech_1.0", "fusion_agent_v1"]
        - mean_auc.loc["noise_speech_1.0", "ufnet"]
    )
    smile_conflict_delta = float(
        mean_auc.loc["conflict_smile", "fusion_agent_v1"]
        - mean_auc.loc["conflict_smile", "ufnet"]
    )
    clean_rows = participant.loc[participant.scenario == "clean"]
    clean_selective = clean_rows.groupby("model").selective_auroc.mean()
    clean_coverage = float(
        clean_rows.loc[clean_rows.model == "fusion_agent_v1", "coverage"].mean()
    )
    stress_selective_seed = (
        stress.groupby(["seed", "model"]).selective_auroc.mean().unstack()
    )
    stress_selective_agent = float(stress_selective_seed.fusion_agent_v1.mean())
    stress_coverage = float(
        stress.loc[stress.model == "fusion_agent_v1"]
        .groupby("seed")
        .coverage.mean()
        .mean()
    )
    selective_candidate = bool(
        clean_delta >= -clean_margin
        and clean_coverage >= 0.80
        and clean_selective["fusion_agent_v1"]
        >= mean_auc.loc["clean", "available_weighted"]
        and stress_coverage >= 0.25
        and stress_selective_agent
        >= max(stress_seed.available_weighted.mean(), stress_seed.ufnet.mean())
    )
    decision = {
        "promote_full_coverage_router_to_external_validation": bool(
            clean_delta >= -clean_margin
            and stress_delta_available > 0
            and stress_delta_ufnet > 0
            and speech_noise_delta > 0
            and smile_conflict_delta >= -0.01
        ),
        "promote_selective_policy_to_external_validation": selective_candidate,
        "clean_auroc_delta_vs_available_weighted": clean_delta,
        "clean_selective_auroc": float(clean_selective["fusion_agent_v1"]),
        "clean_selective_coverage": clean_coverage,
        "stress_macro_auroc": stress_agent,
        "stress_selective_macro_auroc": stress_selective_agent,
        "stress_selective_mean_coverage": stress_coverage,
        "stress_macro_delta_vs_available_weighted": stress_delta_available,
        "stress_macro_delta_vs_ufnet": stress_delta_ufnet,
        "speech_noise_1_auroc_delta_vs_ufnet": speech_noise_delta,
        "smile_conflict_auroc_delta_vs_ufnet": smile_conflict_delta,
        "important": (
            "Internal test has prior analytical exposure. This decision is diagnostic; "
            "a positive result authorizes external validation, not a publication claim."
        ),
    }
    with (output_dir / "selection_decision.json").open("w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2)
    write_report(output_dir / "FUSION_AGENT_V1_REPORT.md", metrics, routes, decision, output_dir)


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    routes: pd.DataFrame,
    decision: Dict[str, Any],
    output_dir: Path,
) -> None:
    participant = metrics.loc[metrics.level == "participant"]
    lines = [
        "# PARK Fusion Agent v1",
        "",
        "Fusion Agent v1 is an auditable conservative router. Quality detectors are fit "
        "on Train synthetic corruptions, router parameters are selected on Dev, and Test "
        "labels are evaluation-only.",
        "",
        "## Decision",
        "",
        f"- Promote full-coverage router to external validation: **{decision['promote_full_coverage_router_to_external_validation']}**.",
        f"- Promote selective policy to external validation: **{decision['promote_selective_policy_to_external_validation']}**.",
        f"- Clean AUROC delta vs available weighted: **{decision['clean_auroc_delta_vs_available_weighted']:+.4f}**.",
        f"- Clean selective AUROC / coverage: **{decision['clean_selective_auroc']:.4f} / {decision['clean_selective_coverage']:.4f}**.",
        f"- Stress macro-AUROC: **{decision['stress_macro_auroc']:.4f}**.",
        f"- Stress selective macro-AUROC / coverage: **{decision['stress_selective_macro_auroc']:.4f} / {decision['stress_selective_mean_coverage']:.4f}**.",
        f"- Stress delta vs available weighted: **{decision['stress_macro_delta_vs_available_weighted']:+.4f}**.",
        f"- Stress delta vs UFNet: **{decision['stress_macro_delta_vs_ufnet']:+.4f}**.",
        f"- Speech noise 1.0 delta vs UFNet: **{decision['speech_noise_1_auroc_delta_vs_ufnet']:+.4f}**.",
        f"- Smile conflict delta vs UFNet: **{decision['smile_conflict_auroc_delta_vs_ufnet']:+.4f}**.",
        "",
        "> The internal test has prior analytical exposure. Passing this gate authorizes "
        "external validation; it is not itself an unbiased final performance claim.",
        "",
        "## Participant-level results",
        "",
        "| Scenario | Model | AUROC mean | AUROC SD | Coverage mean | Selective AUROC |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for scenario in KEY_SCENARIOS:
        for model in MODELS:
            selected = participant.loc[
                (participant.scenario == scenario) & (participant.model == model)
            ]
            if selected.empty:
                continue
            lines.append(
                f"| {scenario} | {model} | {selected.auroc.mean():.4f} | "
                f"{selected.auroc.std(ddof=1):.4f} | {selected.coverage.mean():.4f} | "
                f"{selected.selective_auroc.mean():.4f} |"
            )
    participant_routes = routes.loc[routes.level == "participant"]
    lines.extend(
        [
            "",
            "## Router behavior",
            "",
            "| Scenario | Speech downweight | Smile UFNet fallback | Abstention |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for scenario in KEY_SCENARIOS:
        selected = participant_routes.loc[participant_routes.scenario == scenario]
        if selected.empty:
            continue
        lines.append(
            f"| {scenario} | {selected.speech_downweight_rate.mean():.4f} | "
            f"{selected.smile_ufnet_fallback_rate.mean():.4f} | "
            f"{selected.abstention_rate.mean():.4f} |"
        )
    states = [
        json.load(path.open(encoding="utf-8"))
        for path in sorted(output_dir.glob("*/seed_*/run_complete.json"))
    ]
    lines.extend(
        [
            "",
            "## Frozen per-seed policies",
            "",
            *[
                f"- Seed {state['seed']}: {json.dumps(state['router_config'], sort_keys=True)}, "
                f"risk threshold={state['risk_threshold']:.4f}."
                for state in states
            ],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def collect_protected_paths(
    repo_root: Path, upstream_paths, module, training_dir: Path, datasets, seeds
) -> List[Path]:
    paths = [
        path for item in upstream_paths for key, path in item.items() if key in {"model", "scaler"}
    ] + [Path(module.MODEL_PATH)]
    for dataset in datasets:
        for seed in seeds:
            run = training_dir / dataset / f"seed_{seed}"
            paths.extend(run.glob("*/model.pth"))
            paths.extend(run.glob("scaler_*.pkl"))
    return sorted(set(path.resolve() for path in paths))


def main() -> None:
    args = parse_args()
    if args.mc_trials < 2:
        raise ValueError("--mc-trials must be at least 2")
    if args.clean_margin < 0 or args.smile_conflict_margin < 0:
        raise ValueError("Non-inferiority margins must be non-negative")
    if not 0 <= args.max_clean_route_rate <= 1:
        raise ValueError("--max-clean-route-rate must be in [0, 1]")
    if not 0 < args.target_clean_coverage <= 1:
        raise ValueError("--target-clean-coverage must be in (0, 1]")
    quality_grid = parse_grid(args.quality_threshold_grid)
    speech_factor_grid = parse_grid(args.speech_factor_grid)
    disagreement_grid = parse_grid(args.smile_disagreement_grid)
    repo_root = args.repo_root.resolve()
    data_dir = (args.data_dir or repo_root / "results" / "protocol_alignment_audit").resolve()
    training_dir = (args.training_dir or repo_root / "results" / "paired_retraining").resolve()
    output_dir = (args.output_dir or repo_root / "results" / "fusion_agent_v1").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = paired.parse_csv_list(args.datasets)
    seeds = paired.parse_seeds(args.seeds)
    device = ev.resolve_device(args.device)
    scenarios = robustness.scenario_definitions()
    pd.DataFrame(scenarios).to_csv(output_dir / "scenario_definitions.csv", index=False)

    module = ev.load_upstream_module(repo_root)
    fusion_config = ev.read_json(Path(module.MODEL_CONFIG_PATH))
    selected_models = module.MODEL_SUBSETS[int(fusion_config["model_subset_choice"])]
    module.NUM_MODELS = len(selected_models)
    upstream_paths = ev.checkpoint_paths(module, selected_models)
    configs = [ev.read_json(item["config"]) for item in upstream_paths]
    protected = collect_protected_paths(
        repo_root, upstream_paths, module, training_dir, datasets, seeds
    )
    hashes_before = {str(path.relative_to(repo_root)): ev.sha256_file(path) for path in protected}

    for dataset in datasets:
        exported = paired.load_vector_csv(data_dir / f"{dataset}_aligned.csv")
        masks = paired.split_masks(module, exported)
        raw = paired.inverse_original_scaling(exported, configs, upstream_paths)
        for seed in seeds:
            training_run = training_dir / dataset / f"seed_{seed}"
            if not (training_run / "run_complete.json").exists():
                raise FileNotFoundError(f"Incomplete training run: {training_run}")
            run_output = output_dir / dataset / f"seed_{seed}"
            run_output.mkdir(parents=True, exist_ok=True)
            if (run_output / "run_complete.json").exists() and not args.force:
                print(f"Skipping completed Fusion Agent v1: {dataset} seed={seed}")
                continue
            scaled = calibrated.apply_run_scalers(raw, training_run, configs)
            feature_shapes = [len(scaled.iloc[0][f"features_{index}"]) for index in range(3)]
            predictors, fusion_model = calibrated.load_models(
                module, training_run, configs, fusion_config, feature_shapes, device
            )
            print(f"Fusion Agent v1 {dataset} seed={seed}")
            run_one(
                module,
                dataset,
                seed,
                scaled,
                masks,
                predictors,
                fusion_model,
                device,
                args.mc_trials,
                scenarios,
                quality_grid,
                speech_factor_grid,
                disagreement_grid,
                args.clean_margin,
                args.smile_conflict_margin,
                args.max_clean_route_rate,
                args.target_clean_coverage,
                run_output,
            )
            aggregate_outputs(output_dir, args.clean_margin)

    hashes_after = {path: ev.sha256_file(repo_root / path) for path in hashes_before}
    if hashes_before != hashes_after:
        raise RuntimeError("A protected upstream or paired-training artifact changed")
    aggregate_outputs(output_dir, args.clean_margin)
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            ev.json_ready(
                {
                    "version": "fusion-agent-v1",
                    "git_commit": ev.git_commit(repo_root),
                    "datasets": datasets,
                    "seeds": seeds,
                    "mc_trials": args.mc_trials,
                    "device": str(device),
                    "quality_threshold_grid": quality_grid,
                    "speech_factor_grid": speech_factor_grid,
                    "smile_disagreement_grid": disagreement_grid,
                    "clean_noninferiority_margin": args.clean_margin,
                    "smile_conflict_noninferiority_margin": args.smile_conflict_margin,
                    "max_clean_route_rate": args.max_clean_route_rate,
                    "target_clean_coverage": args.target_clean_coverage,
                    "quality_detector_training_partition": "train",
                    "router_selection_partition": "dev",
                    "evaluation_partition": "internal_test",
                    "protected_artifact_sha256": hashes_after,
                    "protected_artifacts_unchanged": True,
                    "leakage_guard": (
                        "Train synthetic corruption labels fit quality detectors; Dev selects "
                        "router and risk threshold; internal-test labels are evaluation-only."
                    ),
                }
            ),
            handle,
            indent=2,
        )
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
