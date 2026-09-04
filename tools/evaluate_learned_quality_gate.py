#!/usr/bin/env python3
"""Evaluate a learned modality-quality gate on paired PARK runs.

Per-modality quality detectors learn only synthetic corruption labels on Train.
The learned gate is blended with the availability-aware Dev-AUROC baseline; the
blend is selected on perturbed Dev subject to a clean-Dev non-inferiority
constraint. Internal-test labels are evaluation-only.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

import evaluate_calibrated_fusion as calibrated
import evaluate_modality_robustness as robustness
import evaluate_pretrained as ev
import evaluate_quality_gated_fusion as heuristic
import train_paired_baselines as paired


EXPERTS = tuple(calibrated.EXPERT_COLUMNS)
MODELS = (
    "learned_quality_gate",
    "learned_quality_unblended",
    "available_weighted",
    "ufnet",
)
DESCRIPTORS = (
    "mean_abs",
    "rms",
    "std",
    "max_abs",
    "p95_abs",
    "zero_fraction",
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
    parser.add_argument("--blend-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--clean-margin", type=float, default=0.005)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_grid(value: str) -> List[float]:
    grid = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not grid or min(grid) < 0.0 or max(grid) > 1.0:
        raise ValueError("Blend grid must contain values in [0, 1]")
    return sorted(set(grid))


def add_feature_descriptors(prediction: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    output = prediction.copy()
    for index, name in enumerate(EXPERTS):
        matrix = np.stack(frame[f"features_{index}"]).astype(np.float64)
        absolute = np.abs(matrix)
        values = {
            "mean_abs": absolute.mean(axis=1),
            "rms": np.sqrt(np.square(matrix).mean(axis=1)),
            "std": matrix.std(axis=1),
            "max_abs": absolute.max(axis=1),
            "p95_abs": np.quantile(absolute, 0.95, axis=1),
            "zero_fraction": np.isclose(matrix, 0.0, atol=1e-7).mean(axis=1),
        }
        for descriptor, vector in values.items():
            output[f"{name}_feature_{descriptor}"] = vector
    return output


def predict_quality_inputs(
    module,
    frame: pd.DataFrame,
    predictors,
    fusion_model,
    device: torch.device,
    mc_trials: int,
    seed: int,
    missing: Sequence[int],
    conflict: Sequence[int],
) -> pd.DataFrame:
    prediction = heuristic.predict_with_quality(
        module,
        frame,
        predictors,
        fusion_model,
        device,
        mc_trials,
        seed,
        missing,
        conflict,
    )
    return add_feature_descriptors(prediction, frame)


def quality_feature_matrix(frame: pd.DataFrame, modality_index: int) -> np.ndarray:
    name = EXPERTS[modality_index]
    probability = frame[name].to_numpy(dtype=float)
    uncertainty = frame[f"{name}_mc_std"].to_numpy(dtype=float)
    peers = frame[[item for item in EXPERTS if item != name]].to_numpy(dtype=float)
    peer_mean = peers.mean(axis=1)
    peer_spread = peers.std(axis=1)
    columns = [
        frame[f"{name}_feature_{descriptor}"].to_numpy(dtype=float)
        for descriptor in DESCRIPTORS
    ]
    columns.extend(
        [
            probability,
            uncertainty,
            np.abs(probability - 0.5),
            np.abs(probability - peer_mean),
            peer_spread,
        ]
    )
    matrix = np.column_stack(columns)
    return np.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)


def detector_training_scenarios() -> List[Dict[str, Any]]:
    allowed = {"clean", "missing_single", "gaussian_noise", "feature_mask", "expert_conflict"}
    return [
        scenario
        for scenario in robustness.scenario_definitions()
        if scenario["scenario_type"] in allowed
    ]


def generate_scenario_predictions(
    module,
    base_frame: pd.DataFrame,
    predictors,
    fusion_model,
    device: torch.device,
    mc_trials: int,
    seed: int,
    scenarios: Sequence[Dict[str, Any]],
    phase_offset: int,
) -> Dict[str, Dict[str, Any]]:
    outputs: Dict[str, Dict[str, Any]] = {}
    for scenario_index, scenario in enumerate(scenarios):
        perturbed, missing, conflict = robustness.perturb_frame(
            base_frame, scenario, seed + phase_offset
        )
        prediction = predict_quality_inputs(
            module,
            perturbed,
            predictors,
            fusion_model,
            device,
            mc_trials,
            seed * 1000000 + phase_offset + scenario_index,
            missing,
            conflict,
        )
        outputs[scenario["scenario"]] = {
            "definition": scenario,
            "prediction": prediction,
            "missing": missing,
        }
    return outputs


def fit_quality_detectors(
    train_predictions: Dict[str, Dict[str, Any]], seed: int
) -> Tuple[List[HistGradientBoostingClassifier], pd.DataFrame]:
    clean = train_predictions["clean"]["prediction"]
    detectors: List[HistGradientBoostingClassifier] = []
    summaries: List[Dict[str, Any]] = []
    for index, name in enumerate(EXPERTS):
        matrices = [quality_feature_matrix(clean, index)]
        targets = [np.ones(len(clean), dtype=int)]
        source_counts = {"clean": len(clean)}
        for scenario_name, item in train_predictions.items():
            definition = item["definition"]
            if scenario_name == "clean" or definition["modalities"] != name:
                continue
            matrix = quality_feature_matrix(item["prediction"], index)
            matrices.append(matrix)
            targets.append(np.zeros(len(matrix), dtype=int))
            source_counts[scenario_name] = len(matrix)
        x = np.concatenate(matrices)
        y = np.concatenate(targets)
        positive_weight = len(y) / (2.0 * max(1, int(y.sum())))
        negative_weight = len(y) / (2.0 * max(1, int((y == 0).sum())))
        sample_weight = np.where(y == 1, positive_weight, negative_weight)
        detector = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=max(10, len(clean) // 30),
            l2_regularization=1.0,
            random_state=seed + index,
        ).fit(x, y, sample_weight=sample_weight)
        train_score = detector.predict_proba(x)[:, 1]
        detectors.append(detector)
        summaries.append(
            {
                "modality": name,
                "train_rows": len(y),
                "clean_rows": int(y.sum()),
                "corrupted_rows": int((y == 0).sum()),
                "train_quality_auroc": float(roc_auc_score(y, train_score)),
                "train_quality_accuracy": float(accuracy_score(y, train_score >= 0.5)),
                "training_sources": json.dumps(source_counts, sort_keys=True),
            }
        )
    return detectors, pd.DataFrame(summaries)


def quality_detector_validation(
    detectors: Sequence[HistGradientBoostingClassifier],
    predictions: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    clean = predictions["clean"]["prediction"]
    rows: List[Dict[str, Any]] = []
    for index, name in enumerate(EXPERTS):
        matrices = [quality_feature_matrix(clean, index)]
        targets = [np.ones(len(clean), dtype=int)]
        for scenario_name, item in predictions.items():
            definition = item["definition"]
            if scenario_name == "clean" or definition["modalities"] != name:
                continue
            matrices.append(quality_feature_matrix(item["prediction"], index))
            targets.append(np.zeros(len(item["prediction"]), dtype=int))
        x = np.concatenate(matrices)
        y = np.concatenate(targets)
        score = detectors[index].predict_proba(x)[:, 1]
        rows.append(
            {
                "modality": name,
                "rows": len(y),
                "quality_auroc": float(roc_auc_score(y, score)),
                "quality_accuracy": float(accuracy_score(y, score >= 0.5)),
                "clean_quality_mean": float(score[y == 1].mean()),
                "corrupted_quality_mean": float(score[y == 0].mean()),
            }
        )
    return pd.DataFrame(rows)


def quality_gate_components(
    frame: pd.DataFrame,
    detectors: Sequence[HistGradientBoostingClassifier],
    base_weights: np.ndarray,
    missing: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probabilities = frame[list(EXPERTS)].to_numpy(dtype=float)
    qualities = np.column_stack(
        [
            detector.predict_proba(quality_feature_matrix(frame, index))[:, 1]
            for index, detector in enumerate(detectors)
        ]
    )
    available = np.ones_like(qualities)
    if missing:
        available[:, list(missing)] = 0.0
    qualities *= available
    learned_weights = qualities * base_weights.reshape(1, -1)
    denominator = learned_weights.sum(axis=1, keepdims=True)
    failed = denominator[:, 0] <= 1e-12
    if failed.any():
        learned_weights[failed] = available[failed] * base_weights.reshape(1, -1)
        denominator = learned_weights.sum(axis=1, keepdims=True)
    learned_weights /= denominator
    learned = (learned_weights * probabilities).sum(axis=1)

    available_weights = available * base_weights.reshape(1, -1)
    available_weights /= available_weights.sum(axis=1, keepdims=True)
    baseline = (available_weights * probabilities).sum(axis=1)
    return learned, baseline, learned_weights, qualities


def aggregate_scores(frame: pd.DataFrame, columns: Sequence[str], level: str) -> pd.DataFrame:
    selected = frame[["id", "row_id", "label", *columns]].copy()
    if level == "session":
        return selected
    label_counts = selected.groupby("id").label.nunique()
    if (label_counts > 1).any():
        raise ValueError("A participant has inconsistent labels")
    aggregations = {"label": "first", **{column: "mean" for column in columns}}
    return selected.groupby("id", as_index=False).agg(aggregations)


def fit_blend(
    dev_predictions: Dict[str, Dict[str, Any]],
    detectors,
    base_weights: np.ndarray,
    blend_grid: Sequence[float],
    clean_margin: float,
) -> Tuple[float, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for scenario_name, item in dev_predictions.items():
        learned, baseline, _, _ = quality_gate_components(
            item["prediction"], detectors, base_weights, item["missing"]
        )
        temp = item["prediction"][["id", "row_id", "label"]].copy()
        temp["learned"] = learned
        temp["baseline"] = baseline
        participant = aggregate_scores(temp, ["learned", "baseline"], "participant")
        cache[scenario_name] = (
            participant.label.to_numpy(dtype=int),
            participant.learned.to_numpy(dtype=float),
            participant.baseline.to_numpy(dtype=float),
        )
    y_clean, _, base_clean_score = cache["clean"]
    base_clean_auc = float(roc_auc_score(y_clean, base_clean_score))
    candidates: List[Tuple[float, float, float, float, float]] = []
    for blend in blend_grid:
        scenario_aucs = []
        clean_auc = np.nan
        for scenario_name, (labels, learned, baseline) in cache.items():
            score = blend * learned + (1.0 - blend) * baseline
            auc = float(roc_auc_score(labels, score))
            scenario_aucs.append(auc)
            if scenario_name == "clean":
                clean_auc = auc
        stress = [value for name, value in zip(cache, scenario_aucs) if name != "clean"]
        mean_stress = float(np.mean(stress))
        worst_stress = float(np.min(stress))
        feasible = bool(clean_auc >= base_clean_auc - clean_margin)
        objective = 0.2 * clean_auc + 0.5 * mean_stress + 0.3 * worst_stress
        rows.append(
            {
                "blend": blend,
                "base_clean_auroc": base_clean_auc,
                "clean_auroc": clean_auc,
                "mean_stress_auroc": mean_stress,
                "worst_stress_auroc": worst_stress,
                "clean_noninferiority_margin": clean_margin,
                "feasible": feasible,
                "objective": objective,
            }
        )
        if feasible:
            candidates.append((objective, worst_stress, mean_stress, -blend, blend))
    selected = max(candidates)[-1] if candidates else 0.0
    tuning = pd.DataFrame(rows)
    tuning["selected"] = np.isclose(tuning.blend, selected)
    return float(selected), tuning


def add_model_scores(
    prediction: pd.DataFrame,
    detectors,
    base_weights: np.ndarray,
    missing: Sequence[int],
    blend: float,
) -> pd.DataFrame:
    learned, baseline, weights, qualities = quality_gate_components(
        prediction, detectors, base_weights, missing
    )
    output = prediction.copy()
    output["learned_quality_unblended"] = learned
    output["available_weighted"] = baseline
    output["learned_quality_gate"] = blend * learned + (1.0 - blend) * baseline
    for index, name in enumerate(EXPERTS):
        output[f"learned_quality_{name}"] = qualities[:, index]
        output[f"learned_weight_{name}"] = weights[:, index]
    return output


def evaluate_test_predictions(
    dataset: str,
    seed: int,
    test_predictions: Dict[str, Dict[str, Any]],
    detectors,
    base_weights: np.ndarray,
    blend: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, Any]] = []
    prediction_rows: List[pd.DataFrame] = []
    for scenario_name, item in test_predictions.items():
        definition = item["definition"]
        scored = add_model_scores(
            item["prediction"], detectors, base_weights, item["missing"], blend
        )
        compact_columns = [
            "id", "row_id", "label", *EXPERTS, "ufnet",
            "learned_quality_gate", "learned_quality_unblended", "available_weighted",
            *(f"learned_quality_{name}" for name in EXPERTS),
            *(f"learned_weight_{name}" for name in EXPERTS),
        ]
        compact = scored[compact_columns].copy()
        compact.insert(0, "scenario", scenario_name)
        prediction_rows.append(compact)
        for level in ("session", "participant"):
            evaluated = aggregate_scores(scored, MODELS, level)
            labels = evaluated.label.to_numpy(dtype=int)
            for model in MODELS:
                metric_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "scenario": scenario_name,
                        "scenario_type": definition["scenario_type"],
                        "modalities": definition["modalities"],
                        "severity": definition["severity"],
                        "level": level,
                        "model": model,
                        "evaluation_mode": "raw_fixed_0.5",
                        **calibrated.compute_metrics(
                            labels, evaluated[model].to_numpy(dtype=float), 0.5
                        ),
                    }
                )
    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


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
    blend_grid: Sequence[float],
    clean_margin: float,
    output: Path,
) -> None:
    train_frame = scaled.loc[masks["train"]].reset_index(drop=True)
    dev_frame = scaled.loc[masks["dev"]].reset_index(drop=True)
    test_frame = scaled.loc[masks["internal_test"]].reset_index(drop=True)
    detector_scenarios = detector_training_scenarios()
    train_predictions = generate_scenario_predictions(
        module, train_frame, predictors, fusion_model, device, mc_trials,
        seed, detector_scenarios, 10000,
    )
    detectors, detector_train = fit_quality_detectors(train_predictions, seed)
    dev_predictions = generate_scenario_predictions(
        module, dev_frame, predictors, fusion_model, device, mc_trials,
        seed, scenarios, 20000,
    )
    detector_dev = quality_detector_validation(
        detectors,
        {name: item for name, item in dev_predictions.items() if name in train_predictions},
    )
    clean_dev = heuristic.aggregate_quality_level(
        dev_predictions["clean"]["prediction"], "participant"
    )
    base_weights = calibrated.fit_combiners(clean_dev, seed)["weights"]
    blend, tuning = fit_blend(
        dev_predictions, detectors, base_weights, blend_grid, clean_margin
    )
    test_predictions = generate_scenario_predictions(
        module, test_frame, predictors, fusion_model, device, mc_trials,
        seed, scenarios, 30000,
    )
    metrics, predictions = evaluate_test_predictions(
        dataset, seed, test_predictions, detectors, base_weights, blend
    )
    metrics.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "scenario_predictions.csv", index=False)
    detector_train.to_csv(output / "quality_detector_train.csv", index=False)
    detector_dev.to_csv(output / "quality_detector_dev.csv", index=False)
    tuning.to_csv(output / "dev_blend_tuning.csv", index=False)
    with (output / "quality_detectors.pkl").open("wb") as handle:
        pickle.dump(detectors, handle)
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
                    "selected_blend": blend,
                    "base_weights": base_weights,
                    "clean_noninferiority_margin": clean_margin,
                    "metric_rows": len(metrics),
                }
            ),
            handle,
            indent=2,
        )


def add_clean_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    output = metrics.copy()
    keys = ["dataset", "seed", "level", "model"]
    clean = output.loc[output.scenario == "clean", keys + ["auroc", "accuracy"]].rename(
        columns={"auroc": "clean_auroc", "accuracy": "clean_accuracy"}
    )
    output = output.merge(clean, on=keys, validate="many_to_one")
    output["auroc_delta_from_clean"] = output.auroc - output.clean_auroc
    output["accuracy_delta_from_clean"] = output.accuracy - output.clean_accuracy
    return output


def aggregate_outputs(output_dir: Path) -> None:
    metric_files = sorted(output_dir.glob("*/seed_*/metrics.csv"))
    if not metric_files:
        return
    metrics = add_clean_deltas(
        pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    )
    metrics.to_csv(output_dir / "metrics_per_run.csv", index=False)
    numeric = [
        "auroc", "average_precision", "accuracy", "balanced_accuracy", "f1",
        "sensitivity", "specificity", "brier", "ece",
        "auroc_delta_from_clean", "accuracy_delta_from_clean",
    ]
    summary = metrics.groupby(
        ["dataset", "scenario", "scenario_type", "modalities", "severity", "level", "model"],
        as_index=False,
        dropna=False,
    )[numeric].agg(["mean", "std"])
    summary = calibrated.flatten_aggregated_columns(summary)
    summary.to_csv(output_dir / "learned_quality_summary.csv", index=False)

    participant = metrics.loc[metrics.level == "participant"]
    pivot = participant.pivot_table(
        index=["dataset", "seed", "scenario"], columns="model", values="auroc"
    ).reset_index()
    delta_rows = []
    for baseline in ("available_weighted", "ufnet"):
        current = pivot[["dataset", "seed", "scenario", "learned_quality_gate", baseline]].copy()
        current["baseline"] = baseline
        current["auroc_delta"] = current.learned_quality_gate - current[baseline]
        delta_rows.append(current[["dataset", "seed", "scenario", "baseline", "auroc_delta"]])
    pd.concat(delta_rows, ignore_index=True).to_csv(
        output_dir / "paired_model_deltas.csv", index=False
    )
    clean_means = participant.loc[participant.scenario == "clean"].groupby("model").auroc.mean()
    stress_means = (
        participant.loc[participant.scenario != "clean"]
        .groupby(["seed", "model"])
        .auroc.mean()
        .groupby("model")
        .mean()
    )
    clean_delta = float(
        clean_means["learned_quality_gate"] - clean_means["available_weighted"]
    )
    stress_delta_available = float(
        stress_means["learned_quality_gate"] - stress_means["available_weighted"]
    )
    stress_delta_ufnet = float(
        stress_means["learned_quality_gate"] - stress_means["ufnet"]
    )
    decision = {
        "selected_as_replacement": bool(
            clean_delta >= -0.005
            and stress_delta_available > 0
            and stress_delta_ufnet > 0
        ),
        "clean_test_noninferiority_margin": 0.005,
        "clean_auroc_delta_vs_available_weighted": clean_delta,
        "stress_macro_auroc_delta_vs_available_weighted": stress_delta_available,
        "stress_macro_auroc_delta_vs_ufnet": stress_delta_ufnet,
        "decision_rule": (
            "Require clean mean AUROC delta >= -0.005 and positive stress "
            "macro-AUROC deltas versus both available_weighted and UFNet."
        ),
    }
    with (output_dir / "selection_decision.json").open("w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2)
    write_report(output_dir / "LEARNED_QUALITY_GATE_REPORT.md", metrics, output_dir)


def write_report(path: Path, metrics: pd.DataFrame, output_dir: Path) -> None:
    target = metrics.loc[metrics.level == "participant"]
    key_scenarios = [
        "clean", "noise_speech_1.0", "conflict_speech", "conflict_smile",
        "missing_speech", "missing_speech_smile",
    ]
    clean_means = target.loc[target.scenario == "clean"].groupby("model").auroc.mean()
    stress_means = (
        target.loc[target.scenario != "clean"]
        .groupby(["seed", "model"])
        .auroc.mean()
        .groupby("model")
        .mean()
    )
    clean_delta = float(
        clean_means["learned_quality_gate"] - clean_means["available_weighted"]
    )
    stress_delta_available = float(
        stress_means["learned_quality_gate"] - stress_means["available_weighted"]
    )
    stress_delta_ufnet = float(
        stress_means["learned_quality_gate"] - stress_means["ufnet"]
    )
    speech_noise = target.loc[target.scenario == "noise_speech_1.0"].groupby("model").auroc.mean()
    speech_noise_delta_ufnet = float(
        speech_noise["learned_quality_gate"] - speech_noise["ufnet"]
    )
    passes_clean = clean_delta >= -0.005
    passes_stress = stress_delta_available > 0 and stress_delta_ufnet > 0
    selected_as_replacement = passes_clean and passes_stress
    lines = [
        "# PARK learned modality-quality gate",
        "",
        "Quality detectors use synthetic corruption labels on Train only. Blend selection "
        "uses perturbed Dev with a clean-AUROC non-inferiority constraint. Test labels are "
        "used only for the metrics below.",
        "",
        "## Decision",
        "",
        f"- Selected as a replacement fusion rule: **{selected_as_replacement}**.",
        f"- Clean AUROC delta versus available weighted: **{clean_delta:+.4f}** "
        f"(required >= -0.0050; pass={passes_clean}).",
        f"- Stress macro-AUROC delta versus available weighted: **{stress_delta_available:+.4f}**.",
        f"- Stress macro-AUROC delta versus UFNet: **{stress_delta_ufnet:+.4f}** "
        f"(joint stress pass={passes_stress}).",
        f"- Speech-noise-1.0 AUROC delta versus UFNet: **{speech_noise_delta_ufnet:+.4f}**.",
        "",
        "The full learned gate is therefore retained as a diagnostic experiment, not promoted. "
        "Its speech-corruption detector is a useful component for a future modality-specific "
        "fallback, while the unstable blend and smile-conflict behavior require redesign.",
        "",
        "## Participant-level paired results",
        "",
        "| Scenario | Model | AUROC mean | AUROC SD | Accuracy mean |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for scenario in key_scenarios:
        for model in MODELS:
            selected = target.loc[(target.scenario == scenario) & (target.model == model)]
            if selected.empty:
                continue
            lines.append(
                f"| {scenario} | {model} | {selected.auroc.mean():.4f} | "
                f"{selected.auroc.std(ddof=1):.4f} | {selected.accuracy.mean():.4f} |"
            )

    stress = target.loc[target.scenario != "clean"].groupby(["seed", "model"]).auroc.mean().reset_index()
    lines.extend(
        [
            "",
            "## Macro-average across 21 stress scenarios",
            "",
            "| Model | Mean AUROC | Seed SD |",
            "| --- | ---: | ---: |",
        ]
    )
    for model in MODELS:
        selected = stress.loc[stress.model == model]
        lines.append(
            f"| {model} | {selected.auroc.mean():.4f} | {selected.auroc.std(ddof=1):.4f} |"
        )

    states = []
    detector_rows = []
    for complete in sorted(output_dir.glob("*/seed_*/run_complete.json")):
        states.append(json.load(open(complete, encoding="utf-8")))
        dev_path = complete.parent / "quality_detector_dev.csv"
        current = pd.read_csv(dev_path)
        current["seed"] = json.load(open(complete, encoding="utf-8"))["seed"]
        detector_rows.append(current)
    detector = pd.concat(detector_rows, ignore_index=True)
    lines.extend(
        [
            "",
            "## Selection and detector checks",
            "",
            f"Selected blend coefficients: {', '.join(str(item['selected_blend']) for item in states)}.",
            f"Mean synthetic-Dev corruption-detection AUROC: {detector.quality_auroc.mean():.4f}.",
            "",
            "The clean Test result is a post-selection evaluation, not an additional selection rule. "
            "With five seeds, differences are descriptive rather than confirmatory.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def collect_protected_paths(repo_root: Path, upstream_paths, module, training_dir: Path, datasets, seeds):
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
    if args.clean_margin < 0:
        raise ValueError("--clean-margin must be non-negative")
    blend_grid = parse_grid(args.blend_grid)
    repo_root = args.repo_root.resolve()
    data_dir = (args.data_dir or repo_root / "results" / "protocol_alignment_audit").resolve()
    training_dir = (args.training_dir or repo_root / "results" / "paired_retraining").resolve()
    output_dir = (args.output_dir or repo_root / "results" / "learned_quality_gate").resolve()
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
                print(f"Skipping completed learned gate: {dataset} seed={seed}")
                continue
            scaled = calibrated.apply_run_scalers(raw, training_run, configs)
            feature_shapes = [len(scaled.iloc[0][f"features_{index}"]) for index in range(3)]
            predictors, fusion_model = calibrated.load_models(
                module, training_run, configs, fusion_config, feature_shapes, device
            )
            print(f"Learned quality gate {dataset} seed={seed}")
            run_one(
                module, dataset, seed, scaled, masks, predictors, fusion_model,
                device, args.mc_trials, scenarios, blend_grid, args.clean_margin,
                run_output,
            )
            aggregate_outputs(output_dir)

    hashes_after = {path: ev.sha256_file(repo_root / path) for path in hashes_before}
    if hashes_before != hashes_after:
        raise RuntimeError("A protected upstream or paired-training artifact changed")
    aggregate_outputs(output_dir)
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            ev.json_ready(
                {
                    "repo_root": repo_root,
                    "git_commit": ev.git_commit(repo_root),
                    "datasets": datasets,
                    "seeds": seeds,
                    "mc_trials": args.mc_trials,
                    "device": str(device),
                    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                    "blend_grid": blend_grid,
                    "clean_dev_noninferiority_margin": args.clean_margin,
                    "quality_detector_training_partition": "train",
                    "blend_selection_partition": "dev",
                    "evaluation_partition": "internal_test",
                    "scenarios": scenarios,
                    "models": MODELS,
                    "leakage_guard": (
                        "Quality detectors use Train synthetic corruption labels without disease labels; "
                        "blend selection uses perturbed Dev; Test labels are evaluation-only."
                    ),
                    "protected_artifact_sha256": hashes_after,
                    "protected_artifacts_unchanged": True,
                }
            ),
            handle,
            indent=2,
        )
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
