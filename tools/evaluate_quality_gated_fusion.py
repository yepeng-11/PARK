#!/usr/bin/env python3
"""Train and evaluate a quality-aware gate for PARK expert probabilities.

The gate is intentionally small and interpretable. It starts from clean-Dev
expert AUROC weights, removes unavailable experts, and downweights experts with
high MC uncertainty or disagreement with the remaining experts. Gate strengths
are selected on deterministic score-level perturbations of Dev; calibration and
operating thresholds are fitted on clean Dev only. Test labels are never used
for fitting, selection, calibration, or threshold choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

import evaluate_calibrated_fusion as calibrated
import evaluate_modality_robustness as robustness
import evaluate_pretrained as ev
import train_paired_baselines as paired


EXPERTS = ("finger", "speech", "smile")
MODELS = (
    "mean_probability",
    "available_mean",
    "available_weighted",
    "ufnet",
    "quality_gate_full",
    "quality_gate_no_uncertainty",
    "quality_gate_no_consistency",
)
EVALUATION_MODES = robustness.EVALUATION_MODES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Default: <repo>/results/protocol_alignment_audit",
    )
    parser.add_argument(
        "--training-dir", type=Path, default=None,
        help="Default: <repo>/results/paired_retraining",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Default: <repo>/results/quality_gated_fusion",
    )
    parser.add_argument("--datasets", default="cleaned")
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--mc-trials", type=int, default=30)
    parser.add_argument("--alpha-grid", default="0.25,0.5,1,2,4")
    parser.add_argument("--beta-grid", default="1,2,4,8,12")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_float_grid(value: str) -> List[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 for item in values):
        raise ValueError("Gate grids must contain non-negative values")
    return values


def stable_rng(seed: int, name: str) -> np.random.Generator:
    digest = hashlib.sha256(f"quality-gate:{seed}:{name}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))


def predict_with_quality(
    module,
    frame: pd.DataFrame,
    predictors: Sequence[nn.Module],
    fusion_model: nn.Module,
    device: torch.device,
    mc_trials: int,
    seed: int,
    missing: Sequence[int],
    conflict: Sequence[int],
) -> pd.DataFrame:
    """Return expert means, MC standard deviations, and native UFNet score."""
    paired.set_seed(seed)
    loader = paired.make_loader(paired.MultimodalDataset(frame), 1024, False, seed)
    criterion = nn.BCELoss()
    wrappers = [module.ModelWrapper(model, criterion) for model in predictors]
    fusion_wrapper = module.ModelWrapper(fusion_model, criterion)
    mean_pieces: List[List[np.ndarray]] = [[] for _ in predictors]
    std_pieces: List[List[np.ndarray]] = [[] for _ in predictors]
    fusion_pieces: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for features, target in loader:
        features = [value.to(device) for value in features]
        means = []
        deviations = []
        with torch.no_grad():
            for index, (wrapper, values) in enumerate(zip(wrappers, features)):
                samples = wrapper.predict_on_batch(values, iterations=mc_trials)
                mean = samples.mean(dim=-1).reshape(-1)
                deviation = samples.std(dim=-1).reshape(-1)
                if index in conflict:
                    mean = 1.0 - mean
                means.append(mean)
                deviations.append(deviation)
                mean_pieces[index].append(mean.cpu().numpy())
                std_pieces[index].append(deviation.cpu().numpy())
            fusion_samples = fusion_wrapper.predict_on_batch(
                (features, means, deviations), iterations=mc_trials
            )
            fusion_pieces.append(
                fusion_samples.mean(dim=-1).reshape(-1).cpu().numpy()
            )
        labels.append(target.numpy())

    result = frame[["id", "row_id", "label"]].reset_index(drop=True).copy()
    observed = np.concatenate(labels).astype(int)
    if not np.array_equal(observed, result.label.to_numpy(dtype=int)):
        raise ValueError("Prediction order mismatch")
    for index, name in enumerate(EXPERTS):
        result[name] = np.concatenate(mean_pieces[index])
        result[f"{name}_mc_std"] = np.concatenate(std_pieces[index])
        if index in missing:
            result[name] = 0.5
    result["ufnet"] = np.concatenate(fusion_pieces)
    return result


def aggregate_quality_level(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    columns = [
        *EXPERTS,
        *(f"{name}_mc_std" for name in EXPERTS),
        "ufnet",
    ]
    if level == "session":
        return frame[["id", "row_id", "label", *columns]].reset_index(drop=True)
    label_counts = frame.groupby("id").label.nunique()
    if (label_counts > 1).any():
        raise ValueError("A participant has inconsistent labels")
    aggregations: Dict[str, Any] = {
        "label": "first",
        **{column: "mean" for column in columns},
    }
    return frame.groupby("id", as_index=False).agg(aggregations)


def gate_scores(
    frame: pd.DataFrame,
    state: Dict[str, Any],
    missing: Sequence[int],
    alpha_override: float = None,
    beta_override: float = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.clip(frame[list(EXPERTS)].to_numpy(dtype=float), 0.0, 1.0)
    uncertainties = np.maximum(
        frame[[f"{name}_mc_std" for name in EXPERTS]].to_numpy(dtype=float), 0.0
    )
    n_rows, n_experts = probabilities.shape
    available = np.ones((n_rows, n_experts), dtype=float)
    if missing:
        available[:, list(missing)] = 0.0
    if np.any(available.sum(axis=1) == 0):
        raise ValueError("At least one expert must remain available")

    base_weights = np.asarray(state["base_weights"], dtype=float)
    uncertainty_scale = np.asarray(state["uncertainty_scale"], dtype=float)
    alpha = float(state["alpha"] if alpha_override is None else alpha_override)
    beta = float(state["beta"] if beta_override is None else beta_override)

    disagreement = np.zeros_like(probabilities)
    for index in range(n_experts):
        other = available.copy()
        other[:, index] = 0.0
        other_weights = other * base_weights.reshape(1, -1)
        denominator = other_weights.sum(axis=1)
        consensus = probabilities[:, index].copy()
        usable = denominator > 0
        consensus[usable] = (
            (probabilities[usable] * other_weights[usable]).sum(axis=1)
            / denominator[usable]
        )
        disagreement[:, index] = np.abs(probabilities[:, index] - consensus)

    normalized_uncertainty = np.clip(
        uncertainties / uncertainty_scale.reshape(1, -1), 0.0, 20.0
    )
    quality = available * np.exp(
        -alpha * normalized_uncertainty - beta * disagreement
    )
    weights = quality * base_weights.reshape(1, -1)
    denominator = weights.sum(axis=1, keepdims=True)
    failed = denominator[:, 0] <= 1e-15
    if failed.any():
        weights[failed] = available[failed] * base_weights.reshape(1, -1)
        denominator = weights.sum(axis=1, keepdims=True)
    weights = weights / denominator
    scores = (weights * probabilities).sum(axis=1)
    return scores, weights, quality


def tuning_conditions() -> List[Dict[str, Any]]:
    conditions: List[Dict[str, Any]] = [
        {"name": "clean", "kind": "clean", "indices": [], "severity": 0.0}
    ]
    for index, name in enumerate(EXPERTS):
        conditions.append(
            {"name": f"missing_{name}", "kind": "missing", "indices": [index], "severity": 1.0}
        )
    for left in range(3):
        for right in range(left + 1, 3):
            conditions.append(
                {
                    "name": f"missing_{EXPERTS[left]}_{EXPERTS[right]}",
                    "kind": "missing",
                    "indices": [left, right],
                    "severity": 1.0,
                }
            )
    for index, name in enumerate(EXPERTS):
        conditions.append(
            {"name": f"conflict_{name}", "kind": "conflict", "indices": [index], "severity": 1.0}
        )
        for severity in (0.10, 0.20):
            conditions.append(
                {"name": f"score_noise_{name}_{severity}", "kind": "noise", "indices": [index], "severity": severity}
            )
        for severity in (0.25, 0.50):
            conditions.append(
                {"name": f"attenuation_{name}_{severity}", "kind": "attenuation", "indices": [index], "severity": severity}
            )
    return conditions


def perturb_dev_scores(
    frame: pd.DataFrame,
    condition: Dict[str, Any],
    seed: int,
    uncertainty_scale: np.ndarray,
) -> Tuple[pd.DataFrame, List[int]]:
    output = frame.copy()
    missing: List[int] = []
    kind = condition["kind"]
    indices = condition["indices"]
    severity = float(condition["severity"])
    rng = stable_rng(seed, condition["name"])
    if kind == "missing":
        missing = list(indices)
        for index in indices:
            output[EXPERTS[index]] = 0.5
    elif kind == "conflict":
        for index in indices:
            output[EXPERTS[index]] = 1.0 - output[EXPERTS[index]]
    elif kind == "noise":
        for index in indices:
            name = EXPERTS[index]
            noise = rng.normal(0.0, severity, size=len(output))
            output[name] = np.clip(output[name].to_numpy(dtype=float) + noise, 0.0, 1.0)
            old = output[f"{name}_mc_std"].to_numpy(dtype=float)
            output[f"{name}_mc_std"] = np.sqrt(old ** 2 + noise ** 2)
    elif kind == "attenuation":
        for index in indices:
            name = EXPERTS[index]
            probability = output[name].to_numpy(dtype=float)
            output[name] = 0.5 + (probability - 0.5) * (1.0 - severity)
            output[f"{name}_mc_std"] = (
                output[f"{name}_mc_std"].to_numpy(dtype=float)
                + severity * uncertainty_scale[index]
            )
    elif kind != "clean":
        raise ValueError(f"Unknown tuning condition: {kind}")
    return output, missing


def fit_gate(
    dev: pd.DataFrame,
    seed: int,
    base_weights: np.ndarray,
    alpha_grid: Sequence[float],
    beta_grid: Sequence[float],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    labels = dev.label.to_numpy(dtype=int)
    uncertainty_scale = np.median(
        dev[[f"{name}_mc_std" for name in EXPERTS]].to_numpy(dtype=float), axis=0
    )
    uncertainty_scale = np.maximum(uncertainty_scale, 1e-4)
    base_state = {
        "base_weights": np.asarray(base_weights, dtype=float),
        "uncertainty_scale": uncertainty_scale,
        "alpha": 0.0,
        "beta": 0.0,
    }
    base_clean = float(
        roc_auc_score(labels, dev[list(EXPERTS)].to_numpy(dtype=float) @ base_weights)
    )
    conditions = tuning_conditions()
    rows: List[Dict[str, Any]] = []
    candidates: List[Tuple[float, float, float, float, float, float]] = []
    for alpha in alpha_grid:
        for beta in beta_grid:
            state = {**base_state, "alpha": alpha, "beta": beta}
            condition_aurocs: List[float] = []
            condition_ap: List[float] = []
            for condition in conditions:
                perturbed, missing = perturb_dev_scores(
                    dev, condition, seed, uncertainty_scale
                )
                score, _, _ = gate_scores(perturbed, state, missing)
                auroc = float(roc_auc_score(labels, score))
                average_precision = float(average_precision_score(labels, score))
                condition_aurocs.append(auroc)
                condition_ap.append(average_precision)
                rows.append(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "condition": condition["name"],
                        "condition_type": condition["kind"],
                        "auroc": auroc,
                        "average_precision": average_precision,
                    }
                )
            clean_auc = condition_aurocs[0]
            stress = condition_aurocs[1:]
            mean_stress = float(np.mean(stress))
            worst_stress = float(np.min(stress))
            mean_ap = float(np.mean(condition_ap))
            preservation_penalty = max(0.0, base_clean - clean_auc - 0.005)
            objective = (
                0.35 * clean_auc
                + 0.35 * mean_stress
                + 0.20 * worst_stress
                + 0.10 * mean_ap
                - 5.0 * preservation_penalty
            )
            candidates.append(
                (objective, clean_auc, mean_stress, worst_stress, alpha, beta)
            )
    best = max(candidates, key=lambda item: (item[0], -item[4], -item[5]))
    state = {
        **base_state,
        "alpha": best[4],
        "beta": best[5],
        "tuning_objective": best[0],
        "clean_dev_auroc": best[1],
        "mean_stress_dev_auroc": best[2],
        "worst_stress_dev_auroc": best[3],
        "base_clean_dev_auroc": base_clean,
        "tuning_conditions": [condition["name"] for condition in conditions],
    }
    tuning = pd.DataFrame(rows)
    tuning["selected"] = (
        np.isclose(tuning.alpha, state["alpha"])
        & np.isclose(tuning.beta, state["beta"])
    )
    return state, tuning


def model_scores(
    frame: pd.DataFrame,
    state: Dict[str, Any],
    missing: Sequence[int],
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    robust = robustness.robust_scores(frame, state["combiners"], missing)
    full, weights, quality = gate_scores(frame, state["gate"], missing)
    no_uncertainty, _, _ = gate_scores(
        frame, state["gate"], missing, alpha_override=0.0
    )
    no_consistency, _, _ = gate_scores(
        frame, state["gate"], missing, beta_override=0.0
    )
    scores = {
        "mean_probability": robust["mean_probability"],
        "available_mean": robust["available_mean"],
        "available_weighted": robust["available_weighted"],
        "ufnet": robust["ufnet"],
        "quality_gate_full": full,
        "quality_gate_no_uncertainty": no_uncertainty,
        "quality_gate_no_consistency": no_consistency,
    }
    return scores, weights, quality


def fit_dev_state(
    dev: pd.DataFrame,
    seed: int,
    alpha_grid: Sequence[float],
    beta_grid: Sequence[float],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    combiners = calibrated.fit_combiners(dev, seed)
    gate, tuning = fit_gate(
        dev, seed, combiners["weights"], alpha_grid, beta_grid
    )
    state: Dict[str, Any] = {
        "combiners": combiners,
        "gate": gate,
        "models": {},
    }
    scores, _, _ = model_scores(dev, state, [])
    labels = dev.label.to_numpy(dtype=int)
    for model_name, raw in scores.items():
        platt = calibrated.fit_calibrator("platt", raw, labels, seed)
        calibrated_scores = platt.predict(raw)
        threshold = calibrated.select_threshold(
            "specificity_0.90", labels, calibrated_scores
        )
        state["models"][model_name] = {
            "calibrator": platt,
            "specificity_threshold": threshold,
        }
    return state, tuning


def evaluate_scenario(
    dataset_name: str,
    seed: int,
    scenario: Dict[str, Any],
    level: str,
    frame: pd.DataFrame,
    missing: Sequence[int],
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    scores, _, _ = model_scores(frame, state, missing)
    labels = frame.label.to_numpy(dtype=int)
    rows: List[Dict[str, Any]] = []
    for model_name in MODELS:
        raw = scores[model_name]
        platt = state["models"][model_name]["calibrator"].predict(raw)
        for mode in EVALUATION_MODES:
            if mode == "raw_fixed_0.5":
                used = raw
                threshold = 0.5
            elif mode == "platt_fixed_0.5":
                used = platt
                threshold = 0.5
            else:
                used = platt
                threshold = state["models"][model_name]["specificity_threshold"]
            rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "scenario": scenario["scenario"],
                    "scenario_type": scenario["scenario_type"],
                    "modalities": scenario["modalities"],
                    "severity": scenario["severity"],
                    "level": level,
                    "model": model_name,
                    "evaluation_mode": mode,
                    **calibrated.compute_metrics(labels, used, threshold),
                }
            )
    return rows


def evaluate_run(
    module,
    dataset_name: str,
    seed: int,
    frame: pd.DataFrame,
    masks: Dict[str, np.ndarray],
    predictors,
    fusion_model,
    device: torch.device,
    mc_trials: int,
    scenarios: Sequence[Dict[str, Any]],
    alpha_grid: Sequence[float],
    beta_grid: Sequence[float],
    run_output: Path,
) -> None:
    dev_frame = frame.loc[masks["dev"]].reset_index(drop=True)
    dev_prediction = predict_with_quality(
        module, dev_frame, predictors, fusion_model, device, mc_trials,
        seed * 100000 + 1, [], []
    )
    states: Dict[str, Any] = {}
    tuning_rows: List[pd.DataFrame] = []
    for level_index, level in enumerate(("session", "participant")):
        state, tuning = fit_dev_state(
            aggregate_quality_level(dev_prediction, level),
            seed + level_index,
            alpha_grid,
            beta_grid,
        )
        states[level] = state
        tuning.insert(0, "level", level)
        tuning_rows.append(tuning)

    test_frame = frame.loc[masks["internal_test"]].reset_index(drop=True)
    metric_rows: List[Dict[str, Any]] = []
    prediction_rows: List[pd.DataFrame] = []
    for scenario_index, scenario in enumerate(scenarios):
        perturbed, missing, conflict = robustness.perturb_frame(
            test_frame, scenario, seed
        )
        prediction = predict_with_quality(
            module, perturbed, predictors, fusion_model, device, mc_trials,
            seed * 100000 + 1000 + scenario_index, missing, conflict
        )
        session_scores, weights, quality = model_scores(
            aggregate_quality_level(prediction, "session"), states["session"], missing
        )
        compact = prediction.copy()
        compact.insert(0, "scenario", scenario["scenario"])
        compact["quality_gate_full"] = session_scores["quality_gate_full"]
        for index, name in enumerate(EXPERTS):
            compact[f"gate_weight_{name}"] = weights[:, index]
            compact[f"gate_quality_{name}"] = quality[:, index]
        prediction_rows.append(compact)
        for level in ("session", "participant"):
            evaluated = aggregate_quality_level(prediction, level)
            metric_rows.extend(
                evaluate_scenario(
                    dataset_name, seed, scenario, level, evaluated, missing,
                    states[level]
                )
            )

    pd.DataFrame(metric_rows).to_csv(run_output / "metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        run_output / "scenario_predictions.csv", index=False
    )
    pd.concat(tuning_rows, ignore_index=True).to_csv(
        run_output / "dev_gate_grid.csv", index=False
    )
    with (run_output / "dev_gate_states.pkl").open("wb") as handle:
        pickle.dump(states, handle)
    state_summary = {
        level: {
            key: value for key, value in states[level]["gate"].items()
            if key != "tuning_conditions"
        }
        for level in states
    }
    with (run_output / "run_complete.json").open("w", encoding="utf-8") as handle:
        json.dump(
            ev.json_ready(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "mc_trials": mc_trials,
                    "scenarios": len(scenarios),
                    "metric_rows": len(metric_rows),
                    "gate_state": state_summary,
                }
            ),
            handle,
            indent=2,
        )


def write_report(path: Path, metrics: pd.DataFrame) -> None:
    target = metrics[
        (metrics.level == "participant")
        & (metrics.evaluation_mode == "raw_fixed_0.5")
    ]
    lines = [
        "# PARK quality-aware gated fusion report",
        "",
        "All gate hyperparameters originate from deterministic perturbations of Dev. "
        "Calibration and the 90%-specificity threshold use clean Dev only; internal-test "
        "labels are evaluation-only.",
        "",
        "## Clean performance and worst AUROC loss",
        "",
        "| Model | Clean AUROC | Single missing | Pair missing | Noise | Feature mask | Expert conflict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        selected = target[target.model == model]
        clean = selected[selected.scenario == "clean"].auroc.mean()
        values = []
        for scenario_type in (
            "missing_single", "missing_pair", "gaussian_noise",
            "feature_mask", "expert_conflict",
        ):
            group = selected[selected.scenario_type == scenario_type]
            by_scenario = group.groupby("scenario").auroc_delta_from_clean.mean()
            values.append(float(by_scenario.min()))
        lines.append(
            f"| {model} | {clean:.4f} | {values[0]:+.4f} | {values[1]:+.4f} | "
            f"{values[2]:+.4f} | {values[3]:+.4f} | {values[4]:+.4f} |"
        )

    scenario_mean = target.groupby(["scenario", "model"], as_index=False).auroc.mean()
    wide = scenario_mean.pivot(index="scenario", columns="model", values="auroc")
    wide["gate_gain_vs_ufnet"] = wide["quality_gate_full"] - wide["ufnet"]
    wide["gate_gain_vs_available_weighted"] = (
        wide["quality_gate_full"] - wide["available_weighted"]
    )
    lines.extend(
        [
            "",
            "## Largest full-gate gains over UFNet",
            "",
            "| Scenario | Gate AUROC | UFNet AUROC | Gain | Gain vs available weighted |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for scenario, row in wide.sort_values("gate_gain_vs_ufnet", ascending=False).head(8).iterrows():
        lines.append(
            f"| {scenario} | {row['quality_gate_full']:.4f} | {row['ufnet']:.4f} | "
            f"{row['gate_gain_vs_ufnet']:+.4f} | {row['gate_gain_vs_available_weighted']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Availability is supplied by the acquisition pipeline, not inferred from the test label.",
            "- MC standard deviation is normalized by its clean-Dev modality median.",
            "- Consistency is the absolute difference from the Dev-weighted consensus of the other available experts.",
            "- Dev perturbations operate on expert scores and approximate, but do not duplicate, every feature-space corruption.",
            "- The no-uncertainty and no-consistency rows are predeclared ablations of the selected full gate.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate_outputs(output_dir: Path) -> None:
    files = sorted(output_dir.glob("*/seed_*/metrics.csv"))
    if not files:
        return
    metrics = pd.concat([pd.read_csv(item) for item in files], ignore_index=True)
    metrics = robustness.add_clean_deltas(metrics)
    metrics.to_csv(output_dir / "metrics_per_run.csv", index=False)
    numeric = [
        "auroc", "average_precision", "accuracy", "balanced_accuracy", "f1",
        "sensitivity", "specificity", "brier", "ece",
        "auroc_delta_from_clean", "average_precision_delta_from_clean",
        "accuracy_delta_from_clean", "sensitivity_delta_from_clean",
        "specificity_delta_from_clean", "brier_delta_from_clean", "ece_delta_from_clean",
    ]
    summary = metrics.groupby(
        ["dataset", "scenario", "scenario_type", "modalities", "severity", "level", "model", "evaluation_mode"],
        as_index=False,
        dropna=False,
    )[numeric].agg(["mean", "std"])
    calibrated.flatten_aggregated_columns(summary).to_csv(
        output_dir / "quality_gate_summary.csv", index=False
    )
    write_report(output_dir / "QUALITY_GATE_REPORT.md", metrics)


def main() -> None:
    args = parse_args()
    if args.mc_trials < 2:
        raise ValueError("--mc-trials must be at least 2")
    alpha_grid = parse_float_grid(args.alpha_grid)
    beta_grid = parse_float_grid(args.beta_grid)
    repo_root = args.repo_root.resolve()
    data_dir = (args.data_dir or repo_root / "results" / "protocol_alignment_audit").resolve()
    training_dir = (args.training_dir or repo_root / "results" / "paired_retraining").resolve()
    output_dir = (args.output_dir or repo_root / "results" / "quality_gated_fusion").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = paired.parse_csv_list(args.datasets)
    invalid = set(datasets) - {"official", "cleaned"}
    if invalid:
        raise ValueError(f"Invalid dataset names: {sorted(invalid)}")
    seeds = paired.parse_seeds(args.seeds)
    device = ev.resolve_device(args.device)
    scenarios = robustness.scenario_definitions()
    pd.DataFrame(scenarios).to_csv(output_dir / "scenario_definitions.csv", index=False)
    pd.DataFrame(tuning_conditions()).to_csv(
        output_dir / "dev_tuning_conditions.csv", index=False
    )

    module = ev.load_upstream_module(repo_root)
    fusion_config = ev.read_json(Path(module.MODEL_CONFIG_PATH))
    selected_models = module.MODEL_SUBSETS[int(fusion_config["model_subset_choice"])]
    module.NUM_MODELS = len(selected_models)
    paths = ev.checkpoint_paths(module, selected_models)
    configs = [ev.read_json(item["config"]) for item in paths]
    protected_paths = [
        path for item in paths for key, path in item.items() if key in {"model", "scaler"}
    ] + [Path(module.MODEL_PATH)]
    hashes_before = {
        str(path.relative_to(repo_root)): ev.sha256_file(path)
        for path in protected_paths
    }

    for dataset_name in datasets:
        exported = paired.load_vector_csv(data_dir / f"{dataset_name}_aligned.csv")
        masks = paired.split_masks(module, exported)
        raw = paired.inverse_original_scaling(exported, configs, paths)
        for seed in seeds:
            training_run = training_dir / dataset_name / f"seed_{seed}"
            if not (training_run / "run_complete.json").exists():
                raise FileNotFoundError(f"Incomplete training run: {training_run}")
            run_output = output_dir / dataset_name / f"seed_{seed}"
            run_output.mkdir(parents=True, exist_ok=True)
            if (run_output / "run_complete.json").exists() and not args.force:
                print(f"Skipping completed quality-gate run: {dataset_name} seed={seed}")
                continue
            scaled = calibrated.apply_run_scalers(raw, training_run, configs)
            feature_shapes = [
                len(scaled.iloc[0][f"features_{index}"]) for index in range(3)
            ]
            predictors, fusion_model = calibrated.load_models(
                module, training_run, configs, fusion_config, feature_shapes, device
            )
            print(f"Quality-gate evaluation {dataset_name} seed={seed}")
            evaluate_run(
                module, dataset_name, seed, scaled, masks, predictors, fusion_model,
                device, args.mc_trials, scenarios, alpha_grid, beta_grid, run_output
            )
            aggregate_outputs(output_dir)

    hashes_after = {
        path: ev.sha256_file(repo_root / path) for path in hashes_before
    }
    if hashes_before != hashes_after:
        raise RuntimeError("A protected upstream checkpoint changed during gate evaluation")
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
                    "test_scenarios": scenarios,
                    "dev_tuning_conditions": tuning_conditions(),
                    "alpha_grid": alpha_grid,
                    "beta_grid": beta_grid,
                    "models": MODELS,
                    "evaluation_modes": EVALUATION_MODES,
                    "leakage_guard": "Gate selection uses perturbed Dev; calibration and thresholds use clean Dev; test labels are evaluation-only.",
                    "protected_checkpoint_sha256": hashes_after,
                    "protected_checkpoints_unchanged": True,
                }
            ),
            handle,
            indent=2,
        )
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
