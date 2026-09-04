#!/usr/bin/env python3
"""Stress-test PARK fusion models under missing and corrupted modalities.

The benchmark uses clean Dev data to fit classical combiners, Platt calibrators,
and 90%-specificity operating thresholds. Perturbations are applied only to test
features or expert scores. Results are compared with the same seed/model under a
clean test condition, and upstream/training checkpoints are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

import evaluate_calibrated_fusion as calibrated
import evaluate_pretrained as ev
import train_paired_baselines as paired


EXPERTS = ("finger", "speech", "smile")
ROBUST_MODELS = (
    "mean_probability",
    "available_mean",
    "dev_auroc_weighted",
    "available_weighted",
    "logistic_stacking",
    "tree_stacking",
    "ufnet",
)
EVALUATION_MODES = (
    "raw_fixed_0.5",
    "platt_fixed_0.5",
    "platt_dev_specificity_0.90",
)


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
        help="Default: <repo>/results/modality_robustness",
    )
    parser.add_argument("--datasets", default="cleaned")
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--mc-trials", type=int, default=30)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def scenario_definitions() -> List[Dict[str, Any]]:
    scenarios: List[Dict[str, Any]] = [
        {"scenario": "clean", "scenario_type": "clean", "modalities": "", "severity": 0.0}
    ]
    for name in EXPERTS:
        scenarios.append(
            {"scenario": f"missing_{name}", "scenario_type": "missing_single", "modalities": name, "severity": 1.0}
        )
    for left_index in range(len(EXPERTS)):
        for right_index in range(left_index + 1, len(EXPERTS)):
            names = f"{EXPERTS[left_index]}+{EXPERTS[right_index]}"
            scenarios.append(
                {"scenario": f"missing_{EXPERTS[left_index]}_{EXPERTS[right_index]}", "scenario_type": "missing_pair", "modalities": names, "severity": 1.0}
            )
    for name in EXPERTS:
        for severity in (0.5, 1.0):
            scenarios.append(
                {"scenario": f"noise_{name}_{severity:.1f}", "scenario_type": "gaussian_noise", "modalities": name, "severity": severity}
            )
    for name in EXPERTS:
        for severity in (0.25, 0.5):
            scenarios.append(
                {"scenario": f"mask_{name}_{severity:.2f}", "scenario_type": "feature_mask", "modalities": name, "severity": severity}
            )
    for name in EXPERTS:
        scenarios.append(
            {"scenario": f"conflict_{name}", "scenario_type": "expert_conflict", "modalities": name, "severity": 1.0}
        )
    return scenarios


def stable_rng(seed: int, scenario: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{scenario}".encode("utf-8")).digest()
    derived = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(derived)


def modality_indices(value: str) -> List[int]:
    if not value:
        return []
    names = value.split("+")
    return [EXPERTS.index(name) for name in names]


def perturb_frame(
    frame: pd.DataFrame, scenario: Dict[str, Any], seed: int
) -> Tuple[pd.DataFrame, List[int], List[int]]:
    output = frame.copy()
    indices = modality_indices(scenario["modalities"])
    missing: List[int] = []
    conflict: List[int] = []
    rng = stable_rng(seed, scenario["scenario"])
    if scenario["scenario_type"] in {"missing_single", "missing_pair"}:
        missing = indices
        for index in indices:
            column = f"features_{index}"
            output[column] = [np.zeros_like(value) for value in output[column]]
    elif scenario["scenario_type"] == "gaussian_noise":
        index = indices[0]
        column = f"features_{index}"
        severity = float(scenario["severity"])
        output[column] = [
            (value + rng.normal(0.0, severity, size=value.shape)).astype(np.float32)
            for value in output[column]
        ]
    elif scenario["scenario_type"] == "feature_mask":
        index = indices[0]
        column = f"features_{index}"
        severity = float(scenario["severity"])
        output[column] = [
            np.where(rng.random(value.shape) < severity, 0.0, value).astype(np.float32)
            for value in output[column]
        ]
    elif scenario["scenario_type"] == "expert_conflict":
        conflict = indices
    elif scenario["scenario_type"] != "clean":
        raise ValueError(f"Unknown scenario: {scenario}")
    return output, missing, conflict


def predict_with_scenario(
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
    paired.set_seed(seed)
    loader = paired.make_loader(paired.MultimodalDataset(frame), 1024, False, seed)
    criterion = nn.BCELoss()
    predictor_wrappers = [module.ModelWrapper(model, criterion) for model in predictors]
    fusion_wrapper = module.ModelWrapper(fusion_model, criterion)
    expert_pieces: List[List[np.ndarray]] = [[] for _ in predictors]
    fusion_pieces: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for features, target in loader:
        features = [value.to(device) for value in features]
        means = []
        deviations = []
        with torch.no_grad():
            for index, (wrapper, values) in enumerate(zip(predictor_wrappers, features)):
                samples = wrapper.predict_on_batch(values, iterations=mc_trials)
                mean = samples.mean(dim=-1).reshape(-1)
                deviation = samples.std(dim=-1).reshape(-1)
                if index in conflict:
                    mean = 1.0 - mean
                means.append(mean)
                deviations.append(deviation)
                expert_pieces[index].append(mean.cpu().numpy())
            fusion_samples = fusion_wrapper.predict_on_batch(
                (features, means, deviations), iterations=mc_trials
            )
            fusion_pieces.append(fusion_samples.mean(dim=-1).reshape(-1).cpu().numpy())
        labels.append(target.numpy())
    result = frame[["id", "row_id", "label"]].reset_index(drop=True).copy()
    observed = np.concatenate(labels).astype(int)
    if not np.array_equal(observed, result.label.to_numpy(dtype=int)):
        raise ValueError("Prediction order mismatch")
    for index, name in enumerate(EXPERTS):
        result[name] = np.concatenate(expert_pieces[index])
        if index in missing:
            result[name] = 0.5
    result["ufnet"] = np.concatenate(fusion_pieces)
    return result


def fit_dev_state(dev: pd.DataFrame, seed: int):
    combiners = calibrated.fit_combiners(dev, seed)
    clean_scores = robust_scores(dev, combiners, missing=[])
    state: Dict[str, Any] = {"combiners": combiners, "models": {}}
    labels = dev.label.to_numpy(dtype=int)
    for model_name, scores in clean_scores.items():
        calibrator = calibrated.fit_calibrator("platt", scores, labels, seed)
        calibrated_scores = calibrator.predict(scores)
        threshold = calibrated.select_threshold(
            "specificity_0.90", labels, calibrated_scores
        )
        state["models"][model_name] = {
            "calibrator": calibrator,
            "specificity_threshold": threshold,
        }
    return state


def robust_scores(
    frame: pd.DataFrame, combiners: Dict[str, Any], missing: Sequence[int]
) -> Dict[str, np.ndarray]:
    x = frame[list(EXPERTS)].to_numpy(dtype=float)
    available = np.ones(x.shape[1], dtype=bool)
    available[list(missing)] = False
    if not available.any():
        raise ValueError("At least one expert must remain available")
    weights = np.asarray(combiners["weights"], dtype=float)
    available_weights = weights * available
    available_weights = available_weights / available_weights.sum()
    return {
        "mean_probability": x.mean(axis=1),
        "available_mean": x[:, available].mean(axis=1),
        "dev_auroc_weighted": x @ weights,
        "available_weighted": x @ available_weights,
        "logistic_stacking": combiners["logistic"].predict_proba(x)[:, 1],
        "tree_stacking": combiners["tree"].predict_proba(x)[:, 1],
        "ufnet": frame.ufnet.to_numpy(dtype=float),
    }


def evaluate_scores(
    dataset_name: str,
    seed: int,
    scenario: Dict[str, Any],
    level: str,
    frame: pd.DataFrame,
    missing: Sequence[int],
    dev_state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    labels = frame.label.to_numpy(dtype=int)
    scores = robust_scores(frame, dev_state["combiners"], missing)
    for model_name in ROBUST_MODELS:
        raw = scores[model_name]
        calibrator = dev_state["models"][model_name]["calibrator"]
        platt = calibrator.predict(raw)
        for mode in EVALUATION_MODES:
            if mode == "raw_fixed_0.5":
                used_scores = raw
                threshold = 0.5
            elif mode == "platt_fixed_0.5":
                used_scores = platt
                threshold = 0.5
            else:
                used_scores = platt
                threshold = dev_state["models"][model_name]["specificity_threshold"]
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
                    **calibrated.compute_metrics(labels, used_scores, threshold),
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
    run_output: Path,
) -> List[Dict[str, Any]]:
    clean_dev_frame = frame.loc[masks["dev"]].reset_index(drop=True)
    clean_dev_prediction = predict_with_scenario(
        module, clean_dev_frame, predictors, fusion_model, device, mc_trials,
        seed * 100000 + 1, [], []
    )
    dev_states = {
        level: fit_dev_state(calibrated.aggregate_level(clean_dev_prediction, level), seed)
        for level in ("session", "participant")
    }
    test_frame = frame.loc[masks["internal_test"]].reset_index(drop=True)
    rows: List[Dict[str, Any]] = []
    prediction_rows: List[pd.DataFrame] = []
    for scenario_index, scenario in enumerate(scenarios):
        perturbed, missing, conflict = perturb_frame(test_frame, scenario, seed)
        prediction = predict_with_scenario(
            module, perturbed, predictors, fusion_model, device, mc_trials,
            seed * 100000 + 1000 + scenario_index, missing, conflict
        )
        compact = prediction.copy()
        compact.insert(0, "scenario", scenario["scenario"])
        prediction_rows.append(compact)
        for level in ("session", "participant"):
            evaluated = calibrated.aggregate_level(prediction, level)
            rows.extend(
                evaluate_scores(
                    dataset_name, seed, scenario, level, evaluated, missing, dev_states[level]
                )
            )
    pd.DataFrame(rows).to_csv(run_output / "metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        run_output / "scenario_predictions.csv", index=False
    )
    with (run_output / "dev_states.pkl").open("wb") as handle:
        import pickle

        pickle.dump(dev_states, handle)
    with (run_output / "run_complete.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": dataset_name,
                "seed": seed,
                "mc_trials": mc_trials,
                "scenarios": len(scenarios),
                "metric_rows": len(rows),
            },
            handle,
            indent=2,
        )
    return rows


def add_clean_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "seed", "level", "model", "evaluation_mode"]
    clean = metrics[metrics.scenario == "clean"][
        [*keys, "auroc", "average_precision", "accuracy", "sensitivity", "specificity", "brier", "ece"]
    ].rename(
        columns={
            metric: f"clean_{metric}"
            for metric in ("auroc", "average_precision", "accuracy", "sensitivity", "specificity", "brier", "ece")
        }
    )
    merged = metrics.merge(clean, on=keys, how="left", validate="many_to_one")
    for metric in ("auroc", "average_precision", "accuracy", "sensitivity", "specificity"):
        merged[f"{metric}_delta_from_clean"] = merged[metric] - merged[f"clean_{metric}"]
    for metric in ("brier", "ece"):
        merged[f"{metric}_delta_from_clean"] = merged[metric] - merged[f"clean_{metric}"]
    return merged


def write_report(path: Path, metrics: pd.DataFrame) -> None:
    target = metrics[
        (metrics.level == "participant")
        & (metrics.evaluation_mode == "raw_fixed_0.5")
    ]
    lines = [
        "# PARK missing/corrupted modality robustness report",
        "",
        "Perturbations are deterministic per seed and applied only to the pooled internal "
        "test cohort. All classical combiners and calibration choices originate from the "
        "clean Dev partition.",
        "",
        "## Clean discrimination and mean/worst AUROC loss",
        "",
        "| Dataset | Model | Clean AUROC | Single-missing mean delta | Single-missing worst delta | Pair-missing worst delta | Noise worst delta | Conflict worst delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in sorted(target.dataset.unique()):
        for model in ROBUST_MODELS:
            selected = target[(target.dataset == dataset) & (target.model == model)]
            clean = selected[selected.scenario == "clean"].auroc.mean()
            missing_single = selected[selected.scenario_type == "missing_single"]
            missing_pair = selected[selected.scenario_type == "missing_pair"]
            noise = selected[selected.scenario_type == "gaussian_noise"]
            conflict = selected[selected.scenario_type == "expert_conflict"]
            single_by_scenario = missing_single.groupby("scenario").auroc_delta_from_clean.mean()
            pair_by_scenario = missing_pair.groupby("scenario").auroc_delta_from_clean.mean()
            noise_by_scenario = noise.groupby("scenario").auroc_delta_from_clean.mean()
            conflict_by_scenario = conflict.groupby("scenario").auroc_delta_from_clean.mean()
            lines.append(
                f"| {dataset} | {model} | {clean:.4f} | "
                f"{missing_single.auroc_delta_from_clean.mean():+.4f} | "
                f"{single_by_scenario.min():+.4f} | "
                f"{pair_by_scenario.min():+.4f} | "
                f"{noise_by_scenario.min():+.4f} | "
                f"{conflict_by_scenario.min():+.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Missing features are replaced by zero after StandardScaler, which represents the training mean rather than an explicit missingness token.",
            "- Classical fixed-input combiners receive a neutral probability of 0.5 for missing experts; availability-aware variants renormalize over observed experts.",
            "- UFNet has no missingness mask, so this benchmark measures its native behavior without retraining.",
            "- Gaussian noise severity is measured in post-scaling standard-deviation units.",
            "- Expert conflict flips one probability as `1-p` while retaining its uncertainty, intentionally creating an inconsistent adversarial signal.",
            "- These stress tests diagnose failure modes; they do not simulate every real sensor artifact.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate_outputs(output_dir: Path) -> None:
    files = sorted(output_dir.glob("*/seed_*/metrics.csv"))
    if not files:
        return
    metrics = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    metrics = add_clean_deltas(metrics)
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
        output_dir / "robustness_summary.csv", index=False
    )
    write_report(output_dir / "ROBUSTNESS_REPORT.md", metrics)


def main() -> None:
    args = parse_args()
    if args.mc_trials < 2:
        raise ValueError("--mc-trials must be at least 2")
    repo_root = args.repo_root.resolve()
    data_dir = (args.data_dir or repo_root / "results" / "protocol_alignment_audit").resolve()
    training_dir = (args.training_dir or repo_root / "results" / "paired_retraining").resolve()
    output_dir = (args.output_dir or repo_root / "results" / "modality_robustness").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = paired.parse_csv_list(args.datasets)
    seeds = paired.parse_seeds(args.seeds)
    device = ev.resolve_device(args.device)
    scenarios = scenario_definitions()
    pd.DataFrame(scenarios).to_csv(output_dir / "scenario_definitions.csv", index=False)

    module = ev.load_upstream_module(repo_root)
    fusion_config = ev.read_json(Path(module.MODEL_CONFIG_PATH))
    selected_models = module.MODEL_SUBSETS[int(fusion_config["model_subset_choice"])]
    module.NUM_MODELS = len(selected_models)
    paths = ev.checkpoint_paths(module, selected_models)
    configs = [ev.read_json(item["config"]) for item in paths]
    protected_paths = [
        path for item in paths for key, path in item.items() if key in {"model", "scaler"}
    ] + [Path(module.MODEL_PATH)]
    hashes_before = {str(path.relative_to(repo_root)): ev.sha256_file(path) for path in protected_paths}

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
                print(f"Skipping completed robustness run: {dataset_name} seed={seed}")
                continue
            scaled = calibrated.apply_run_scalers(raw, training_run, configs)
            feature_shapes = [len(scaled.iloc[0][f"features_{index}"]) for index in range(3)]
            predictors, fusion_model = calibrated.load_models(
                module, training_run, configs, fusion_config, feature_shapes, device
            )
            print(f"Robustness evaluation {dataset_name} seed={seed}")
            evaluate_run(
                module, dataset_name, seed, scaled, masks, predictors, fusion_model,
                device, args.mc_trials, scenarios, run_output
            )
            aggregate_outputs(output_dir)

    hashes_after = {path: ev.sha256_file(repo_root / path) for path in hashes_before}
    if hashes_before != hashes_after:
        raise RuntimeError("A protected upstream checkpoint changed during robustness evaluation")
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
                    "scenarios": scenarios,
                    "models": ROBUST_MODELS,
                    "evaluation_modes": EVALUATION_MODES,
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
