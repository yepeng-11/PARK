#!/usr/bin/env python3
"""Evaluate calibrated classical fusion baselines for paired PARK runs.

All combiners, probability calibrators, and operating thresholds are fitted on
the development partition only. Test labels are used exclusively for metrics.
The script reads saved paired-retraining models and writes to a separate output
tree without modifying upstream data, checkpoints, or training artifacts.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import evaluate_pretrained as ev
import train_paired_baselines as paired


EXPERT_COLUMNS = ("finger", "speech", "smile")
TEST_SPLITS = ("internal_test", "validation_1", "validation_2", "global")
CALIBRATIONS = ("raw", "platt", "isotonic")
THRESHOLD_RULES = ("fixed_0.5", "youden", "specificity_0.80", "specificity_0.90", "specificity_0.95")


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
        help="Default: <repo>/results/calibrated_fusion",
    )
    parser.add_argument("--datasets", default="official,cleaned")
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--mc-trials", type=int, default=30)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def apply_run_scalers(
    frame: pd.DataFrame, run_dir: Path, configs: Sequence[Dict[str, Any]]
) -> pd.DataFrame:
    output = frame.copy()
    for index, (name, config) in enumerate(zip(EXPERT_COLUMNS, configs)):
        column = f"features_{index}"
        matrix = np.stack(output[column]).astype(np.float64)
        with (run_dir / f"scaler_{name}.pkl").open("rb") as handle:
            scaler = pickle.load(handle)
        if config["use_feature_scaling"] == "yes":
            if scaler is None:
                raise ValueError(f"Missing fitted scaler for {name}: {run_dir}")
            matrix = scaler.transform(matrix)
        elif scaler is not None:
            raise ValueError(f"Unexpected scaler for unscaled modality {name}")
        output[column] = list(matrix.astype(np.float32))
    return output


def load_models(
    module,
    run_dir: Path,
    configs: Sequence[Dict[str, Any]],
    fusion_config: Dict[str, Any],
    feature_shapes: Sequence[int],
    device: torch.device,
):
    predictors = []
    for index, (name, config) in enumerate(zip(EXPERT_COLUMNS, configs)):
        model = paired.instantiate_predictor(module, config, feature_shapes[index])
        model.load_state_dict(torch.load(run_dir / name / "model.pth", map_location="cpu"))
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        predictors.append(model)
    fusion = module.HybridFusionNetworkWithUncertainty(feature_shapes, fusion_config)
    fusion.load_state_dict(torch.load(run_dir / "fusion" / "model.pth", map_location="cpu"))
    fusion.to(device).eval()
    for parameter in fusion.parameters():
        parameter.requires_grad = False
    return predictors, fusion


def predict_partition(
    module,
    frame: pd.DataFrame,
    predictors: Sequence[torch.nn.Module],
    fusion_model: torch.nn.Module,
    device: torch.device,
    mc_trials: int,
    seed: int,
) -> pd.DataFrame:
    result = frame[["id", "row_id", "label"]].reset_index(drop=True).copy()
    for index, (name, model) in enumerate(zip(EXPERT_COLUMNS, predictors)):
        paired.set_seed(seed + index * 101)
        dataset = paired.UnimodalDataset(
            np.stack(frame[f"features_{index}"]).astype(np.float32),
            frame.label.to_numpy(dtype=np.float32),
        )
        loader = paired.make_loader(dataset, 1024, False, seed)
        labels, scores, _ = paired.mc_unimodal_scores(
            module, model, loader, device, mc_trials
        )
        if not np.array_equal(labels.astype(int), result.label.to_numpy(dtype=int)):
            raise ValueError(f"Prediction order mismatch for {name}")
        result[name] = scores

    paired.set_seed(seed + 10000)
    loader = paired.make_loader(paired.MultimodalDataset(frame), 1024, False, seed)
    labels, scores, _ = paired.evaluate_fusion(
        module, fusion_model, predictors, loader, device, mc_trials
    )
    if not np.array_equal(labels.astype(int), result.label.to_numpy(dtype=int)):
        raise ValueError("Prediction order mismatch for UFNet")
    result["ufnet"] = scores
    return result


def aggregate_level(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    score_columns = [*EXPERT_COLUMNS, "ufnet"]
    if level == "session":
        return frame[["id", "row_id", "label", *score_columns]].reset_index(drop=True)
    label_counts = frame.groupby("id").label.nunique()
    if (label_counts > 1).any():
        raise ValueError("A participant has inconsistent labels")
    aggregations: Dict[str, Any] = {"label": "first", **{column: "mean" for column in score_columns}}
    return frame.groupby("id", as_index=False).agg(aggregations)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else 0.5


def fit_combiners(dev: pd.DataFrame, seed: int):
    x = dev[list(EXPERT_COLUMNS)].to_numpy(dtype=float)
    y = dev.label.to_numpy(dtype=int)
    aucs = np.asarray([safe_auc(y, x[:, index]) for index in range(x.shape[1])])
    weights = np.maximum(aucs - 0.5, 1e-6)
    weights = weights / weights.sum()
    logistic = LogisticRegression(
        class_weight="balanced", C=1.0, solver="lbfgs", max_iter=1000, random_state=seed
    ).fit(x, y)
    tree = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=100,
        max_depth=2,
        min_samples_leaf=max(5, len(dev) // 20),
        l2_regularization=1.0,
        random_state=seed,
    ).fit(x, y)
    return {"weights": weights, "logistic": logistic, "tree": tree, "dev_auroc": aucs}


def combined_scores(frame: pd.DataFrame, combiners: Dict[str, Any]) -> Dict[str, np.ndarray]:
    x = frame[list(EXPERT_COLUMNS)].to_numpy(dtype=float)
    binary = (x >= 0.5).astype(float)
    return {
        "finger": x[:, 0],
        "speech": x[:, 1],
        "smile": x[:, 2],
        "ufnet": frame.ufnet.to_numpy(dtype=float),
        "mean_probability": x.mean(axis=1),
        "majority_vote": binary.mean(axis=1),
        "dev_auroc_weighted": x @ combiners["weights"],
        "logistic_stacking": combiners["logistic"].predict_proba(x)[:, 1],
        "tree_stacking": combiners["tree"].predict_proba(x)[:, 1],
    }


class IdentityCalibrator:
    def fit(self, scores: np.ndarray, labels: np.ndarray):
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, dtype=float)


class PlattCalibrator:
    def __init__(self, seed: int):
        self.model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)

    @staticmethod
    def logits(scores: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6)
        return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        self.model.fit(self.logits(scores), labels)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.logits(scores))[:, 1]


class IsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        self.model.fit(scores, labels)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(scores), dtype=float)


def fit_calibrator(name: str, scores: np.ndarray, labels: np.ndarray, seed: int):
    if name == "raw":
        calibrator = IdentityCalibrator()
    elif name == "platt":
        calibrator = PlattCalibrator(seed)
    elif name == "isotonic":
        calibrator = IsotonicCalibrator()
    else:
        raise ValueError(name)
    return calibrator.fit(scores, labels)


def confusion_rates(labels: np.ndarray, scores: np.ndarray, threshold: float):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return sensitivity, specificity


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(scores, dtype=float))
    return np.concatenate(([np.nextafter(1.0, 2.0)], unique[::-1], [np.nextafter(0.0, -1.0)]))


def select_threshold(rule: str, labels: np.ndarray, scores: np.ndarray) -> float:
    if rule == "fixed_0.5":
        return 0.5
    candidates = candidate_thresholds(scores)
    values = []
    for threshold in candidates:
        sensitivity, specificity = confusion_rates(labels, scores, float(threshold))
        values.append((float(threshold), sensitivity, specificity))
    if rule == "youden":
        best_value = max(sensitivity + specificity - 1.0 for _, sensitivity, specificity in values)
        tied = [row for row in values if np.isclose(row[1] + row[2] - 1.0, best_value)]
        return min(tied, key=lambda row: abs(row[0] - 0.5))[0]
    if rule.startswith("specificity_"):
        target = float(rule.split("_", 1)[1])
        valid = [row for row in values if row[2] + 1e-12 >= target]
        best_sensitivity = max(row[1] for row in valid)
        tied = [row for row in valid if np.isclose(row[1], best_sensitivity)]
        return min(row[0] for row in tied)
    raise ValueError(rule)


def compute_metrics(labels: Iterable[int], scores: Iterable[float], threshold: float) -> Dict[str, Any]:
    labels = np.asarray(list(labels), dtype=int)
    scores = np.clip(np.asarray(list(scores), dtype=float), 0.0, 1.0)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = recall_score(labels, predictions, zero_division=0)
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = precision_score(labels, predictions, zero_division=0)
    npv = tn / (tn + fn) if tn + fn else 0.0
    return {
        "n": len(labels),
        "positives": int(labels.sum()),
        "threshold": threshold,
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "auroc": roc_auc_score(labels, scores),
        "average_precision": average_precision_score(labels, scores),
        "f1": f1_score(labels, predictions, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "brier": brier_score_loss(labels, scores),
        "ece": ev.calibration_error(labels, scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


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
    run_output: Path,
):
    raw_predictions: Dict[str, pd.DataFrame] = {}
    for offset, split in enumerate(("dev", *TEST_SPLITS)):
        selected = frame.loc[masks[split]].reset_index(drop=True)
        prediction = predict_partition(
            module, selected, predictors, fusion_model, device, mc_trials,
            seed * 1000 + offset * 100,
        )
        raw_predictions[split] = prediction
        prediction.to_csv(run_output / f"raw_predictions_{split}.csv", index=False)

    metric_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []
    fitted_objects: Dict[str, Any] = {}
    for level_index, level in enumerate(("session", "participant")):
        dev = aggregate_level(raw_predictions["dev"], level)
        combiners = fit_combiners(dev, seed + level_index)
        dev_models = combined_scores(dev, combiners)
        fitted_objects[level] = {"combiners": combiners, "calibrators": {}, "thresholds": {}}
        for model_name, dev_raw_score in dev_models.items():
            fitted_objects[level]["calibrators"][model_name] = {}
            fitted_objects[level]["thresholds"][model_name] = {}
            for calibration in CALIBRATIONS:
                calibrator = fit_calibrator(
                    calibration, dev_raw_score, dev.label.to_numpy(dtype=int), seed + level_index
                )
                dev_score = calibrator.predict(dev_raw_score)
                fitted_objects[level]["calibrators"][model_name][calibration] = calibrator
                fitted_objects[level]["thresholds"][model_name][calibration] = {}
                for rule in THRESHOLD_RULES:
                    threshold = select_threshold(
                        rule, dev.label.to_numpy(dtype=int), dev_score
                    )
                    dev_sensitivity, dev_specificity = confusion_rates(
                        dev.label.to_numpy(dtype=int), dev_score, threshold
                    )
                    fitted_objects[level]["thresholds"][model_name][calibration][rule] = threshold
                    threshold_rows.append(
                        {
                            "dataset": dataset_name,
                            "seed": seed,
                            "level": level,
                            "model": model_name,
                            "calibration": calibration,
                            "threshold_rule": rule,
                            "threshold": threshold,
                            "dev_sensitivity": dev_sensitivity,
                            "dev_specificity": dev_specificity,
                            "expert_weight_finger": combiners["weights"][0],
                            "expert_weight_speech": combiners["weights"][1],
                            "expert_weight_smile": combiners["weights"][2],
                        }
                    )
                    for split in TEST_SPLITS:
                        test = aggregate_level(raw_predictions[split], level)
                        test_raw_score = combined_scores(test, combiners)[model_name]
                        test_score = calibrator.predict(test_raw_score)
                        metric_rows.append(
                            {
                                "dataset": dataset_name,
                                "seed": seed,
                                "split": split,
                                "level": level,
                                "model": model_name,
                                "calibration": calibration,
                                "threshold_rule": rule,
                                **compute_metrics(test.label, test_score, threshold),
                            }
                        )
    pd.DataFrame(metric_rows).to_csv(run_output / "metrics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(run_output / "thresholds.csv", index=False)
    with (run_output / "fitted_dev_models.pkl").open("wb") as handle:
        pickle.dump(fitted_objects, handle)
    with (run_output / "run_complete.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": dataset_name,
                "seed": seed,
                "mc_trials": mc_trials,
                "metric_rows": len(metric_rows),
                "threshold_rows": len(threshold_rows),
            },
            handle,
            indent=2,
        )
    return metric_rows, threshold_rows


def flatten_aggregated_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else column
        for column in frame.columns
    ]
    return frame


def write_report(path: Path, metrics: pd.DataFrame) -> None:
    primary = metrics[
        (metrics.split == "internal_test")
        & (metrics.level == "participant")
        & (metrics.calibration == "raw")
        & (metrics.threshold_rule == "fixed_0.5")
    ]
    operating = metrics[
        (metrics.split == "internal_test")
        & (metrics.level == "participant")
        & (metrics.calibration == "platt")
        & (metrics.threshold_rule == "specificity_0.90")
    ]
    lines = [
        "# PARK calibrated fusion baseline report",
        "",
        "All combiners, calibrators, and thresholds were fitted using Dev only. "
        "The test labels were not used for model fitting or operating-point selection.",
        "",
        "## Raw participant-level internal-test discrimination",
        "",
        "| Dataset | Model | AUROC | AUPRC | Accuracy@0.5 | ECE | Brier |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    model_order = [*EXPERT_COLUMNS, "mean_probability", "majority_vote", "dev_auroc_weighted", "logistic_stacking", "tree_stacking", "ufnet"]
    for dataset in ("official", "cleaned"):
        for model in model_order:
            selected = primary[(primary.dataset == dataset) & (primary.model == model)]
            if selected.empty:
                continue
            lines.append(
                f"| {dataset} | {model} | {selected.auroc.mean():.4f} | "
                f"{selected.average_precision.mean():.4f} | {selected.accuracy.mean():.4f} | "
                f"{selected.ece.mean():.4f} | {selected.brier.mean():.4f} |"
            )
    lines.extend(
        [
            "",
            "## Calibration effect at the default 0.5 threshold",
            "",
            "| Dataset | Model | Calibration | AUROC | Brier | ECE | Accuracy |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    calibration_models = ["mean_probability", "dev_auroc_weighted", "logistic_stacking", "ufnet"]
    calibrated = metrics[
        (metrics.split == "internal_test")
        & (metrics.level == "participant")
        & (metrics.threshold_rule == "fixed_0.5")
    ]
    for dataset in ("official", "cleaned"):
        for model in calibration_models:
            for calibration in CALIBRATIONS:
                selected = calibrated[
                    (calibrated.dataset == dataset)
                    & (calibrated.model == model)
                    & (calibrated.calibration == calibration)
                ]
                lines.append(
                    f"| {dataset} | {model} | {calibration} | {selected.auroc.mean():.4f} | "
                    f"{selected.brier.mean():.4f} | {selected.ece.mean():.4f} | "
                    f"{selected.accuracy.mean():.4f} |"
                )
    lines.extend(
        [
            "",
            "## Platt-calibrated Dev-specificity-0.90 operating point",
            "",
            "| Dataset | Model | Test sensitivity | Test specificity | Accuracy |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for dataset in ("official", "cleaned"):
        for model in model_order:
            selected = operating[(operating.dataset == dataset) & (operating.model == model)]
            if selected.empty:
                continue
            lines.append(
                f"| {dataset} | {model} | {selected.sensitivity.mean():.4f} | "
                f"{selected.specificity.mean():.4f} | {selected.accuracy.mean():.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- AUROC and AUPRC compare discrimination independently of the operating threshold.",
            "- Fixed-specificity thresholds are selected on Dev; achieved test specificity may differ because of sampling and distribution shift.",
            "- Isotonic calibration is included as a sensitivity analysis but can overfit the small Dev cohort.",
            "- Tree stacking uses scikit-learn HistGradientBoosting because XGBoost is not installed in the validated PARK environment.",
            "- These five-seed results are development evidence; final claims should use 30 seeds and confidence intervals.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate_outputs(output_dir: Path) -> None:
    metric_files = sorted(output_dir.glob("*/seed_*/metrics.csv"))
    threshold_files = sorted(output_dir.glob("*/seed_*/thresholds.csv"))
    if not metric_files:
        return
    metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    thresholds = pd.concat([pd.read_csv(path) for path in threshold_files], ignore_index=True)
    metrics.to_csv(output_dir / "metrics_per_run.csv", index=False)
    thresholds.to_csv(output_dir / "thresholds_per_run.csv", index=False)
    numeric = [
        "threshold", "accuracy", "balanced_accuracy", "auroc", "average_precision",
        "f1", "sensitivity", "specificity", "precision", "npv", "brier", "ece",
    ]
    summary = metrics.groupby(
        ["dataset", "split", "level", "model", "calibration", "threshold_rule"],
        as_index=False,
    )[numeric].agg(["mean", "std"])
    flatten_aggregated_columns(summary).to_csv(output_dir / "metrics_summary.csv", index=False)
    write_report(output_dir / "CALIBRATION_REPORT.md", metrics)


def main() -> None:
    args = parse_args()
    if args.mc_trials < 2:
        raise ValueError("--mc-trials must be at least 2")
    repo_root = args.repo_root.resolve()
    data_dir = (args.data_dir or repo_root / "results" / "protocol_alignment_audit").resolve()
    training_dir = (args.training_dir or repo_root / "results" / "paired_retraining").resolve()
    output_dir = (args.output_dir or repo_root / "results" / "calibrated_fusion").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = paired.parse_csv_list(args.datasets)
    seeds = paired.parse_seeds(args.seeds)
    device = ev.resolve_device(args.device)

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
        source = data_dir / f"{dataset_name}_aligned.csv"
        exported = paired.load_vector_csv(source)
        masks = paired.split_masks(module, exported)
        raw = paired.inverse_original_scaling(exported, configs, paths)
        for seed in seeds:
            training_run = training_dir / dataset_name / f"seed_{seed}"
            if not (training_run / "run_complete.json").exists():
                raise FileNotFoundError(f"Incomplete training run: {training_run}")
            run_output = output_dir / dataset_name / f"seed_{seed}"
            run_output.mkdir(parents=True, exist_ok=True)
            if (run_output / "run_complete.json").exists() and not args.force:
                print(f"Skipping completed evaluation: {dataset_name} seed={seed}")
                continue
            scaled = apply_run_scalers(raw, training_run, configs)
            feature_shapes = [len(scaled.iloc[0][f"features_{index}"]) for index in range(3)]
            predictors, fusion_model = load_models(
                module, training_run, configs, fusion_config, feature_shapes, device
            )
            print(f"Evaluating {dataset_name} seed={seed}")
            evaluate_run(
                module, dataset_name, seed, scaled, masks, predictors, fusion_model,
                device, args.mc_trials, run_output
            )
            aggregate_outputs(output_dir)

    hashes_after = {
        path: ev.sha256_file(repo_root / path) for path in hashes_before
    }
    if hashes_before != hashes_after:
        raise RuntimeError("A protected upstream checkpoint changed during evaluation")
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
                    "models": [*EXPERT_COLUMNS, "ufnet", "mean_probability", "majority_vote", "dev_auroc_weighted", "logistic_stacking", "tree_stacking"],
                    "calibrations": CALIBRATIONS,
                    "threshold_rules": THRESHOLD_RULES,
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
