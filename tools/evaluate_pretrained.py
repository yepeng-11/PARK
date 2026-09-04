#!/usr/bin/env python3
"""Non-destructive evaluation of the pretrained PARK/UFNet checkpoints.

The upstream training script rewrites checkpoints, intermediate CSVs, and
``data/predictions.csv`` during evaluation.  This utility loads the same data,
scalers, model classes, and checkpoints without changing upstream artifacts.

It reports both session-level and participant-level metrics.  For fusion, it
also reports the repository's original abstention rule and a corrected rule:
the original implementation divides the Monte-Carlo standard deviation by the
square root of the batch size, whereas a confidence interval for the MC mean
must use the square root of the number of MC trials.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import pickle
import random
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
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
from torch import nn
from torch.utils.data import DataLoader


MODALITY_NAMES = ("finger", "speech", "smile")
PAPER_EXACT_SPLIT_COUNTS = {"global": 162, "validation_1": 91, "validation_2": 67}
PAPER_EXACT_MANIFEST_SHA256 = (
    "6f8424c91357d68042a6012498256d98a5541b21f61cf7d79b565db359122c34"
)
SPLIT_SEED_OFFSETS = {
    "internal_test": 0,
    "validation_1": 1,
    "validation_2": 2,
    "global": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root of the PARK repository.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/results/pretrained_eval).",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="MC-dropout trials; default uses the fusion checkpoint config (30).",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=289)
    parser.add_argument(
        "--protocol",
        choices=("source", "paper-exact"),
        default="source",
        help="Evaluate source cohorts or the frozen 162/91/67 paper cohorts.",
    )
    parser.add_argument(
        "--paper-manifest",
        type=Path,
        default=None,
        help=(
            "Frozen cohort manifest; default: "
            "<repo>/results/paper_score_provenance/paper_exact_cohort_manifest.csv"
        ),
    )
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def load_upstream_module(repo_root: Path):
    """Import upstream fusion code with the working directory it assumes."""
    module_dir = (repo_root / "code" / "fusion_model").resolve()
    if not module_dir.is_dir():
        raise FileNotFoundError(f"Fusion module directory not found: {module_dir}")
    os.chdir(module_dir)
    sys.path.insert(0, str(module_dir))
    return importlib.import_module("uncertainty_aware_fusion_wavlm")


def checkpoint_paths(module, selected_models: Sequence[str]) -> List[Dict[str, Path]]:
    paths: List[Dict[str, Path]] = []
    for model_name in selected_models:
        base = Path(module.MODEL_BASE_PATH) / model_name
        item = {
            "config": base / "predictive_model" / "model_config.json",
            "model": base / "predictive_model" / "model.pth",
            "scaler": base / "scaler" / "scaler.pth",
        }
        for path in item.values():
            if not path.exists():
                raise FileNotFoundError(path)
        paths.append(item)
    return paths


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_aligned_dataframe(module, selected_models, paths):
    """Reproduce upstream modality preprocessing without writing intermediate CSVs."""
    processed: List[pd.DataFrame] = []
    configs: List[Dict[str, Any]] = []
    alignment_audit: List[Dict[str, Any]] = []

    for index, (model_name, model_paths) in enumerate(zip(selected_models, paths)):
        config = read_json(model_paths["config"])
        configs.append(config)
        drop_correlated = config["drop_correlated"] == "yes"

        if "finger_model" in model_name:
            right = module.load_finger_data(
                drop_correlated=drop_correlated,
                corr_thr=config["corr_thr"],
                hand="right",
            )
            left = module.load_finger_data(
                drop_correlated=drop_correlated,
                corr_thr=config["corr_thr"],
                hand="left",
            )
            features_right, labels_right, ids_right, _, rows_right = right
            features_left, labels_left, ids_left, _, rows_left = left
            df_right = pd.DataFrame(
                {
                    "features_right": list(features_right),
                    "id_right": list(ids_right),
                    "row_id": list(rows_right),
                    "label_right": list(labels_right),
                }
            )
            df_left = pd.DataFrame(
                {
                    "features_left": list(features_left),
                    "id_left": list(ids_left),
                    "row_id": list(rows_left),
                    "label_left": list(labels_left),
                }
            )
            both = pd.merge(df_right, df_left, how="inner", on="row_id")
            if not np.array_equal(both["label_right"], both["label_left"]):
                raise ValueError("Finger left/right labels disagree after alignment")
            if not np.array_equal(both["id_right"], both["id_left"]):
                raise ValueError("Finger left/right participant IDs disagree after alignment")
            features = np.stack(
                [np.concatenate((r, l)) for r, l in zip(both.features_right, both.features_left)]
            )
            labels = both.label_right.to_numpy()
            ids = both.id_right.to_numpy()
            row_ids = both.row_id.to_numpy()
        elif "fox_model" in model_name:
            features, labels, ids, _, row_ids = module.load_qbf_data(
                drop_correlated=drop_correlated, corr_thr=config["corr_thr"]
            )
        elif "facial_expression_smile" in model_name:
            features, labels, ids, _, row_ids = module.load_smile_data(
                drop_correlated=drop_correlated, corr_thr=config["corr_thr"]
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        features = np.asarray(features)
        labels = np.asarray(labels)
        ids = np.asarray(ids)
        row_ids = np.asarray(row_ids)
        if config["use_feature_scaling"] == "yes":
            with model_paths["scaler"].open("rb") as handle:
                scaler = pickle.load(handle)
            features = scaler.transform(features)

        current = pd.DataFrame(
            {
                f"features_{index}": list(features),
                f"label_{index}": labels,
                f"id_{index}": ids,
                "row_id": row_ids,
            }
        )
        if index:
            current = pd.merge(processed[-1], current, on="row_id", how="inner")
            label_mismatch = current[f"label_{index}"] != current["label_0"]
            id_mismatch = current[f"id_{index}"] != current["id_0"]
            if id_mismatch.any():
                raise ValueError(
                    f"Cross-modality alignment mismatch for {model_name}: "
                    f"labels={int(label_mismatch.sum())}, ids={int(id_mismatch.sum())}"
                )
            mismatch_rows = int(label_mismatch.sum())
            mismatch_participants = int(current.loc[label_mismatch, "id_0"].nunique())
            alignment_audit.append(
                {
                    "model": model_name,
                    "overlapping_rows": len(current),
                    "label_mismatch_rows": mismatch_rows,
                    "label_mismatch_participants": mismatch_participants,
                }
            )
            if mismatch_rows:
                warnings.warn(
                    f"{model_name} has {mismatch_rows} labels that disagree with the "
                    "first (finger) modality after row_id alignment; preserving label_0 "
                    "to reproduce the upstream fusion pipeline.",
                    RuntimeWarning,
                )
            current = current.drop(columns=[f"label_{index}", f"id_{index}"])
        processed.append(current)

    aligned = processed[-1].rename(columns={"label_0": "label", "id_0": "id"})
    duplicate_rows = int(aligned.row_id.duplicated().sum())
    alignment_audit.append(
        {
            "model": "final_merge",
            "overlapping_rows": len(aligned),
            "unique_row_ids": int(aligned.row_id.nunique()),
            "duplicate_row_id_rows": duplicate_rows,
        }
    )
    if duplicate_rows:
        warnings.warn(
            f"The upstream modality joins produce {duplicate_rows} duplicate row_id "
            "rows; preserving them to reproduce the shipped full_fusion_dataset.csv.",
            RuntimeWarning,
        )
    return aligned.reset_index(drop=True), configs, alignment_audit


def make_splits(module, dataframe: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    split_ids = {
        "internal_test": module.test_ids,
        "validation_1": module.test_ids_validation_1,
        "validation_2": module.test_ids_validation_2,
        "global": module.test_ids_global,
    }
    return {
        name: dataframe[dataframe.id.isin(ids)].reset_index(drop=True)
        for name, ids in split_ids.items()
    }


def load_paper_exact_manifest(path: Path) -> Tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen paper manifest not found: {path}. Run "
            "tools/trace_paper_score_provenance.py first."
        )
    digest = sha256_file(path)
    if digest != PAPER_EXACT_MANIFEST_SHA256:
        raise ValueError(
            "Frozen paper manifest hash mismatch: "
            f"expected {PAPER_EXACT_MANIFEST_SHA256}, observed {digest}"
        )
    manifest = pd.read_csv(path)
    required = {"paper_order", "split", "row_id", "id", "label"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Frozen paper manifest missing columns: {sorted(missing)}")
    if manifest.duplicated(["split", "row_id"]).any():
        raise ValueError("Frozen paper manifest contains duplicate (split, row_id) keys")
    counts = manifest.split.value_counts().to_dict()
    if counts != PAPER_EXACT_SPLIT_COUNTS:
        raise ValueError(f"Unexpected frozen paper cohort counts: {counts}")
    return manifest, digest


def apply_paper_exact_manifest(
    predictions: pd.DataFrame, split_name: str, manifest: pd.DataFrame
) -> pd.DataFrame:
    reference = manifest.loc[
        manifest.split == split_name, ["paper_order", "row_id", "id", "label"]
    ]
    joined = reference.merge(
        predictions,
        on="row_id",
        how="left",
        suffixes=("_manifest", ""),
        validate="one_to_one",
    )
    if len(joined) != len(reference) or joined.fusion_score.isna().any():
        raise ValueError(f"Incomplete paper-exact prediction join for {split_name}")
    if not (joined.id_manifest.astype(str) == joined.id.astype(str)).all():
        raise ValueError(f"Participant mismatch against frozen manifest in {split_name}")
    if not (joined.label_manifest.astype(int) == joined.label.astype(int)).all():
        raise ValueError(f"Label mismatch against frozen manifest in {split_name}")
    return (
        joined.drop(columns=["id_manifest", "label_manifest"])
        .sort_values("paper_order")
        .reset_index(drop=True)
    )


def instantiate_models(module, selected_models, paths, configs, feature_shapes, device):
    prediction_models = []
    for model_name, model_paths, config, n_features in zip(
        selected_models, paths, configs, feature_shapes
    ):
        if config["model"] == "ShallowANN":
            model = module.ShallowANN(n_features, drop_prob=config["dropout_prob"])
        elif config["model"] == "ANN":
            model = module.ANN(n_features, drop_prob=config["dropout_prob"])
        else:
            raise ValueError(f"Unsupported predictor class for {model_name}: {config['model']}")
        state = torch.load(model_paths["model"], map_location="cpu")
        model.load_state_dict(state)
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        prediction_models.append(model)
    return prediction_models


def predict_split(
    module,
    dataframe: pd.DataFrame,
    prediction_models: Sequence[nn.Module],
    fusion_model: nn.Module,
    device: torch.device,
    batch_size: int,
    num_trials: int,
    seed: int,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = module.TensorDataset(dataframe)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.BCELoss()
    predictor_wrappers = [module.ModelWrapper(model, criterion) for model in prediction_models]
    fusion_wrapper = module.ModelWrapper(fusion_model, criterion)
    t_critical = stats.t.ppf(q=0.975, df=num_trials - 1)

    labels: List[np.ndarray] = []
    modality_scores: List[List[np.ndarray]] = [[] for _ in prediction_models]
    fusion_scores: List[np.ndarray] = []
    fusion_std: List[np.ndarray] = []
    official_uncertain: List[np.ndarray] = []
    corrected_uncertain: List[np.ndarray] = []

    with torch.no_grad():
        for features, target in loader:
            features = [x.to(device) for x in features]
            target = target.to(device)
            means = []
            deviations = []
            for index, (wrapper, x) in enumerate(zip(predictor_wrappers, features)):
                samples = wrapper.predict_on_batch(x, iterations=num_trials)
                mean = samples.mean(dim=-1).reshape(-1)
                deviation = samples.std(dim=-1).reshape(-1)
                means.append(mean)
                deviations.append(deviation)
                modality_scores[index].append(mean.cpu().numpy())

            fusion_samples = fusion_wrapper.predict_on_batch(
                (features, means, deviations), iterations=num_trials
            )
            mean = fusion_samples.mean(dim=-1).reshape(-1)
            deviation = fusion_samples.std(dim=-1).reshape(-1)
            original_half_width = t_critical * deviation / math.sqrt(len(mean))
            corrected_half_width = t_critical * deviation / math.sqrt(num_trials)

            labels.append(target.cpu().numpy())
            fusion_scores.append(mean.cpu().numpy())
            fusion_std.append(deviation.cpu().numpy())
            official_uncertain.append(
                ((mean - original_half_width <= 0.5) & (mean + original_half_width >= 0.5))
                .cpu()
                .numpy()
            )
            corrected_uncertain.append(
                ((mean - corrected_half_width <= 0.5) & (mean + corrected_half_width >= 0.5))
                .cpu()
                .numpy()
            )

    output = dataframe[["row_id", "id", "label"]].copy()
    observed_labels = np.concatenate(labels)
    if not np.array_equal(output.label.to_numpy(), observed_labels):
        raise ValueError("Prediction order does not match dataframe labels")
    for name, pieces in zip(MODALITY_NAMES, modality_scores):
        output[f"{name}_score"] = np.concatenate(pieces)
    output["fusion_score"] = np.concatenate(fusion_scores)
    output["fusion_mc_std"] = np.concatenate(fusion_std)
    output["official_uncertain"] = np.concatenate(official_uncertain).astype(bool)
    output["corrected_uncertain"] = np.concatenate(corrected_uncertain).astype(bool)
    return output


def calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 20) -> float:
    confidences = np.maximum(scores, 1.0 - scores)
    correct = (scores >= 0.5).astype(int) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            error += abs(float(correct[mask].mean()) - float(confidences[mask].mean())) * float(
                mask.mean()
            )
    return error


def compute_metrics(labels: Iterable[float], scores: Iterable[float]) -> Dict[str, Any]:
    labels = np.asarray(list(labels), dtype=int)
    scores = np.asarray(list(scores), dtype=float)
    if len(labels) == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    predictions = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = recall_score(labels, predictions, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = precision_score(labels, predictions, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    has_two_classes = len(np.unique(labels)) == 2
    return {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "official_weighted_accuracy": (precision + npv) / 2.0,
        "auroc": roc_auc_score(labels, scores) if has_two_classes else float("nan"),
        "average_precision": average_precision_score(labels, scores)
        if has_two_classes
        else float("nan"),
        "f1": f1_score(labels, predictions, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "brier": brier_score_loss(labels, scores),
        "ece": calibration_error(labels, scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def participant_scores(dataframe: pd.DataFrame, score_column: str) -> pd.DataFrame:
    label_counts = dataframe.groupby("id").label.nunique()
    if (label_counts > 1).any():
        raise ValueError("At least one participant has inconsistent labels")
    return (
        dataframe.groupby("id", as_index=False)
        .agg(label=("label", "first"), score=(score_column, "mean"), sessions=("row_id", "size"))
        .reset_index(drop=True)
    )


def bootstrap_intervals(
    dataframe: pd.DataFrame, iterations: int, seed: int
) -> Dict[str, Tuple[float, float]]:
    if iterations <= 0:
        return {}
    rng = np.random.default_rng(seed)
    values: Dict[str, List[float]] = {}
    n = len(dataframe)
    for _ in range(iterations):
        sample = dataframe.iloc[rng.integers(0, n, size=n)]
        if sample.label.nunique() < 2:
            continue
        metrics = compute_metrics(sample.label, sample.score)
        for key in (
            "accuracy",
            "balanced_accuracy",
            "auroc",
            "f1",
            "sensitivity",
            "specificity",
            "precision",
            "npv",
            "brier",
            "ece",
        ):
            values.setdefault(key, []).append(metrics[key])
    return {
        key: (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))
        for key, samples in values.items()
        if samples
    }


def add_summary_rows(
    rows: List[Dict[str, Any]],
    split_name: str,
    predictions: pd.DataFrame,
    model_name: str,
    score_column: str,
    abstention: str,
    uncertainty_column: str | None,
    bootstrap: int,
    seed: int,
) -> None:
    retained = predictions
    if uncertainty_column:
        retained = predictions[~predictions[uncertainty_column]].copy()

    for level in ("session", "participant"):
        if level == "session":
            evaluation = retained[["label", score_column]].rename(columns={score_column: "score"})
            original_n = len(predictions)
        else:
            evaluation = participant_scores(retained, score_column)
            original_n = predictions.id.nunique()
        metrics = compute_metrics(evaluation.label, evaluation.score)
        row = {
            "split": split_name,
            "level": level,
            "model": model_name,
            "abstention": abstention,
            "coverage": len(evaluation) / original_n,
            **metrics,
        }
        if level == "participant":
            intervals = bootstrap_intervals(evaluation, bootstrap, seed)
            for key, (low, high) in intervals.items():
                row[f"{key}_ci_low"] = low
                row[f"{key}_ci_high"] = high
        rows.append(row)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    default_output_name = (
        "pretrained_eval_paper_exact"
        if args.protocol == "paper-exact"
        else "pretrained_eval"
    )
    output_dir = (args.output_dir or repo_root / "results" / default_output_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    paper_manifest = None
    paper_manifest_path = None
    paper_manifest_hash = None
    if args.protocol == "paper-exact":
        paper_manifest_path = (
            args.paper_manifest
            or repo_root
            / "results"
            / "paper_score_provenance"
            / "paper_exact_cohort_manifest.csv"
        ).resolve()
        paper_manifest, paper_manifest_hash = load_paper_exact_manifest(
            paper_manifest_path
        )

    module = load_upstream_module(repo_root)
    fusion_config_path = Path(module.MODEL_CONFIG_PATH)
    fusion_model_path = Path(module.MODEL_PATH)
    fusion_config = read_json(fusion_config_path)
    num_trials = args.num_trials or int(fusion_config["num_trials"])
    if num_trials < 2:
        raise ValueError("num_trials must be at least 2")

    selected_models = module.MODEL_SUBSETS[int(fusion_config["model_subset_choice"])]
    if len(selected_models) != 3:
        raise ValueError("This evaluator expects the full three-modality checkpoint")
    module.NUM_MODELS = len(selected_models)
    paths = checkpoint_paths(module, selected_models)
    aligned, predictor_configs, alignment_audit = build_aligned_dataframe(
        module, selected_models, paths
    )
    splits = make_splits(module, aligned)
    feature_shapes = [len(aligned.iloc[0][f"features_{i}"]) for i in range(module.NUM_MODELS)]

    prediction_models = instantiate_models(
        module, selected_models, paths, predictor_configs, feature_shapes, device
    )
    fusion_model = module.HybridFusionNetworkWithUncertainty(feature_shapes, fusion_config)
    fusion_model.load_state_dict(torch.load(fusion_model_path, map_location="cpu"))
    fusion_model.to(device).eval()
    for parameter in fusion_model.parameters():
        parameter.requires_grad = False

    summary_rows: List[Dict[str, Any]] = []
    split_counts: Dict[str, Dict[str, int]] = {}
    split_names = (
        list(PAPER_EXACT_SPLIT_COUNTS)
        if args.protocol == "paper-exact"
        else list(splits)
    )
    for split_name in split_names:
        split_df = splits[split_name]
        if split_df.empty:
            raise ValueError(f"Split is empty: {split_name}")
        split_seed = args.seed + SPLIT_SEED_OFFSETS[split_name]
        predictions = predict_split(
            module,
            split_df,
            prediction_models,
            fusion_model,
            device,
            args.batch_size,
            num_trials,
            split_seed,
        )
        if paper_manifest is not None:
            predictions = apply_paper_exact_manifest(
                predictions, split_name, paper_manifest
            )
        predictions.to_csv(output_dir / f"predictions_{split_name}.csv", index=False)
        split_counts[split_name] = {
            "sessions": len(predictions),
            "participants": predictions.id.nunique(),
            "positive_sessions": int(predictions.label.sum()),
            "positive_participants": int(
                predictions.groupby("id").label.first().sum()
            ),
        }

        for model_name, score_column in (
            ("finger", "finger_score"),
            ("speech", "speech_score"),
            ("smile", "smile_score"),
            ("fusion", "fusion_score"),
        ):
            add_summary_rows(
                summary_rows,
                split_name,
                predictions,
                model_name,
                score_column,
                "none",
                None,
                args.bootstrap,
                split_seed,
            )
        for label, uncertainty_column in (
            ("official_batch_denominator", "official_uncertain"),
            ("corrected_mc_denominator", "corrected_uncertain"),
        ):
            add_summary_rows(
                summary_rows,
                split_name,
                predictions,
                "fusion",
                "fusion_score",
                label,
                uncertainty_column,
                args.bootstrap,
                split_seed,
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(summary_rows), handle, ensure_ascii=False, indent=2)

    checkpoint_hashes = {
        str(path.relative_to(repo_root)): sha256_file(path)
        for item in paths
        for key, path in item.items()
        if key in {"model", "scaler"}
    }
    checkpoint_hashes[str(fusion_model_path.relative_to(repo_root))] = sha256_file(
        fusion_model_path
    )
    manifest = {
        "repo_root": repo_root,
        "git_commit": git_commit(repo_root),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "seed": args.seed,
        "protocol": args.protocol,
        "paper_manifest": paper_manifest_path,
        "paper_manifest_sha256": paper_manifest_hash,
        "num_trials": num_trials,
        "batch_size": args.batch_size,
        "bootstrap_iterations": args.bootstrap,
        "aligned_sessions": len(aligned),
        "aligned_participants": aligned.id.nunique(),
        "alignment_audit": alignment_audit,
        "split_counts": split_counts,
        "checkpoint_sha256": checkpoint_hashes,
        "notes": [
            "Upstream checkpoints and data files were opened read-only by this evaluator.",
            "official_weighted_accuracy reproduces upstream (PPV + NPV) / 2; balanced_accuracy is standard (sensitivity + specificity) / 2.",
            "official_batch_denominator reproduces upstream abstention CI; corrected_mc_denominator divides by sqrt(num_trials).",
            "The upstream fusion pipeline keeps labels from the first (finger) modality after row_id joins; cross-modality label disagreements are recorded in alignment_audit.",
            "Duplicate row_id rows produced by the upstream many-to-many modality joins are preserved and recorded in alignment_audit for faithful reproduction.",
            (
                "paper-exact mode filters full source-cohort inference by the frozen "
                "manifest, validates participants and labels, and restores paper order."
                if args.protocol == "paper-exact"
                else "source mode evaluates the unfiltered source cohorts."
            ),
        ],
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(manifest), handle, ensure_ascii=False, indent=2)

    display = summary[
        (summary.level == "participant")
        & (summary.model == "fusion")
        & (summary.abstention.isin(["none", "corrected_mc_denominator"]))
    ][
        [
            "split",
            "abstention",
            "n",
            "coverage",
            "accuracy",
            "balanced_accuracy",
            "auroc",
            "f1",
            "sensitivity",
            "specificity",
        ]
    ]
    print("\nParticipant-level fusion summary")
    print(display.to_string(index=False))
    print(f"\nResults written to: {output_dir}")


if __name__ == "__main__":
    main()
