#!/usr/bin/env python3
"""Non-destructive paired retraining on official and cleaned PARK tables.

The runner trains the three shipped unimodal architectures followed by UFNet on
the same complete-case participants. Official and cleaned variants use identical
runtime seeds and hyperparameters. Every run is isolated below ``--output-dir``;
the upstream ``models/`` and ``data/`` trees are never written.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

import evaluate_pretrained as ev


MODALITIES = ("finger", "speech", "smile")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Default: <repo>/results/protocol_alignment_audit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <repo>/results/paired_retraining",
    )
    parser.add_argument("--datasets", default="official,cleaned")
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--mc-trials",
        type=int,
        default=30,
        help="MC-dropout trials used for dev selection and final evaluation.",
    )
    parser.add_argument(
        "--unimodal-epochs",
        type=int,
        default=None,
        help="Override each shipped unimodal epoch count (useful for smoke tests).",
    )
    parser.add_argument(
        "--fusion-epochs",
        type=int,
        default=None,
        help="Override the shipped UFNet epoch count (useful for smoke tests).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Rerun completed dataset/seed pairs."
    )
    return parser.parse_args()


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seeds(value: str) -> List[int]:
    seeds = [int(item) for item in parse_csv_list(value)]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_vector_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"features_0", "features_1", "features_2", "label", "id", "row_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    for column in ("features_0", "features_1", "features_2"):
        frame[column] = frame[column].map(
            lambda value: np.asarray(json.loads(value), dtype=np.float32)
        )
        if frame[column].map(lambda value: not np.isfinite(value).all()).any():
            raise ValueError(f"{path} contains non-finite values in {column}")
    if frame[["label", "id", "row_id"]].isna().any().any():
        raise ValueError(f"{path} contains null labels or identifiers")
    frame["label"] = pd.to_numeric(frame.label).astype(int)
    frame["id"] = frame.id.astype(str)
    frame["row_id"] = frame.row_id.astype(str)
    return frame


def split_masks(module, frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    ids = frame.id.astype(str)
    test_ids = set(map(str, module.test_ids))
    dev_ids = set(map(str, module.dev_ids))
    masks = {
        "train": (~ids.isin(test_ids | dev_ids)).to_numpy(),
        "dev": ids.isin(dev_ids).to_numpy(),
        "internal_test": ids.isin(test_ids).to_numpy(),
        "validation_1": ids.isin(set(map(str, module.test_ids_validation_1))).to_numpy(),
        "validation_2": ids.isin(set(map(str, module.test_ids_validation_2))).to_numpy(),
        "global": ids.isin(set(map(str, module.test_ids_global))).to_numpy(),
    }
    if any(not mask.any() for mask in masks.values()):
        empty = [name for name, mask in masks.items() if not mask.any()]
        raise ValueError(f"Empty data partitions: {empty}")
    if (masks["train"] & masks["dev"]).any() or (
        masks["train"] & masks["internal_test"]
    ).any() or (masks["dev"] & masks["internal_test"]).any():
        raise ValueError("Train/dev/test participant masks overlap")
    return masks


def inverse_original_scaling(
    frame: pd.DataFrame,
    configs: Sequence[Dict[str, Any]],
    paths: Sequence[Dict[str, Path]],
) -> pd.DataFrame:
    output = frame.copy()
    for index, (config, model_paths) in enumerate(zip(configs, paths)):
        column = f"features_{index}"
        matrix = np.stack(output[column].to_numpy()).astype(np.float64)
        if config["use_feature_scaling"] == "yes":
            with model_paths["scaler"].open("rb") as handle:
                scaler = pickle.load(handle)
            matrix = scaler.inverse_transform(matrix)
        output[column] = list(matrix.astype(np.float32))
    return output


def fit_training_scalers(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    configs: Sequence[Dict[str, Any]],
) -> Tuple[pd.DataFrame, List[Any]]:
    output = frame.copy()
    scalers: List[Any] = []
    for index, config in enumerate(configs):
        column = f"features_{index}"
        matrix = np.stack(output[column].to_numpy()).astype(np.float64)
        if config["use_feature_scaling"] == "yes":
            if config.get("scaling_method", "StandardScaler") != "StandardScaler":
                raise ValueError(f"Unsupported scaling method: {config['scaling_method']}")
            scaler = StandardScaler().fit(matrix[train_mask])
            matrix = scaler.transform(matrix)
        else:
            scaler = None
        output[column] = list(matrix.astype(np.float32))
        scalers.append(scaler)
    return output, scalers


class UnimodalDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


class MultimodalDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.features = [
            torch.as_tensor(np.stack(frame[f"features_{index}"]), dtype=torch.float32)
            for index in range(3)
        ]
        self.labels = torch.as_tensor(frame.label.to_numpy(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return [features[index] for features in self.features], self.labels[index]


def make_loader(
    dataset: Dataset, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def instantiate_predictor(module, config: Dict[str, Any], n_features: int) -> nn.Module:
    if config["model"] == "ShallowANN":
        return module.ShallowANN(n_features, drop_prob=config["dropout_prob"])
    if config["model"] == "ANN":
        return module.ANN(n_features, drop_prob=config["dropout_prob"])
    raise ValueError(f"Unsupported predictor: {config['model']}")


def make_optimizer(model: nn.Module, config: Dict[str, Any]):
    name = config["optimizer"]
    common = {"lr": config["learning_rate"], "weight_decay": config["weight_decay"]}
    if name == "AdamW":
        return torch.optim.AdamW(
            model.parameters(), betas=(config["beta1"], config["beta2"]), **common
        )
    if name == "SGD":
        return torch.optim.SGD(model.parameters(), momentum=config["momentum"], **common)
    if name == "RMSprop":
        return torch.optim.RMSprop(
            model.parameters(), momentum=config["momentum"], **common
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def mc_unimodal_scores(
    module, model: nn.Module, loader: DataLoader, device: torch.device, trials: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    criterion = nn.BCELoss()
    wrapper = module.ModelWrapper(model, criterion)
    scores: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for features, target in loader:
            features = features.to(device)
            target = target.to(device)
            samples = wrapper.predict_on_batch(features, iterations=trials)
            mean = samples.mean(dim=-1).reshape(-1)
            total_loss += float(criterion(mean, target).item()) * len(target)
            total += len(target)
            scores.append(mean.cpu().numpy())
            labels.append(target.cpu().numpy())
    return np.concatenate(labels), np.concatenate(scores), total_loss / total


def train_predictor(
    module,
    train_frame: pd.DataFrame,
    dev_frame: pd.DataFrame,
    modality_index: int,
    config: Dict[str, Any],
    device: torch.device,
    seed: int,
    mc_trials: int,
    epochs_override: int | None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    column = f"features_{modality_index}"
    x_train = np.stack(train_frame[column]).astype(np.float32)
    y_train = train_frame.label.to_numpy(dtype=np.float32)
    if config["minority_oversample"] == "yes":
        x_train, y_train = SMOTE(random_state=seed).fit_resample(x_train, y_train)
    train_dataset = UnimodalDataset(x_train, y_train)
    dev_dataset = UnimodalDataset(
        np.stack(dev_frame[column]).astype(np.float32),
        dev_frame.label.to_numpy(dtype=np.float32),
    )
    train_loader = make_loader(train_dataset, int(config["batch_size"]), True, seed)
    dev_loader = make_loader(dev_dataset, int(config["batch_size"]), False, seed)

    model = instantiate_predictor(module, config, x_train.shape[1]).to(device)
    optimizer = make_optimizer(model, config)
    criterion = nn.BCELoss()
    epochs = epochs_override or int(config["num_epochs"])
    best_loss = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for features, target in train_loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            prediction = model(features).reshape(-1)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(target)
            count += len(target)
        _, _, dev_loss = mc_unimodal_scores(module, model, dev_loader, device, mc_trials)
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / count, "dev_loss": dev_loss})
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, {
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_dev_loss": best_loss,
        "history": history,
        "train_rows_after_smote": len(train_dataset),
    }


def metrics_at_levels(
    frame: pd.DataFrame, scores: np.ndarray, dataset_name: str, seed: int,
    split: str, model_name: str
) -> List[Dict[str, Any]]:
    predictions = frame[["id", "row_id", "label"]].copy()
    predictions["score"] = scores
    rows = []
    for level in ("session", "participant"):
        if level == "session":
            evaluation = predictions
        else:
            evaluation = ev.participant_scores(predictions, "score")
        rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "split": split,
                "level": level,
                "model": model_name,
                **ev.compute_metrics(evaluation.label, evaluation.score),
            }
        )
    return rows


def evaluate_predictor_splits(
    module, model: nn.Module, frame: pd.DataFrame, masks: Dict[str, np.ndarray],
    modality_index: int, device: torch.device, seed: int, mc_trials: int,
    dataset_name: str
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    rows: List[Dict[str, Any]] = []
    scores_by_split: Dict[str, np.ndarray] = {}
    config_batch_size = 1024
    for offset, split in enumerate(("internal_test", "validation_1", "validation_2", "global")):
        selected = frame.loc[masks[split]].reset_index(drop=True)
        loader = make_loader(
            UnimodalDataset(
                np.stack(selected[f"features_{modality_index}"]),
                selected.label.to_numpy(dtype=np.float32),
            ),
            config_batch_size,
            False,
            seed + offset,
        )
        _, scores, _ = mc_unimodal_scores(module, model, loader, device, mc_trials)
        rows.extend(metrics_at_levels(selected, scores, dataset_name, seed, split, MODALITIES[modality_index]))
        scores_by_split[split] = scores
    return rows, scores_by_split


def fusion_batch_inputs(module, wrappers, features, trials: int):
    means = []
    deviations = []
    with torch.no_grad():
        for wrapper, values in zip(wrappers, features):
            samples = wrapper.predict_on_batch(values, iterations=trials)
            means.append(samples.mean(dim=-1).reshape(-1))
            deviations.append(samples.std(dim=-1).reshape(-1))
    return means, deviations


def evaluate_fusion(
    module, fusion_model: nn.Module, predictors: Sequence[nn.Module], loader: DataLoader,
    device: torch.device, trials: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    criterion = nn.BCELoss()
    predictor_wrappers = [module.ModelWrapper(model, criterion) for model in predictors]
    fusion_wrapper = module.ModelWrapper(fusion_model, criterion)
    labels: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    total_loss = 0.0
    total = 0
    fusion_model.eval()
    for features, target in loader:
        features = [values.to(device) for values in features]
        target = target.to(device)
        means, deviations = fusion_batch_inputs(module, predictor_wrappers, features, trials)
        with torch.no_grad():
            samples = fusion_wrapper.predict_on_batch(
                (features, means, deviations), iterations=trials
            )
            mean = samples.mean(dim=-1).reshape(-1)
            total_loss += float(criterion(mean, target).item()) * len(target)
        total += len(target)
        labels.append(target.cpu().numpy())
        scores.append(mean.cpu().numpy())
    return np.concatenate(labels), np.concatenate(scores), total_loss / total


def train_fusion(
    module, predictors: Sequence[nn.Module], train_frame: pd.DataFrame,
    dev_frame: pd.DataFrame, feature_shapes: Sequence[int], config: Dict[str, Any],
    device: torch.device, seed: int, mc_trials: int, epochs_override: int | None
) -> Tuple[nn.Module, Dict[str, Any]]:
    train_data = train_frame
    if config["minority_oversample"] == "yes":
        raise ValueError("This paired runner expects the shipped UFNet config without oversampling")
    batch_size = int(config["batch_size"])
    train_loader = make_loader(MultimodalDataset(train_data), batch_size, True, seed)
    dev_loader = make_loader(MultimodalDataset(dev_frame), batch_size, False, seed)
    for model in predictors:
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
    criterion = nn.BCELoss()
    wrappers = [module.ModelWrapper(model, criterion) for model in predictors]
    model = module.HybridFusionNetworkWithUncertainty(feature_shapes, config).to(device)
    optimizer = make_optimizer(model, config)
    epochs = epochs_override or int(config["num_epochs"])
    best_loss = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for features, target in train_loader:
            features = [values.to(device) for values in features]
            target = target.to(device)
            means, deviations = fusion_batch_inputs(module, wrappers, features, mc_trials)
            optimizer.zero_grad()
            prediction = model((features, means, deviations)).reshape(-1)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(target)
            count += len(target)
        _, _, dev_loss = evaluate_fusion(
            module, model, predictors, dev_loader, device, mc_trials
        )
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / count, "dev_loss": dev_loss})
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, {
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_dev_loss": best_loss,
        "history": history,
        "train_rows": len(train_frame),
    }


def evaluate_fusion_splits(
    module, model: nn.Module, predictors: Sequence[nn.Module], frame: pd.DataFrame,
    masks: Dict[str, np.ndarray], device: torch.device, seed: int, mc_trials: int,
    dataset_name: str, output_dir: Path
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for offset, split in enumerate(("internal_test", "validation_1", "validation_2", "global")):
        selected = frame.loc[masks[split]].reset_index(drop=True)
        loader = make_loader(MultimodalDataset(selected), 1024, False, seed + offset)
        _, scores, _ = evaluate_fusion(module, model, predictors, loader, device, mc_trials)
        predictions = selected[["id", "row_id", "label"]].copy()
        predictions["score"] = scores
        predictions.to_csv(output_dir / f"predictions_{split}.csv", index=False)
        rows.extend(metrics_at_levels(selected, scores, dataset_name, seed, split, "fusion"))
    return rows


def save_model(path: Path, model: nn.Module, metadata: Dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(state, path / "model.pth")
    with (path / "training.json").open("w", encoding="utf-8") as handle:
        json.dump(ev.json_ready(metadata), handle, indent=2)


def train_one(
    module, dataset_name: str, frame: pd.DataFrame, masks: Dict[str, np.ndarray],
    configs: Sequence[Dict[str, Any]], fusion_config: Dict[str, Any], scalers: Sequence[Any],
    device: torch.device, seed: int, mc_trials: int, unimodal_epochs: int | None,
    fusion_epochs: int | None, run_dir: Path
) -> List[Dict[str, Any]]:
    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_frame = frame.loc[masks["train"]].reset_index(drop=True)
    dev_frame = frame.loc[masks["dev"]].reset_index(drop=True)
    feature_shapes = [len(frame.iloc[0][f"features_{index}"]) for index in range(3)]
    for index, scaler in enumerate(scalers):
        with (run_dir / f"scaler_{MODALITIES[index]}.pkl").open("wb") as handle:
            pickle.dump(scaler, handle)

    predictors: List[nn.Module] = []
    rows: List[Dict[str, Any]] = []
    training_metadata: Dict[str, Any] = {"unimodal": {}}
    for index, (name, config) in enumerate(zip(MODALITIES, configs)):
        set_seed(seed + index * 1000)
        model, metadata = train_predictor(
            module, train_frame, dev_frame, index, config, device, seed + index,
            mc_trials, unimodal_epochs
        )
        predictors.append(model)
        training_metadata["unimodal"][name] = metadata
        save_model(run_dir / name, model, {"config": config, **metadata})
        metric_rows, _ = evaluate_predictor_splits(
            module, model, frame, masks, index, device, seed, mc_trials, dataset_name
        )
        rows.extend(metric_rows)

    set_seed(seed + 10000)
    trained_fusion, fusion_metadata = train_fusion(
        module, predictors, train_frame, dev_frame, feature_shapes, fusion_config,
        device, seed + 10000, mc_trials, fusion_epochs
    )
    training_metadata["fusion"] = fusion_metadata
    save_model(run_dir / "fusion", trained_fusion, {"config": fusion_config, **fusion_metadata})
    rows.extend(
        evaluate_fusion_splits(
            module, trained_fusion, predictors, frame, masks, device, seed, mc_trials,
            dataset_name, run_dir
        )
    )

    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
    with (run_dir / "run_complete.json").open("w", encoding="utf-8") as handle:
        json.dump(
            ev.json_ready(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "mc_trials": mc_trials,
                    "rows": len(frame),
                    "participants": frame.id.nunique(),
                    "partition_rows": {name: int(mask.sum()) for name, mask in masks.items()},
                    "training": training_metadata,
                }
            ),
            handle,
            indent=2,
        )
    return rows


def aggregate_metrics(output_dir: Path) -> pd.DataFrame:
    files = sorted(output_dir.glob("*/seed_*/metrics.csv"))
    if not files:
        return pd.DataFrame()
    metrics = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    metrics.to_csv(output_dir / "metrics_per_run.csv", index=False)
    numeric = [
        "accuracy", "balanced_accuracy", "auroc", "average_precision", "f1",
        "sensitivity", "specificity", "brier", "ece"
    ]
    summary = (
        metrics.groupby(["dataset", "split", "level", "model"], as_index=False)[numeric]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    keys = ["seed", "split", "level", "model"]
    wide = metrics.pivot(index=keys, columns="dataset", values=numeric)
    if {"official", "cleaned"}.issubset(set(metrics.dataset.unique())):
        paired = wide.xs("cleaned", axis=1, level=1) - wide.xs(
            "official", axis=1, level=1
        )
        paired = paired.reset_index()
        paired = paired.rename(
            columns={column: f"{column}_delta_cleaned_minus_official" for column in numeric}
        )
        paired.to_csv(output_dir / "paired_deltas.csv", index=False)
        delta_columns = [column for column in paired.columns if column.endswith("_delta_cleaned_minus_official")]
        delta_summary = paired.groupby(["split", "level", "model"], as_index=False)[
            delta_columns
        ].agg(["mean", "std"])
        delta_summary.columns = [
            "_".join(str(part) for part in column if str(part))
            if isinstance(column, tuple)
            else column
            for column in delta_summary.columns
        ]
        delta_summary.to_csv(output_dir / "paired_delta_summary.csv", index=False)
        write_paired_report(output_dir / "PAIRED_REPORT.md", metrics, paired)
    return summary


def write_paired_report(path: Path, metrics: pd.DataFrame, paired: pd.DataFrame) -> None:
    target = metrics[
        (metrics.split == "internal_test") & (metrics.level == "participant")
    ]
    delta_target = paired[
        (paired.split == "internal_test") & (paired.level == "participant")
    ]
    lines = [
        "# PARK official-versus-cleaned paired retraining",
        "",
        "Five runtime seeds use identical architectures and hyperparameters for both "
        "datasets. Values below are participant-level means on the pooled internal test "
        "cohort; deltas are cleaned minus official. With only five seeds, these are "
        "descriptive paired results rather than confirmatory significance tests.",
        "",
        "| Model | Official accuracy | Cleaned accuracy | Accuracy delta | Official AUROC | Cleaned AUROC | AUROC delta | Sensitivity delta | Specificity delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODALITIES + ("fusion",):
        official = target[(target.model == model) & (target.dataset == "official")]
        cleaned = target[(target.model == model) & (target.dataset == "cleaned")]
        delta = delta_target[delta_target.model == model]
        lines.append(
            "| {model} | {oa:.4f} | {ca:.4f} | {da:+.4f} | {ou:.4f} | {cu:.4f} | "
            "{du:+.4f} | {ds:+.4f} | {dp:+.4f} |".format(
                model=model,
                oa=official.accuracy.mean(),
                ca=cleaned.accuracy.mean(),
                da=delta.accuracy_delta_cleaned_minus_official.mean(),
                ou=official.auroc.mean(),
                cu=cleaned.auroc.mean(),
                du=delta.auroc_delta_cleaned_minus_official.mean(),
                ds=delta.sensitivity_delta_cleaned_minus_official.mean(),
                dp=delta.specificity_delta_cleaned_minus_official.mean(),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- The two datasets have identical test rows; only train/dev duplicate handling and refitted scalers differ.",
            "- The cleaned majority-vote labels equal the official finger labels for every unique row, so label replacement is not driving the difference.",
            "- Accuracy at threshold 0.5 can move differently from AUROC because cleaning changes calibration and the sensitivity/specificity trade-off.",
            "- This complete-case paired experiment is not an exact reproduction of the paper's modality-specific training cohorts or 30-seed protocol.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.mc_trials < 2:
        raise ValueError("--mc-trials must be at least 2")
    repo_root = args.repo_root.resolve()
    data_dir = (args.data_dir or repo_root / "results" / "protocol_alignment_audit").resolve()
    output_dir = (args.output_dir or repo_root / "results" / "paired_retraining").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = parse_csv_list(args.datasets)
    invalid = set(datasets) - {"official", "cleaned"}
    if invalid:
        raise ValueError(f"Invalid dataset names: {sorted(invalid)}")
    seeds = parse_seeds(args.seeds)
    device = ev.resolve_device(args.device)

    module = ev.load_upstream_module(repo_root)
    fusion_config = ev.read_json(Path(module.MODEL_CONFIG_PATH))
    selected_models = module.MODEL_SUBSETS[int(fusion_config["model_subset_choice"])]
    module.NUM_MODELS = len(selected_models)
    paths = ev.checkpoint_paths(module, selected_models)
    configs = [ev.read_json(item["config"]) for item in paths]
    source_hashes_before = {
        str(path.relative_to(repo_root)): ev.sha256_file(path)
        for item in paths for key, path in item.items() if key in {"model", "scaler"}
    }
    source_hashes_before[str(Path(module.MODEL_PATH).relative_to(repo_root))] = ev.sha256_file(
        Path(module.MODEL_PATH)
    )

    started = time.time()
    for dataset_name in datasets:
        source = data_dir / f"{dataset_name}_aligned.csv"
        frame = load_vector_csv(source)
        masks = split_masks(module, frame)
        raw_frame = inverse_original_scaling(frame, configs, paths)
        scaled_frame, scalers = fit_training_scalers(raw_frame, masks["train"], configs)
        for seed in seeds:
            run_dir = output_dir / dataset_name / f"seed_{seed}"
            if (run_dir / "run_complete.json").exists() and not args.force:
                print(f"Skipping completed run: {dataset_name} seed={seed}")
                continue
            print(f"Training {dataset_name} seed={seed}")
            train_one(
                module, dataset_name, scaled_frame, masks, configs, fusion_config, scalers,
                device, seed, args.mc_trials, args.unimodal_epochs, args.fusion_epochs,
                run_dir
            )
            aggregate_metrics(output_dir)

    source_hashes_after = {
        path: ev.sha256_file(repo_root / path) for path in source_hashes_before
    }
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("An upstream checkpoint or scaler changed during paired training")
    summary = aggregate_metrics(output_dir)
    manifest = {
        "repo_root": repo_root,
        "git_commit": ev.git_commit(repo_root),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "datasets": datasets,
        "seeds": seeds,
        "mc_trials": args.mc_trials,
        "unimodal_epochs_override": args.unimodal_epochs,
        "fusion_epochs_override": args.fusion_epochs,
        "elapsed_seconds": time.time() - started,
        "source_checkpoint_sha256": source_hashes_after,
        "source_checkpoints_unchanged": True,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(ev.json_ready(manifest), handle, indent=2)
    print(f"Results written to: {output_dir}")
    if not summary.empty:
        selected = summary[
            (summary["level"] == "participant")
            & (summary["model"] == "fusion")
            & (summary["split"] == "internal_test")
        ]
        print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
