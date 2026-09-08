#!/usr/bin/env python3
"""Phase 2 development of feature-adapter fusion on the frozen UFNet cache.

Only train and validation files are loaded. The paper test cache is deliberately
not accepted by the command line, preventing accidental test-guided development.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


MODALITIES = ("finger", "speech", "smile")
QUALITY_FIELDS = ("mean", "std", "predictive_entropy", "mutual_information")
SCALAR_FIELDS = ("mean", "std", "deterministic_probability", "predictive_entropy", "mutual_information")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(cache: Path, split: str) -> dict[str, np.ndarray]:
    if split not in {"train", "validation"}:
        raise ValueError("Phase 2 may load only train and validation")
    payload = dict(np.load(cache / f"expert_cache_{split}.npz", allow_pickle=False))
    expected = {"manifest_row_id", "participant_id", "label"}
    expected |= {f"{modality}_features" for modality in MODALITIES}
    expected |= {f"{modality}_{field}" for modality in MODALITIES for field in SCALAR_FIELDS}
    missing = expected - set(payload)
    if missing:
        raise ValueError(f"Missing cache arrays: {sorted(missing)}")
    if not all(np.isfinite(value).all() for value in payload.values() if value.dtype.kind in "fiu"):
        raise ValueError(f"Non-finite value in {split} cache")
    return payload


class CacheDataset(Dataset):
    def __init__(self, payload: dict[str, np.ndarray]):
        self.features = [torch.tensor(payload[f"{m}_features"], dtype=torch.float32) for m in MODALITIES]
        self.quality = torch.tensor(
            np.stack([np.stack([payload[f"{m}_{f}"] for f in QUALITY_FIELDS], axis=1) for m in MODALITIES], axis=1),
            dtype=torch.float32,
        )
        self.scalars = torch.tensor(
            np.concatenate([np.stack([payload[f"{m}_{f}"] for f in SCALAR_FIELDS], axis=1) for m in MODALITIES], axis=1),
            dtype=torch.float32,
        )
        self.labels = torch.tensor(payload["label"], dtype=torch.float32)
        self.participant_ids = payload["participant_id"].astype(str)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return tuple(value[index] for value in self.features), self.quality[index], self.scalars[index], self.labels[index], index


class ConcatMLP(nn.Module):
    def __init__(self, dims: list[int], width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(sum(dims)), nn.Linear(sum(dims), width * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width * 2, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1),
        )

    def forward(self, features, quality, scalars):
        return self.net(torch.cat(features, dim=1)).squeeze(1), None


class ScalarMLP(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(len(MODALITIES) * len(SCALAR_FIELDS)), nn.Linear(len(MODALITIES) * len(SCALAR_FIELDS), width),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1),
        )

    def forward(self, features, quality, scalars):
        return self.net(scalars).squeeze(1), None


class AdapterTransformer(nn.Module):
    def __init__(self, dims: list[int], width: int, layers: int, heads: int, dropout: float, use_uncertainty: bool):
        super().__init__()
        self.use_uncertainty = use_uncertainty
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(width))
            for dim in dims
        ])
        self.quality_adapters = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(len(QUALITY_FIELDS)), nn.Linear(len(QUALITY_FIELDS), width), nn.Tanh())
            for _ in dims
        ])
        self.modality_embedding = nn.Parameter(torch.randn(1, len(dims), width) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.final_norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, 1)
        self.auxiliary = nn.ModuleList([nn.Linear(width, 1) for _ in dims])

    def forward(self, features, quality, scalars):
        tokens = torch.stack([adapter(value) for adapter, value in zip(self.adapters, features)], dim=1)
        if self.use_uncertainty:
            quality_tokens = torch.stack(
                [adapter(quality[:, i]) for i, adapter in enumerate(self.quality_adapters)], dim=1
            )
            tokens = tokens + quality_tokens
        tokens = tokens + self.modality_embedding
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        hidden = self.encoder(torch.cat([cls, tokens], dim=1))
        logits = self.classifier(self.final_norm(hidden[:, 0])).squeeze(1)
        auxiliary = torch.stack([head(tokens[:, i]).squeeze(1) for i, head in enumerate(self.auxiliary)], dim=1)
        return logits, auxiliary


def make_model(name: str, dims: list[int], config: dict) -> nn.Module:
    width, dropout = int(config["token_dim"]), float(config["dropout"])
    if name == "scalar_mlp":
        return ScalarMLP(width, dropout)
    if name == "concat_mlp":
        return ConcatMLP(dims, width, dropout)
    if name in {"adapter_transformer", "uncertainty_adapter_transformer"}:
        return AdapterTransformer(
            dims, width, int(config["transformer_layers"]), int(config["attention_heads"]), dropout,
            use_uncertainty=name == "uncertainty_adapter_transformer",
        )
    raise ValueError(f"Unknown model {name}")


def participant_frame(dataset: CacheDataset, scores: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"participant_id": dataset.participant_ids, "label": dataset.labels.numpy().astype(int), "score": scores})
    if (frame.groupby("participant_id").label.nunique() > 1).any():
        raise ValueError("Inconsistent participant labels")
    return frame.groupby("participant_id", as_index=False).agg(label=("label", "first"), score=("score", "mean"))


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    y, p = frame.label.to_numpy(int), np.clip(frame.score.to_numpy(float), 1e-6, 1 - 1e-6)
    hard = (p >= 0.5).astype(int)
    return {
        "participants": int(len(y)), "positives": int(y.sum()), "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)), "accuracy": float(accuracy_score(y, hard)),
        "balanced_accuracy": float(balanced_accuracy_score(y, hard)), "F1": float(f1_score(y, hard)),
        "sensitivity": float(recall_score(y, hard)), "specificity": float(recall_score(y, hard, pos_label=0)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p)),
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, dataset: CacheDataset, device: torch.device) -> np.ndarray:
    model.eval()
    result = np.empty(len(dataset), dtype=np.float64)
    for features, quality, scalars, _, indices in loader:
        features = tuple(value.to(device) for value in features)
        logits, _ = model(features, quality.to(device), scalars.to(device))
        result[indices.numpy()] = torch.sigmoid(logits).cpu().numpy()
    return result


def make_loaders(train: CacheDataset, validation: CacheDataset, config: dict, seed: int):
    counts = pd.Series(train.participant_ids).value_counts()
    weights = np.asarray([1.0 / counts[pid] for pid in train.participant_ids], dtype=np.float64)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.tensor(weights), len(train), replacement=True, generator=generator)
    batch = int(config["batch_size"])
    train_loader = DataLoader(train, batch_size=batch, sampler=sampler, num_workers=0)
    train_eval = DataLoader(train, batch_size=batch, shuffle=False, num_workers=0)
    validation_loader = DataLoader(validation, batch_size=batch, shuffle=False, num_workers=0)
    return train_loader, train_eval, validation_loader


def fit_model(name: str, seed: int, train: CacheDataset, validation: CacheDataset, config: dict, device: torch.device):
    set_seed(seed)
    dims = [value.shape[1] for value in train.features]
    model = make_model(name, dims, config).to(device)
    train_loader, train_eval, validation_loader = make_loaders(train, validation, config, seed)
    labels = train.labels.numpy().astype(int)
    pos_weight = torch.tensor([(labels == 0).sum() / max((labels == 1).sum(), 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    best_state, best_epoch, best_loss, stale = copy.deepcopy(model.state_dict()), 0, float("inf"), 0
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        for features, quality, scalars, target, _ in train_loader:
            features = tuple(value.to(device) for value in features)
            target, quality, scalars = target.to(device), quality.to(device), scalars.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, auxiliary = model(features, quality, scalars)
            loss = criterion(logits, target)
            if auxiliary is not None:
                aux = sum(criterion(auxiliary[:, i], target) for i in range(auxiliary.shape[1])) / auxiliary.shape[1]
                loss = loss + float(config["auxiliary_loss_weight"]) * aux
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        val_scores = predict(model, validation_loader, validation, device)
        val_loss = metrics(participant_frame(validation, val_scores))["log_loss"]
        if val_loss < best_loss - 1e-5:
            best_state, best_epoch, best_loss, stale = copy.deepcopy(model.state_dict()), epoch, val_loss, 0
        else:
            stale += 1
            if stale >= int(config["early_stopping_patience"]):
                break
    model.load_state_dict(best_state)
    return model, best_epoch, predict(model, train_eval, train, device), predict(model, validation_loader, validation, device)


def bootstrap_delta(reference: pd.DataFrame, candidate: pd.DataFrame, replicates: int, seed: int) -> dict[str, float]:
    merged = reference.merge(candidate, on=["participant_id", "label"], suffixes=("_reference", "_candidate"), validate="one_to_one")
    y = merged.label.to_numpy(int)
    ref, cand = merged.score_reference.to_numpy(float), merged.score_candidate.to_numpy(float)
    rng, deltas = np.random.default_rng(seed), []
    for _ in range(replicates):
        indices = rng.integers(0, len(y), len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        deltas.append(roc_auc_score(y[indices], cand[indices]) - roc_auc_score(y[indices], ref[indices]))
    values = np.asarray(deltas)
    return {"AUROC_delta": float(roc_auc_score(y, cand) - roc_auc_score(y, ref)), "CI_low": float(np.quantile(values, 0.025)), "CI_high": float(np.quantile(values, 0.975))}


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("test_policy") != "sealed_not_loaded":
        raise ValueError("Phase 2 config must seal the test split")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train, validation = CacheDataset(load_split(args.cache, "train")), CacheDataset(load_split(args.cache, "validation"))
    if set(train.participant_ids) & set(validation.participant_ids):
        raise ValueError("Train/validation participant overlap")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    baseline_scores = np.mean(np.stack([load_split(args.cache, "validation")[f"{m}_mean"] for m in MODALITIES]), axis=0)
    baseline = participant_frame(validation, baseline_scores)
    predictions = [baseline.assign(model="available_mean", seed=-1)]
    per_seed_rows = [{"model": "available_mean", "seed": -1, "best_epoch": 0, **metrics(baseline)}]
    ensemble_rows, bootstrap_rows = [], []
    for model_name in config["models"]:
        seed_frames = []
        for seed in config["seeds"]:
            model, epoch, train_scores, val_scores = fit_model(model_name, int(seed), train, validation, config, device)
            frame = participant_frame(validation, val_scores)
            predictions.append(frame.assign(model=model_name, seed=int(seed)))
            seed_frames.append(frame.set_index("participant_id").score.rename(str(seed)))
            per_seed_rows.append({"model": model_name, "seed": int(seed), "best_epoch": epoch, **metrics(frame)})
            torch.save({"model": model.state_dict(), "model_name": model_name, "seed": int(seed), "best_epoch": epoch}, output / f"{model_name}_seed{seed}.pth")
        ensemble_scores = pd.concat(seed_frames, axis=1).mean(axis=1)
        ensemble = baseline[["participant_id", "label"]].copy()
        ensemble["score"] = ensemble.participant_id.map(ensemble_scores)
        ensemble_rows.append({"model": model_name, **metrics(ensemble)})
        bootstrap_rows.append({"model": model_name, **bootstrap_delta(baseline, ensemble, int(config["bootstrap_replicates"]), int(config["bootstrap_seed"]))})
        predictions.append(ensemble.assign(model=model_name + "_ensemble", seed=-1))
    baseline_metrics = {"model": "available_mean", **metrics(baseline)}
    summary = pd.DataFrame([baseline_metrics, *ensemble_rows]).sort_values("AUROC", ascending=False)
    pd.DataFrame(per_seed_rows).to_csv(output / "per_seed_metrics.csv", index=False)
    summary.to_csv(output / "ensemble_metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output / "bootstrap_vs_available_mean.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "private_validation_predictions.csv", index=False)
    audit = {
        "status": "PASS", "device": str(device), "test_cache_loaded": False,
        "train_rows": len(train), "train_participants": len(set(train.participant_ids)),
        "validation_rows": len(validation), "validation_participants": len(set(validation.participant_ids)),
        "participant_overlap": 0, "config": config,
    }
    (output / "run_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nBootstrap AUROC deltas vs available mean")
    print(pd.DataFrame(bootstrap_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
