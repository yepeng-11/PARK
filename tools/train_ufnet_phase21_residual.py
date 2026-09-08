#!/usr/bin/env python3
"""Nested train-only selection for constrained residual Feature Adapters."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import train_ufnet_phase2_adapters as phase2


MODALITIES = phase2.MODALITIES


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def subset_payload(payload: dict[str, np.ndarray], participant_ids: set[str]) -> dict[str, np.ndarray]:
    mask = np.isin(payload["participant_id"].astype(str), list(participant_ids))
    return {key: value[mask] for key, value in payload.items()}


def participant_table(payload: dict[str, np.ndarray]) -> pd.DataFrame:
    frame = pd.DataFrame({"participant_id": payload["participant_id"].astype(str), "label": payload["label"].astype(int)})
    # One training participant converts from control to PD longitudinally. The
    # ever-PD label is used only to stratify participant-disjoint folds; session
    # labels remain unchanged for training and primary evaluation.
    return frame.groupby("participant_id", as_index=False).label.max().sort_values("participant_id").reset_index(drop=True)


def make_dataset(payload: dict[str, np.ndarray]) -> phase2.CacheDataset:
    dataset = phase2.CacheDataset(payload)
    dataset.row_ids = payload["manifest_row_id"].astype(str)
    return dataset


def session_frame(dataset: phase2.CacheDataset, scores: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "manifest_row_id": dataset.row_ids,
        "participant_id": dataset.participant_ids,
        "label": dataset.labels.numpy().astype(int),
        "score": scores,
    })


def session_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    compact = frame[["label", "score"]].copy()
    compact["participant_id"] = np.arange(len(compact)).astype(str)
    result = phase2.metrics(compact[["participant_id", "label", "score"]])
    result["sessions"] = result.pop("participants")
    return result


class ResidualAdapter(nn.Module):
    def __init__(self, scalar: nn.Module, dims: list[int], width: int, dropout: float, scale: float, gated: bool):
        super().__init__()
        self.scalar = scalar
        for parameter in self.scalar.parameters():
            parameter.requires_grad = False
        self.scale = scale
        self.gated = gated
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(width))
            for dim in dims
        ])
        self.gates = nn.ModuleList([nn.Linear(len(phase2.QUALITY_FIELDS), 1) for _ in dims])
        self.residual = nn.Sequential(
            nn.Linear(width * len(dims), width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1)
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.scalar.eval()
        return self

    def forward(self, features, quality, scalars):
        with torch.no_grad():
            base, _ = self.scalar(features, quality, scalars)
        tokens = []
        for index, (adapter, value) in enumerate(zip(self.adapters, features)):
            token = adapter(value)
            if self.gated:
                token = token * torch.sigmoid(self.gates[index](quality[:, index]))
            tokens.append(token)
        residual = self.residual(torch.cat(tokens, dim=1)).squeeze(1)
        return base + self.scale * residual, None


def loaders(train: phase2.CacheDataset, evaluation: phase2.CacheDataset, config: dict, seed: int):
    counts = pd.Series(train.participant_ids).value_counts()
    weights = torch.tensor([1.0 / counts[pid] for pid in train.participant_ids], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, len(train), replacement=True, generator=torch.Generator().manual_seed(seed))
    batch = int(config["batch_size"])
    return (
        DataLoader(train, batch_size=batch, sampler=sampler, num_workers=0),
        DataLoader(train, batch_size=batch, shuffle=False, num_workers=0),
        DataLoader(evaluation, batch_size=batch, shuffle=False, num_workers=0),
    )


def participant_pos_weight(dataset: phase2.CacheDataset, device: torch.device) -> torch.Tensor:
    frame = pd.DataFrame({"id": dataset.participant_ids, "label": dataset.labels.numpy().astype(int)}).groupby("id").label.max()
    return torch.tensor([(frame == 0).sum() / max((frame == 1).sum(), 1)], dtype=torch.float32, device=device)


def train_epochs(model, dataset, config, device, seed, epochs):
    train_loader, _, _ = loaders(dataset, dataset, config, seed)
    criterion = nn.BCEWithLogitsLoss(pos_weight=participant_pos_weight(dataset, device))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    model.to(device)
    for _ in range(int(epochs)):
        model.train()
        for features, quality, scalars, target, _ in train_loader:
            features = tuple(value.to(device) for value in features)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(features, quality.to(device), scalars.to(device))
            loss = criterion(logits, target.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
    return model


def tune_epochs(model, optimize, tune, config, device, seed):
    train_loader, _, tune_loader = loaders(optimize, tune, config, seed)
    criterion = nn.BCEWithLogitsLoss(pos_weight=participant_pos_weight(optimize, device))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    model.to(device)
    best_state, best_epoch, best_loss, stale = copy.deepcopy(model.state_dict()), 1, float("inf"), 0
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        for features, quality, scalars, target, _ in train_loader:
            features = tuple(value.to(device) for value in features)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(features, quality.to(device), scalars.to(device))
            loss = criterion(logits, target.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
        scores = phase2.predict(model, tune_loader, tune, device)
        loss_value = session_metrics(session_frame(tune, scores))["log_loss"]
        if loss_value < best_loss - 1e-5:
            best_state, best_epoch, best_loss, stale = copy.deepcopy(model.state_dict()), epoch, loss_value, 0
        else:
            stale += 1
            if stale >= int(config["early_stopping_patience"]):
                break
    model.load_state_dict(best_state)
    return model, best_epoch


def new_scalar(config, seed):
    phase2.set_seed(seed)
    return phase2.ScalarMLP(int(config["scalar_width"]), float(config["dropout"]))


def new_residual(scalar, dims, config, scale, gated, seed):
    phase2.set_seed(seed)
    return ResidualAdapter(
        scalar, dims, int(config["adapter_width"]), float(config["dropout"]), float(scale), gated
    )


def paired_bootstrap(reference, candidate, replicates, seed):
    joined = reference.merge(
        candidate, on=["manifest_row_id", "participant_id", "label"],
        suffixes=("_reference", "_candidate"), validate="one_to_one",
    )
    y = joined.label.to_numpy(int)
    ref, cand = joined.score_reference.to_numpy(float), joined.score_candidate.to_numpy(float)
    clusters = {pid: np.flatnonzero(joined.participant_id.to_numpy(str) == pid) for pid in joined.participant_id.unique()}
    participant_ids = np.asarray(list(clusters), dtype=str)
    rng, roc_values, pr_values = np.random.default_rng(seed), [], []
    for _ in range(int(replicates)):
        sampled = rng.choice(participant_ids, size=len(participant_ids), replace=True)
        index = np.concatenate([clusters[pid] for pid in sampled])
        if len(np.unique(y[index])) < 2:
            continue
        roc_values.append(roc_auc_score(y[index], cand[index]) - roc_auc_score(y[index], ref[index]))
        pr_values.append(average_precision_score(y[index], cand[index]) - average_precision_score(y[index], ref[index]))
    return {
        "AUROC_delta": float(roc_auc_score(y, cand) - roc_auc_score(y, ref)),
        "AUROC_CI_low": float(np.quantile(roc_values, 0.025)), "AUROC_CI_high": float(np.quantile(roc_values, 0.975)),
        "AUPRC_delta": float(average_precision_score(y, cand) - average_precision_score(y, ref)),
        "AUPRC_CI_low": float(np.quantile(pr_values, 0.025)), "AUPRC_CI_high": float(np.quantile(pr_values, 0.975)),
    }


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("test_policy") != "sealed_not_loaded":
        raise ValueError("The official test must remain sealed")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train_payload = phase2.load_split(args.cache, "train")
    validation_payload = phase2.load_split(args.cache, "validation")
    train_people, validation_people = participant_table(train_payload), participant_table(validation_payload)
    if set(train_people.participant_id) & set(validation_people.participant_id):
        raise ValueError("Train/validation participant overlap")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    splitter = StratifiedKFold(n_splits=int(config["inner_folds"]), shuffle=True, random_state=int(config["inner_fold_seed"]))
    fold_assignments = np.empty(len(train_people), dtype=int)
    for fold, (_, heldout_index) in enumerate(splitter.split(train_people.participant_id, train_people.label)):
        fold_assignments[heldout_index] = fold
    folds = dict(zip(train_people.participant_id, fold_assignments))
    pd.DataFrame({"participant_id": train_people.participant_id, "label": train_people.label, "fold": fold_assignments}).to_csv(output / "private_train_fold_manifest.csv", index=False)
    dims = [train_payload[f"{m}_features"].shape[1] for m in MODALITIES]
    oof_rows, epoch_rows = [], []
    candidate_keys = [(model, float(scale)) for model in config["models"] for scale in config["residual_scales"]]
    for fold in range(int(config["inner_folds"])):
        heldout_ids = {pid for pid, value in folds.items() if value == fold}
        inner_ids = set(train_people.participant_id) - heldout_ids
        inner_table = train_people.loc[train_people.participant_id.isin(inner_ids)].reset_index(drop=True)
        tune_split = StratifiedShuffleSplit(n_splits=1, test_size=float(config["tune_fraction"]), random_state=int(config["inner_fold_seed"]) + fold)
        optimize_index, tune_index = next(tune_split.split(inner_table.participant_id, inner_table.label))
        optimize_ids = set(inner_table.iloc[optimize_index].participant_id)
        tune_ids = set(inner_table.iloc[tune_index].participant_id)
        datasets = {
            "optimize": make_dataset(subset_payload(train_payload, optimize_ids)),
            "tune": make_dataset(subset_payload(train_payload, tune_ids)),
            "inner": make_dataset(subset_payload(train_payload, inner_ids)),
            "heldout": make_dataset(subset_payload(train_payload, heldout_ids)),
        }
        heldout_loader = DataLoader(datasets["heldout"], batch_size=int(config["batch_size"]), shuffle=False)
        for seed in config["seeds"]:
            base_seed = int(seed) * 100 + fold * 10
            tuned_scalar, scalar_epoch = tune_epochs(new_scalar(config, base_seed), datasets["optimize"], datasets["tune"], config, device, base_seed)
            full_scalar = train_epochs(new_scalar(config, base_seed), datasets["inner"], config, device, base_seed, scalar_epoch)
            scalar_scores = phase2.predict(full_scalar, heldout_loader, datasets["heldout"], device)
            scalar_frame = session_frame(datasets["heldout"], scalar_scores).assign(model="scalar_mlp", scale=0.0, seed=int(seed), fold=fold)
            oof_rows.append(scalar_frame)
            epoch_rows.append({"model": "scalar_mlp", "scale": 0.0, "seed": int(seed), "fold": fold, "epoch": scalar_epoch})
            for model_index, (model_name, scale) in enumerate(candidate_keys, start=1):
                gated = model_name == "uncertainty_gated_residual_adapter"
                residual_seed = base_seed + model_index
                tune_model, residual_epoch = tune_epochs(
                    new_residual(copy.deepcopy(tuned_scalar), dims, config, scale, gated, residual_seed),
                    datasets["optimize"], datasets["tune"], config, device, residual_seed,
                )
                final_model = train_epochs(
                    new_residual(copy.deepcopy(full_scalar), dims, config, scale, gated, residual_seed),
                    datasets["inner"], config, device, residual_seed, residual_epoch,
                )
                scores = phase2.predict(final_model, heldout_loader, datasets["heldout"], device)
                frame = session_frame(datasets["heldout"], scores).assign(model=model_name, scale=scale, seed=int(seed), fold=fold)
                oof_rows.append(frame)
                epoch_rows.append({"model": model_name, "scale": scale, "seed": int(seed), "fold": fold, "epoch": residual_epoch})
    oof = pd.concat(oof_rows, ignore_index=True)
    oof.to_csv(output / "private_train_oof_predictions.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(output / "selected_epochs.csv", index=False)
    aggregate_rows, ensemble_frames = [], {}
    for (model_name, scale), frame in oof.groupby(["model", "scale"]):
        ensemble = frame.groupby(["manifest_row_id", "participant_id"], as_index=False).agg(label=("label", "first"), score=("score", "mean"))
        ensemble_frames[(model_name, float(scale))] = ensemble
        aggregate_rows.append({"model": model_name, "scale": float(scale), **session_metrics(ensemble)})
    oof_metrics = pd.DataFrame(aggregate_rows).sort_values(["model", "AUROC"], ascending=[True, False])
    oof_metrics.to_csv(output / "train_oof_candidate_metrics.csv", index=False)
    selected = {}
    for model_name in config["models"]:
        part = oof_metrics.loc[oof_metrics.model.eq(model_name)].sort_values(["AUROC", "AUPRC"], ascending=False)
        selected[model_name] = float(part.iloc[0].scale)
    validation = make_dataset(validation_payload)
    full_train = make_dataset(train_payload)
    validation_loader = DataLoader(validation, batch_size=int(config["batch_size"]), shuffle=False)
    validation_seed_frames: dict[str, list[pd.DataFrame]] = {"scalar_mlp": []}
    validation_seed_frames.update({model: [] for model in config["models"]})
    epoch_frame = pd.DataFrame(epoch_rows)
    for seed in config["seeds"]:
        scalar_epochs = int(round(epoch_frame.loc[(epoch_frame.model == "scalar_mlp") & (epoch_frame.seed == int(seed)), "epoch"].median()))
        scalar = train_epochs(new_scalar(config, int(seed) * 1000), full_train, config, device, int(seed) * 1000, scalar_epochs)
        scores = phase2.predict(scalar, validation_loader, validation, device)
        validation_seed_frames["scalar_mlp"].append(session_frame(validation, scores))
        for model_index, model_name in enumerate(config["models"], start=1):
            scale = selected[model_name]
            residual_epochs = int(round(epoch_frame.loc[(epoch_frame.model == model_name) & (epoch_frame.scale == scale) & (epoch_frame.seed == int(seed)), "epoch"].median()))
            model = train_epochs(
                new_residual(copy.deepcopy(scalar), dims, config, scale, model_name.startswith("uncertainty"), int(seed) * 1000 + model_index),
                full_train, config, device, int(seed) * 1000 + model_index, residual_epochs,
            )
            scores = phase2.predict(model, validation_loader, validation, device)
            validation_seed_frames[model_name].append(session_frame(validation, scores))
            torch.save({"model": model.state_dict(), "model_name": model_name, "scale": scale, "seed": int(seed), "epochs": residual_epochs}, output / f"{model_name}_seed{seed}.pth")
    validation_rows, participant_validation_rows, private_validation, validation_ensembles = [], [], [], {}
    for model_name, frames in validation_seed_frames.items():
        combined = pd.concat([frame.assign(seed=int(seed)) for frame, seed in zip(frames, config["seeds"])])
        private_validation.append(combined.assign(model=model_name))
        ensemble = combined.groupby(["manifest_row_id", "participant_id"], as_index=False).agg(label=("label", "first"), score=("score", "mean"))
        validation_ensembles[model_name] = ensemble
        validation_rows.append({"model": model_name, "selected_scale": selected.get(model_name, 0.0), **session_metrics(ensemble)})
        participant_ensemble = ensemble.groupby("participant_id", as_index=False).agg(label=("label", "first"), score=("score", "mean"))
        participant_validation_rows.append({"model": model_name, "selected_scale": selected.get(model_name, 0.0), **phase2.metrics(participant_ensemble)})
    pd.concat(private_validation, ignore_index=True).to_csv(output / "private_validation_predictions.csv", index=False)
    validation_metrics = pd.DataFrame(validation_rows).sort_values("AUROC", ascending=False)
    validation_metrics.to_csv(output / "validation_metrics.csv", index=False)
    pd.DataFrame(participant_validation_rows).sort_values("AUROC", ascending=False).to_csv(output / "validation_participant_metrics.csv", index=False)
    bootstrap_rows = []
    for index, model_name in enumerate(config["models"]):
        bootstrap_rows.append({"candidate": model_name, "reference": "scalar_mlp", **paired_bootstrap(
            validation_ensembles["scalar_mlp"], validation_ensembles[model_name], config["bootstrap_replicates"], int(config["bootstrap_seed"]) + index
        )})
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(output / "validation_bootstrap_vs_scalar.csv", index=False)
    audit = {
        "status": "PASS", "test_loaded": False, "device": str(device), "train_participants": len(train_people),
        "validation_participants": len(validation_people), "participant_overlap": 0, "selected_scales": selected,
        "selection_source": "train-only participant-disjoint OOF with unchanged session labels",
        "validation_role": "frozen check for this run; cohort was already exposed in Phase 2",
        "longitudinal_label_changes_in_train": int((pd.DataFrame({"id": train_payload["participant_id"], "label": train_payload["label"]}).groupby("id").label.nunique() > 1).sum()),
        "primary_evaluation_unit": "session", "bootstrap_cluster": "participant",
    }
    (output / "run_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("Train-only OOF candidate selection")
    print(oof_metrics.to_string(index=False))
    print("\nFrozen validation evaluation")
    print(validation_metrics.to_string(index=False))
    print("\nPaired validation bootstrap vs scalar MLP")
    print(bootstrap.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
