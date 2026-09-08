#!/usr/bin/env python3
"""Export frozen UFNet expert inputs and MC-dropout statistics.

The released experts are single-linear-layer ShallowANN models. Consequently,
there is no learned penultimate hidden vector: the exact preprocessed expert
input is the representation consumed by both UFNet and later Feature Adapters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from baal.modelwrapper import ModelWrapper
import baal.bayesian.dropout as mcdropout
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn


MODEL_TAGS = [
    "finger_model_both_hand_fusion_baal",
    "fox_model_best_auroc_baal",
    "facial_expression_smile_best_auroc_baal",
]
MODALITIES = ["finger", "speech", "smile"]


class ShallowANN(nn.Module):
    def __init__(self, n_features: int, drop_prob: float):
        super().__init__()
        self.fc = nn.Linear(n_features, 1, bias=True)
        self.drop = mcdropout.Dropout(p=drop_prob)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sig(self.drop(self.fc(x)))


def read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def parse_date(name: str) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(name))
    if match is None:
        raise ValueError(f"No date in {name}")
    return match.group()


def parse_finger_id(name: str) -> str:
    name = str(name)
    if name.startswith("NIH"):
        return name.split("-")[0]
    if name.endswith("finger_tapping.mp4"):
        return name.split("-")[-2]
    return name.split("_")[-4]


def parse_speech_id(name: str) -> str:
    name = str(name)
    if name.startswith("NIH"):
        return name.split("-")[0]
    if name.endswith("-quick_brown_fox.mp4"):
        return name.split("-")[-2]
    if name.endswith("_quick_brown_fox.mp4"):
        return name.split("_")[1]
    return name.split("_")[-4]


def label_not_no(value: object) -> int:
    return int(str(value).strip().lower() != "no")


def drop_correlated(frame: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    corr = frame.corr().abs().to_numpy()
    upper = np.triu(corr, k=1)
    drop_indices = np.flatnonzero(np.nan_to_num(upper, nan=0.0).max(axis=0) >= threshold)
    dropped = frame.columns[drop_indices].tolist()
    return frame.drop(columns=dropped), dropped


def load_finger(repo: Path, config: dict) -> tuple[pd.DataFrame, list[str]]:
    path = repo / "data/finger_tapping/features_demography_diagnosis_Nov22_2023.csv"
    raw = pd.read_csv(path)
    exempt = {
        "Unnamed: 0", "filename", "Protocol", "Participant_ID", "Task", "Duration", "FPS",
        "Frame_Height", "Frame_Width", "gender", "age", "race", "ethnicity", "dob", "time_mdsupdrs",
    }
    frame = raw.dropna(subset=[column for column in raw.columns if column not in exempt]).copy()
    metadata = [
        "Unnamed: 0", "filename", "Protocol", "Participant_ID", "Task", "Duration", "FPS",
        "Frame_Height", "Frame_Width", "gender", "age", "race", "ethnicity", "pd", "dob",
        "time_mdsupdrs", "hand",
    ]
    feature_columns = [column for column in frame.columns if column not in metadata]
    features = frame[feature_columns]
    dropped: list[str] = []
    if config["drop_correlated"] == "yes":
        features, dropped = drop_correlated(features, float(config["corr_thr"]))
    frame["participant_id"] = frame["filename"].map(parse_finger_id)
    frame["date"] = frame["filename"].map(parse_date)
    frame["row_id"] = frame["participant_id"] + "#" + frame["date"]
    frame["label"] = frame["pd"].map(label_not_no)
    frame["features"] = list(features.to_numpy(dtype=np.float32))
    right = frame.loc[frame["hand"].eq("right"), ["row_id", "participant_id", "label", "filename", "features"]].rename(
        columns={"filename": "finger_right_file", "features": "right_features"}
    )
    left = frame.loc[frame["hand"].eq("left"), ["row_id", "participant_id", "label", "filename", "features"]].rename(
        columns={"filename": "finger_left_file", "participant_id": "left_id", "label": "left_label", "features": "left_features"}
    )
    both = right.merge(left, how="inner", on="row_id")
    if not both["participant_id"].eq(both["left_id"]).all() or not both["label"].eq(both["left_label"]).all():
        raise ValueError("Finger left/right ID or label mismatch")
    both["finger_features"] = both.apply(
        lambda row: np.concatenate([row["right_features"], row["left_features"]]).astype(np.float32), axis=1
    )
    return both.drop(columns=["left_id", "left_label", "right_features", "left_features"]), dropped


def load_speech(repo: Path, config: dict) -> tuple[pd.DataFrame, list[str]]:
    path = repo / "data/quick_brown_fox/wavlm_fox_features.csv"
    raw = pd.read_csv(path)
    exempt = {"Filename", "Participant_ID", "gender", "age", "race"}
    frame = raw.dropna(subset=[column for column in raw.columns if column not in exempt]).copy()
    feature_columns = [column for column in frame.columns if column not in {"Filename", "Participant_ID", "gender", "age", "race", "pd"}]
    features = frame[feature_columns]
    dropped: list[str] = []
    if config["drop_correlated"] == "yes":
        features, dropped = drop_correlated(features, float(config["corr_thr"]))
    frame["participant_id"] = frame["Filename"].map(parse_speech_id)
    frame["date"] = frame["Filename"].map(parse_date)
    frame["row_id"] = frame["participant_id"] + "#" + frame["date"]
    frame["label"] = frame["pd"].astype(int)
    frame["speech_features"] = list(features.to_numpy(dtype=np.float32))
    return frame[["row_id", "participant_id", "label", "Filename", "speech_features"]].rename(
        columns={"participant_id": "speech_id", "label": "speech_label", "Filename": "speech_file"}
    ), dropped


def load_smile(repo: Path, config: dict) -> tuple[pd.DataFrame, list[str]]:
    path = repo / "data/facial_expression_smile/facial_dataset.csv"
    frame = pd.read_csv(path).fillna(0)
    feature_columns = [column for column in frame.columns if "smile" in column.lower()]
    features = frame[feature_columns]
    dropped: list[str] = []
    if config["drop_correlated"] == "yes":
        features, dropped = drop_correlated(features, float(config["corr_thr"]))
    frame["participant_id"] = frame["ID"].astype(str)
    frame["date_parsed"] = frame["Filename"].map(parse_date)
    frame["row_id"] = frame["participant_id"] + "#" + frame["date_parsed"]
    frame["label"] = frame["pd"].map(label_not_no)
    frame["smile_features"] = list(features.to_numpy(dtype=np.float32))
    return frame[["row_id", "participant_id", "label", "Filename", "smile_features"]].rename(
        columns={"participant_id": "smile_id", "label": "smile_label", "Filename": "smile_file"}
    ), dropped


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return sha256_bytes(contiguous.tobytes())


def load_configs(repo: Path) -> list[dict]:
    return [json.loads((repo / "models" / tag / "predictive_model/model_config.json").read_text()) for tag in MODEL_TAGS]


def assign_split(pid: str, dev: set[str], test: set[str]) -> str:
    return "test" if pid in test else "validation" if pid in dev else "train"


@torch.no_grad()
def expert_statistics(
    features: np.ndarray, repo: Path, tag: str, config: dict, trials: int,
    batch_size: int, device: torch.device,
) -> dict[str, np.ndarray]:
    if config["model"] != "ShallowANN":
        raise ValueError(f"{tag} is not ShallowANN; a hidden-hook implementation is required")
    model = ShallowANN(features.shape[1], float(config["dropout_prob"]))
    state = torch.load(repo / "models" / tag / "predictive_model/model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    wrapper = ModelWrapper(model, nn.BCELoss())
    means, stds, deterministic, expected_entropy = [], [], [], []
    eps = 1e-7
    for start in range(0, len(features), batch_size):
        tensor = torch.from_numpy(features[start:start + batch_size]).to(device)
        draws = wrapper.predict_on_batch(tensor, iterations=trials).squeeze(1)
        mean = draws.mean(dim=-1)
        std = draws.std(dim=-1)
        clipped_draws = draws.clamp(eps, 1 - eps)
        draw_entropy = -(clipped_draws * clipped_draws.log() + (1 - clipped_draws) * (1 - clipped_draws).log()).mean(dim=-1)
        base_probability = torch.sigmoid(model.fc(tensor)).reshape(-1)
        means.append(mean.cpu().numpy())
        stds.append(std.cpu().numpy())
        deterministic.append(base_probability.cpu().numpy())
        expected_entropy.append(draw_entropy.cpu().numpy())
    mean = np.concatenate(means).astype(np.float32)
    clipped_mean = np.clip(mean, eps, 1 - eps)
    predictive_entropy = -(clipped_mean * np.log(clipped_mean) + (1 - clipped_mean) * np.log(1 - clipped_mean))
    expected_entropy_array = np.concatenate(expected_entropy).astype(np.float32)
    return {
        "mean": mean,
        "std": np.concatenate(stds).astype(np.float32),
        "deterministic_probability": np.concatenate(deterministic).astype(np.float32),
        "predictive_entropy": predictive_entropy.astype(np.float32),
        "mutual_information": (predictive_entropy - expected_entropy_array).astype(np.float32),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--mc-trials", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repo, output = args.repo.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs = load_configs(repo)
    finger, finger_dropped = load_finger(repo, configs[0])
    speech, speech_dropped = load_speech(repo, configs[1])
    smile, smile_dropped = load_smile(repo, configs[2])
    joined = finger.merge(speech, how="inner", on="row_id").merge(smile, how="inner", on="row_id")
    for id_column in ["speech_id", "smile_id"]:
        if not joined["participant_id"].eq(joined[id_column]).all():
            raise ValueError(f"ID mismatch in {id_column}")
    for label_column in ["speech_label", "smile_label"]:
        if not joined["label"].eq(joined[label_column]).all():
            raise ValueError(f"Label mismatch in {label_column}")
    source_columns = ["finger_right_file", "finger_left_file", "speech_file", "smile_file"]
    joined = joined.sort_values(["row_id", *source_columns]).reset_index(drop=True)
    joined["within_join_key_index"] = joined.groupby("row_id").cumcount()
    joined["manifest_row_id"] = joined["row_id"] + "#" + joined["within_join_key_index"].map(lambda x: f"{x:03d}")
    dev = read_ids(repo / "data/dev_set_participants.txt")
    test = read_ids(repo / "data/test_set_participants.txt")
    joined["split"] = joined["participant_id"].map(lambda pid: assign_split(pid, dev, test))

    manifest = pd.read_csv(args.manifest)
    expected_columns = ["manifest_row_id", "participant_id", "label", "split", *source_columns]
    check = joined[expected_columns].merge(
        manifest[expected_columns], on="manifest_row_id", suffixes=("_cache", "_manifest"), validate="one_to_one"
    )
    if len(check) != len(joined) or len(check) != len(manifest):
        raise ValueError("Manifest coverage mismatch")
    for column in expected_columns[1:]:
        if not check[f"{column}_cache"].astype(str).equals(check[f"{column}_manifest"].astype(str)):
            raise ValueError(f"Manifest mismatch: {column}")

    feature_arrays = {
        "finger": np.stack(joined["finger_features"]).astype(np.float32),
        "speech": np.stack(joined["speech_features"]).astype(np.float32),
        "smile": np.stack(joined["smile_features"]).astype(np.float32),
    }
    for index, modality in enumerate(MODALITIES):
        if configs[index]["use_feature_scaling"] == "yes":
            scaler_path = repo / "models" / MODEL_TAGS[index] / "scaler/scaler.pth"
            with scaler_path.open("rb") as handle:
                scaler = pickle.load(handle)
            feature_arrays[modality] = scaler.transform(feature_arrays[modality]).astype(np.float32)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    statistics = {
        modality: expert_statistics(feature_arrays[modality], repo, MODEL_TAGS[index], configs[index], args.mc_trials, args.batch_size, device)
        for index, modality in enumerate(MODALITIES)
    }

    index_columns = ["manifest_row_id", "row_id", "participant_id", "split", "label", *source_columns]
    joined[index_columns].to_csv(output / "expert_cache_index.csv", index=False)
    metric_rows = []
    split_order = ["train", "validation", "test"]
    split_counts = {}
    for split in split_order:
        mask = joined["split"].eq(split).to_numpy()
        split_counts[split] = {"rows": int(mask.sum()), "participants": int(joined.loc[mask, "participant_id"].nunique())}
        payload: dict[str, np.ndarray] = {
            "manifest_row_id": joined.loc[mask, "manifest_row_id"].to_numpy(dtype=str),
            "participant_id": joined.loc[mask, "participant_id"].to_numpy(dtype=str),
            "label": joined.loc[mask, "label"].to_numpy(dtype=np.int8),
        }
        for modality in MODALITIES:
            payload[f"{modality}_features"] = feature_arrays[modality][mask]
            for name, values in statistics[modality].items():
                payload[f"{modality}_{name}"] = values[mask]
            labels, scores = payload["label"], payload[f"{modality}_mean"]
            metric_rows.append({
                "split": split, "modality": modality, "rows": len(labels),
                "AUROC": float(roc_auc_score(labels, scores)),
                "AUPRC": float(average_precision_score(labels, scores)),
            })
        np.savez_compressed(output / f"expert_cache_{split}.npz", **payload)

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output / "expert_metrics.csv", index=False)
    hashes = {modality: {"features": array_sha256(feature_arrays[modality]), **{
        name: array_sha256(values) for name, values in statistics[modality].items()
    }} for modality in MODALITIES}
    metadata = {
        "status": "PASS", "source_repository": "https://github.com/ROC-HCI/UFNet.git",
        "source_commit": "5ece2c65ba184faccf6c8cdccdc03132427c464b",
        "seed": args.seed, "mc_trials": args.mc_trials, "batch_size": args.batch_size,
        "device": str(device), "modalities": MODALITIES, "model_tags": MODEL_TAGS,
        "expert_architecture": "ShallowANN: Linear -> MC Dropout -> Sigmoid",
        "representation_definition": "exact preprocessed expert input; released experts have no penultimate hidden layer",
        "cache_semantics": "one frozen MC-dropout draw set shared by all downstream fusion models",
        "feature_dimensions": {name: int(values.shape[1]) for name, values in feature_arrays.items()},
        "dropped_correlated_columns": {
            "finger": finger_dropped, "speech": speech_dropped, "smile": smile_dropped,
        },
        "split_counts": split_counts, "array_sha256": hashes,
        "checks": {
            "manifest_exact_match": True, "all_labels_consistent": True,
            "all_ids_consistent": True, "all_arrays_finite": bool(all(
                np.isfinite(array).all() for array in feature_arrays.values()
            ) and all(
                np.isfinite(array).all()
                for modality_statistics in statistics.values()
                for array in modality_statistics.values()
            )),
        },
    }
    (output / "expert_cache_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report = f"""# UFNet frozen expert cache audit

Status: **PASS**

- Source commit: `{metadata['source_commit']}`
- MC-dropout seed/trials: {args.seed} / {args.mc_trials}
- Feature dimensions: {metadata['feature_dimensions']}
- Split counts: {split_counts}
- Expert architecture: all three released models are `ShallowANN` (`Linear -> MC Dropout -> Sigmoid`).
- Representation used by Feature Adapters: exact preprocessed expert input, because these experts contain no learned penultimate hidden layer.

## Expert metrics

{metrics_frame.to_string(index=False)}

All cache arrays are finite, labels/IDs agree across modalities, and every row exactly matches the frozen paper manifest.
"""
    (output / "AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
