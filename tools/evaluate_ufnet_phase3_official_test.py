#!/usr/bin/env python3
"""One-time evaluation of frozen fusion candidates on the UFNet official test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader

import train_ufnet_phase2_adapters as phase2
import train_ufnet_phase21_residual as phase21


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--phase2-results", required=True, type=Path)
    parser.add_argument("--phase21-results", required=True, type=Path)
    parser.add_argument("--phase2-config", required=True, type=Path)
    parser.add_argument("--phase21-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def session_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    y, p = frame.label.to_numpy(int), np.clip(frame.score.to_numpy(float), 1e-7, 1 - 1e-7)
    hard = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, hard, labels=[0, 1]).ravel()
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    return {
        "sessions": int(len(y)), "positives": int(y.sum()),
        "AUROC": float(roc_auc_score(y, p)), "AUPRC": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, hard)), "balanced_accuracy": float(balanced_accuracy_score(y, hard)),
        "paper_weighted_accuracy": float((ppv + npv) / 2),
        "F1": float(f1_score(y, hard)), "sensitivity": float(recall_score(y, hard)),
        "specificity": float(recall_score(y, hard, pos_label=0)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def frame_from_scores(dataset, scores):
    return pd.DataFrame({
        "manifest_row_id": dataset.row_ids, "participant_id": dataset.participant_ids,
        "label": dataset.labels.numpy().astype(int), "score": scores,
    })


def participant_metrics(frame: pd.DataFrame, threshold: float):
    if (frame.groupby("participant_id").label.nunique() > 1).any():
        raise ValueError("Official test contains inconsistent participant labels")
    part = frame.groupby("participant_id", as_index=False).agg(label=("label", "first"), score=("score", "mean"))
    result = session_metrics(part, threshold)
    result["participants"] = result.pop("sessions")
    return result


def bootstrap(all_frames, threshold, replicates, seed):
    reference_names = ("available_mean", "scalar_mlp")
    base = all_frames["available_mean"][["manifest_row_id", "participant_id", "label"]]
    table = base.copy()
    for name, frame in all_frames.items():
        table = table.merge(frame[["manifest_row_id", "score"]].rename(columns={"score": name}), on="manifest_row_id", validate="one_to_one")
    clusters = {pid: np.flatnonzero(table.participant_id.to_numpy(str) == pid) for pid in table.participant_id.unique()}
    people = np.asarray(list(clusters), dtype=str)
    rng = np.random.default_rng(seed)
    absolute = {name: {metric: [] for metric in ("AUROC", "AUPRC")} for name in all_frames}
    deltas = {(name, ref): {metric: [] for metric in ("AUROC", "AUPRC")} for name in all_frames for ref in reference_names if name != ref}
    for _ in range(replicates):
        sampled = rng.choice(people, len(people), replace=True)
        index = np.concatenate([clusters[pid] for pid in sampled])
        y = table.label.to_numpy(int)[index]
        if len(np.unique(y)) < 2:
            continue
        values = {}
        for name in all_frames:
            score = table[name].to_numpy(float)[index]
            values[name] = {"AUROC": roc_auc_score(y, score), "AUPRC": average_precision_score(y, score)}
            for metric in values[name]:
                absolute[name][metric].append(values[name][metric])
        for (name, ref), metrics in deltas.items():
            for metric in metrics:
                metrics[metric].append(values[name][metric] - values[ref][metric])
    absolute_rows, delta_rows = [], []
    for name, metrics in absolute.items():
        absolute_rows.append({"model": name, **{
            f"{metric}_{bound}": float(np.quantile(values, 0.025 if bound == "CI_low" else 0.975))
            for metric, values in metrics.items() for bound in ("CI_low", "CI_high")
        }})
    for (name, ref), metrics in deltas.items():
        point_candidate, point_reference = session_metrics(all_frames[name], threshold), session_metrics(all_frames[ref], threshold)
        delta_rows.append({"candidate": name, "reference": ref, **{
            f"{metric}_delta": point_candidate[metric] - point_reference[metric] for metric in metrics
        }, **{
            f"{metric}_{bound}": float(np.quantile(values, 0.025 if bound == "CI_low" else 0.975))
            for metric, values in metrics.items() for bound in ("CI_low", "CI_high")
        }})
    return pd.DataFrame(absolute_rows), pd.DataFrame(delta_rows)


def main():
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status_before_run") != "FROZEN_UNOPENED":
        raise ValueError("Protocol is not frozen and unopened")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    # Phase 2 intentionally rejects test loading, so the one-time evaluator
    # performs the same validated NPZ load directly under this frozen protocol.
    payload = dict(np.load(args.cache / "expert_cache_test.npz", allow_pickle=False))
    dataset = phase2.CacheDataset(payload)
    dataset.row_ids = payload["manifest_row_id"].astype(str)
    expected = protocol["test_contract"]
    if len(dataset) != expected["sessions"] or len(set(dataset.participant_ids)) != expected["participants"]:
        raise ValueError("Official test count mismatch")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    dims = [value.shape[1] for value in dataset.features]
    phase2_config = json.loads(args.phase2_config.read_text(encoding="utf-8"))
    phase21_config = json.loads(args.phase21_config.read_text(encoding="utf-8"))
    all_frames = {}
    expert_mean = np.mean(np.stack([payload[f"{m}_mean"] for m in phase2.MODALITIES]), axis=0)
    all_frames["available_mean"] = frame_from_scores(dataset, expert_mean)
    for name, spec in protocol["models"].items():
        if name == "available_mean":
            continue
        seed_scores = []
        for seed in spec["seeds"]:
            if spec["source"] == "phase2":
                model = phase2.make_model(name, dims, phase2_config)
                path = args.phase2_results / f"{name}_seed{seed}.pth"
            else:
                scalar = phase2.ScalarMLP(int(phase21_config["scalar_width"]), float(phase21_config["dropout"]))
                model = phase21.ResidualAdapter(
                    scalar, dims, int(phase21_config["adapter_width"]), float(phase21_config["dropout"]),
                    float(spec["scale"]), name == "uncertainty_gated_residual_adapter",
                )
                path = args.phase21_results / f"{name}_seed{seed}.pth"
            checkpoint = torch.load(path, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=True)
            model.to(device)
            seed_scores.append(phase2.predict(model, loader, dataset, device))
        all_frames[name] = frame_from_scores(dataset, np.mean(seed_scores, axis=0))
    threshold = float(expected["threshold"])
    session_rows = [{"model": name, **session_metrics(frame, threshold)} for name, frame in all_frames.items()]
    participant_rows = [{"model": name, **participant_metrics(frame, threshold)} for name, frame in all_frames.items()]
    session_table = pd.DataFrame(session_rows).sort_values("AUROC", ascending=False)
    participant_table = pd.DataFrame(participant_rows).sort_values("AUROC", ascending=False)
    absolute_ci, delta_ci = bootstrap(all_frames, threshold, int(protocol["bootstrap_replicates"]), int(protocol["bootstrap_seed"]))
    session_table.to_csv(output / "official_test_session_metrics.csv", index=False)
    participant_table.to_csv(output / "official_test_participant_metrics.csv", index=False)
    absolute_ci.to_csv(output / "participant_cluster_bootstrap_absolute.csv", index=False)
    delta_ci.to_csv(output / "participant_cluster_bootstrap_deltas.csv", index=False)
    pd.concat([frame.assign(model=name) for name, frame in all_frames.items()], ignore_index=True).to_csv(output / "private_official_test_predictions.csv", index=False)
    primary = session_table.loc[session_table.model.eq(protocol["primary_candidate"])].iloc[0].to_dict()
    reproduced = protocol["reproduced_UFNet_30_seed_mean"]
    comparison = {
        metric: float(primary[metric] - reproduced[metric])
        for metric in ("AUROC", "AUPRC", "accuracy", "F1", "sensitivity", "specificity")
    }
    comparison["paper_weighted_accuracy"] = float(
        primary["paper_weighted_accuracy"] - reproduced["balanced_accuracy"]
    )
    audit = {
        "status": "PASS", "protocol_sha256": file_sha256(args.protocol), "test_sessions": len(dataset),
        "test_participants": len(set(dataset.participant_ids)), "coverage": 1.0, "threshold": threshold,
        "primary_candidate": protocol["primary_candidate"], "primary_minus_reproduced_UFNet": comparison,
        "no_test_time_selection_or_retuning": True, "device": str(device),
    }
    (output / "run_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("Official test session-level metrics")
    print(session_table.to_string(index=False))
    print("\nPrimary minus reproduced UFNet")
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
