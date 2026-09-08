#!/usr/bin/env python3
"""Evaluate frozen PD probabilities under the UFNet paper-aligned contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


HIGHER_IS_BETTER = {
    "AUROC": True, "AUPRC": True, "F1": True, "balanced_accuracy": True,
    "sensitivity": True, "specificity": True, "Brier": False, "ECE": False,
}


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float, ece_bins: int) -> dict[str, float]:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    edges = np.linspace(0.0, 1.0, ece_bins + 1)
    bin_ids = np.minimum(np.digitize(scores, edges[1:-1]), ece_bins - 1)
    ece = 0.0
    for bin_id in range(ece_bins):
        mask = bin_ids == bin_id
        if mask.any():
            ece += mask.mean() * abs(labels[mask].mean() - scores[mask].mean())
    return {
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPRC": float(average_precision_score(labels, scores)),
        "F1": float(f1_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "Brier": float(brier_score_loss(labels, scores)),
        "ECE": float(ece),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "coverage": 1.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--baseline-predictions", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--key-col", default="manifest_row_id")
    parser.add_argument("--participant-col", default="participant_id")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--score-col", default="probability")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--ece-bins", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260908)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    required = {args.key_col, args.participant_col, args.label_col, args.score_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if frame[args.key_col].duplicated().any():
        raise ValueError("Prediction keys are not unique")
    if frame[args.score_col].isna().any() or not frame[args.score_col].between(0, 1).all():
        raise ValueError("Probabilities must be complete and within [0, 1]")

    if args.manifest:
        manifest = pd.read_csv(args.manifest)
        manifest = manifest.loc[manifest["split"].eq("test"), [args.key_col, args.participant_col, args.label_col]]
        if set(frame[args.key_col]) != set(manifest[args.key_col]):
            raise ValueError("Predictions do not exactly cover the frozen test manifest")
        checked = manifest.merge(frame, on=args.key_col, suffixes=("_manifest", "_prediction"), validate="one_to_one")
        for column in [args.participant_col, args.label_col]:
            if not checked[f"{column}_manifest"].astype(str).equals(checked[f"{column}_prediction"].astype(str)):
                raise ValueError(f"Manifest mismatch in {column}")

    labels = frame[args.label_col].to_numpy(dtype=int)
    scores = frame[args.score_col].to_numpy(dtype=float)
    result = {"point_estimates": metrics(labels, scores, args.threshold, args.ece_bins)}

    baseline_scores = None
    if args.baseline_predictions:
        baseline = pd.read_csv(args.baseline_predictions)
        baseline_required = {args.key_col, args.label_col, args.score_col}
        if baseline_required - set(baseline.columns):
            raise ValueError("Baseline predictions are missing required columns")
        if baseline[args.key_col].duplicated().any() or set(baseline[args.key_col]) != set(frame[args.key_col]):
            raise ValueError("Baseline keys must uniquely and exactly match candidate keys")
        aligned = frame[[args.key_col, args.label_col]].merge(
            baseline[[args.key_col, args.label_col, args.score_col]], on=args.key_col,
            suffixes=("_candidate", "_baseline"), validate="one_to_one",
        )
        if not aligned[f"{args.label_col}_candidate"].equals(aligned[f"{args.label_col}_baseline"]):
            raise ValueError("Candidate and baseline labels differ")
        baseline_scores = aligned[args.score_col].to_numpy(dtype=float)
        baseline_point = metrics(labels, baseline_scores, args.threshold, args.ece_bins)
        result["baseline_point_estimates"] = baseline_point
        result["candidate_minus_baseline"] = {
            name: result["point_estimates"][name] - baseline_point[name]
            for name in ["AUROC", "AUPRC", "F1", "balanced_accuracy", "sensitivity", "specificity", "Brier", "ECE"]
        }

    rng = np.random.default_rng(args.bootstrap_seed)
    participants = frame[args.participant_col].astype(str).unique()
    bootstrap_metrics = []
    bootstrap_deltas = []
    for _ in range(args.bootstrap):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        indices = np.concatenate([np.flatnonzero(frame[args.participant_col].astype(str).to_numpy() == pid) for pid in sampled])
        sampled_labels, sampled_scores = labels[indices], scores[indices]
        if np.unique(sampled_labels).size == 2:
            candidate_item = metrics(sampled_labels, sampled_scores, args.threshold, args.ece_bins)
            bootstrap_metrics.append(candidate_item)
            if baseline_scores is not None:
                baseline_item = metrics(sampled_labels, baseline_scores[indices], args.threshold, args.ece_bins)
                bootstrap_deltas.append({name: candidate_item[name] - baseline_item[name] for name in candidate_item})
    ci = {}
    for name in ["AUROC", "AUPRC", "F1", "balanced_accuracy", "sensitivity", "specificity", "Brier", "ECE"]:
        values = np.asarray([item[name] for item in bootstrap_metrics])
        ci[name] = {"low": float(np.quantile(values, 0.025)), "high": float(np.quantile(values, 0.975))}
    result["participant_cluster_bootstrap_95CI"] = ci
    if bootstrap_deltas:
        delta_ci = {}
        for name in HIGHER_IS_BETTER:
            values = np.asarray([item[name] for item in bootstrap_deltas])
            delta_ci[name] = {
                "low": float(np.quantile(values, 0.025)),
                "high": float(np.quantile(values, 0.975)),
                "higher_is_better": HIGHER_IS_BETTER[name],
                "probability_candidate_better": float(
                    np.mean(values > 0) if HIGHER_IS_BETTER[name] else np.mean(values < 0)
                ),
            }
        result["paired_participant_cluster_bootstrap_delta_95CI"] = delta_ci
    result["contract"] = {
        "rows": len(frame), "participants": int(frame[args.participant_col].nunique()),
        "threshold": args.threshold, "ece_bins": args.ece_bins,
        "bootstrap_resamples_requested": args.bootstrap,
        "bootstrap_resamples_valid": len(bootstrap_metrics), "bootstrap_seed": args.bootstrap_seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
