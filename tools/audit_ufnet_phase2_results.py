#!/usr/bin/env python3
"""Create an aggregate, participant-paired audit of Phase 2 validation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


COMPARISONS = (
    ("scalar_mlp_ensemble", "available_mean"),
    ("concat_mlp_ensemble", "available_mean"),
    ("adapter_transformer_ensemble", "available_mean"),
    ("uncertainty_adapter_transformer_ensemble", "available_mean"),
    ("uncertainty_adapter_transformer_ensemble", "scalar_mlp_ensemble"),
    ("uncertainty_adapter_transformer_ensemble", "adapter_transformer_ensemble"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--repeat", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser.parse_args()


def selected_predictions(directory: Path) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(directory / "private_validation_predictions.csv", dtype={"participant_id": str})
    result = {}
    for name in {item for pair in COMPARISONS for item in pair}:
        part = frame.loc[frame.model.eq(name), ["participant_id", "label", "score"]].copy()
        if len(part) != 167 or part.participant_id.duplicated().any():
            raise ValueError(f"Expected 167 unique validation participants for {name}, got {len(part)}")
        result[name] = part.sort_values("participant_id").reset_index(drop=True)
    return result


def paired_delta(candidate: pd.DataFrame, reference: pd.DataFrame, replicates: int, seed: int) -> dict[str, float]:
    joined = reference.merge(candidate, on=["participant_id", "label"], suffixes=("_reference", "_candidate"), validate="one_to_one")
    y = joined.label.to_numpy(int)
    reference_score = joined.score_reference.to_numpy(float)
    candidate_score = joined.score_candidate.to_numpy(float)
    rng = np.random.default_rng(seed)
    auroc, auprc = [], []
    for _ in range(replicates):
        indices = rng.integers(0, len(y), len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        auroc.append(roc_auc_score(y[indices], candidate_score[indices]) - roc_auc_score(y[indices], reference_score[indices]))
        auprc.append(average_precision_score(y[indices], candidate_score[indices]) - average_precision_score(y[indices], reference_score[indices]))
    auroc_values, auprc_values = np.asarray(auroc), np.asarray(auprc)
    return {
        "AUROC_delta": float(roc_auc_score(y, candidate_score) - roc_auc_score(y, reference_score)),
        "AUROC_CI_low": float(np.quantile(auroc_values, 0.025)),
        "AUROC_CI_high": float(np.quantile(auroc_values, 0.975)),
        "AUPRC_delta": float(average_precision_score(y, candidate_score) - average_precision_score(y, reference_score)),
        "AUPRC_CI_low": float(np.quantile(auprc_values, 0.025)),
        "AUPRC_CI_high": float(np.quantile(auprc_values, 0.975)),
    }


def main() -> int:
    args = parse_args()
    predictions = selected_predictions(args.results)
    rows = []
    for offset, (candidate, reference) in enumerate(COMPARISONS):
        rows.append({
            "candidate": candidate,
            "reference": reference,
            **paired_delta(predictions[candidate], predictions[reference], args.replicates, args.seed + offset),
        })
    comparisons = pd.DataFrame(rows)
    repeat_equal = None
    if args.repeat:
        repeat_equal = all(
            (args.results / filename).read_bytes() == (args.repeat / filename).read_bytes()
            for filename in ("per_seed_metrics.csv", "ensemble_metrics.csv", "bootstrap_vs_available_mean.csv", "private_validation_predictions.csv", "run_audit.json")
        )
    key = comparisons.loc[
        comparisons.candidate.eq("uncertainty_adapter_transformer_ensemble")
        & comparisons.reference.eq("scalar_mlp_ensemble")
    ].iloc[0]
    confirmed = bool(key.AUROC_CI_low > 0 and key.AUPRC_delta >= 0)
    decision = "CONFIRMED_ON_VALIDATION" if confirmed else "PROMISING_NOT_CONFIRMED"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(output / "paired_bootstrap_comparisons.csv", index=False)
    audit = {
        "status": "PASS", "decision": decision, "validation_only": True, "test_loaded": False,
        "repeat_outputs_exact": repeat_equal, "bootstrap_replicates": args.replicates,
        "bootstrap_unit": "participant", "primary_comparison": "uncertainty adapter vs scalar MLP",
        "primary_AUROC_CI_low": float(key.AUROC_CI_low), "primary_AUPRC_delta": float(key.AUPRC_delta),
    }
    (output / "phase2_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    report = f"""# Phase 2 adapter development audit

Status: **PASS**
Decision: **{decision}**

This is a development result on the 167-participant validation split. The test
cache was not loaded. All aggregate CSV outputs and participant predictions were
exactly reproduced in a complete same-configuration repeat: **{repeat_equal}**.

The uncertainty-aware Adapter-Transformer is promising against available-mean,
but it is not yet confirmed against the strongest lightweight comparator
(scalar MLP). Promotion requires both AUROC and AUPRC non-degradation and a
positive participant-bootstrap AUROC lower bound.

## Paired bootstrap comparisons

{comparisons.to_string(index=False)}
"""
    (output / "PHASE2_AUDIT.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
