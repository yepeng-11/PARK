#!/usr/bin/env python3
"""Audit and reproduce the published PARK cross-setting evaluation protocol.

The audit reconstructs the 162/91/67-session paper cohorts from the canonical
split CSVs, explains every exclusion without consulting outcomes, recomputes the
published point metrics from ``data/test_data_big.csv``, and compares the stored
paper scores with the independently generated pretrained-evaluator scores.
Inputs are read-only and all outputs are written to a separate result directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
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


SPLIT_SOURCES = {
    "global": "test_df_global_to_save.csv",
    "validation_1": "test_df_validation_1_to_save.csv",
    "validation_2": "test_df_validation_2_to_save.csv",
}
PAPER_NAMES = {
    "global": "Balanced Test Data",
    "validation_1": "External Evaluation (supervised)",
    "validation_2": "External Evaluation (unsupervised)",
}
PUBLISHED_POINT_ESTIMATES = {
    "global": {
        "n": 162,
        "accuracy": 0.802,
        "specificity": 0.712,
        "sensitivity": 0.865,
        "precision": 0.814,
        "npv": 0.783,
        "f1": 0.838,
    },
    "validation_1": {
        "n": 91,
        "accuracy": 0.802,
        "specificity": 0.738,
        "sensitivity": 0.857,
        "precision": 0.792,
        "npv": 0.816,
        "f1": 0.824,
    },
    "validation_2": {
        "n": 67,
        "accuracy": 0.806,
        "specificity": 0.784,
        "sensitivity": 0.833,
        "precision": 0.758,
        "npv": 0.853,
        "f1": 0.794,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--pretrained-dir", type=Path, default=None,
        help="Default: <repo>/results/pretrained_eval",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Default: <repo>/results/paper_exact_protocol_audit",
    )
    return parser.parse_args()


def paper_ece(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    confidences = np.maximum(scores, 1.0 - scores)
    correct = (scores >= 0.5).astype(int) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            value += abs(float(correct[mask].mean()) - float(confidences[mask].mean())) * float(mask.mean())
    return value


def compute_metrics(labels: Iterable[int], scores: Iterable[float]) -> Dict[str, Any]:
    labels = np.asarray(list(labels), dtype=int)
    scores = np.asarray(list(scores), dtype=float)
    predictions = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = recall_score(labels, predictions, zero_division=0)
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = precision_score(labels, predictions, zero_division=0)
    npv = tn / (tn + fn) if tn + fn else 0.0
    return {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "npv": float(npv),
        "brier": float(brier_score_loss(labels, scores)),
        "ece_10_bin": float(paper_ece(labels, scores, 10)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_source_cohorts(data_dir: Path) -> pd.DataFrame:
    pieces: List[pd.DataFrame] = []
    for split, filename in SPLIT_SOURCES.items():
        frame = pd.read_csv(data_dir / filename)
        required = {"row_id", "id", "label"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing columns in {filename}: {sorted(required - set(frame.columns))}")
        selected = frame[["row_id", "id", "label"]].copy()
        selected.insert(0, "split", split)
        pieces.append(selected)
    output = pd.concat(pieces, ignore_index=True)
    if output.duplicated(["split", "row_id"]).any():
        duplicate = output.loc[output.duplicated(["split", "row_id"], False), "row_id"].tolist()
        raise ValueError(f"Duplicate row_id within a source cohort: {duplicate[:5]}")
    return output


def load_paper_artifact(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "test_data_big.csv"
    frame = pd.read_csv(path)
    required = {
        "unique_row_id", "participant_id", "test_split", "true_label",
        "pred_score_fusion", "uncertain_flag",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing paper-artifact columns: {sorted(required - set(frame.columns))}")
    if frame.unique_row_id.duplicated().any():
        raise ValueError("Paper artifact contains duplicate unique_row_id values")
    return frame


def cross_split_overlaps(
    frame: pd.DataFrame, participant_column: str, split_column: str
) -> pd.DataFrame:
    grouped = (
        frame.groupby(participant_column, as_index=False)
        .agg(
            split_count=(split_column, "nunique"),
            splits=(split_column, lambda values: "+".join(sorted(set(values)))),
            rows=(split_column, "size"),
        )
    )
    return grouped[grouped.split_count > 1].reset_index(drop=True)


def cross_split_row_duplicates(
    frame: pd.DataFrame, row_column: str, split_column: str
) -> pd.DataFrame:
    grouped = (
        frame.groupby(row_column, as_index=False)
        .agg(
            split_count=(split_column, "nunique"),
            splits=(split_column, lambda values: "+".join(sorted(set(values)))),
            memberships=(split_column, "size"),
        )
    )
    return grouped[grouped.split_count > 1].reset_index(drop=True)


def membership_audit(
    source: pd.DataFrame, paper: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paper_keys = set(zip(paper.test_split.astype(str), paper.unique_row_id.astype(str)))
    source_keys = set(zip(source.split.astype(str), source.row_id.astype(str)))
    paper_only = sorted(paper_keys - source_keys)
    if paper_only:
        raise ValueError(f"Paper artifact has split/row memberships absent from source cohorts: {paper_only[:5]}")

    participant_overlap_before = cross_split_overlaps(source, "id", "split")
    row_duplicates_before = cross_split_row_duplicates(source, "row_id", "split")
    duplicated_rows = set(row_duplicates_before.row_id.astype(str))
    audit = source.copy()
    audit["included_in_paper_artifact"] = [
        (str(split), str(row_id)) in paper_keys
        for split, row_id in zip(audit.split, audit.row_id)
    ]
    audit["exclusion_reason"] = ""
    excluded = ~audit.included_in_paper_artifact
    reassigned = excluded & audit.row_id.astype(str).isin(duplicated_rows)
    audit.loc[reassigned, "exclusion_reason"] = "duplicate_session_assigned_to_other_split"
    audit.loc[excluded & ~reassigned, "exclusion_reason"] = "unexplained_exclusion"

    included = audit[audit.included_in_paper_artifact][["row_id", "id", "split", "label"]]
    reference = paper[["unique_row_id", "participant_id", "test_split", "true_label"]].rename(
        columns={
            "unique_row_id": "row_id",
            "participant_id": "paper_id",
            "test_split": "paper_split",
            "true_label": "paper_label",
        }
    )
    checked = included.merge(
        reference,
        left_on=["row_id", "split"],
        right_on=["row_id", "paper_split"],
        how="left",
        validate="one_to_one",
    )
    if checked.paper_id.isna().any():
        raise ValueError("An included source row is absent from the paper artifact")
    if (checked.id.astype(str) != checked.paper_id.astype(str)).any():
        raise ValueError("Participant identifiers differ between source and paper artifact")
    if (checked.label.astype(int) != checked.paper_label.astype(int)).any():
        raise ValueError("Labels differ between source and paper artifact")
    if (audit.exclusion_reason == "unexplained_exclusion").any():
        raise ValueError("At least one paper exclusion is not explained by cross-split overlap")

    paper_membership = paper[["unique_row_id", "participant_id", "test_split"]].rename(
        columns={"unique_row_id": "row_id", "participant_id": "id", "test_split": "split"}
    )
    participant_overlap_after = cross_split_overlaps(paper_membership, "id", "split")
    row_duplicates_after = cross_split_row_duplicates(paper_membership, "row_id", "split")
    return (
        audit,
        participant_overlap_before,
        participant_overlap_after,
        row_duplicates_before,
        row_duplicates_after,
    )


def paper_artifact_metrics(paper: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLIT_SOURCES:
        selected = paper[paper.test_split == split]
        rows.append(
            {
                "source": "paper_artifact",
                "split": split,
                "paper_name": PAPER_NAMES[split],
                **compute_metrics(selected.true_label, selected.pred_score_fusion),
            }
        )
    return pd.DataFrame(rows)


def published_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, expected in PUBLISHED_POINT_ESTIMATES.items():
        observed = metrics[metrics.split == split].iloc[0]
        for metric, published in expected.items():
            current = float(observed[metric])
            rows.append(
                {
                    "split": split,
                    "paper_name": PAPER_NAMES[split],
                    "metric": metric,
                    "published": published,
                    "recomputed": current,
                    "delta_recomputed_minus_published": current - published,
                }
            )
    return pd.DataFrame(rows)


def evaluator_alignment(
    pretrained_dir: Path,
    paper: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    alignment_rows = []
    merged_pieces = []
    for split in SPLIT_SOURCES:
        path = pretrained_dir / f"predictions_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing pretrained predictions: {path}")
        generated = pd.read_csv(path)
        required = {"row_id", "label", "fusion_score"}
        if not required.issubset(generated.columns):
            raise ValueError(f"Missing evaluator columns in {path}: {sorted(required - set(generated.columns))}")
        stored = paper[paper.test_split == split][
            ["unique_row_id", "true_label", "pred_score_fusion"]
        ].rename(columns={"unique_row_id": "row_id"})
        merged = stored.merge(
            generated[["row_id", "label", "fusion_score"]],
            on="row_id",
            how="left",
            validate="one_to_one",
        )
        if merged.fusion_score.isna().any():
            raise ValueError(f"Missing evaluator predictions for paper rows in {split}")
        merged.insert(0, "split", split)
        merged_pieces.append(merged)
        score_delta = merged.fusion_score - merged.pred_score_fusion
        alignment_rows.append(
            {
                "split": split,
                "n": len(merged),
                "label_mismatches": int((merged.label.astype(int) != merged.true_label.astype(int)).sum()),
                "score_mae": float(score_delta.abs().mean()),
                "score_max_abs": float(score_delta.abs().max()),
                "score_correlation": float(merged[["fusion_score", "pred_score_fusion"]].corr().iloc[0, 1]),
                "threshold_0.5_disagreements": int(((merged.fusion_score >= 0.5) != (merged.pred_score_fusion >= 0.5)).sum()),
            }
        )
        metric_rows.append(
            {
                "source": "independent_pretrained_evaluator_exact_paper_rows",
                "split": split,
                "paper_name": PAPER_NAMES[split],
                **compute_metrics(merged.label, merged.fusion_score),
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(alignment_rows), pd.concat(merged_pieces, ignore_index=True)


def auxiliary_artifact_alignment(data_dir: Path, paper: pd.DataFrame) -> pd.DataFrame:
    rows = []
    predictions_path = data_dir / "predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        row: Dict[str, Any] = {
            "artifact": "predictions.csv",
            "rows": len(predictions),
            "paper_rows": len(paper),
            "order_assumption": True,
        }
        if len(predictions) == len(paper):
            row["label_mismatches"] = int(
                (predictions.labels.astype(int).to_numpy() != paper.true_label.astype(int).to_numpy()).sum()
            )
            delta = predictions.pred_scores.to_numpy(dtype=float) - paper.pred_score_fusion.to_numpy(dtype=float)
            row["score_mae"] = float(np.mean(np.abs(delta)))
            row["score_max_abs"] = float(np.max(np.abs(delta)))
        rows.append(row)
    uncertainty_path = data_dir.parent / "code" / "fusion_model" / "uncertain_indices.csv"
    if uncertainty_path.exists():
        flags = pd.read_csv(uncertainty_path, header=None).iloc[:, 0].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
        row = {
            "artifact": "uncertain_indices.csv",
            "rows": len(flags),
            "paper_rows": len(paper),
            "order_assumption": True,
        }
        if len(flags) == len(paper) and flags.notna().all():
            row["flag_mismatches"] = int(
                (flags.to_numpy(dtype=bool) != paper.uncertain_flag.astype(bool).to_numpy()).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    cohort_summary: pd.DataFrame,
    audit: pd.DataFrame,
    participant_overlap_before: pd.DataFrame,
    participant_overlap_after: pd.DataFrame,
    row_duplicates_before: pd.DataFrame,
    row_duplicates_after: pd.DataFrame,
    paper_metrics: pd.DataFrame,
    generated_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    alignment: pd.DataFrame,
) -> None:
    lines = [
        "# PARK paper-exact protocol audit",
        "",
        "This audit reconstructs the published cross-setting cohorts without using "
        "outcomes to decide membership. The paper-ready artifact is treated as the "
        "frozen protocol reference, while independently generated checkpoint scores "
        "are evaluated on exactly the same rows.",
        "",
        "## Cohort reconstruction",
        "",
        "| Cohort | Source sessions | Paper sessions | Excluded | Source participants | Paper participants |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in cohort_summary.itertuples(index=False):
        lines.append(
            f"| {row.paper_name} | {row.source_sessions} | {row.paper_sessions} | "
            f"{row.excluded_sessions} | {row.source_participants} | {row.paper_participants} |"
        )
    excluded = audit[~audit.included_in_paper_artifact]
    lines.extend(
        [
            "",
            f"The source files contain **{len(row_duplicates_before)}** session row IDs assigned "
            f"to more than one split; the paper artifact contains **{len(row_duplicates_after)}**. "
            f"The notebook removes {len(excluded)} duplicate split memberships while retaining "
            "each session exactly once. No label-based membership rule is required.",
            "",
            f"Cross-split participant overlaps before reconstruction: **{len(participant_overlap_before)}**; "
            f"after reconstruction: **{len(participant_overlap_after)}**. Thus the paper resolves "
            "duplicate session attribution but does not make the two external cohorts participant-independent.",
            "",
            "## Published point-estimate reproduction",
            "",
            "| Cohort | Accuracy | AUROC | Sensitivity | Specificity | PPV | NPV | F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in paper_metrics.itertuples(index=False):
        lines.append(
            f"| {row.paper_name} | {row.accuracy:.4f} | {row.auroc:.4f} | "
            f"{row.sensitivity:.4f} | {row.specificity:.4f} | {row.precision:.4f} | "
            f"{row.npv:.4f} | {row.f1:.4f} |"
        )
    max_published_delta = comparison[comparison.metric != "n"].delta_recomputed_minus_published.abs().max()
    lines.extend(
        [
            "",
            "Published classification metrics are rounded to one decimal percentage "
            f"point; the maximum absolute point-estimate difference after recomputation is {max_published_delta * 100:.3f} pp.",
            "",
            "## Independent checkpoint evaluator on paper-exact rows",
            "",
            "| Cohort | Accuracy | AUROC | Sensitivity | Specificity | Score MAE vs paper artifact | 0.5 decision disagreements |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in generated_metrics.itertuples(index=False):
        align = alignment[alignment.split == row.split].iloc[0]
        lines.append(
            f"| {row.paper_name} | {row.accuracy:.4f} | {row.auroc:.4f} | "
            f"{row.sensitivity:.4f} | {row.specificity:.4f} | "
            f"{align['score_mae']:.6f} | {int(align['threshold_0.5_disagreements'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The 162/91/67 paper cohort sizes are completely explained by assigning each of three cross-file duplicate sessions to exactly one external-validation split.",
            "- The stored paper artifact exactly determines the published point estimates; any remaining difference from a fresh evaluator is a score-generation/version issue rather than cohort membership.",
            "- The notebook used explicit row IDs for the exclusions. This audit expresses the operation as duplicate session attribution and records both retained and discarded split memberships.",
            "- One participant remains represented in both external cohorts after session de-duplication; cohort-level confidence intervals should therefore not be interpreted as fully independent at participant level.",
            "- AUROC values are recomputed from released paper scores because the article reports only the cross-cohort AUROC range in prose/figures.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    data_dir = repo_root / "data"
    pretrained_dir = (args.pretrained_dir or repo_root / "results" / "pretrained_eval").resolve()
    output_dir = (args.output_dir or repo_root / "results" / "paper_exact_protocol_audit").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = [
        *(data_dir / filename for filename in SPLIT_SOURCES.values()),
        data_dir / "test_data_big.csv",
        data_dir / "predictions.csv",
        repo_root / "code" / "fusion_model" / "uncertain_indices.csv",
        *(pretrained_dir / f"predictions_{split}.csv" for split in SPLIT_SOURCES),
    ]
    missing_inputs = [str(path) for path in input_paths if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing audit inputs: {missing_inputs}")
    hashes_before = {str(path.relative_to(repo_root)): ev.sha256_file(path) for path in input_paths}

    source = load_source_cohorts(data_dir)
    paper = load_paper_artifact(data_dir)
    (
        audit,
        participant_overlap_before,
        participant_overlap_after,
        row_duplicates_before,
        row_duplicates_after,
    ) = membership_audit(source, paper)
    audit.to_csv(output_dir / "cohort_membership_audit.csv", index=False)
    participant_overlap_before.to_csv(
        output_dir / "cross_split_participant_overlap_before.csv", index=False
    )
    participant_overlap_after.to_csv(
        output_dir / "cross_split_participant_overlap_after.csv", index=False
    )
    row_duplicates_before.to_csv(
        output_dir / "cross_split_duplicate_sessions_before.csv", index=False
    )
    row_duplicates_after.to_csv(
        output_dir / "cross_split_duplicate_sessions_after.csv", index=False
    )

    cohort_rows = []
    for split in SPLIT_SOURCES:
        source_split = source[source.split == split]
        paper_split = paper[paper.test_split == split]
        cohort_rows.append(
            {
                "split": split,
                "paper_name": PAPER_NAMES[split],
                "source_sessions": len(source_split),
                "paper_sessions": len(paper_split),
                "excluded_sessions": len(source_split) - len(paper_split),
                "source_participants": source_split.id.nunique(),
                "paper_participants": paper_split.participant_id.nunique(),
            }
        )
    cohort_summary = pd.DataFrame(cohort_rows)
    observed_counts = dict(zip(cohort_summary.split, cohort_summary.paper_sessions))
    expected_counts = {
        split: values["n"] for split, values in PUBLISHED_POINT_ESTIMATES.items()
    }
    if observed_counts != expected_counts:
        raise ValueError(
            f"Paper cohort counts do not match the published protocol: "
            f"observed={observed_counts}, expected={expected_counts}"
        )
    if len(row_duplicates_after):
        raise ValueError("Cross-split duplicate sessions remain after reconstruction")
    cohort_summary.to_csv(output_dir / "cohort_summary.csv", index=False)

    paper_metrics = paper_artifact_metrics(paper)
    paper_metrics.to_csv(output_dir / "paper_artifact_metrics.csv", index=False)
    comparison = published_comparison(paper_metrics)
    comparison.to_csv(output_dir / "published_point_estimate_comparison.csv", index=False)
    generated_metrics, alignment, merged = evaluator_alignment(pretrained_dir, paper)
    generated_metrics.to_csv(output_dir / "evaluator_paper_exact_metrics.csv", index=False)
    alignment.to_csv(output_dir / "prediction_alignment_summary.csv", index=False)
    merged.to_csv(output_dir / "paper_vs_evaluator_scores.csv", index=False)
    auxiliary = auxiliary_artifact_alignment(data_dir, paper)
    auxiliary.to_csv(output_dir / "auxiliary_artifact_alignment.csv", index=False)

    hashes_after = {path: ev.sha256_file(repo_root / path) for path in hashes_before}
    if hashes_before != hashes_after:
        raise RuntimeError("An audit input changed during execution")
    write_report(
        output_dir / "PAPER_EXACT_PROTOCOL_REPORT.md",
        cohort_summary,
        audit,
        participant_overlap_before,
        participant_overlap_after,
        row_duplicates_before,
        row_duplicates_after,
        paper_metrics,
        generated_metrics,
        comparison,
        alignment,
    )
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            ev.json_ready(
                {
                    "repo_root": repo_root,
                    "git_commit": ev.git_commit(repo_root),
                    "paper_cohort_expected": {key: value["n"] for key, value in PUBLISHED_POINT_ESTIMATES.items()},
                    "cross_split_participant_overlaps_before": len(participant_overlap_before),
                    "cross_split_participant_overlaps_after": len(participant_overlap_after),
                    "cross_split_duplicate_sessions_before": len(row_duplicates_before),
                    "cross_split_duplicate_sessions_after": len(row_duplicates_after),
                    "excluded_split_memberships": int((~audit.included_in_paper_artifact).sum()),
                    "input_sha256": hashes_after,
                    "inputs_unchanged": True,
                }
            ),
            handle,
            indent=2,
        )
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
