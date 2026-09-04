#!/usr/bin/env python3
"""Freeze the leakage-safe development protocol for PARK Fusion Agent v2.

The script never reads model predictions and never computes Test metrics. It
uses participant identifiers and labels only to create stratified nested folds
inside the original Train+Dev pool, verifies isolation from every released test
cohort, freezes specialist corruption tasks and acceptance criteria, and hashes
all protocol artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import evaluate_pretrained as ev
import train_paired_baselines as paired


PROTOCOL_VERSION = "fusion-agent-v2-protocol-1.0"
MODALITIES = ("finger", "speech", "smile")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=20260904)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        ev.json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def participant_table(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    label_counts = frame.groupby("id").label.nunique()
    conflict_ids = set(label_counts[label_counts > 1].index.astype(str))
    conflicts = frame.loc[
        frame.id.astype(str).isin(conflict_ids), ["id", "row_id", "label"]
    ].copy()
    conflicts = conflicts.rename(columns={"id": "participant_id"}).sort_values(
        ["participant_id", "row_id"]
    )
    eligible = frame.loc[~frame.id.astype(str).isin(conflict_ids)]
    output = (
        eligible.groupby("id", as_index=False)
        .agg(label=("label", "first"), sessions=("row_id", "nunique"))
        .sort_values("id")
        .reset_index(drop=True)
    )
    output["label"] = output.label.astype(int)
    return output, conflicts.reset_index(drop=True)


def validate_fold_count(labels: np.ndarray, folds: int, name: str) -> None:
    if folds < 2:
        raise ValueError(f"{name} folds must be at least 2")
    counts = pd.Series(labels).value_counts()
    if len(counts) != 2 or int(counts.min()) < folds:
        raise ValueError(
            f"{name}={folds} is not possible with class counts {counts.to_dict()}"
        )


def nested_manifest(
    development: pd.DataFrame, outer_folds: int, inner_folds: int, seed: int
) -> pd.DataFrame:
    labels = development.label.to_numpy(dtype=int)
    validate_fold_count(labels, outer_folds, "outer")
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    rows: List[Dict[str, Any]] = []
    dummy = np.zeros(len(development), dtype=np.int8)
    for outer_fold, (outer_train_index, outer_validation_index) in enumerate(
        outer.split(dummy, labels)
    ):
        outer_train = development.iloc[outer_train_index].reset_index(drop=True)
        outer_validation = development.iloc[outer_validation_index]
        inner_labels = outer_train.label.to_numpy(dtype=int)
        validate_fold_count(inner_labels, inner_folds, f"inner outer_fold={outer_fold}")
        inner = StratifiedKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=seed + 1000 + outer_fold,
        )
        inner_assignment = np.full(len(outer_train), -1, dtype=int)
        for inner_fold, (_, inner_validation_index) in enumerate(
            inner.split(np.zeros(len(outer_train), dtype=np.int8), inner_labels)
        ):
            inner_assignment[inner_validation_index] = inner_fold
        if (inner_assignment < 0).any():
            raise RuntimeError("At least one outer-training participant lacks an inner fold")
        for position, item in outer_train.iterrows():
            rows.append(
                {
                    "participant_id": str(item.id),
                    "label": int(item.label),
                    "sessions": int(item.sessions),
                    "outer_fold": outer_fold,
                    "outer_role": "train",
                    "inner_validation_fold": int(inner_assignment[position]),
                }
            )
        for _, item in outer_validation.iterrows():
            rows.append(
                {
                    "participant_id": str(item.id),
                    "label": int(item.label),
                    "sessions": int(item.sessions),
                    "outer_fold": outer_fold,
                    "outer_role": "validation",
                    "inner_validation_fold": -1,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["outer_fold", "outer_role", "participant_id"]
    ).reset_index(drop=True)


def fold_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for outer_fold, outer_rows in manifest.groupby("outer_fold"):
        for role, selected in outer_rows.groupby("outer_role"):
            rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "inner_fold": -1,
                    "partition": f"outer_{role}",
                    "participants": selected.participant_id.nunique(),
                    "sessions": int(selected.sessions.sum()),
                    "positives": int(selected.groupby("participant_id").label.first().sum()),
                    "negatives": int(selected.participant_id.nunique() - selected.groupby("participant_id").label.first().sum()),
                }
            )
        outer_train = outer_rows.loc[outer_rows.outer_role == "train"]
        for inner_fold in sorted(outer_train.inner_validation_fold.unique()):
            for role, mask in (
                ("inner_train", outer_train.inner_validation_fold != inner_fold),
                ("inner_validation", outer_train.inner_validation_fold == inner_fold),
            ):
                selected = outer_train.loc[mask]
                positives = int(selected.groupby("participant_id").label.first().sum())
                rows.append(
                    {
                        "outer_fold": int(outer_fold),
                        "inner_fold": int(inner_fold),
                        "partition": role,
                        "participants": selected.participant_id.nunique(),
                        "sessions": int(selected.sessions.sum()),
                        "positives": positives,
                        "negatives": int(selected.participant_id.nunique() - positives),
                    }
                )
    return pd.DataFrame(rows)


def leakage_audit(
    manifest: pd.DataFrame,
    development_ids: set,
    protected_sets: Dict[str, set],
    excluded_conflict_ids: set,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add(check: str, observed: int, expected: int, detail: str) -> None:
        rows.append(
            {
                "check": check,
                "status": "PASS" if observed == expected else "FAIL",
                "observed": observed,
                "expected": expected,
                "detail": detail,
            }
        )

    manifest_ids = set(manifest.participant_id.astype(str))
    add(
        "manifest_contains_only_development_pool",
        len(manifest_ids - development_ids),
        0,
        "No identifier outside original Train+Dev may enter nested CV.",
    )
    add(
        "all_development_participants_represented",
        len(development_ids - manifest_ids),
        0,
        "Every eligible Train+Dev participant after quarantine must be represented.",
    )
    add(
        "label_conflict_participants_quarantined",
        len(manifest_ids & excluded_conflict_ids),
        0,
        "Any identifier with inconsistent session labels is excluded from all folds.",
    )
    for name, protected in protected_sets.items():
        add(
            f"development_disjoint_from_{name}",
            len(development_ids & protected),
            0,
            f"Original Train+Dev must not overlap locked partition {name}.",
        )
    for outer_fold, outer_rows in manifest.groupby("outer_fold"):
        train = set(
            outer_rows.loc[outer_rows.outer_role == "train", "participant_id"].astype(str)
        )
        validation = set(
            outer_rows.loc[
                outer_rows.outer_role == "validation", "participant_id"
            ].astype(str)
        )
        add(
            f"outer_{outer_fold}_train_validation_disjoint",
            len(train & validation),
            0,
            "Participant is the indivisible split unit.",
        )
        add(
            f"outer_{outer_fold}_covers_development_once",
            len((train | validation) ^ development_ids),
            0,
            "Outer train plus validation must exactly cover the development pool.",
        )
        training_rows = outer_rows.loc[outer_rows.outer_role == "train"]
        add(
            f"outer_{outer_fold}_inner_assignment_complete",
            int((training_rows.inner_validation_fold < 0).sum()),
            0,
            "Each outer-training participant is inner-validation exactly once.",
        )
    participant_outer_validation_counts = (
        manifest.loc[manifest.outer_role == "validation"]
        .groupby("participant_id")
        .size()
    )
    add(
        "each_participant_outer_validation_once",
        int((participant_outer_validation_counts != 1).sum()),
        0,
        "Every development participant must be outer-validation exactly once.",
    )
    return pd.DataFrame(rows)


def perturbation_registry() -> pd.DataFrame:
    rows = [
        {
            "task": "baseline",
            "scenario": "clean",
            "target": "quality=1",
            "operator": "identity",
            "severity": 0.0,
            "fit_scope": "inner_train_only",
        }
    ]
    for severity in (0.25, 0.5, 1.0):
        rows.append(
            {
                "task": "speech_noise_detector",
                "scenario": f"speech_gaussian_{severity:.2f}",
                "target": "speech_noise=1",
                "operator": "add Gaussian noise after train-fold scaling",
                "severity": severity,
                "fit_scope": "inner_train_only",
            }
        )
    for severity in (0.10, 0.25, 0.5):
        rows.append(
            {
                "task": "speech_noise_detector",
                "scenario": f"speech_mask_{severity:.2f}",
                "target": "speech_noise=1",
                "operator": "independent feature masking after train-fold scaling",
                "severity": severity,
                "fit_scope": "inner_train_only",
            }
        )
    for operator in (
        "invert smile probability p -> 1-p",
        "replace smile probability with opposite extreme",
        "permute smile probability within inner-train label-blindly",
    ):
        rows.append(
            {
                "task": "smile_conflict_detector",
                "scenario": "smile_conflict",
                "target": "smile_conflict=1",
                "operator": operator,
                "severity": 1.0,
                "fit_scope": "inner_train_only",
            }
        )
    rows.extend(
        [
            {
                "task": "missingness_router",
                "scenario": f"missing_{name}",
                "target": "availability=0",
                "operator": "explicit modality availability mask; never infer missingness from zero values",
                "severity": 1.0,
                "fit_scope": "no_fit",
            }
            for name in MODALITIES
        ]
    )
    return pd.DataFrame(rows)


def acceptance_criteria() -> Dict[str, Any]:
    return {
        "selection_unit": "participant",
        "aggregation": "mean session score per participant before metrics",
        "required": [
            {
                "id": "clean_noninferiority",
                "metric": "outer-CV mean clean AUROC delta vs available_weighted",
                "operator": ">=",
                "value": -0.005,
            },
            {
                "id": "stress_superiority_available",
                "metric": "outer-CV stress macro-AUROC delta vs available_weighted",
                "operator": ">",
                "value": 0.0,
            },
            {
                "id": "stress_superiority_ufnet",
                "metric": "outer-CV stress macro-AUROC delta vs UFNet",
                "operator": ">",
                "value": 0.0,
            },
            {
                "id": "speech_noise_gain",
                "metric": "outer-CV speech-noise macro-AUROC delta vs max baseline",
                "operator": ">",
                "value": 0.0,
            },
            {
                "id": "smile_conflict_noninferiority",
                "metric": "outer-CV smile-conflict AUROC delta vs UFNet",
                "operator": ">=",
                "value": -0.01,
            },
            {
                "id": "clean_coverage",
                "metric": "outer-CV clean selective coverage",
                "operator": ">=",
                "value": 0.80,
            },
            {
                "id": "calibration_guard",
                "metric": "outer-CV clean ECE delta vs available_weighted",
                "operator": "<=",
                "value": 0.02,
            },
            {
                "id": "fold_stability",
                "metric": "outer folds satisfying clean noninferiority",
                "operator": ">=",
                "value": 4,
                "denominator": 5,
            },
        ],
        "external_evaluation_rule": (
            "Freeze the complete v2 policy after nested CV. Evaluate once on a genuinely "
            "unseen participant-disjoint external cohort; do not revise thresholds afterward."
        ),
    }


def write_report(
    path: Path,
    development: pd.DataFrame,
    summary: pd.DataFrame,
    audit: pd.DataFrame,
    protocol: Dict[str, Any],
) -> None:
    outer = summary.loc[summary.inner_fold == -1]
    lines = [
        "# Fusion Agent v2 frozen development protocol",
        "",
        f"Protocol version: `{PROTOCOL_VERSION}`  ",
        f"Protocol SHA-256: `{protocol['protocol_sha256']}`",
        "",
        "## Scope",
        "",
        "Only eligible original Train+Dev participants enter model development. All released "
        "internal and external test cohorts remain locked. The script creates no predictions "
        "and reads no test outcomes for metric computation.",
        "",
        f"Development participants: **{development.id.nunique()}**; sessions: "
        f"**{int(development.sessions.sum())}**; positives: **{int(development.label.sum())}**.",
        f"Quarantined inconsistent-label identifiers: **{protocol['excluded_label_conflict_participants']}**.",
        "",
        "## Nested cross-validation",
        "",
        "| Outer fold | Train participants | Validation participants | Train positives | Validation positives |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for outer_fold in sorted(outer.outer_fold.unique()):
        train = outer.loc[
            (outer.outer_fold == outer_fold) & (outer.partition == "outer_train")
        ].iloc[0]
        validation = outer.loc[
            (outer.outer_fold == outer_fold)
            & (outer.partition == "outer_validation")
        ].iloc[0]
        lines.append(
            f"| {outer_fold} | {train.participants} | {validation.participants} | "
            f"{train.positives} | {validation.positives} |"
        )
    lines.extend(
        [
            "",
            "Each outer-training partition has four stratified inner folds. Specialist "
            "detectors fit only inner-train synthetic corruptions; router and abstention "
            "parameters use inner-validation; outer-validation estimates development "
            "generalization.",
            "",
            "## Leakage audit",
            "",
            f"Checks passed: **{int((audit.status == 'PASS').sum())}/{len(audit)}**.",
            "",
            "## Frozen specialist tasks",
            "",
            "- Preserve Fusion Agent v1 risk rejection and fail-closed behavior.",
            "- Train a speech feature-corruption detector independently of disease labels.",
            "- Train a separate smile score-conflict detector using label-blind conflict synthesis.",
            "- Missing modalities use explicit availability masks and deterministic weight renormalization.",
            "- No Test or external-cohort result may select a feature, threshold, route, or checkpoint.",
            "",
            "## Next gate",
            "",
            "Implement and train v2 only against this protocol. Promotion requires every "
            "criterion in `acceptance_criteria.json`; passing nested CV authorizes exactly "
            "one evaluation on a genuinely unseen external cohort.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    source = (
        args.data
        or repo_root / "results" / "protocol_alignment_audit" / "cleaned_aligned.csv"
    ).resolve()
    output = (args.output_dir or repo_root / "results" / "fusion_agent_v2_protocol").resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = paired.load_vector_csv(source)
    module = ev.load_upstream_module(repo_root)
    masks = paired.split_masks(module, frame)
    people, conflicts = participant_table(frame)
    original_development_ids = set(
        frame.loc[masks["train"] | masks["dev"], "id"].astype(str)
    )
    conflict_ids = set(conflicts.participant_id.astype(str))
    development_ids = original_development_ids - conflict_ids
    development = people.loc[people.id.astype(str).isin(development_ids)].reset_index(drop=True)
    protected_sets = {
        name: set(frame.loc[masks[name], "id"].astype(str))
        for name in ("internal_test", "validation_1", "validation_2", "global")
    }
    manifest = nested_manifest(
        development, args.outer_folds, args.inner_folds, args.split_seed
    )
    summary = fold_summary(manifest)
    audit = leakage_audit(manifest, development_ids, protected_sets, conflict_ids)
    if (audit.status != "PASS").any():
        failed = audit.loc[audit.status != "PASS", "check"].tolist()
        raise RuntimeError(f"Protocol leakage audit failed: {failed}")
    perturbations = perturbation_registry()
    criteria = acceptance_criteria()

    manifest_path = output / "participant_fold_manifest.csv"
    summary_path = output / "fold_summary.csv"
    audit_path = output / "leakage_audit.csv"
    perturbation_path = output / "perturbation_registry.csv"
    criteria_path = output / "acceptance_criteria.json"
    conflicts_path = output / "excluded_label_conflicts.csv"
    manifest.to_csv(manifest_path, index=False)
    summary.to_csv(summary_path, index=False)
    audit.to_csv(audit_path, index=False)
    perturbations.to_csv(perturbation_path, index=False)
    conflicts.to_csv(conflicts_path, index=False)
    with criteria_path.open("w", encoding="utf-8") as handle:
        json.dump(criteria, handle, indent=2)

    protocol: Dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN",
        "source_dataset": str(source.relative_to(repo_root)),
        "source_dataset_sha256": sha256_file(source),
        "source_git_commit": ev.git_commit(repo_root),
        "generator_script": str(Path(__file__).resolve().relative_to(repo_root)),
        "generator_script_sha256": sha256_file(Path(__file__).resolve()),
        "split_unit": "participant",
        "stratification": "binary disease label",
        "development_source_partitions": ["original_train", "original_dev"],
        "locked_partitions": list(protected_sets),
        "development_participants": len(development),
        "development_sessions": int(development.sessions.sum()),
        "original_development_participants_before_quarantine": len(original_development_ids),
        "excluded_label_conflict_participants": len(conflict_ids),
        "label_conflict_policy": (
            "Quarantine every session from any identifier with inconsistent labels; "
            "never resolve by majority vote or latest-session selection."
        ),
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "split_seed": args.split_seed,
        "specialists": ["speech_noise_detector", "smile_conflict_detector"],
        "preserved_v1_components": [
            "risk_abstention",
            "availability_renormalization",
            "fail_closed_available_weighted",
        ],
        "artifact_sha256": {
            "participant_fold_manifest.csv": sha256_file(manifest_path),
            "fold_summary.csv": sha256_file(summary_path),
            "leakage_audit.csv": sha256_file(audit_path),
            "perturbation_registry.csv": sha256_file(perturbation_path),
            "acceptance_criteria.json": sha256_file(criteria_path),
            "excluded_label_conflicts.csv": sha256_file(conflicts_path),
        },
        "privacy": (
            "participant_fold_manifest.csv and excluded_label_conflicts.csv contain "
            "identifiers and must remain private; other root artifacts are aggregate."
        ),
    }
    protocol["protocol_sha256"] = canonical_hash(protocol)
    protocol_path = output / "protocol.json"
    with protocol_path.open("w", encoding="utf-8") as handle:
        json.dump(ev.json_ready(protocol), handle, indent=2)
    (output / "protocol.sha256").write_text(
        f"{protocol['protocol_sha256']}  canonical-protocol-payload\n", encoding="utf-8"
    )
    write_report(output / "FUSION_AGENT_V2_PROTOCOL.md", development, summary, audit, protocol)
    print(f"Protocol frozen: {protocol['protocol_sha256']}")
    print(f"Leakage checks: {(audit.status == 'PASS').sum()}/{len(audit)} PASS")
    print(f"Outputs written to: {output}")


if __name__ == "__main__":
    main()
