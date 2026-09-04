#!/usr/bin/env python3
"""Freeze the failure-driven development contract for PARK Fusion Agent v3.

V2 outer-fold outcomes are already exposed, so this protocol explicitly treats
all reused Train+Dev cross-validation as development evidence only. It creates
no predictions. Promotion requires a later, genuinely unseen external cohort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import evaluate_pretrained as ev
import freeze_fusion_agent_v2_protocol as v2
import train_paired_baselines as paired


PROTOCOL_VERSION = "fusion-agent-v3-protocol-1.0"
DEFAULT_SEED = 20260905


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--v2-router-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(ev.json_ready(value), indent=2) + "\n", encoding="utf-8")


def v2_exposure_register(v2_router_dir: Path) -> Dict[str, Any]:
    required = [
        "acceptance_checks.csv",
        "outer_metrics.csv",
        "outer_routes.csv",
        "selection_decision.json",
        "run_manifest.json",
    ]
    missing = [name for name in required if not (v2_router_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing v2 outcome artifacts: {missing}")
    decision = json.loads((v2_router_dir / "selection_decision.json").read_text(encoding="utf-8"))
    return {
        "prior_protocol": "fusion-agent-v2-protocol-1.0",
        "prior_outer_results_exposed": True,
        "exposure_consequence": (
            "Every reused Train+Dev participant has contributed to observed v2 outer-fold "
            "results. New fold assignments do not restore statistical independence."
        ),
        "permitted_use": "development, debugging, ablation, and rejection of v3 only",
        "prohibited_use": (
            "claiming unbiased performance, promoting v3, or selecting a final clinical "
            "operating point without a genuinely unseen participant-disjoint cohort"
        ),
        "v2_passed_criteria": int(decision["passed_criteria"]),
        "v2_total_criteria": int(decision["total_criteria"]),
        "v2_promoted": bool(decision["promote_to_unseen_external_evaluation"]),
        "artifact_sha256": {
            name: v2.sha256_file(v2_router_dir / name) for name in required
        },
    }


def router_contract() -> Dict[str, Any]:
    return {
        "router_family": "cross_fitted_expected_regret_router",
        "motivation": (
            "V2 detected synthetic corruption but could not identify which prediction was "
            "more trustworthy; v3 learns action utility rather than mapping detector score "
            "directly to a route."
        ),
        "inference_features": [
            "calibrated finger, speech, smile, UFNet, and available-weighted probabilities",
            "pairwise absolute score disagreements",
            "per-expert entropy and distance from 0.5",
            "explicit modality availability masks",
            "speech-noise and smile-conflict specialist probabilities",
            "Monte-Carlo predictive mean and variance when available",
        ],
        "forbidden_inference_features": [
            "disease label",
            "participant identifier",
            "partition name or fold number",
            "released Test or external-cohort outcomes",
        ],
        "actions": [
            "available_weighted_fail_closed",
            "drop_smile_and_renormalize",
            "UFNet_fallback",
            "shrink_smile_toward_available_non_smile_consensus",
            "abstain_without_emitting_a_disease_decision",
        ],
        "supervision": (
            "Generate action-specific binary log loss from disease labels only on inner-train "
            "out-of-fold predictions. Fit one expected-loss model per non-abstention action."
        ),
        "selection": (
            "Choose a non-default action only when its cross-fitted expected-regret improvement "
            "over available_weighted is positive and its one-sided participant-bootstrap 95% "
            "lower confidence bound exceeds zero. Otherwise fail closed."
        ),
        "calibration": (
            "All disease scores and action-utility estimates are calibrated inside inner folds; "
            "no outer-validation labels may fit a calibrator or threshold."
        ),
        "risk_rejection": {
            "preserved": True,
            "clean_coverage_floor": 0.80,
            "missing_modality_is_not_quality_failure": True,
            "abstention_is_reported_separately_from_score_routing": True,
        },
        "complexity_guards": {
            "maximum_non_default_action_rate_on_clean": 0.20,
            "maximum_router_depth": 3,
            "minimum_leaf_participants": 30,
            "permitted_model_families": [
                "regularized multinomial logistic regression",
                "shallow histogram gradient boosting",
            ],
        },
    }


def perturbation_registry() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = [
        {
            "family": "reference",
            "scenario": "clean",
            "operator": "identity",
            "severity": 0.0,
            "purpose": "clean non-inferiority and calibration",
        },
        {
            "family": "missingness",
            "scenario": "each_single_modality_missing",
            "operator": "explicit availability mask",
            "severity": 1.0,
            "purpose": "deterministic renormalization",
        },
    ]
    for temperature in (0.5, 2.0, 4.0):
        rows.append(
            {
                "family": "smile_reliability",
                "scenario": f"smile_logit_temperature_{temperature:.1f}",
                "operator": "multiply centered smile logit by fixed temperature",
                "severity": temperature,
                "purpose": "over/under-confidence shift without using labels",
            }
        )
    for shift in (-1.0, -0.5, 0.5, 1.0):
        rows.append(
            {
                "family": "smile_reliability",
                "scenario": f"smile_logit_shift_{shift:+.1f}",
                "operator": "add fixed offset to smile logit",
                "severity": abs(shift),
                "purpose": "systematic calibration drift",
            }
        )
    rows.extend(
        [
            {
                "family": "smile_conflict",
                "scenario": "smile_rank_permutation",
                "operator": "label-blind permutation within evaluation partition",
                "severity": 1.0,
                "purpose": "score-source mismatch",
            },
            {
                "family": "smile_conflict",
                "scenario": "smile_opposite_consensus",
                "operator": "place smile score on the opposite side of non-smile consensus",
                "severity": 1.0,
                "purpose": "directional disagreement stress test only; never a training target",
            },
            {
                "family": "speech_quality",
                "scenario": "speech_noise_registry_v2",
                "operator": "reuse frozen v2 Gaussian and masking operators",
                "severity": -1.0,
                "purpose": "verify that utility routing does not repeat v2 harmful downweighting",
            },
        ]
    )
    output = pd.DataFrame(rows)
    output["fit_scope"] = "inner_train_only_or_no_fit"
    return output


def acceptance_contract() -> Dict[str, Any]:
    return {
        "internal_status": "DEVELOPMENT_ONLY_NOT_A_PROMOTION_GATE",
        "internal_required": [
            {"id": "clean_noninferiority", "metric": "mean AUROC delta vs available_weighted", "operator": ">=", "value": -0.005},
            {"id": "smile_reliability_gain", "metric": "macro AUROC delta vs best fixed comparator", "operator": ">", "value": 0.0},
            {"id": "stress_regret_guard", "metric": "mean binary-log-loss regret vs available_weighted", "operator": "<=", "value": 0.0},
            {"id": "clean_coverage", "metric": "selective coverage", "operator": ">=", "value": 0.80},
            {"id": "clean_action_rate", "metric": "non-default non-abstention route rate", "operator": "<=", "value": 0.20},
            {"id": "calibration_guard", "metric": "ECE delta vs available_weighted", "operator": "<=", "value": 0.02},
            {"id": "fold_stability", "metric": "folds passing clean noninferiority", "operator": ">=", "value": 4, "denominator": 5},
            {"id": "bootstrap_route_evidence", "metric": "one-sided 95% lower bound for selected-action regret improvement", "operator": ">", "value": 0.0},
        ],
        "external_cohort_eligibility": {
            "genuinely_unseen": True,
            "participant_disjoint_from_all_released_and_development_cohorts": True,
            "minimum_participants": 100,
            "minimum_positive_participants": 25,
            "minimum_negative_participants": 25,
            "labels_hidden_until_policy_and_analysis_code_are_frozen": True,
            "single_evaluation_only": True,
        },
        "promotion_rule": (
            "Internal criteria may reject v3 but cannot promote it. Promotion requires every "
            "predeclared internal criterion plus confirmatory success on one eligible unseen "
            "external cohort. Failure ends v3; thresholds may not be revised after unblinding."
        ),
    }


def write_report(path: Path, protocol: Dict[str, Any], audit: pd.DataFrame) -> None:
    lines = [
        "# Fusion Agent v3 frozen development protocol",
        "",
        f"Protocol version: `{PROTOCOL_VERSION}`  ",
        f"Protocol SHA-256: `{protocol['protocol_sha256']}`",
        "",
        "## Decision boundary",
        "",
        "V2 outer-fold results are exposed. Repartitioning the same participants cannot create "
        "a new unbiased test set. V3 cross-validation is therefore development evidence only: "
        "it may reject v3 but cannot promote it.",
        "",
        "## Router change",
        "",
        "V3 predicts cross-fitted expected loss for each allowed action. It changes the default "
        "availability-weighted route only when the participant-bootstrap lower confidence bound "
        "for expected-regret improvement is positive. Risk rejection and fail-closed behavior "
        "remain mandatory.",
        "",
        "Synthetic opposite-consensus smile cases are stress tests, not positive labels for a "
        "direct switch rule. This addresses the v2 failure mode: corruption was detectable, but "
        "the detector did not establish which disease prediction was correct.",
        "",
        "## Data isolation",
        "",
        f"Development participants: **{protocol['development_participants']}**; sessions: "
        f"**{protocol['development_sessions']}**. Leakage checks: "
        f"**{int((audit.status == 'PASS').sum())}/{len(audit)} PASS**.",
        "",
        "No released Test or external-cohort predictions are created by this script. Final "
        "promotion requires one newly collected, participant-disjoint cohort with labels hidden "
        "until the policy and analysis code are frozen.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    source = (args.data or repo_root / "results/protocol_alignment_audit/cleaned_aligned.csv").resolve()
    v2_router_dir = (args.v2_router_dir or repo_root / "results/fusion_agent_v2_router").resolve()
    output = (args.output_dir or repo_root / "results/fusion_agent_v3_protocol").resolve()
    output.mkdir(parents=True, exist_ok=True)

    frame = paired.load_vector_csv(source)
    module = ev.load_upstream_module(repo_root)
    masks = paired.split_masks(module, frame)
    people, conflicts = v2.participant_table(frame)
    original_development_ids = set(frame.loc[masks["train"] | masks["dev"], "id"].astype(str))
    conflict_ids = set(conflicts.participant_id.astype(str))
    development_ids = original_development_ids - conflict_ids
    development = people.loc[people.id.astype(str).isin(development_ids)].reset_index(drop=True)
    protected_sets = {
        name: set(frame.loc[masks[name], "id"].astype(str))
        for name in ("internal_test", "validation_1", "validation_2", "global")
    }
    manifest = v2.nested_manifest(development, args.outer_folds, args.inner_folds, args.split_seed)
    summary = v2.fold_summary(manifest)
    audit = v2.leakage_audit(manifest, development_ids, protected_sets, conflict_ids)
    if (audit.status != "PASS").any():
        raise RuntimeError(f"Protocol leakage audit failed: {audit.loc[audit.status != 'PASS', 'check'].tolist()}")

    exposure = v2_exposure_register(v2_router_dir)
    contract = router_contract()
    perturbations = perturbation_registry()
    criteria = acceptance_contract()
    artifacts = {
        "participant_fold_manifest.csv": manifest,
        "fold_summary.csv": summary,
        "leakage_audit.csv": audit,
        "perturbation_registry.csv": perturbations,
        "excluded_label_conflicts.csv": conflicts,
    }
    for name, table in artifacts.items():
        table.to_csv(output / name, index=False)
    write_json(output / "v2_exposure_register.json", exposure)
    write_json(output / "router_contract.json", contract)
    write_json(output / "acceptance_contract.json", criteria)

    hashed_names = [*artifacts, "v2_exposure_register.json", "router_contract.json", "acceptance_contract.json"]
    protocol: Dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN",
        "source_dataset": str(source.relative_to(repo_root)),
        "source_dataset_sha256": v2.sha256_file(source),
        "source_git_commit": ev.git_commit(repo_root),
        "generator_script": str(Path(__file__).resolve().relative_to(repo_root)),
        "generator_script_sha256": v2.sha256_file(Path(__file__).resolve()),
        "v2_protocol_generator_sha256": v2.sha256_file(Path(v2.__file__).resolve()),
        "development_source_partitions": ["original_train", "original_dev"],
        "locked_partitions": list(protected_sets),
        "prior_outer_results_exposed": True,
        "internal_evaluation_role": "development_and_rejection_only",
        "development_participants": len(development),
        "development_sessions": int(development.sessions.sum()),
        "excluded_label_conflict_participants": len(conflict_ids),
        "split_unit": "participant",
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "split_seed": args.split_seed,
        "artifact_sha256": {name: v2.sha256_file(output / name) for name in hashed_names},
        "privacy": (
            "participant_fold_manifest.csv and excluded_label_conflicts.csv contain identifiers "
            "and must remain on the experiment server; all other listed artifacts are aggregate."
        ),
        "locked_test_predictions_generated": False,
    }
    protocol["protocol_sha256"] = v2.canonical_hash(protocol)
    write_json(output / "protocol.json", protocol)
    (output / "protocol.sha256").write_text(
        f"{protocol['protocol_sha256']}  canonical-protocol-payload\n", encoding="utf-8"
    )
    write_report(output / "FUSION_AGENT_V3_PROTOCOL.md", protocol, audit)
    print(f"Protocol frozen: {protocol['protocol_sha256']}")
    print(f"Leakage checks: {(audit.status == 'PASS').sum()}/{len(audit)} PASS")
    print("Internal role: development/rejection only; external unseen cohort required")
    print(f"Outputs written to: {output}")


if __name__ == "__main__":
    main()
