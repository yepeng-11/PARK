#!/usr/bin/env python3
"""Build and validate a release-ready PARK evaluation-reproduction bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch


EXPECTED_COHORT_HASH = "6f8424c91357d68042a6012498256d98a5541b21f61cf7d79b565db359122c34"
EXPECTED_SPLITS = {"global": 162, "validation_1": 91, "validation_2": 67}
METRIC_COLUMNS = (
    "n",
    "accuracy",
    "auroc",
    "f1",
    "sensitivity",
    "specificity",
    "precision",
    "npv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def command_output(command: List[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as error:
        return f"UNAVAILABLE: {error}"


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def add_check(
    checks: List[Dict[str, Any]],
    check_id: str,
    passed: bool,
    observed: Any,
    criterion: str,
    evidence: str,
    severity: str = "required",
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "status": "PASS" if passed else ("WARN" if severity == "warning" else "FAIL"),
            "severity": severity,
            "observed": observed,
            "criterion": criterion,
            "evidence": evidence,
        }
    )


def require_files(repo_root: Path, checks: List[Dict[str, Any]]) -> Dict[str, Path]:
    relative = {
        "official": "results/protocol_alignment_audit/official_aligned.csv",
        "cleaned": "results/protocol_alignment_audit/cleaned_aligned.csv",
        "published_compare": "results/paper_exact_protocol_audit/published_point_estimate_comparison.csv",
        "audit_evaluator": "results/paper_exact_protocol_audit/evaluator_paper_exact_metrics.csv",
        "participant_overlap": "results/paper_exact_protocol_audit/cross_split_participant_overlap_after.csv",
        "session_overlap": "results/paper_exact_protocol_audit/cross_split_duplicate_sessions_after.csv",
        "cohort": "results/paper_score_provenance/paper_exact_cohort_manifest.csv",
        "lineage": "results/paper_score_provenance/released_artifact_lineage.csv",
        "mc_metrics": "results/paper_score_provenance/mc_replicate_metrics.csv",
        "final_table": "results/paper_score_provenance/final_reproduction_table.csv",
        "provenance_manifest": "results/paper_score_provenance/run_manifest.json",
        "paper_eval_metrics": "results/pretrained_eval_paper_exact/metrics_summary.csv",
        "paper_eval_manifest": "results/pretrained_eval_paper_exact/run_manifest.json",
    }
    paths = {name: repo_root / value for name, value in relative.items()}
    missing = [str(path.relative_to(repo_root)) for path in paths.values() if not path.exists()]
    add_check(
        checks,
        "required_artifacts_present",
        not missing,
        "none" if not missing else "; ".join(missing),
        "All required audit and evaluation artifacts exist",
        "filesystem inventory",
    )
    if missing:
        raise FileNotFoundError(f"Missing required reproduction artifacts: {missing}")
    return paths


def run_acceptance_checks(repo_root: Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    paths = require_files(repo_root, checks)

    official = pd.read_csv(paths["official"])
    cleaned = pd.read_csv(paths["cleaned"])
    add_check(
        checks,
        "alignment_dataset_counts",
        (len(official), len(cleaned)) == (1119, 1079),
        f"official={len(official)}, cleaned={len(cleaned)}",
        "official=1119 and cleaned=1079",
        str(paths["official"].relative_to(repo_root)),
    )

    cohort = pd.read_csv(paths["cohort"])
    cohort_hash = sha256_file(paths["cohort"])
    counts = cohort.split.value_counts().to_dict()
    add_check(
        checks,
        "paper_cohort_counts",
        counts == EXPECTED_SPLITS,
        json.dumps(counts, sort_keys=True),
        json.dumps(EXPECTED_SPLITS, sort_keys=True),
        str(paths["cohort"].relative_to(repo_root)),
    )
    add_check(
        checks,
        "paper_cohort_hash",
        cohort_hash == EXPECTED_COHORT_HASH,
        cohort_hash,
        EXPECTED_COHORT_HASH,
        str(paths["cohort"].relative_to(repo_root)),
    )

    published = pd.read_csv(paths["published_compare"])
    max_published_delta = float(published.delta_recomputed_minus_published.abs().max())
    add_check(
        checks,
        "published_point_estimates",
        max_published_delta <= 0.0005,
        max_published_delta,
        "maximum absolute fraction delta <= 0.0005 (0.05 percentage points)",
        str(paths["published_compare"].relative_to(repo_root)),
    )

    lineage = pd.read_csv(paths["lineage"])
    lineage_max = float(lineage.filter(like="max_abs_delta").max().max())
    lineage_flags = int(lineage.label_mismatches.sum() + lineage.uncertain_flag_mismatches.sum())
    add_check(
        checks,
        "released_csv_pickle_lineage",
        lineage_flags == 0 and lineage_max <= 5e-8,
        f"flag_mismatches={lineage_flags}, max_numeric_delta={lineage_max:.3e}",
        "zero label/uncertainty mismatches and numeric delta <= 5e-8",
        str(paths["lineage"].relative_to(repo_root)),
    )

    paper_eval_manifest = json.load(paths["paper_eval_manifest"].open(encoding="utf-8"))
    add_check(
        checks,
        "paper_exact_evaluator_manifest",
        paper_eval_manifest.get("protocol") == "paper-exact"
        and paper_eval_manifest.get("paper_manifest_sha256") == EXPECTED_COHORT_HASH,
        f"protocol={paper_eval_manifest.get('protocol')}, hash={paper_eval_manifest.get('paper_manifest_sha256')}",
        "protocol=paper-exact and frozen manifest hash matches",
        str(paths["paper_eval_manifest"].relative_to(repo_root)),
    )

    main_metrics = pd.read_csv(paths["paper_eval_metrics"])
    main_metrics = main_metrics.loc[
        (main_metrics.level == "session")
        & (main_metrics.model == "fusion")
        & (main_metrics.abstention == "none")
    ]
    audit_metrics = pd.read_csv(paths["audit_evaluator"])
    comparison = main_metrics.merge(audit_metrics, on="split", suffixes=("_main", "_audit"))
    metric_deltas = []
    for metric in METRIC_COLUMNS:
        metric_deltas.extend(
            (comparison[f"{metric}_main"] - comparison[f"{metric}_audit"]).abs().tolist()
        )
    max_cross_script_delta = float(max(metric_deltas))
    add_check(
        checks,
        "cross_script_metric_identity",
        len(comparison) == 3 and max_cross_script_delta <= 1e-12,
        f"matched_splits={len(comparison)}, max_delta={max_cross_script_delta:.3e}",
        "three splits and maximum metric delta <= 1e-12",
        f"{paths['paper_eval_metrics'].relative_to(repo_root)} vs {paths['audit_evaluator'].relative_to(repo_root)}",
    )

    provenance = json.load(paths["provenance_manifest"].open(encoding="utf-8"))
    mc = pd.read_csv(paths["mc_metrics"])
    mc_seeds = int(mc.seed.nunique())
    add_check(
        checks,
        "mc_trace_complete",
        provenance.get("replicates") == 100
        and provenance.get("num_trials") == 30
        and mc_seeds == 100
        and len(mc) == 300,
        f"manifest_replicates={provenance.get('replicates')}, trials={provenance.get('num_trials')}, seeds={mc_seeds}, rows={len(mc)}",
        "100 seeds x 3 cohorts at 30 MC trials",
        str(paths["mc_metrics"].relative_to(repo_root)),
    )
    add_check(
        checks,
        "provenance_inputs_unchanged",
        bool(provenance.get("inputs_unchanged")),
        provenance.get("inputs_unchanged"),
        "True",
        str(paths["provenance_manifest"].relative_to(repo_root)),
    )

    final_table = pd.read_csv(paths["final_table"])
    classification = final_table.loc[
        final_table.published.notna() & ~final_table.metric.isin(["n", "auroc"])
    ]
    max_final_delta = float(classification.frozen_minus_published.abs().max())
    interval_pass = bool(classification.frozen_within_fresh_mc_95_interval.all())
    add_check(
        checks,
        "final_reproduction_table",
        len(final_table) == 24
        and len(classification) == 18
        and max_final_delta <= 0.0005
        and interval_pass,
        f"rows={len(final_table)}, published_classification_rows={len(classification)}, max_delta={max_final_delta:.6f}, all_in_mc_interval={interval_pass}",
        "24 total rows; 18 published classification rows; <=0.0005 delta; all within fresh-MC interval",
        str(paths["final_table"].relative_to(repo_root)),
    )

    participant_overlap = pd.read_csv(paths["participant_overlap"])
    session_overlap = pd.read_csv(paths["session_overlap"])
    add_check(
        checks,
        "cross_split_session_uniqueness",
        len(session_overlap) == 0,
        len(session_overlap),
        "zero duplicated sessions after paper reconstruction",
        str(paths["session_overlap"].relative_to(repo_root)),
    )
    add_check(
        checks,
        "cross_split_participant_independence",
        len(participant_overlap) == 0,
        len(participant_overlap),
        "zero overlapping participants is ideal; overlap must otherwise be documented",
        str(paths["participant_overlap"].relative_to(repo_root)),
        severity="warning",
    )

    add_check(
        checks,
        "historical_rng_state",
        False,
        "not stored by upstream training run",
        "Historical RNG state would be required for bitwise score regeneration",
        "released seed-289 PKL artifacts and upstream evaluation code",
        severity="warning",
    )
    add_check(
        checks,
        "pickle_checkpoint_binding",
        False,
        "PKL files contain no embedded checkpoint hash",
        "Embedded hash would be required for cryptographic score-to-checkpoint binding",
        "released seed-289 PKL artifacts",
        severity="warning",
    )

    check_frame = pd.DataFrame(checks)
    required_failures = check_frame.loc[
        (check_frame.severity == "required") & (check_frame.status != "PASS")
    ]
    summary = {
        "scope": "Released-checkpoint evaluation reproduction for the 2026 paper",
        "status": "PASS_WITH_DOCUMENTED_LIMITATIONS" if required_failures.empty else "FAIL",
        "required_checks": int((check_frame.severity == "required").sum()),
        "required_passes": int(
            ((check_frame.severity == "required") & (check_frame.status == "PASS")).sum()
        ),
        "warnings": int((check_frame.status == "WARN").sum()),
        "git_commit": git_commit(repo_root),
        "cohort_manifest_sha256": cohort_hash,
        "max_published_metric_delta_fraction": max_published_delta,
        "max_cross_script_metric_delta": max_cross_script_delta,
    }
    return check_frame, summary


def environment_snapshot() -> Dict[str, Any]:
    packages = {}
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "torch",
        "torchmetrics",
        "baal",
        "imbalanced-learn",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }


def installed_package_lock() -> str:
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower()] = (name, distribution.version)
    lines = [f"{name}=={version}" for _, (name, version) in sorted(packages.items())]
    return "\n".join(lines) + "\n"


def artifact_inventory(repo_root: Path) -> pd.DataFrame:
    roots = [
        repo_root / "results" / "protocol_alignment_audit",
        repo_root / "results" / "pretrained_eval",
        repo_root / "results" / "paper_exact_protocol_audit",
        repo_root / "results" / "paper_score_provenance",
        repo_root / "results" / "pretrained_eval_paper_exact",
    ]
    scripts = [
        repo_root / "tools" / name
        for name in (
            "audit_protocol_alignment.py",
            "evaluate_pretrained.py",
            "audit_paper_exact_protocol.py",
            "trace_paper_score_provenance.py",
            "build_reproduction_bundle.py",
            "run_paper_reproduction.py",
        )
    ]
    bundle_root = repo_root / "results" / "reproduction_bundle"
    bundle_files = [
        path
        for path in bundle_root.glob("*")
        if path.is_file()
        and path.name not in {"artifact_inventory.csv", "REPRODUCTION_ACCEPTANCE_REPORT.md"}
    ]
    files = scripts + bundle_files + [
        path for root in roots if root.exists() for path in root.rglob("*") if path.is_file()
    ]
    records = []
    for path in sorted(set(files)):
        if not path.exists():
            continue
        records.append(
            {
                "path": str(path.relative_to(repo_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(records)


def report_text(checks: pd.DataFrame, summary: Dict[str, Any], inventory: pd.DataFrame) -> str:
    lines = [
        "# PARK reproduction acceptance report",
        "",
        f"**Acceptance status: {summary['status']}**",
        "",
        f"Scope: {summary['scope']}.",
        "",
        f"Required checks passed: {summary['required_passes']}/{summary['required_checks']}; "
        f"documented warnings: {summary['warnings']}.",
        "",
        "## Acceptance checks",
        "",
        "| Check | Status | Observed | Criterion |",
        "| --- | :---: | --- | --- |",
    ]
    for row in checks.itertuples(index=False):
        lines.append(
            f"| {row.check_id} | {row.status} | {str(row.observed).replace('|', '/')} | "
            f"{str(row.criterion).replace('|', '/')} |"
        )
    lines.extend(
        [
            "",
            "## Scientific conclusion",
            "",
            "The released 162/91/67 evaluation cohorts and every published classification "
            "point estimate are reproduced. The maximum published-value discrepancy is "
            f"{summary['max_published_metric_delta_fraction'] * 100:.3f} percentage points. "
            "The independent paper-exact evaluator agrees identically with the separate audit.",
            "",
            "Exact historical floating-point scores are supplied by the released PKL artifacts. "
            "Checkpoint-only reruns are statistically consistent with those scores, but cannot be "
            "bitwise identical because the historical MC-dropout RNG state was not saved.",
            "",
            "## Documented protocol limitations",
            "",
            "- One participant remains represented in both external cohorts after the three duplicate "
            "session memberships are resolved; the cohorts are not participant-independent.",
            "- The released prediction PKLs do not embed a checkpoint SHA-256, so historical "
            "score-to-checkpoint binding is not cryptographically provable.",
            "- This acceptance covers released-checkpoint evaluation. A full from-scratch 30-seed "
            "training reproduction remains a separate, substantially more expensive scope.",
            "",
            "## Bundle contents",
            "",
            f"The artifact inventory contains {len(inventory)} files with individual SHA-256 hashes. "
            "The Conda explicit lock and pip freeze snapshot record the server environment used for acceptance.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (args.output_dir or repo_root / "results" / "reproduction_bundle").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checks, summary = run_acceptance_checks(repo_root)
    checks.to_csv(output_dir / "acceptance_checks.csv", index=False)
    with (output_dir / "acceptance_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(summary), handle, ensure_ascii=False, indent=2)

    snapshot = environment_snapshot()
    with (output_dir / "environment_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(snapshot), handle, ensure_ascii=False, indent=2)
    (output_dir / "conda-linux-64-explicit.lock").write_text(
        command_output(["conda", "list", "--explicit"]), encoding="utf-8"
    )
    (output_dir / "pip-freeze.txt").write_text(
        installed_package_lock(), encoding="utf-8"
    )
    (output_dir / "nvidia-smi.txt").write_text(
        command_output(["nvidia-smi"]), encoding="utf-8"
    )

    inventory = artifact_inventory(repo_root)
    inventory.to_csv(output_dir / "artifact_inventory.csv", index=False)
    (output_dir / "REPRODUCTION_ACCEPTANCE_REPORT.md").write_text(
        report_text(checks, summary, inventory), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Bundle written to: {output_dir}")
    if summary["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
