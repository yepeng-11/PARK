#!/usr/bin/env python3
"""One-command PARK released-checkpoint paper reproduction pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--mc-replicates", type=int, default=100)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun every stage instead of reusing existing stage artifacts.",
    )
    return parser.parse_args()


def stages(repo_root: Path, device: str, replicates: int) -> List[Dict[str, object]]:
    python = sys.executable
    return [
        {
            "name": "protocol_alignment",
            "command": [python, "tools/audit_protocol_alignment.py"],
            "expected": ["results/protocol_alignment_audit/DIFFERENCE_REPORT.md"],
        },
        {
            "name": "source_pretrained_evaluation",
            "command": [python, "tools/evaluate_pretrained.py", "--device", device],
            "expected": ["results/pretrained_eval/run_manifest.json"],
        },
        {
            "name": "paper_exact_protocol_audit",
            "command": [python, "tools/audit_paper_exact_protocol.py"],
            "expected": ["results/paper_exact_protocol_audit/PAPER_EXACT_PROTOCOL_REPORT.md"],
        },
        {
            "name": "paper_score_provenance",
            "command": [
                python,
                "tools/trace_paper_score_provenance.py",
                "--replicates",
                str(replicates),
                "--device",
                device,
            ],
            "expected": [
                "results/paper_score_provenance/FINAL_REPRODUCTION_TABLE.md",
                "results/paper_score_provenance/paper_exact_cohort_manifest.csv",
            ],
        },
        {
            "name": "paper_exact_pretrained_evaluation",
            "command": [
                python,
                "tools/evaluate_pretrained.py",
                "--protocol",
                "paper-exact",
                "--device",
                device,
            ],
            "expected": ["results/pretrained_eval_paper_exact/run_manifest.json"],
        },
        {
            "name": "acceptance_bundle",
            "command": [python, "tools/build_reproduction_bundle.py"],
            "expected": ["results/reproduction_bundle/REPRODUCTION_ACCEPTANCE_REPORT.md"],
            "always_run": True,
        },
    ]


def main() -> None:
    args = parse_args()
    if args.mc_replicates != 100:
        raise ValueError("The locked acceptance protocol requires exactly 100 MC replicates")
    repo_root = args.repo_root.resolve()
    log_dir = repo_root / "results" / "reproduction_bundle" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    state = []
    for stage in stages(repo_root, args.device, args.mc_replicates):
        expected = [repo_root / value for value in stage["expected"]]
        reusable = all(path.exists() for path in expected)
        always_run = bool(stage.get("always_run", False))
        if reusable and not args.force and not always_run:
            print(f"[reuse] {stage['name']}")
            state.append({"stage": stage["name"], "status": "reused"})
            continue
        print(f"[run] {stage['name']}")
        completed = subprocess.run(
            stage["command"],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output = completed.stdout or ""
        print(output, end="")
        (log_dir / f"{stage['name']}.log").write_text(output, encoding="utf-8")
        if completed.returncode != 0:
            state.append(
                {"stage": stage["name"], "status": "failed", "returncode": completed.returncode}
            )
            (log_dir / "pipeline_state.json").write_text(
                json.dumps(state, indent=2), encoding="utf-8"
            )
            raise SystemExit(completed.returncode)
        if not all(path.exists() for path in expected):
            raise RuntimeError(f"Stage {stage['name']} completed without expected outputs")
        state.append({"stage": stage["name"], "status": "completed"})
    (log_dir / "pipeline_state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    print("PARK released-checkpoint reproduction pipeline accepted.")


if __name__ == "__main__":
    main()
