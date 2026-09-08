#!/usr/bin/env python3
"""Run the 30 final UFNet seeds reported by the official repository."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SEEDS = [870, 782, 563, 879, 585, 800, 485, 710, 821, 622, 524, 987, 812,
         784, 733, 819, 900, 604, 122, 958, 346, 986, 701, 595, 253, 771,
         548, 580, 856, 242]

FIXED_ARGS = [
    "--learning_rate", "0.020724443604128343",
    "--dropout_prob", "0.495989214406461",
    "--num_buckets", "20", "--num_trials", "30",
    "--uncertainty_weight", "81.81790352752515",
    "--minority_oversample", "no", "--sampler", "SMOTE",
    "--train_random_noise", "no", "--validation_random_noise", "no",
    "--increase_variance", "no", "--temperature", "0.05",
    "--noise_variance", "0.01", "--random_state", "357",
    "--model_subset_choice", "0", "--batch_size", "1024",
    "--num_epochs", "164", "--hidden_dim", "512", "--query_dim", "64",
    "--last_hidden_dim", "128", "--optimizer", "SGD", "--beta1", "0.9",
    "--beta2", "0.999", "--weight_decay", "0.0001",
    "--momentum", "0.6897821582954526", "--use_scheduler", "no",
    "--scheduler", "reduce", "--step_size", "5",
    "--gamma", "0.57143922410234", "--patience", "5",
]


def extract_metrics(output: str) -> dict:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{'accuracy':"):
            value = ast.literal_eval(line)
            if isinstance(value, dict) and "auroc" in value:
                return value
    raise RuntimeError("Final metrics dictionary not found in output")


def flatten(seed: int, metrics: dict, seconds: float) -> dict:
    result = {"seed": seed, "runtime_seconds": seconds}
    for key, value in metrics.items():
        if key == "confusion_matrix":
            for subkey, subvalue in value.items():
                result[f"confusion_{subkey}"] = subvalue
        else:
            result[key] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    args = parser.parse_args()

    repo, output = args.repo.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    script = repo / "code" / "fusion_models" / "ufnet" / "UFNet_no_withhold.py"
    expected_count_text = [
        "Number of training samples: 690.",
        "Number of validation samples: 215.",
        "Number of test samples: 197.",
    ]

    started = datetime.now(timezone.utc).isoformat()
    rows = []
    for index, seed in enumerate(SEEDS, start=1):
        result_path = output / f"seed_{seed}.json"
        log_path = output / f"seed_{seed}.log"
        if result_path.exists():
            row = json.loads(result_path.read_text(encoding="utf-8"))
            rows.append(row)
            print(f"[{index:02d}/30] seed={seed} resume AUROC={row['auroc']:.6f}", flush=True)
            continue

        print(f"[{index:02d}/30] seed={seed} starting", flush=True)
        command = [str(args.python), str(script), *FIXED_ARGS, "--seed", str(seed)]
        start = time.monotonic()
        proc = subprocess.run(
            command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        seconds = time.monotonic() - start
        log_path.write_text(proc.stdout, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"seed={seed} failed with exit code {proc.returncode}; see {log_path}")
        missing = [text for text in expected_count_text if text not in proc.stdout]
        if missing:
            raise RuntimeError(f"seed={seed} protocol count gate failed: {missing}")
        row = flatten(seed, extract_metrics(proc.stdout), seconds)
        result_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        rows.append(row)
        print(f"[{index:02d}/30] seed={seed} done AUROC={row['auroc']:.6f} time={seconds:.1f}s", flush=True)

    frame = pd.DataFrame(rows).sort_values("seed")
    frame.to_csv(output / "per_seed_metrics.csv", index=False)
    metric_names = [
        "auroc", "average_precision", "accuracy", "weighted_accuracy", "f1_score",
        "sensitivity", "specificity", "BS", "ECE", "loss",
    ]
    summaries = {}
    for name in metric_names:
        values = pd.to_numeric(frame[name], errors="raise")
        mean, std = float(values.mean()), float(values.std(ddof=1))
        half_width = 1.96 * std / math.sqrt(len(values))
        summaries[name] = {
            "mean": mean, "std": std, "ci95_low_normal": mean - half_width,
            "ci95_high_normal": mean + half_width, "min": float(values.min()),
            "max": float(values.max()),
        }

    reference_path = repo / "code" / "performance_analysis" / "wandb_reports" / "final_fusion_no_drop_preds_May132024.csv"
    comparison = {}
    if reference_path.exists():
        reference = pd.read_csv(reference_path)
        reference["seed"] = reference["seed"].astype(int)
        joined = frame.merge(reference, on="seed", suffixes=("_reproduced", "_official_log"))
        for name in ["auroc", "average_precision", "accuracy", "f1_score"]:
            delta = (joined[f"{name}_reproduced"] - joined[f"{name}_official_log"]).abs()
            comparison[name] = {"max_absolute_delta": float(delta.max()), "mean_absolute_delta": float(delta.mean())}

    summary = {
        "status": "PASS",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo), "seeds": SEEDS, "num_seeds": len(frame),
        "protocol": {"train_sessions": 690, "validation_sessions": 215, "test_sessions": 197},
        "metrics": summaries, "comparison_to_official_wandb_export": comparison,
        "total_runtime_seconds": float(frame["runtime_seconds"].sum()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
