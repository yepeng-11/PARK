#!/usr/bin/env python3
"""Verify one frozen UFNet expert cache and an optional deterministic repeat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SPLITS = ("train", "validation", "test")
EXPECTED_ROWS = {"train": 690, "validation": 215, "test": 197}
EXPECTED_PARTICIPANTS = {"train": 516, "validation": 167, "test": 162}
EXPECTED_DIMENSIONS = {"finger": 232, "speech": 1024, "smile": 42}


def load_cache(directory: Path) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    metadata = json.loads((directory / "expert_cache_metadata.json").read_text(encoding="utf-8"))
    arrays = {
        split: dict(np.load(directory / f"expert_cache_{split}.npz", allow_pickle=False))
        for split in SPLITS
    }
    return metadata, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--repeat", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    metadata, arrays = load_cache(args.cache)
    participants = {split: set(payload["participant_id"].tolist()) for split, payload in arrays.items()}
    checks: dict[str, bool] = {
        "metadata_pass": metadata["status"] == "PASS",
        "row_counts_exact": all(len(arrays[split]["label"]) == EXPECTED_ROWS[split] for split in SPLITS),
        "participant_counts_exact": all(
            len(participants[split]) == EXPECTED_PARTICIPANTS[split] for split in SPLITS
        ),
        "participant_disjoint": not (
            participants["train"] & participants["validation"]
            or participants["train"] & participants["test"]
            or participants["validation"] & participants["test"]
        ),
        "feature_dimensions_exact": all(
            arrays[split][f"{modality}_features"].shape
            == (EXPECTED_ROWS[split], EXPECTED_DIMENSIONS[modality])
            for split in SPLITS
            for modality in EXPECTED_DIMENSIONS
        ),
        "all_numeric_arrays_finite": all(
            np.isfinite(value).all()
            for payload in arrays.values()
            for value in payload.values()
            if value.dtype.kind in "fiu"
        ),
        "labels_binary": all(set(payload["label"].tolist()) <= {0, 1} for payload in arrays.values()),
        "row_ids_unique": len(set().union(*(set(payload["manifest_row_id"].tolist()) for payload in arrays.values())))
        == sum(EXPECTED_ROWS.values()),
    }
    repeat_hashes_equal = None
    if args.repeat:
        repeat_metadata, repeat_arrays = load_cache(args.repeat)
        repeat_hashes_equal = metadata["array_sha256"] == repeat_metadata["array_sha256"]
        checks["repeat_metadata_array_hashes_equal"] = repeat_hashes_equal
        checks["repeat_split_arrays_equal"] = all(
            set(arrays[split]) == set(repeat_arrays[split])
            and all(np.array_equal(arrays[split][key], repeat_arrays[split][key]) for key in arrays[split])
            for split in SPLITS
        )

    passed = all(checks.values())
    result = {
        "status": "PASS" if passed else "FAIL",
        "cache": str(args.cache.resolve()),
        "repeat": str(args.repeat.resolve()) if args.repeat else None,
        "expected_rows": EXPECTED_ROWS,
        "expected_participants": EXPECTED_PARTICIPANTS,
        "expected_feature_dimensions": EXPECTED_DIMENSIONS,
        "repeat_hashes_equal": repeat_hashes_equal,
        "checks": checks,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Phase 1 expert-cache verification", "", f"Status: **{result['status']}**", ""]
    lines.extend(f"- {name}: **{'PASS' if value else 'FAIL'}**" for name, value in checks.items())
    (args.output / "VERIFICATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
