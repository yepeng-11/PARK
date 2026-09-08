#!/usr/bin/env python3
"""Build and validate the participant/session manifest used by the AAAI UFNet paper.

This intentionally mirrors the released UFNet data-loading code: complete-case
filtering, filename-derived participant/date keys, left/right finger-tapping
inner join, and three-task inner joins on participant-plus-calendar-date.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd


EXPECTED = {
    "all": {"participants": 845, "sessions": 1102},
    "train": {"participants": 516, "sessions": 690},
    "validation": {"participants": 167, "sessions": 215},
    "test": {"participants": 162, "sessions": 197},
}


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_date(name: str) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(name))
    if match is None:
        raise ValueError(f"No YYYY-MM-DD date in filename: {name}")
    return match.group()


def parse_finger_id(name: str) -> str:
    name = str(name)
    if name.startswith("NIH"):
        return name.split("-")[0]
    if name.endswith("finger_tapping.mp4"):
        return name.split("-")[-2]
    return name.split("_")[-4]


def parse_speech_id(name: str) -> str:
    name = str(name)
    if name.startswith("NIH"):
        return name.split("-")[0]
    if name.endswith("-quick_brown_fox.mp4"):
        return name.split("-")[-2]
    if name.endswith("_quick_brown_fox.mp4"):
        return name.split("_")[1]
    return name.split("_")[-4]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label01(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"no", "0", "0.0", "false"}:
        return 0
    if text in {"yes", "1", "1.0", "true"}:
        return 1
    raise ValueError(f"Unrecognized PD label: {value!r}")


def label_not_no(value: object) -> int:
    """Mirror released finger/smile code: 1.0 * (pd != 'no')."""
    return int(str(value).strip().lower() != "no")


def build_modality_tables(data_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    finger_path = data_dir / "finger_tapping" / "features_demography_diagnosis_Nov22_2023.csv"
    smile_path = data_dir / "facial_expression_smile" / "facial_dataset.csv"
    speech_path = data_dir / "quick_brown_fox" / "wavlm_fox_features.csv"

    finger_raw = pd.read_csv(finger_path)
    finger_exempt = {
        "Unnamed: 0", "filename", "Protocol", "Participant_ID", "Task", "Duration", "FPS",
        "Frame_Height", "Frame_Width", "gender", "age", "race", "ethnicity", "dob",
        "time_mdsupdrs",
    }
    finger = finger_raw.dropna(subset=[c for c in finger_raw.columns if c not in finger_exempt]).copy()
    finger["participant_id"] = finger["filename"].map(parse_finger_id)
    finger["date"] = finger["filename"].map(parse_date)
    finger["row_id"] = finger["participant_id"] + "#" + finger["date"]
    finger["label"] = finger["pd"].map(label_not_no)
    right = finger.loc[finger["hand"].eq("right"), ["row_id", "participant_id", "label", "filename"]].rename(
        columns={"filename": "finger_right_file", "label": "finger_right_label"}
    )
    left = finger.loc[finger["hand"].eq("left"), ["row_id", "participant_id", "label", "filename"]].rename(
        columns={"filename": "finger_left_file", "participant_id": "finger_left_id", "label": "finger_left_label"}
    )
    both = right.merge(left, how="inner", on="row_id")

    smile_raw = pd.read_csv(smile_path)
    smile = smile_raw.fillna(0).copy()
    smile["participant_id"] = smile["ID"].astype(str)
    smile["date"] = smile["Filename"].map(parse_date)
    smile["row_id"] = smile["participant_id"] + "#" + smile["date"]
    smile["label"] = smile["pd"].map(label_not_no)
    smile = smile[["row_id", "participant_id", "label", "Filename"]].rename(
        columns={"participant_id": "smile_id", "label": "smile_label", "Filename": "smile_file"}
    )

    speech_raw = pd.read_csv(speech_path)
    speech_exempt = {"Filename", "Participant_ID", "gender", "age", "race"}
    speech = speech_raw.dropna(subset=[c for c in speech_raw.columns if c not in speech_exempt]).copy()
    speech["participant_id"] = speech["Filename"].map(parse_speech_id)
    speech["date"] = speech["Filename"].map(parse_date)
    speech["row_id"] = speech["participant_id"] + "#" + speech["date"]
    speech["label"] = speech["pd"].map(label01)
    speech = speech[["row_id", "participant_id", "label", "Filename"]].rename(
        columns={"participant_id": "speech_id", "label": "speech_label", "Filename": "speech_file"}
    )

    counts = {
        "finger_raw_rows": len(finger_raw),
        "finger_complete_rows": len(finger),
        "finger_both_hand_sessions": len(both),
        "finger_both_hand_participants": both["participant_id"].nunique(),
        "smile_raw_rows": len(smile_raw),
        "smile_complete_rows": len(smile),
        "smile_participants": smile["smile_id"].nunique(),
        "speech_raw_rows": len(speech_raw),
        "speech_complete_rows": len(speech),
        "speech_participants": speech["speech_id"].nunique(),
    }
    return {"finger": both, "smile": smile, "speech": speech}, counts


def assign_split(participant_id: str, dev_ids: set[str], test_ids: set[str]) -> str:
    if participant_id in test_ids:
        return "test"
    if participant_id in dev_ids:
        return "validation"
    return "train"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="Official ROC-HCI/UFNet checkout")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    data_dir = source / "data"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    all_ids_list = read_ids(data_dir / "all_task_ids.txt")
    dev_ids_list = read_ids(data_dir / "dev_set_participants.txt")
    test_ids_list = read_ids(data_dir / "test_set_participants.txt")
    all_ids, dev_ids, test_ids = set(all_ids_list), set(dev_ids_list), set(test_ids_list)

    tables, modality_counts = build_modality_tables(data_dir)
    sessions = tables["finger"].merge(tables["speech"], how="inner", on="row_id")
    sessions = sessions.merge(tables["smile"], how="inner", on="row_id")
    sessions = sessions.rename(columns={"participant_id": "participant_id", "finger_right_label": "label"})
    sessions["split"] = sessions["participant_id"].map(lambda x: assign_split(x, dev_ids, test_ids))
    sessions["date"] = sessions["row_id"].str.rsplit("#", n=1).str[-1]
    sessions["label_consistent"] = (
        sessions["label"].eq(sessions["finger_left_label"])
        & sessions["label"].eq(sessions["speech_label"])
        & sessions["label"].eq(sessions["smile_label"])
    )
    sessions["id_consistent"] = (
        sessions["participant_id"].eq(sessions["finger_left_id"])
        & sessions["participant_id"].eq(sessions["speech_id"])
        & sessions["participant_id"].eq(sessions["smile_id"])
    )
    sessions["listed_all_tasks"] = sessions["participant_id"].isin(all_ids)
    sessions = sessions.sort_values([
        "row_id", "finger_right_file", "finger_left_file", "speech_file", "smile_file"
    ]).reset_index(drop=True)
    sessions["within_join_key_index"] = sessions.groupby("row_id").cumcount()
    sessions["manifest_row_id"] = sessions["row_id"] + "#" + sessions["within_join_key_index"].map(lambda x: f"{x:03d}")
    session_columns = [
        "manifest_row_id", "row_id", "within_join_key_index", "participant_id", "date", "split", "label", "listed_all_tasks",
        "label_consistent", "id_consistent", "finger_right_file", "finger_left_file",
        "speech_file", "smile_file",
    ]
    sessions = sessions[session_columns].sort_values(["split", "participant_id", "date", "manifest_row_id"])

    demography = pd.read_csv(data_dir / "demography_details.csv")
    demo = demography.rename(columns={"id": "participant_id", "Diagnosis": "diagnosis"}).copy()
    demo["participant_id"] = demo["participant_id"].astype(str)
    demo["label"] = demo["diagnosis"].map(label01)
    participants = pd.DataFrame({"participant_id": sorted(all_ids)})
    participants["split"] = participants["participant_id"].map(lambda x: assign_split(x, dev_ids, test_ids))
    participants = participants.merge(demo, how="left", on="participant_id", suffixes=("", "_demography"))
    session_stats = sessions.groupby("participant_id").agg(
        session_count=("row_id", "size"), session_label=("label", "first")
    ).reset_index()
    participants = participants.merge(session_stats, how="left", on="participant_id")
    participants["session_count"] = participants["session_count"].fillna(0).astype(int)
    participants["present_in_reconstructed_sessions"] = participants["session_count"].gt(0)
    participants = participants.sort_values(["split", "participant_id"])

    summary_rows = []
    for split in ["train", "validation", "test"]:
        part_count = int(participants["split"].eq(split).sum())
        session_count = int(sessions["split"].eq(split).sum())
        summary_rows.append({
            "split": split,
            "participants_observed": part_count,
            "participants_expected": EXPECTED[split]["participants"],
            "sessions_observed": session_count,
            "sessions_expected": EXPECTED[split]["sessions"],
            "participants_match": part_count == EXPECTED[split]["participants"],
            "sessions_match": session_count == EXPECTED[split]["sessions"],
        })
    summary_rows.append({
        "split": "all",
        "participants_observed": int(participants["participant_id"].nunique()),
        "participants_expected": EXPECTED["all"]["participants"],
        "sessions_observed": len(sessions),
        "sessions_expected": EXPECTED["all"]["sessions"],
        "participants_match": participants["participant_id"].nunique() == EXPECTED["all"]["participants"],
        "sessions_match": len(sessions) == EXPECTED["all"]["sessions"],
    })
    summary = pd.DataFrame(summary_rows)

    union_keys = set().union(*(set(frame["row_id"]) for frame in tables.values()))
    presence = pd.DataFrame({"row_id": sorted(union_keys)})
    for name, frame in tables.items():
        presence[f"has_{name}"] = presence["row_id"].isin(set(frame["row_id"]))
    presence["included_three_task"] = presence[["has_finger", "has_smile", "has_speech"]].all(axis=1)
    presence["participant_id"] = presence["row_id"].str.rsplit("#", n=1).str[0]
    presence["listed_all_tasks"] = presence["participant_id"].isin(all_ids)
    presence["exclusion_reason"] = presence.apply(
        lambda row: "included" if row["included_three_task"] else ";".join(
            f"missing_{name}" for name in ["finger", "smile", "speech"] if not row[f"has_{name}"]
        ), axis=1,
    )

    source_files = [
        data_dir / "all_task_ids.txt", data_dir / "dev_set_participants.txt",
        data_dir / "test_set_participants.txt", data_dir / "demography_details.csv",
        data_dir / "finger_tapping" / "features_demography_diagnosis_Nov22_2023.csv",
        data_dir / "facial_expression_smile" / "facial_dataset.csv",
        data_dir / "quick_brown_fox" / "wavlm_fox_features.csv",
        source / "code" / "fusion_models" / "ufnet" / "UFNet_no_withhold.py",
    ]
    provenance = {
        "source_repository": "https://github.com/ROC-HCI/UFNet.git",
        "source_checkout": source.name,
        "git_commit": "unknown",
        "hash_algorithm": "sha256",
        "files": {str(path.relative_to(source)).replace("\\", "/"): sha256(path) for path in source_files},
        "official_list_sizes": {
            "all_task_ids": len(all_ids_list), "dev_all_tasks_or_unimodal": len(dev_ids_list),
            "test_all_tasks_or_unimodal": len(test_ids_list),
            "all_task_intersection_dev": len(all_ids & dev_ids),
            "all_task_intersection_test": len(all_ids & test_ids),
            "all_task_train_remainder": len(all_ids - dev_ids - test_ids),
        },
        "modality_counts": modality_counts,
    }
    try:
        provenance["git_commit"] = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass

    checks = {
        "exact_paper_counts": bool(summary["participants_match"].all() and summary["sessions_match"].all()),
        "participant_lists_unique": len(all_ids_list) == len(all_ids),
        "dev_test_disjoint_within_fusion_cohort": not bool((all_ids & dev_ids) & (all_ids & test_ids)),
        "all_session_ids_consistent": bool(sessions["id_consistent"].all()),
        "all_session_labels_consistent": bool(sessions["label_consistent"].all()),
        "all_sessions_in_all_task_list": bool(sessions["listed_all_tasks"].all()),
        "all_listed_participants_have_sessions": bool(participants["present_in_reconstructed_sessions"].all()),
        "unique_manifest_row_ids": not bool(sessions["manifest_row_id"].duplicated().any()),
        "no_duplicate_source_combinations": not bool(sessions.duplicated([
            "finger_right_file", "finger_left_file", "speech_file", "smile_file"
        ]).any()),
    }
    provenance["checks"] = checks

    sessions.to_csv(output / "paper_session_manifest.csv", index=False)
    participants.to_csv(output / "paper_participant_manifest.csv", index=False)
    summary.to_csv(output / "paper_partition_summary.csv", index=False)
    presence.to_csv(output / "paper_modality_presence_and_exclusions.csv", index=False)
    (output / "source_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    status = "PASS" if all(checks.values()) else "FAIL"
    headers = list(summary.columns)
    table_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in summary.itertuples(index=False, name=None):
        table_lines.append("| " + " | ".join(str(value) for value in row) + " |")
    table = "\n".join(table_lines)
    report = f"""# UFNet paper-aligned manifest audit

Status: **{status}**

Official source commit: `{provenance['git_commit']}`

## Paper count gate

{table}

## Interpretation

- `all_task_ids.txt` defines the 845-participant three-task fusion cohort.
- The released validation/test lists each contain 267 participants because they also cover the larger unimodal cohorts.
- Intersecting those lists with the fusion cohort yields {len(all_ids & dev_ids)} validation and {len(all_ids & test_ids)} test participants; the remaining {len(all_ids - dev_ids - test_ids)} are training participants.
- Sessions are reconstructed with the exact released-code key: parsed `participant_id#YYYY-MM-DD`; finger tapping first requires both left and right hands, then finger/speech/smile are inner-joined.
- The released join key is not unique when a participant repeats tasks on the same day. The manifest preserves the paper code's Cartesian combinations and adds a deterministic `manifest_row_id`; deduplicating the date key would incorrectly reduce 1,102 sessions to 1,013.
- No row was deleted merely to force a paper count. Any failed check makes this manifest non-authoritative.

## Integrity checks

```json
{json.dumps(checks, indent=2)}
```

## Files

- `paper_session_manifest.csv`: immutable session-level split and source-file mapping.
- `paper_participant_manifest.csv`: immutable participant-level split, labels, demographics, and session counts.
- `paper_partition_summary.csv`: expected-versus-observed gate.
- `paper_modality_presence_and_exclusions.csv`: modality availability and explicit exclusion reason.
- `source_provenance.json`: source hashes, commit, raw list sizes, and checks.
"""
    (output / "AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
