#!/usr/bin/env python3
"""Audit PARK/UFNet evaluation protocol and build official/clean aligned tables.

The script is deliberately non-destructive: it imports the upstream loaders and
checkpoints, writes only below ``--output-dir``, and verifies that the shipped
fusion CSV has not changed while the audit runs.

Two datasets are exported:

* ``official_aligned.csv`` reproduces the upstream many-to-many ``row_id`` joins
  and keeps the first (finger-tapping) modality label.
* ``cleaned_aligned.csv`` contains one row per participant/date. Repeated rows
  within each modality are mean-aggregated and the binary label is resolved by
  majority vote across finger, speech, and smile.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import evaluate_pretrained as ev


MODALITY_EXPORT_NAMES = ("finger", "speech", "smile")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root of the PARK repository.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/results/protocol_alignment_audit).",
    )
    return parser.parse_args()


def vector_array(values: Iterable[Any]) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in values]
    if not arrays:
        raise ValueError("Cannot aggregate an empty feature group")
    dimensions = {len(value) for value in arrays}
    if len(dimensions) != 1:
        raise ValueError(f"Feature dimensions disagree inside one row_id: {dimensions}")
    return np.mean(np.stack(arrays, axis=0), axis=0)


def binary_vote(values: Iterable[Any]) -> Tuple[float, str, bool]:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().astype(int)
    numeric = numeric[numeric.isin([0, 1])]
    unique = sorted(numeric.unique().tolist())
    rendered = "|".join(map(str, unique))
    if numeric.empty:
        return float("nan"), rendered, True
    counts = numeric.value_counts()
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return float("nan"), rendered, True
    return float(counts.index[0]), rendered, False


def raw_frame(
    features: Iterable[Any], labels: Iterable[Any], ids: Iterable[Any], row_ids: Iterable[Any]
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "features": list(features),
            "label": np.asarray(labels),
            "id": np.asarray(ids).astype(str),
            "row_id": np.asarray(row_ids).astype(str),
        }
    )
    frame["source_row"] = np.arange(len(frame), dtype=int)
    return frame


def scale_frame(frame: pd.DataFrame, scaler_path: Path, enabled: bool) -> pd.DataFrame:
    if not enabled:
        return frame
    import pickle

    with scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)
    output = frame.copy()
    output["features"] = list(scaler.transform(np.stack(output.features.to_numpy())))
    return output


def collapse_modality(frame: pd.DataFrame, name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    for row_id, group in frame.groupby("row_id", sort=False, dropna=False):
        ids = sorted(group.id.astype(str).unique().tolist())
        label, label_values, unresolved = binary_vote(group.label)
        audit.append(
            {
                "row_id": row_id,
                f"{name}_rows": len(group),
                f"{name}_ids": "|".join(ids),
                f"{name}_label_values": label_values,
                f"{name}_label_unresolved": unresolved,
            }
        )
        if len(ids) != 1 or unresolved:
            continue
        rows.append(
            {
                "row_id": row_id,
                "id": ids[0],
                "label": label,
                "features": vector_array(group.features),
                "source_rows": len(group),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(audit)


def load_raw_modalities(module, selected_models, paths):
    raw: Dict[str, pd.DataFrame] = {}
    configs: List[Dict[str, Any]] = []
    for model_name, model_paths in zip(selected_models, paths):
        config = ev.read_json(model_paths["config"])
        configs.append(config)
        drop_correlated = config["drop_correlated"] == "yes"
        if "finger_model" in model_name:
            right = module.load_finger_data(
                drop_correlated=drop_correlated,
                corr_thr=config["corr_thr"],
                hand="right",
            )
            left = module.load_finger_data(
                drop_correlated=drop_correlated,
                corr_thr=config["corr_thr"],
                hand="left",
            )
            raw["finger_right"] = scale_frame(
                raw_frame(right[0], right[1], right[2], right[4]),
                model_paths["scaler"],
                False,
            )
            raw["finger_left"] = raw_frame(left[0], left[1], left[2], left[4])
            # The official finger scaler expects concatenated left/right vectors,
            # so scaling is applied after the two clean sides are combined.
        elif "fox_model" in model_name:
            values = module.load_qbf_data(
                drop_correlated=drop_correlated, corr_thr=config["corr_thr"]
            )
            raw["speech"] = scale_frame(
                raw_frame(values[0], values[1], values[2], values[4]),
                model_paths["scaler"],
                config["use_feature_scaling"] == "yes",
            )
        elif "facial_expression_smile" in model_name:
            values = module.load_smile_data(
                drop_correlated=drop_correlated, corr_thr=config["corr_thr"]
            )
            raw["smile"] = scale_frame(
                raw_frame(values[0], values[1], values[2], values[4]),
                model_paths["scaler"],
                config["use_feature_scaling"] == "yes",
            )
        else:
            raise ValueError(f"Unknown upstream model: {model_name}")
    return raw, configs


def build_cleaned_dataset(raw, finger_scaler_path: Path, finger_scale: bool):
    right, right_audit = collapse_modality(raw["finger_right"], "finger_right")
    left, left_audit = collapse_modality(raw["finger_left"], "finger_left")
    finger = pd.merge(right, left, on="row_id", how="inner", suffixes=("_right", "_left"))
    finger_id_conflict = finger.id_right != finger.id_left
    finger_label_conflict = finger.label_right != finger.label_left
    finger["id"] = finger.id_right
    finger["label"] = np.where(finger_label_conflict, np.nan, finger.label_right)
    finger["features"] = [
        np.concatenate((right_value, left_value))
        for right_value, left_value in zip(finger.features_right, finger.features_left)
    ]
    if finger_scale:
        finger = scale_frame(finger, finger_scaler_path, True)
    finger = finger.loc[~finger_id_conflict & finger.label.notna(), [
        "row_id", "id", "label", "features"
    ]].copy()

    speech, speech_audit = collapse_modality(raw["speech"], "speech")
    smile, smile_audit = collapse_modality(raw["smile"], "smile")

    combined = finger.rename(
        columns={"id": "id_finger", "label": "label_finger", "features": "features_0"}
    )
    combined = pd.merge(
        combined,
        speech.rename(
            columns={"id": "id_speech", "label": "label_speech", "features": "features_1"}
        )[["row_id", "id_speech", "label_speech", "features_1"]],
        on="row_id",
        how="inner",
        validate="one_to_one",
    )
    combined = pd.merge(
        combined,
        smile.rename(
            columns={"id": "id_smile", "label": "label_smile", "features": "features_2"}
        )[["row_id", "id_smile", "label_smile", "features_2"]],
        on="row_id",
        how="inner",
        validate="one_to_one",
    )

    id_conflict = ~(
        (combined.id_finger == combined.id_speech)
        & (combined.id_finger == combined.id_smile)
    )
    label_rows: List[Dict[str, Any]] = []
    resolved_labels: List[float] = []
    unresolved_labels: List[bool] = []
    for values in combined[["label_finger", "label_speech", "label_smile"]].to_numpy():
        label, rendered, unresolved = binary_vote(values)
        resolved_labels.append(label)
        unresolved_labels.append(unresolved)
        label_rows.append({"clean_label_values": rendered})
    combined["label"] = resolved_labels
    combined["label_unresolved"] = unresolved_labels
    combined["cross_modal_label_conflict"] = (
        combined[["label_finger", "label_speech", "label_smile"]].nunique(axis=1) > 1
    )
    combined["id_conflict"] = id_conflict
    combined["id"] = combined.id_finger
    combined["clean_label_values"] = [row["clean_label_values"] for row in label_rows]

    clean = combined.loc[
        ~combined.id_conflict & ~combined.label_unresolved,
        ["features_0", "label", "id", "row_id", "features_1", "features_2"],
    ].reset_index(drop=True)

    row_audit = right_audit
    for audit in (left_audit, speech_audit, smile_audit):
        row_audit = pd.merge(row_audit, audit, on="row_id", how="outer")
    clean_flags = combined[[
        "row_id",
        "id",
        "label",
        "label_finger",
        "label_speech",
        "label_smile",
        "clean_label_values",
        "cross_modal_label_conflict",
        "id_conflict",
        "label_unresolved",
    ]].rename(columns={"label": "clean_label", "id": "clean_id"})
    row_audit = pd.merge(row_audit, clean_flags, on="row_id", how="outer")
    return clean, row_audit, {
        "finger_side_id_conflict_rows": int(finger_id_conflict.sum()),
        "finger_side_label_conflict_rows": int(finger_label_conflict.sum()),
        "cross_modal_id_conflict_rows": int(id_conflict.sum()),
        "cross_modal_label_conflict_rows": int(combined.cross_modal_label_conflict.sum()),
        "unresolved_clean_label_rows": int(combined.label_unresolved.sum()),
    }


def serialize_vectors(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()
    for column in ("features_0", "features_1", "features_2"):
        output[column] = output[column].map(
            lambda value: json.dumps(np.asarray(value, dtype=float).tolist(), separators=(",", ":"))
        )
    output["label"] = output.label.astype(int)
    return output


def split_definitions(module) -> Dict[str, Sequence[str]]:
    return {
        "internal_test": list(map(str, module.test_ids)),
        "validation_1": list(map(str, module.test_ids_validation_1)),
        "validation_2": list(map(str, module.test_ids_validation_2)),
        "global": list(map(str, module.test_ids_global)),
    }


def split_membership_table(definitions: Dict[str, Sequence[str]]) -> pd.DataFrame:
    all_ids = sorted(set().union(*(set(values) for values in definitions.values())))
    rows = []
    for participant_id in all_ids:
        item: Dict[str, Any] = {"id": participant_id}
        memberships = []
        for name, values in definitions.items():
            present = participant_id in set(values)
            item[name] = present
            if present:
                memberships.append(name)
        item["membership_count"] = len(memberships)
        item["memberships"] = "|".join(memberships)
        rows.append(item)
    return pd.DataFrame(rows)


def pairwise_split_overlap(definitions: Dict[str, Sequence[str]]) -> List[Dict[str, Any]]:
    rows = []
    for left, right in combinations(definitions, 2):
        overlap = set(definitions[left]) & set(definitions[right])
        rows.append({"left": left, "right": right, "participants": len(overlap)})
    return rows


def partition_membership_table(module, participant_ids: Iterable[str]) -> pd.DataFrame:
    test_ids = set(map(str, module.test_ids))
    dev_ids = set(map(str, module.dev_ids))
    validation_1 = set(map(str, module.test_ids_validation_1))
    validation_2 = set(map(str, module.test_ids_validation_2))
    global_ids = set(map(str, module.test_ids_global))
    rows = []
    for participant_id in sorted(set(map(str, participant_ids))):
        in_test = participant_id in test_ids
        in_dev = participant_id in dev_ids
        rows.append(
            {
                "id": participant_id,
                "partition": "test" if in_test else ("dev" if in_dev else "train"),
                "in_test": in_test,
                "in_dev": in_dev,
                "in_validation_1": participant_id in validation_1,
                "in_validation_2": participant_id in validation_2,
                "in_global": participant_id in global_ids,
            }
        )
    return pd.DataFrame(rows)


def dataset_summary(name: str, dataframe: pd.DataFrame) -> Dict[str, Any]:
    return {
        "dataset": name,
        "rows": len(dataframe),
        "unique_row_ids": int(dataframe.row_id.nunique()),
        "duplicate_row_id_rows": int(dataframe.row_id.duplicated().sum()),
        "participants": int(dataframe.id.nunique()),
        "positive_rows": int(dataframe.label.sum()),
        "positive_participants": int(dataframe.groupby("id").label.first().sum()),
    }


def markdown_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, rule, *body])


def write_report(
    output_path: Path,
    summaries: List[Dict[str, Any]],
    clean_audit: Dict[str, Any],
    official_audit: List[Dict[str, Any]],
    shipped_comparison: Dict[str, Any],
    split_counts: List[Dict[str, Any]],
    partition_counts: List[Dict[str, Any]],
    overlaps: List[Dict[str, Any]],
    split_structure: Dict[str, Any],
) -> None:
    lines = [
        "# PARK/UFNet evaluation protocol and alignment audit",
        "",
        "## Executive finding",
        "",
        "The official table reproduces the upstream many-to-many joins and keeps the "
        "finger-tapping label. The cleaned table has exactly one row per participant/date, "
        "mean-aggregates repeated records within each modality, and resolves the final "
        "label by three-modality majority vote.",
        "",
        "## Dataset difference",
        "",
        markdown_table(
            summaries,
            ["dataset", "rows", "unique_row_ids", "duplicate_row_id_rows", "participants", "positive_rows"],
        ),
        "",
        "## Official alignment findings",
        "",
        markdown_table(
            official_audit,
            ["model", "overlapping_rows", "label_mismatch_rows", "label_mismatch_participants", "unique_row_ids", "duplicate_row_id_rows"],
        ),
        "",
        "## Clean alignment findings",
        "",
        markdown_table([clean_audit], list(clean_audit)),
        "",
        "## Shipped table comparison",
        "",
        markdown_table([shipped_comparison], list(shipped_comparison)),
        "",
        "## Evaluation split counts on exported datasets",
        "",
        markdown_table(split_counts, ["dataset", "split", "rows", "participants"]),
        "",
        "## Official train/dev/test partition counts",
        "",
        markdown_table(partition_counts, ["dataset", "partition", "rows", "participants"]),
        "",
        "## Pairwise overlap among upstream evaluation ID lists",
        "",
        markdown_table(overlaps, ["left", "right", "participants"]),
        "",
        "The upstream `internal_test` list is the pooled union of the three named "
        "evaluation cohorts, so those four results are not independent. The one-person "
        "overlap between validation 1 and validation 2 should be retained only for exact "
        "upstream reproduction and removed when constructing independent future cohorts.",
        "",
        markdown_table([split_structure], list(split_structure)),
        "",
        "## Recommended use",
        "",
        "- Use `official_aligned.csv` only when checking compatibility with the shipped checkpoint and upstream code.",
        "- Use `cleaned_aligned.csv` for new baselines and dynamic-routing experiments.",
        "- Keep all sessions from one participant in one data partition.",
        "- Treat rows marked in `alignment_row_audit.csv` as provenance issues, not model errors.",
        "- Report both session-level and participant-level metrics and state the aggregation rule explicitly.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (args.output_dir or repo_root / "results" / "protocol_alignment_audit").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    module = ev.load_upstream_module(repo_root)
    fusion_config_path = Path(module.MODEL_CONFIG_PATH)
    fusion_config = ev.read_json(fusion_config_path)
    selected_models = module.MODEL_SUBSETS[int(fusion_config["model_subset_choice"])]
    paths = ev.checkpoint_paths(module, selected_models)

    shipped_path = repo_root / "data" / "full_fusion_dataset.csv"
    shipped_hash_before = ev.sha256_file(shipped_path) if shipped_path.exists() else None

    official, predictor_configs, official_audit = ev.build_aligned_dataframe(
        module, selected_models, paths
    )
    raw, raw_configs = load_raw_modalities(module, selected_models, paths)
    if predictor_configs != raw_configs:
        raise ValueError("Predictor configurations changed between audit passes")
    cleaned, row_audit, clean_audit = build_cleaned_dataset(
        raw,
        paths[0]["scaler"],
        predictor_configs[0]["use_feature_scaling"] == "yes",
    )

    official_multiplicity = (
        official.groupby("row_id").size().rename("official_rows").reset_index()
    )
    row_audit = pd.merge(row_audit, official_multiplicity, on="row_id", how="outer")
    row_audit["present_in_cleaned"] = row_audit.row_id.isin(set(cleaned.row_id))

    official_unique = official.drop_duplicates("row_id")[["row_id", "label"]]
    label_comparison = pd.merge(
        official_unique,
        cleaned[["row_id", "label"]],
        on="row_id",
        suffixes=("_official", "_clean"),
        validate="one_to_one",
    )
    clean_audit["labels_changed_by_majority_vote"] = int(
        (label_comparison.label_official != label_comparison.label_clean).sum()
    )

    official_export = serialize_vectors(official)
    cleaned_export = serialize_vectors(cleaned)
    official_export.to_csv(output_dir / "official_aligned.csv", index=False)
    cleaned_export.to_csv(output_dir / "cleaned_aligned.csv", index=False)
    row_audit.sort_values("row_id").to_csv(output_dir / "alignment_row_audit.csv", index=False)

    definitions = split_definitions(module)
    membership = split_membership_table(definitions)
    membership.to_csv(output_dir / "split_membership_audit.csv", index=False)
    overlaps = pairwise_split_overlap(definitions)
    subset_union = (
        set(definitions["validation_1"])
        | set(definitions["validation_2"])
        | set(definitions["global"])
    )
    split_structure = {
        "internal_test_equals_subgroup_union": set(definitions["internal_test"])
        == subset_union,
        "validation_1_validation_2_overlap": len(
            set(definitions["validation_1"]) & set(definitions["validation_2"])
        ),
        "validation_1_global_overlap": len(
            set(definitions["validation_1"]) & set(definitions["global"])
        ),
        "validation_2_global_overlap": len(
            set(definitions["validation_2"]) & set(definitions["global"])
        ),
    }

    all_participants = set(official.id.astype(str)) | set(cleaned.id.astype(str))
    partition_membership = partition_membership_table(module, all_participants)
    partition_membership.to_csv(output_dir / "partition_membership_audit.csv", index=False)
    partition_counts: List[Dict[str, Any]] = []
    for dataset_name, frame in (("official", official), ("cleaned", cleaned)):
        id_to_partition = partition_membership.set_index("id").partition
        assigned = frame.assign(
            partition=frame.id.astype(str).map(id_to_partition).fillna("unassigned")
        )
        for partition, selected in assigned.groupby("partition", sort=False):
            partition_counts.append(
                {
                    "dataset": dataset_name,
                    "partition": partition,
                    "rows": len(selected),
                    "participants": int(selected.id.nunique()),
                }
            )

    shipped_comparison: Dict[str, Any] = {
        "exists": shipped_path.exists(),
        "sha256_unchanged": None,
        "shape_match": None,
        "key_columns_exact_match": None,
    }
    if shipped_path.exists():
        shipped = pd.read_csv(shipped_path)
        shipped_comparison["shape_match"] = list(shipped.shape) == list(official_export.shape)
        labels_match = np.array_equal(
            pd.to_numeric(shipped.label).to_numpy(),
            pd.to_numeric(official_export.label).to_numpy(),
        )
        ids_match = np.array_equal(
            shipped.id.astype(str).to_numpy(), official_export.id.astype(str).to_numpy()
        )
        row_ids_match = np.array_equal(
            shipped.row_id.astype(str).to_numpy(), official_export.row_id.astype(str).to_numpy()
        )
        shipped_comparison["key_columns_exact_match"] = bool(
            labels_match and ids_match and row_ids_match
        )
        shipped_comparison["sha256_unchanged"] = shipped_hash_before == ev.sha256_file(shipped_path)

    summaries = [dataset_summary("official", official), dataset_summary("cleaned", cleaned)]
    split_counts: List[Dict[str, Any]] = []
    for dataset_name, frame in (("official", official), ("cleaned", cleaned)):
        for split_name, ids in definitions.items():
            selected = frame[frame.id.astype(str).isin(set(ids))]
            split_counts.append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "rows": len(selected),
                    "participants": int(selected.id.nunique()),
                }
            )

    protocol = {
        "repo_root": repo_root,
        "git_commit": ev.git_commit(repo_root),
        "fusion_config": fusion_config_path,
        "selected_models": selected_models,
        "decision_threshold": 0.5,
        "official_join_key": "row_id = participant_id#YYYY-MM-DD",
        "official_join_cardinality": "many-to-many",
        "official_label_source": "label_0 (finger tapping)",
        "clean_join_cardinality": "one-to-one after per-modality aggregation",
        "clean_duplicate_feature_rule": "arithmetic mean within modality and row_id",
        "clean_label_rule": "majority vote across finger, speech, and smile",
        "dataset_summary": summaries,
        "official_alignment_audit": official_audit,
        "clean_alignment_audit": clean_audit,
        "shipped_dataset_comparison": shipped_comparison,
        "split_counts": split_counts,
        "partition_counts": partition_counts,
        "pairwise_split_overlap": overlaps,
        "evaluation_split_structure": split_structure,
        "feature_dimensions": {
            name: int(len(cleaned.iloc[0][f"features_{index}"]))
            for index, name in enumerate(MODALITY_EXPORT_NAMES)
        },
        "checkpoint_sha256": {
            str(path.relative_to(repo_root)): ev.sha256_file(path)
            for item in paths
            for key, path in item.items()
            if key in {"model", "scaler"}
        },
    }
    with (output_dir / "protocol_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(ev.json_ready(protocol), handle, ensure_ascii=False, indent=2)

    pd.DataFrame(summaries).to_csv(output_dir / "dataset_summary.csv", index=False)
    pd.DataFrame(split_counts).to_csv(output_dir / "split_counts.csv", index=False)
    pd.DataFrame(partition_counts).to_csv(output_dir / "partition_counts.csv", index=False)
    pd.DataFrame(overlaps).to_csv(output_dir / "split_overlap.csv", index=False)
    write_report(
        output_dir / "DIFFERENCE_REPORT.md",
        summaries,
        clean_audit,
        official_audit,
        shipped_comparison,
        split_counts,
        partition_counts,
        overlaps,
        split_structure,
    )

    print("\nDataset summary")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("\nClean alignment audit")
    print(json.dumps(clean_audit, indent=2))
    print("\nShipped table comparison")
    print(json.dumps(shipped_comparison, indent=2))
    print(f"\nAudit artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
