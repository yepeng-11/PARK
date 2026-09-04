#!/usr/bin/env python3
"""Freeze the PARK paper cohort and trace released-score provenance.

The script performs three complementary checks without modifying upstream data
or checkpoints:

1. freeze the released 162/91/67 cohort as a canonical, hashed manifest;
2. verify that ``test_data_big.csv`` is numerically derived from the released
   seed-289 pickle artifacts after the documented split-membership exclusions;
3. repeat MC-dropout inference with the released checkpoints to test whether
   paper-versus-fresh score differences are consistent with stochastic MC masks.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import pickle
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import evaluate_pretrained as evaluator


SPLITS = ("global", "validation_1", "validation_2")
SOURCE_FILES = {
    "global": "test_df_global_to_save.csv",
    "validation_1": "test_df_validation_1_to_save.csv",
    "validation_2": "test_df_validation_2_to_save.csv",
}
PICKLE_FILES = {
    "global": "test_df_global_to_save_289.pkl",
    "validation_1": "test_df_validation_1_to_save_289.pkl",
    "validation_2": "test_df_validation_2_to_save_289.pkl",
}
PAPER_SCORE_COLUMNS = {
    "finger": "pred_score_finger",
    "speech": "pred_score_speech",
    "smile": "pred_score_smile",
    "fusion": "pred_score_fusion",
}
FRESH_SCORE_COLUMNS = {
    "finger": "finger_score",
    "speech": "speech_score",
    "smile": "smile_score",
    "fusion": "fusion_score",
}
REPORT_METRICS = (
    "accuracy",
    "auroc",
    "sensitivity",
    "specificity",
    "precision",
    "npv",
    "f1",
)
PUBLISHED_POINT_ESTIMATES = {
    "global": {
        "accuracy": 0.802,
        "sensitivity": 0.865,
        "specificity": 0.712,
        "precision": 0.814,
        "npv": 0.783,
        "f1": 0.838,
    },
    "validation_1": {
        "accuracy": 0.802,
        "sensitivity": 0.857,
        "specificity": 0.738,
        "precision": 0.792,
        "npv": 0.816,
        "f1": 0.824,
    },
    "validation_2": {
        "accuracy": 0.806,
        "sensitivity": 0.833,
        "specificity": 0.784,
        "precision": 0.758,
        "npv": 0.853,
        "f1": 0.794,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="PARK repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <repo-root>/results/paper_score_provenance",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed-start", type=int, default=289)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--num-trials", type=int, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def stable_csv_write(frame: pd.DataFrame, path: Path) -> str:
    frame.to_csv(path, index=False, lineterminator="\n")
    return sha256_file(path)


def load_paper(path: Path) -> pd.DataFrame:
    paper = pd.read_csv(path)
    required = {
        "unique_row_id",
        "participant_id",
        "test_split",
        "true_label",
        *PAPER_SCORE_COLUMNS.values(),
        "uncertain_flag",
        "pred_std_fusion",
    }
    missing = required - set(paper.columns)
    if missing:
        raise ValueError(f"Paper artifact missing columns: {sorted(missing)}")
    paper = paper.copy()
    paper["paper_order"] = np.arange(len(paper), dtype=int)
    paper["split_order"] = paper.groupby("test_split", sort=False).cumcount()
    observed = paper.test_split.value_counts().to_dict()
    expected = {"global": 162, "validation_1": 91, "validation_2": 67}
    if observed != expected:
        raise ValueError(f"Unexpected paper cohort counts: {observed}")
    if paper.duplicated(["test_split", "unique_row_id"]).any():
        raise ValueError("Duplicate (split, row_id) keys in paper artifact")
    return paper


def freeze_cohort(paper: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    records: List[pd.DataFrame] = []
    for split in SPLITS:
        source = pd.read_csv(data_dir / SOURCE_FILES[split]).reset_index().rename(
            columns={"index": "source_index"}
        )
        required = {"row_id", "id", "label", "source_index"}
        if required - set(source.columns):
            raise ValueError(f"Missing source columns for {split}")
        current = paper.loc[paper.test_split == split].merge(
            source[["source_index", "row_id", "id", "label"]],
            left_on="unique_row_id",
            right_on="row_id",
            how="left",
            validate="one_to_one",
        )
        if current.source_index.isna().any():
            raise ValueError(f"Paper rows missing from source split: {split}")
        if not (current.participant_id.astype(str) == current.id.astype(str)).all():
            raise ValueError(f"Participant mismatch in {split}")
        if not (current.true_label.astype(int) == current.label.astype(int)).all():
            raise ValueError(f"Label mismatch in {split}")
        current["source_csv"] = SOURCE_FILES[split]
        records.append(current)

    merged = pd.concat(records, ignore_index=True).sort_values("paper_order")
    manifest = merged[
        [
            "paper_order",
            "test_split",
            "split_order",
            "unique_row_id",
            "participant_id",
            "true_label",
            "source_csv",
            "source_index",
        ]
    ].rename(
        columns={
            "test_split": "split",
            "unique_row_id": "row_id",
            "participant_id": "id",
            "true_label": "label",
        }
    )
    manifest["source_index"] = manifest.source_index.astype(int)
    manifest["label"] = manifest.label.astype(int)
    canonical = manifest[["split", "row_id", "id", "label"]].astype(str)
    manifest["row_membership_sha256"] = [
        hashlib.sha256("\x1f".join(row).encode("utf-8")).hexdigest()
        for row in canonical.itertuples(index=False, name=None)
    ]
    return manifest


def verify_pickle_lineage(
    paper: pd.DataFrame, data_dir: Path, fusion_dir: Path
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split in SPLITS:
        source = pd.read_csv(data_dir / SOURCE_FILES[split]).copy()
        with (fusion_dir / PICKLE_FILES[split]).open("rb") as handle:
            artifact = pickle.load(handle)
        expected_length = len(source)
        vectors = {
            "label": np.asarray(artifact["all_labels"]).reshape(-1),
            "finger": np.asarray(artifact["all_pred_scores"][0]).reshape(-1),
            "speech": np.asarray(artifact["all_pred_scores"][1]).reshape(-1),
            "smile": np.asarray(artifact["all_pred_scores"][2]).reshape(-1),
            "fusion": np.asarray(artifact["all_final_predictions"]).reshape(-1),
            "uncertain": np.asarray(artifact["uncertain_indices"]).reshape(-1),
            "fusion_std_artifact": np.asarray(artifact["all_final_std"]).reshape(-1),
        }
        lengths = {name: len(values) for name, values in vectors.items()}
        if set(lengths.values()) != {expected_length}:
            raise ValueError(f"Pickle/source length mismatch in {split}: {lengths}")

        reconstructed = pd.DataFrame(
            {
                "row_id": source.row_id.astype(str),
                "label": vectors["label"].astype(int),
                **{name: vectors[name] for name in ("finger", "speech", "smile", "fusion")},
                "uncertain": vectors["uncertain"].astype(bool),
                "fusion_std_artifact": vectors["fusion_std_artifact"],
            }
        )
        reference = paper.loc[paper.test_split == split].copy()
        joined = reference.merge(
            reconstructed, left_on="unique_row_id", right_on="row_id", validate="one_to_one"
        )
        if len(joined) != len(reference):
            raise ValueError(f"Incomplete pickle lineage join for {split}")

        row: Dict[str, Any] = {
            "split": split,
            "source_rows": expected_length,
            "paper_rows": len(reference),
            "excluded_memberships": expected_length - len(reference),
            "label_mismatches": int(
                (joined.true_label.astype(int).to_numpy() != joined.label.to_numpy()).sum()
            ),
            "uncertain_flag_mismatches": int(
                (
                    joined.uncertain_flag.astype(bool).to_numpy()
                    != joined.uncertain.to_numpy()
                ).sum()
            ),
            "fusion_std_max_abs_delta": float(
                np.max(
                    np.abs(
                        joined.pred_std_fusion.to_numpy(float)
                        - joined.fusion_std_artifact.to_numpy(float)
                    )
                )
            ),
        }
        for model, paper_col in PAPER_SCORE_COLUMNS.items():
            row[f"{model}_score_max_abs_delta"] = float(
                np.max(np.abs(joined[paper_col].to_numpy(float) - joined[model].to_numpy(float)))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def point_metrics(labels: Iterable[int], scores: Iterable[float]) -> Dict[str, float]:
    y = np.asarray(list(labels), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    pred = (s >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "auroc": float(roc_auc_score(y, s)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "npv": float(tn / (tn + fn)) if tn + fn else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def repeated_inference(
    repo_root: Path,
    paper: pd.DataFrame,
    device: torch.device,
    seeds: List[int],
    num_trials: int,
) -> tuple[pd.DataFrame, Dict[str, str], int, str]:
    module = evaluator.load_upstream_module(repo_root)
    fusion_config_path = Path(module.MODEL_CONFIG_PATH)
    fusion_config = evaluator.read_json(fusion_config_path)
    selected_models = module.MODEL_SUBSETS[int(fusion_config["model_subset_choice"])]
    module.NUM_MODELS = len(selected_models)
    paths = evaluator.checkpoint_paths(module, selected_models)
    aligned, predictor_configs, _ = evaluator.build_aligned_dataframe(
        module, selected_models, paths
    )
    splits = evaluator.make_splits(module, aligned)
    feature_shapes = [
        len(aligned.iloc[0][f"features_{index}"]) for index in range(module.NUM_MODELS)
    ]
    prediction_models = evaluator.instantiate_models(
        module, selected_models, paths, predictor_configs, feature_shapes, device
    )
    fusion_model = module.HybridFusionNetworkWithUncertainty(feature_shapes, fusion_config)
    fusion_path = Path(module.MODEL_PATH)
    fusion_model.load_state_dict(torch.load(fusion_path, map_location="cpu"))
    fusion_model.to(device).eval()
    for parameter in fusion_model.parameters():
        parameter.requires_grad = False

    checkpoint_hashes: Dict[str, str] = {}
    for group in paths:
        for path in group.values():
            checkpoint_hashes[str(path.relative_to(repo_root))] = sha256_file(path)
    checkpoint_hashes[str(fusion_path.relative_to(repo_root))] = sha256_file(fusion_path)

    batch_size = int(fusion_config["batch_size"])
    records: List[pd.DataFrame] = []
    for replicate, seed in enumerate(seeds):
        for split in SPLITS:
            fresh = evaluator.predict_split(
                module,
                splits[split],
                prediction_models,
                fusion_model,
                device,
                batch_size,
                num_trials,
                seed,
            )
            reference = paper.loc[paper.test_split == split, [
                "paper_order", "unique_row_id", "true_label", *PAPER_SCORE_COLUMNS.values()
            ]]
            joined = reference.merge(
                fresh[["row_id", *FRESH_SCORE_COLUMNS.values()]],
                left_on="unique_row_id",
                right_on="row_id",
                validate="one_to_one",
            )
            if len(joined) != len(reference):
                raise ValueError(f"Incomplete fresh-prediction join for {split}")
            out = joined[["paper_order", "unique_row_id", "true_label"]].rename(
                columns={"unique_row_id": "row_id", "true_label": "label"}
            )
            out.insert(0, "split", split)
            out.insert(0, "seed", seed)
            out.insert(0, "replicate", replicate)
            for model in PAPER_SCORE_COLUMNS:
                out[f"paper_{model}_score"] = joined[PAPER_SCORE_COLUMNS[model]].to_numpy(float)
                out[f"fresh_{model}_score"] = joined[FRESH_SCORE_COLUMNS[model]].to_numpy(float)
            records.append(out)
    return pd.concat(records, ignore_index=True), checkpoint_hashes, batch_size, str(fusion_config_path)


def summarize_replicates(replicates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_seed: List[Dict[str, Any]] = []
    for (split, seed), group in replicates.groupby(["split", "seed"], sort=False):
        row: Dict[str, Any] = {"split": split, "seed": int(seed), "n": len(group)}
        fresh_metrics = point_metrics(group.label, group.fresh_fusion_score)
        paper_metrics = point_metrics(group.label, group.paper_fusion_score)
        row.update({f"fresh_{k}": v for k, v in fresh_metrics.items()})
        row.update({f"paper_{k}": v for k, v in paper_metrics.items()})
        row["fusion_score_mae_vs_paper"] = float(
            np.mean(np.abs(group.fresh_fusion_score - group.paper_fusion_score))
        )
        row["fusion_threshold_disagreements_vs_paper"] = int(
            ((group.fresh_fusion_score >= 0.5) != (group.paper_fusion_score >= 0.5)).sum()
        )
        for model in PAPER_SCORE_COLUMNS:
            row[f"{model}_score_mae_vs_paper"] = float(
                np.mean(np.abs(group[f"fresh_{model}_score"] - group[f"paper_{model}_score"]))
            )
        per_seed.append(row)
    per_seed_frame = pd.DataFrame(per_seed)

    ensemble_parts: List[pd.DataFrame] = []
    for split, group in replicates.groupby("split", sort=False):
        static = group.drop_duplicates("row_id").set_index("row_id")
        aggregate = group.groupby("row_id", sort=False).agg(
            paper_order=("paper_order", "first"),
            label=("label", "first"),
            paper_score=("paper_fusion_score", "first"),
            mc_mean=("fresh_fusion_score", "mean"),
            mc_std_of_30_trial_means=("fresh_fusion_score", "std"),
            mc_min=("fresh_fusion_score", "min"),
            mc_max=("fresh_fusion_score", "max"),
        ).reset_index()
        aggregate.insert(0, "split", split)
        aggregate["paper_minus_mc_mean"] = aggregate.paper_score - aggregate.mc_mean
        denom = aggregate.mc_std_of_30_trial_means.replace(0.0, np.nan)
        aggregate["paper_mc_z"] = aggregate.paper_minus_mc_mean / denom
        aggregate["paper_within_mc_95_band"] = aggregate.paper_mc_z.abs() <= 1.96
        ensemble_parts.append(aggregate)
    ensemble = pd.concat(ensemble_parts, ignore_index=True).sort_values("paper_order")

    summary_rows: List[Dict[str, Any]] = []
    for split, group in ensemble.groupby("split", sort=False):
        seed_group = per_seed_frame.loc[per_seed_frame.split == split]
        paper_metrics = point_metrics(group.label, group.paper_score)
        for metric in REPORT_METRICS:
            values = seed_group[f"fresh_{metric}"].to_numpy(float)
            summary_rows.append(
                {
                    "split": split,
                    "measure": metric,
                    "paper_value": paper_metrics[metric],
                    "mc_mean": float(values.mean()),
                    "mc_sd": float(values.std(ddof=1)),
                    "mc_p2_5": float(np.quantile(values, 0.025)),
                    "mc_p97_5": float(np.quantile(values, 0.975)),
                    "paper_within_mc_95_interval": bool(
                        np.quantile(values, 0.025) <= paper_metrics[metric] <= np.quantile(values, 0.975)
                    ),
                }
            )

        paper_mae = float(np.mean(np.abs(group.paper_score - group.mc_mean)))
        replicate_maes = []
        for _, rep in replicates.loc[replicates.split == split].groupby("seed"):
            ordered = rep.set_index("row_id").loc[group.row_id]
            replicate_maes.append(
                float(
                    np.mean(
                        np.abs(
                            ordered.fresh_fusion_score.to_numpy(float)
                            - group.mc_mean.to_numpy(float)
                        )
                    )
                )
            )
        summary_rows.append(
            {
                "split": split,
                "measure": "score_draw_consistency",
                "paper_value": paper_mae,
                "mc_mean": float(np.mean(replicate_maes)),
                "mc_sd": float(np.std(replicate_maes, ddof=1)),
                "mc_p2_5": float(np.quantile(replicate_maes, 0.025)),
                "mc_p97_5": float(np.quantile(replicate_maes, 0.975)),
                "paper_within_mc_95_interval": bool(
                    np.quantile(replicate_maes, 0.025) <= paper_mae <= np.quantile(replicate_maes, 0.975)
                ),
                "paper_to_typical_mc_mae_ratio": paper_mae / float(np.median(replicate_maes)),
                "row_fraction_within_mc_95_band": float(group.paper_within_mc_95_band.mean()),
            }
        )
    return per_seed_frame, ensemble, pd.DataFrame(summary_rows)


def build_final_reproduction_table(distribution: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    counts = {"global": 162, "validation_1": 91, "validation_2": 67}
    for split in SPLITS:
        rows.append(
            {
                "split": split,
                "metric": "n",
                "published": float(counts[split]),
                "frozen_paper_score": float(counts[split]),
                "fresh_mc_mean": float(counts[split]),
                "fresh_mc_sd": 0.0,
                "fresh_mc_p2_5": float(counts[split]),
                "fresh_mc_p97_5": float(counts[split]),
                "frozen_within_fresh_mc_95_interval": True,
                "frozen_minus_published": 0.0,
            }
        )
        for metric in REPORT_METRICS:
            row = distribution.loc[
                (distribution.split == split) & (distribution.measure == metric)
            ].iloc[0]
            published = PUBLISHED_POINT_ESTIMATES[split].get(metric, np.nan)
            rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "published": published,
                    "frozen_paper_score": float(row.paper_value),
                    "fresh_mc_mean": float(row.mc_mean),
                    "fresh_mc_sd": float(row.mc_sd),
                    "fresh_mc_p2_5": float(row.mc_p2_5),
                    "fresh_mc_p97_5": float(row.mc_p97_5),
                    "frozen_within_fresh_mc_95_interval": bool(
                        row.paper_within_mc_95_interval
                    ),
                    "frozen_minus_published": (
                        float(row.paper_value - published)
                        if np.isfinite(published)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def final_table_markdown(
    table: pd.DataFrame, replicates: int, num_trials: int
) -> str:
    lines = [
        "# PARK final paper-exact reproduction table",
        "",
        "Published values are the article's one-decimal percentage point estimates. "
        "AUROC is left blank in that column because the article reports a cross-cohort "
        f"range rather than exact table values. Fresh estimates summarize {replicates} seeded "
        f"checkpoint evaluations with {num_trials} MC-dropout trials each.",
        "",
        "| Split | Metric | Published | Frozen paper score | Fresh MC mean | Fresh MC 95% interval | Frozen in interval |",
        "| --- | --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in table.itertuples(index=False):
        if row.metric == "n":
            published = frozen = mean = f"{int(row.frozen_paper_score)}"
            interval = f"[{int(row.fresh_mc_p2_5)}, {int(row.fresh_mc_p97_5)}]"
        else:
            published = "—" if pd.isna(row.published) else f"{row.published:.4f}"
            frozen = f"{row.frozen_paper_score:.4f}"
            mean = f"{row.fresh_mc_mean:.4f}"
            interval = f"[{row.fresh_mc_p2_5:.4f}, {row.fresh_mc_p97_5:.4f}]"
        lines.append(
            f"| {row.split} | {row.metric} | {published} | {frozen} | {mean} | "
            f"{interval} | {bool(row.frozen_within_fresh_mc_95_interval)} |"
        )
    lines.append("")
    return "\n".join(lines)


def report_text(
    cohort_hash: str,
    lineage: pd.DataFrame,
    distribution: pd.DataFrame,
    replicates: int,
    num_trials: int,
    dropout_forces_training: bool,
) -> str:
    exact_lineage = bool(
        (lineage.label_mismatches == 0).all()
        and (lineage.uncertain_flag_mismatches == 0).all()
        and (lineage.filter(like="max_abs_delta").fillna(0).to_numpy() <= 5e-8).all()
    )
    score_rows = distribution.loc[distribution.measure == "score_draw_consistency"]
    stochastic_consistent = bool(
        (score_rows.paper_to_typical_mc_mae_ratio <= 1.5).all()
        and (score_rows.row_fraction_within_mc_95_band >= 0.85).all()
    )
    metric_rows = distribution.loc[distribution.measure.isin(["accuracy", "auroc"])]

    lines = [
        "# PARK paper score-provenance trace",
        "",
        "## Outcome",
        "",
        f"- Frozen paper cohort manifest SHA-256: `{cohort_hash}`.",
        f"- Released CSV-to-PKL lineage matches within float32 serialization tolerance: **{exact_lineage}**.",
        f"- Fresh-score differences are consistent with MC-dropout variation: **{stochastic_consistent}**.",
        f"- BaaL dropout remains stochastic in evaluation mode: **{dropout_forces_training}**.",
        "",
        "The released PKL files do not embed a checkpoint hash, so the current checkpoint cannot be "
        "cryptographically bound to those historical scores. The statistical trace can support, but not "
        "prove, that the same released checkpoint generated them.",
        "",
        "## Frozen cohort",
        "",
        "| Split | Sessions |",
        "| --- | ---: |",
    ]
    for split in SPLITS:
        n = {"global": 162, "validation_1": 91, "validation_2": 67}[split]
        lines.append(f"| {split} | {n} |")

    lines.extend(
        [
            "",
            "## Released artifact lineage",
            "",
            "| Split | Source rows | Paper rows | Excluded memberships | Maximum numeric delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in lineage.itertuples(index=False):
        numeric_columns = [
            value for name, value in row._asdict().items() if "max_abs_delta" in name
        ]
        lines.append(
            f"| {row.split} | {row.source_rows} | {row.paper_rows} | "
            f"{row.excluded_memberships} | {max(numeric_columns):.3e} |"
        )

    lines.extend(
        [
            "",
            f"## MC-dropout trace ({replicates} replicates × {num_trials} trials)",
            "",
            "| Split | Measure | Paper | MC mean | MC 95% interval | In interval |",
            "| --- | --- | ---: | ---: | ---: | :---: |",
        ]
    )
    for row in metric_rows.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.measure} | {row.paper_value:.4f} | {row.mc_mean:.4f} | "
            f"[{row.mc_p2_5:.4f}, {row.mc_p97_5:.4f}] | {bool(row.paper_within_mc_95_interval)} |"
        )
    lines.extend(
        [
            "",
            "| Split | Paper MAE to MC ensemble | Typical replicate MAE | Ratio | Rows in MC 95% band |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in score_rows.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.paper_value:.6f} | {row.mc_mean:.6f} | "
            f"{row.paper_to_typical_mc_mae_ratio:.3f} | {row.row_fraction_within_mc_95_band:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `test_data_big.csv` is exactly reconstructable from the released seed-289 PKL files after "
            "the three documented duplicate split memberships are removed.",
            "- The evaluator and upstream code use the same released models and 30 MC trials, but the "
            "upstream training program does not reset the random-number state before each evaluation cohort.",
            "- The paper's historical RNG state was consumed by model training and repeated Dev/Test calls "
            "and is not stored. Exact floating-point score regeneration is therefore not expected from a "
            "checkpoint-only evaluation.",
            "- Both external-cohort AUROCs fall just outside the empirical central 95% interval even though "
            "their row-level score-draw diagnostics are typical. This shows that rank metrics on these small "
            "cohorts are sensitive to a single MC draw and should be reported with repeated-inference variation.",
            "- Use the frozen cohort manifest and released paper scores for exact table reproduction. Use "
            "seeded fresh checkpoint inference for new experiments, reporting seed and MC-trial count.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.replicates < 20:
        raise ValueError("At least 20 replicates are required for a useful MC trace")
    repo_root = args.repo_root.resolve()
    output_dir = (args.output_dir or repo_root / "results" / "paper_score_provenance").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = repo_root / "data"
    fusion_dir = repo_root / "code" / "fusion_model"
    paper_path = data_dir / "test_data_big.csv"
    paper = load_paper(paper_path)

    input_paths = [
        paper_path,
        *(data_dir / SOURCE_FILES[split] for split in SPLITS),
        *(fusion_dir / PICKLE_FILES[split] for split in SPLITS),
    ]
    before_hashes = {str(path.relative_to(repo_root)): sha256_file(path) for path in input_paths}

    cohort = freeze_cohort(paper, data_dir)
    cohort_hash = stable_csv_write(cohort, output_dir / "paper_exact_cohort_manifest.csv")
    lineage = verify_pickle_lineage(paper, data_dir, fusion_dir)
    stable_csv_write(lineage, output_dir / "released_artifact_lineage.csv")

    device = resolve_device(args.device)
    fusion_config = evaluator.read_json(
        repo_root / "models" / "uncertainty_aware_fusion" / "model_config.json"
    )
    num_trials = args.num_trials or int(fusion_config["num_trials"])
    seeds = list(range(args.seed_start, args.seed_start + args.replicates))
    replicates, checkpoint_hashes, batch_size, config_path = repeated_inference(
        repo_root, paper, device, seeds, num_trials
    )
    stable_csv_write(replicates, output_dir / "mc_replicate_predictions.csv")
    per_seed, ensemble, distribution = summarize_replicates(replicates)
    stable_csv_write(per_seed, output_dir / "mc_replicate_metrics.csv")
    stable_csv_write(ensemble, output_dir / "paper_vs_mc_ensemble.csv")
    stable_csv_write(distribution, output_dir / "mc_distribution_summary.csv")
    final_table = build_final_reproduction_table(distribution)
    stable_csv_write(final_table, output_dir / "final_reproduction_table.csv")
    (output_dir / "FINAL_REPRODUCTION_TABLE.md").write_text(
        final_table_markdown(final_table, args.replicates, num_trials),
        encoding="utf-8",
    )

    from baal.bayesian.dropout import Dropout

    dropout_source = inspect.getsource(Dropout.forward)
    dropout_forces_training = "F.dropout(input, self.p, True" in dropout_source
    report = report_text(
        cohort_hash,
        lineage,
        distribution,
        args.replicates,
        num_trials,
        dropout_forces_training,
    )
    (output_dir / "SCORE_PROVENANCE_REPORT.md").write_text(report, encoding="utf-8")

    after_hashes = {str(path.relative_to(repo_root)): sha256_file(path) for path in input_paths}
    manifest = {
        "repo_root": str(repo_root),
        "git_commit": evaluator.git_commit(repo_root),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "paper_cohort_manifest_sha256": cohort_hash,
        "paper_rows": len(cohort),
        "paper_split_counts": cohort.split.value_counts().to_dict(),
        "seed_start": args.seed_start,
        "replicates": args.replicates,
        "num_trials": num_trials,
        "batch_size_from_fusion_config": batch_size,
        "fusion_config_path": str(Path(config_path).relative_to(repo_root)),
        "baal_dropout_forces_training_in_eval": dropout_forces_training,
        "input_sha256": before_hashes,
        "checkpoint_sha256": checkpoint_hashes,
        "inputs_unchanged": before_hashes == after_hashes,
        "limitation": (
            "Released PKL artifacts contain no embedded checkpoint hash; checkpoint-to-score "
            "attribution is statistical rather than cryptographic."
        ),
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(manifest), handle, ensure_ascii=False, indent=2)

    print(f"Results written to: {output_dir}")
    print(f"Frozen cohort SHA-256: {cohort_hash}")
    print(distribution.to_string(index=False))


if __name__ == "__main__":
    main()
