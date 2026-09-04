#!/usr/bin/env python3
"""Train Fusion Agent v2 speech-noise and smile-conflict specialists.

Training follows the frozen participant-level 5x4 nested-CV protocol. Synthetic
corruption labels are the only detector targets. Existing PARK disease labels
are passed through the inference loader but are never used to fit, select, or
score either specialist. No released test partition is loaded for prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

import evaluate_calibrated_fusion as calibrated
import evaluate_learned_quality_gate as learned
import evaluate_pretrained as ev
import train_paired_baselines as paired


EXPECTED_PROTOCOL_HASH = "c07666342e8271ec53cf2d0d6ded0463701eba1c03d7ed86d3e296a535bc380c"
PAIRED_SEEDS = (101, 202, 303, 404, 505)
SPEECH_INDEX = 1
SMILE_INDEX = 2
DESCRIPTORS = tuple(learned.DESCRIPTORS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--protocol-dir", type=Path, default=None)
    parser.add_argument("--training-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--paired-seeds", default=",".join(map(str, PAIRED_SEEDS)))
    parser.add_argument(
        "--outer-folds", default="", help="Optional comma-separated subset for smoke tests"
    )
    parser.add_argument("--mc-trials", type=int, default=30)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def canonical_hash(value: Dict[str, Any]) -> str:
    payload = json.dumps(
        ev.json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_and_validate_protocol(protocol_dir: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    protocol = json.loads((protocol_dir / "protocol.json").read_text(encoding="utf-8"))
    recorded = protocol.pop("protocol_sha256")
    observed = canonical_hash(protocol)
    protocol["protocol_sha256"] = recorded
    if observed != recorded or recorded != EXPECTED_PROTOCOL_HASH:
        raise ValueError(
            f"Frozen protocol hash mismatch: recorded={recorded}, observed={observed}, "
            f"expected={EXPECTED_PROTOCOL_HASH}"
        )
    manifest_path = protocol_dir / "participant_fold_manifest.csv"
    if ev.sha256_file(manifest_path) != protocol["artifact_sha256"][manifest_path.name]:
        raise ValueError("Participant fold manifest hash mismatch")
    manifest = pd.read_csv(manifest_path, dtype={"participant_id": str})
    return protocol, manifest


def stable_rng(*parts: Any) -> np.random.Generator:
    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))


def speech_scenarios() -> List[Dict[str, Any]]:
    rows = [{"scenario": "clean", "kind": "clean", "severity": 0.0}]
    rows.extend(
        {"scenario": f"speech_gaussian_{value:.2f}", "kind": "gaussian", "severity": value}
        for value in (0.25, 0.5, 1.0)
    )
    rows.extend(
        {"scenario": f"speech_mask_{value:.2f}", "kind": "mask", "severity": value}
        for value in (0.10, 0.25, 0.5)
    )
    return rows


def perturb_speech(
    frame: pd.DataFrame, scenario: Dict[str, Any], outer_fold: int
) -> pd.DataFrame:
    output = frame.copy()
    if scenario["kind"] == "clean":
        return output
    rng = stable_rng("fusion-agent-v2", outer_fold, scenario["scenario"])
    severity = float(scenario["severity"])
    if scenario["kind"] == "gaussian":
        output["features_1"] = [
            (value + rng.normal(0.0, severity, size=value.shape)).astype(np.float32)
            for value in output.features_1
        ]
    elif scenario["kind"] == "mask":
        output["features_1"] = [
            np.where(rng.random(value.shape) < severity, 0.0, value).astype(np.float32)
            for value in output.features_1
        ]
    else:
        raise ValueError(scenario)
    return output


def predict_speech_scenarios(
    module,
    frame: pd.DataFrame,
    predictors,
    fusion_model,
    device: torch.device,
    mc_trials: int,
    model_seed: int,
    outer_fold: int,
) -> Dict[str, pd.DataFrame]:
    outputs: Dict[str, pd.DataFrame] = {}
    for index, scenario in enumerate(speech_scenarios()):
        perturbed = perturb_speech(frame, scenario, outer_fold)
        outputs[scenario["scenario"]] = learned.predict_quality_inputs(
            module,
            perturbed,
            predictors,
            fusion_model,
            device,
            mc_trials,
            model_seed * 100000 + outer_fold * 1000 + index,
            [],
            [],
        )
    return outputs


def speech_dataset(
    predictions: Dict[str, pd.DataFrame], participant_ids: set
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    scenarios: List[np.ndarray] = []
    for scenario, frame in predictions.items():
        selected = frame.loc[frame.id.astype(str).isin(participant_ids)].reset_index(drop=True)
        matrix = learned.quality_feature_matrix(selected, SPEECH_INDEX)
        target = np.zeros(len(selected), dtype=int) if scenario == "clean" else np.ones(len(selected), dtype=int)
        matrices.append(matrix)
        targets.append(target)
        scenarios.append(np.repeat(scenario, len(selected)))
    return np.concatenate(matrices), np.concatenate(targets), np.concatenate(scenarios)


def smile_feature_matrix(frame: pd.DataFrame, smile_score: np.ndarray) -> np.ndarray:
    speech = frame.speech.to_numpy(dtype=float)
    finger = frame.finger.to_numpy(dtype=float)
    peers = np.column_stack([finger, speech])
    peer_mean = peers.mean(axis=1)
    columns = [
        np.asarray(smile_score, dtype=float),
        frame.smile_mc_std.to_numpy(dtype=float),
        peer_mean,
        peers.std(axis=1),
        np.abs(smile_score - peer_mean),
        np.abs(smile_score - 0.5),
        np.abs(finger - speech),
    ]
    columns.extend(
        frame[f"smile_feature_{descriptor}"].to_numpy(dtype=float)
        for descriptor in DESCRIPTORS
    )
    return np.nan_to_num(np.column_stack(columns), nan=0.0, posinf=1e6, neginf=-1e6)


def smile_dataset(
    clean_prediction: pd.DataFrame, participant_ids: set, seed_parts: Sequence[Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = clean_prediction.loc[
        clean_prediction.id.astype(str).isin(participant_ids)
    ].reset_index(drop=True)
    clean = selected.smile.to_numpy(dtype=float)
    rng = stable_rng("smile-conflict", *seed_parts)
    permuted = clean.copy()
    rng.shuffle(permuted)
    variants = {
        "clean": clean,
        "smile_invert": 1.0 - clean,
        "smile_opposite_extreme": np.where(clean >= 0.5, 0.02, 0.98),
        "smile_permute": permuted,
    }
    matrices: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    scenarios: List[np.ndarray] = []
    for scenario, score in variants.items():
        matrices.append(smile_feature_matrix(selected, score))
        targets.append(
            np.zeros(len(selected), dtype=int)
            if scenario == "clean"
            else np.ones(len(selected), dtype=int)
        )
        scenarios.append(np.repeat(scenario, len(selected)))
    return np.concatenate(matrices), np.concatenate(targets), np.concatenate(scenarios)


def parameter_grid() -> List[Dict[str, Any]]:
    return [
        {
            "learning_rate": learning_rate,
            "max_leaf_nodes": leaves,
            "l2_regularization": l2,
        }
        for learning_rate, leaves, l2 in itertools.product(
            (0.03, 0.05), (7, 15), (0.5, 2.0)
        )
    ]


def balanced_weights(target: np.ndarray) -> np.ndarray:
    positives = max(1, int(target.sum()))
    negatives = max(1, int((target == 0).sum()))
    return np.where(
        target == 1,
        len(target) / (2.0 * positives),
        len(target) / (2.0 * negatives),
    )


def fit_detector(
    matrix: np.ndarray, target: np.ndarray, params: Dict[str, Any], seed: int
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=params["learning_rate"],
        max_iter=150,
        max_leaf_nodes=params["max_leaf_nodes"],
        min_samples_leaf=max(10, len(target) // 100),
        l2_regularization=params["l2_regularization"],
        random_state=seed,
    ).fit(matrix, target, sample_weight=balanced_weights(target))


def detector_metrics(
    target: np.ndarray, score: np.ndarray, scenarios: np.ndarray
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.append(
        {
            "scenario": "all",
            "n": len(target),
            "corrupted": int(target.sum()),
            "auroc": float(roc_auc_score(target, score)),
            "accuracy": float(accuracy_score(target, score >= 0.5)),
            "detection_rate": float((score[target == 1] >= 0.5).mean()),
            "false_positive_rate": float((score[target == 0] >= 0.5).mean()),
            "mean_score": float(score.mean()),
        }
    )
    for scenario in sorted(set(scenarios)):
        mask = scenarios == scenario
        scenario_target = target[mask]
        rows.append(
            {
                "scenario": scenario,
                "n": int(mask.sum()),
                "corrupted": int(scenario_target.sum()),
                "auroc": float("nan"),
                "accuracy": float(accuracy_score(scenario_target, score[mask] >= 0.5)),
                "detection_rate": (
                    float((score[mask] >= 0.5).mean())
                    if scenario != "clean"
                    else float("nan")
                ),
                "false_positive_rate": (
                    float((score[mask] >= 0.5).mean())
                    if scenario == "clean"
                    else float("nan")
                ),
                "mean_score": float(score[mask].mean()),
            }
        )
    return rows


def tune_detector(
    specialist: str,
    outer_rows: pd.DataFrame,
    dataset_builder,
    outer_fold: int,
    base_seed: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for candidate_index, params in enumerate(parameter_grid()):
        for inner_fold in sorted(outer_rows.inner_validation_fold.unique()):
            train_ids = set(
                outer_rows.loc[
                    outer_rows.inner_validation_fold != inner_fold, "participant_id"
                ].astype(str)
            )
            validation_ids = set(
                outer_rows.loc[
                    outer_rows.inner_validation_fold == inner_fold, "participant_id"
                ].astype(str)
            )
            x_train, y_train, _ = dataset_builder(
                train_ids, (outer_fold, inner_fold, "train")
            )
            x_validation, y_validation, _ = dataset_builder(
                validation_ids, (outer_fold, inner_fold, "validation")
            )
            detector = fit_detector(
                x_train, y_train, params, base_seed + candidate_index * 100 + inner_fold
            )
            score = detector.predict_proba(x_validation)[:, 1]
            rows.append(
                {
                    "specialist": specialist,
                    "outer_fold": outer_fold,
                    "candidate": candidate_index,
                    "inner_fold": int(inner_fold),
                    **params,
                    "auroc": float(roc_auc_score(y_validation, score)),
                    "accuracy": float(accuracy_score(y_validation, score >= 0.5)),
                    "clean_false_positive_rate": float(
                        (score[y_validation == 0] >= 0.5).mean()
                    ),
                }
            )
    tuning = pd.DataFrame(rows)
    aggregate = (
        tuning.groupby(
            ["candidate", "learning_rate", "max_leaf_nodes", "l2_regularization"],
            as_index=False,
        )
        .agg(
            mean_auroc=("auroc", "mean"),
            min_auroc=("auroc", "min"),
            mean_accuracy=("accuracy", "mean"),
            mean_clean_fpr=("clean_false_positive_rate", "mean"),
        )
        .sort_values(
            ["mean_auroc", "min_auroc", "mean_clean_fpr"],
            ascending=[False, False, True],
        )
    )
    selected = aggregate.iloc[0]
    selected_candidate = int(selected.candidate)
    tuning["selected"] = tuning.candidate == selected_candidate
    params = {
        "learning_rate": float(selected.learning_rate),
        "max_leaf_nodes": int(selected.max_leaf_nodes),
        "l2_regularization": float(selected.l2_regularization),
    }
    return params, tuning


def load_outer_models(
    module,
    raw: pd.DataFrame,
    training_dir: Path,
    seed: int,
    configs,
    fusion_config,
    device: torch.device,
):
    run = training_dir / "cleaned" / f"seed_{seed}"
    if not (run / "run_complete.json").exists():
        raise FileNotFoundError(f"Incomplete paired model run: {run}")
    scaled = calibrated.apply_run_scalers(raw, run, configs)
    shapes = [len(scaled.iloc[0][f"features_{index}"]) for index in range(3)]
    predictors, fusion_model = calibrated.load_models(
        module, run, configs, fusion_config, shapes, device
    )
    return run, scaled, predictors, fusion_model


def write_report(path: Path, metrics: pd.DataFrame, tuning: pd.DataFrame) -> None:
    overall = metrics.loc[metrics.scenario == "all"]
    lines = [
        "# Fusion Agent v2 specialist training",
        "",
        f"Frozen protocol: `{EXPECTED_PROTOCOL_HASH}`.",
        "",
        "The speech-noise and smile-conflict specialists use synthetic corruption "
        "targets only. Hyperparameters are selected by four inner folds; the five "
        "outer folds below are held out from fitting and selection. No released Test "
        "partition is predicted or scored.",
        "",
        "## Outer-fold corruption detection",
        "",
        "| Specialist | Mean AUROC | SD | Mean accuracy | Clean FPR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for specialist, selected in overall.groupby("specialist"):
        lines.append(
            f"| {specialist} | {selected.auroc.mean():.4f} | "
            f"{selected.auroc.std(ddof=1):.4f} | {selected.accuracy.mean():.4f} | "
            f"{selected.false_positive_rate.mean():.4f} |"
        )
    lines.extend(
        [
            "",
            "## Per-corruption detection rate",
            "",
            "| Specialist | Scenario | Mean detection rate | Mean score |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for (specialist, scenario), selected in metrics.loc[
        ~metrics.scenario.isin(["all", "clean"])
    ].groupby(["specialist", "scenario"]):
        lines.append(
            f"| {specialist} | {scenario} | {selected.detection_rate.mean():.4f} | "
            f"{selected.mean_score.mean():.4f} |"
        )
    lines.extend(
        [
            "",
            "These are detector-level engineering results, not disease-classification "
            "results. The next stage must train and evaluate the v2 router inside the "
            "same outer folds against the frozen acceptance criteria.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.mc_trials < 2:
        raise ValueError("--mc-trials must be at least 2")
    repo_root = args.repo_root.resolve()
    data_path = (
        args.data
        or repo_root / "results" / "protocol_alignment_audit" / "cleaned_aligned.csv"
    ).resolve()
    protocol_dir = (
        args.protocol_dir or repo_root / "results" / "fusion_agent_v2_protocol"
    ).resolve()
    training_dir = (
        args.training_dir or repo_root / "results" / "paired_retraining"
    ).resolve()
    output = (
        args.output_dir or repo_root / "results" / "fusion_agent_v2_specialists"
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol, manifest = load_and_validate_protocol(protocol_dir)
    if ev.sha256_file(data_path) != protocol["source_dataset_sha256"]:
        raise ValueError("Development dataset does not match the frozen protocol hash")
    seeds = paired.parse_seeds(args.paired_seeds)
    protocol_outer_folds = sorted(manifest.outer_fold.unique())
    outer_folds = (
        paired.parse_seeds(args.outer_folds)
        if args.outer_folds.strip()
        else protocol_outer_folds
    )
    if not set(outer_folds).issubset(set(protocol_outer_folds)):
        raise ValueError(f"Unknown outer folds: {outer_folds}")
    if len(seeds) != len(outer_folds):
        raise ValueError("Exactly one predeclared paired seed is required per outer fold")

    module = ev.load_upstream_module(repo_root)
    masks = paired.split_masks(module, paired.load_vector_csv(data_path))
    frame = paired.load_vector_csv(data_path)
    allowed_ids = set(manifest.participant_id.astype(str))
    development = frame.loc[frame.id.astype(str).isin(allowed_ids)].reset_index(drop=True)
    if len(set(development.id.astype(str)) - allowed_ids) != 0:
        raise RuntimeError("Unexpected participant entered development data")
    locked_mask = masks["internal_test"] | masks["validation_1"] | masks["validation_2"] | masks["global"]
    locked_ids = set(frame.loc[locked_mask, "id"].astype(str))
    if allowed_ids & locked_ids:
        raise RuntimeError("Frozen development IDs overlap a locked test partition")

    fusion_config = ev.read_json(Path(module.MODEL_CONFIG_PATH))
    selected_models = module.MODEL_SUBSETS[int(fusion_config["model_subset_choice"])]
    module.NUM_MODELS = len(selected_models)
    upstream_paths = ev.checkpoint_paths(module, selected_models)
    configs = [ev.read_json(item["config"]) for item in upstream_paths]
    raw = paired.inverse_original_scaling(development, configs, upstream_paths)
    device = ev.resolve_device(args.device)
    protected_paths: List[Path] = [
        path
        for item in upstream_paths
        for path in item.values()
        if path.suffix in {".pth", ".pkl"}
    ]
    for seed in seeds:
        run = training_dir / "cleaned" / f"seed_{seed}"
        protected_paths.extend(run.glob("*/model.pth"))
        protected_paths.extend(run.glob("scaler_*.pkl"))
    protected_before = {
        str(path.resolve().relative_to(repo_root)): ev.sha256_file(path.resolve())
        for path in sorted(set(protected_paths))
    }

    for outer_fold, model_seed in zip(outer_folds, seeds):
        fold_output = output / f"outer_{outer_fold}"
        fold_output.mkdir(parents=True, exist_ok=True)
        complete = fold_output / "run_complete.json"
        if complete.exists() and not args.force:
            print(f"Skipping completed outer fold {outer_fold}")
            continue
        outer_rows = manifest.loc[
            (manifest.outer_fold == outer_fold) & (manifest.outer_role == "train")
        ].copy()
        outer_train_ids = set(outer_rows.participant_id.astype(str))
        outer_validation_ids = set(
            manifest.loc[
                (manifest.outer_fold == outer_fold)
                & (manifest.outer_role == "validation"),
                "participant_id",
            ].astype(str)
        )
        if outer_train_ids & outer_validation_ids:
            raise RuntimeError(f"Outer fold {outer_fold} participant leakage")
        run, scaled, predictors, fusion_model = load_outer_models(
            module, raw, training_dir, model_seed, configs, fusion_config, device
        )
        print(f"Outer fold {outer_fold}: paired seed {model_seed}; generating scenarios")
        speech_predictions = predict_speech_scenarios(
            module,
            scaled,
            predictors,
            fusion_model,
            device,
            args.mc_trials,
            model_seed,
            int(outer_fold),
        )
        clean_prediction = speech_predictions["clean"]

        def build_speech(ids: set, seed_parts: Sequence[Any]):
            return speech_dataset(speech_predictions, ids)

        def build_smile(ids: set, seed_parts: Sequence[Any]):
            return smile_dataset(clean_prediction, ids, seed_parts)

        fold_state: Dict[str, Any] = {
            "outer_fold": int(outer_fold),
            "paired_seed": model_seed,
            "outer_train_participants": len(outer_train_ids),
            "outer_validation_participants": len(outer_validation_ids),
            "specialists": {},
        }
        fold_metric_rows: List[Dict[str, Any]] = []
        fold_tuning_rows: List[pd.DataFrame] = []
        for specialist, builder in (
            ("speech_noise_detector", build_speech),
            ("smile_conflict_detector", build_smile),
        ):
            params, tuning = tune_detector(
                specialist,
                outer_rows,
                builder,
                int(outer_fold),
                model_seed * 10000,
            )
            fold_tuning_rows.append(tuning)
            x_train, y_train, _ = builder(
                outer_train_ids, (outer_fold, "outer_train")
            )
            x_validation, y_validation, validation_scenarios = builder(
                outer_validation_ids, (outer_fold, "outer_validation")
            )
            detector = fit_detector(
                x_train, y_train, params, model_seed * 10000 + int(outer_fold)
            )
            score = detector.predict_proba(x_validation)[:, 1]
            for row in detector_metrics(y_validation, score, validation_scenarios):
                fold_metric_rows.append(
                    {
                        "specialist": specialist,
                        "outer_fold": int(outer_fold),
                        "paired_seed": model_seed,
                        **row,
                    }
                )
            with (fold_output / f"{specialist}.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        "specialist": specialist,
                        "protocol_sha256": EXPECTED_PROTOCOL_HASH,
                        "outer_fold": int(outer_fold),
                        "paired_seed": model_seed,
                        "parameters": params,
                        "model": detector,
                    },
                    handle,
                )
            fold_state["specialists"][specialist] = {
                "parameters": params,
                "validation_auroc": float(roc_auc_score(y_validation, score)),
            }
        pd.DataFrame(fold_metric_rows).to_csv(
            fold_output / "outer_fold_metrics.csv", index=False
        )
        pd.concat(fold_tuning_rows, ignore_index=True).to_csv(
            fold_output / "inner_tuning_metrics.csv", index=False
        )
        with complete.open("w", encoding="utf-8") as handle:
            json.dump(ev.json_ready(fold_state), handle, indent=2)

    metric_files = [output / f"outer_{fold}" / "outer_fold_metrics.csv" for fold in outer_folds]
    tuning_files = [output / f"outer_{fold}" / "inner_tuning_metrics.csv" for fold in outer_folds]
    missing_outputs = [path for path in (*metric_files, *tuning_files) if not path.exists()]
    if missing_outputs:
        raise RuntimeError(f"Missing fold outputs: {missing_outputs}")
    metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    tuning = pd.concat([pd.read_csv(path) for path in tuning_files], ignore_index=True)
    metrics.to_csv(output / "outer_fold_metrics.csv", index=False)
    tuning.to_csv(output / "inner_tuning_metrics.csv", index=False)
    if len(metrics.loc[metrics.scenario == "all"]) != 2 * len(outer_folds):
        raise RuntimeError("Incomplete outer-fold specialist metrics")
    write_report(output / "FUSION_AGENT_V2_SPECIALIST_REPORT.md", metrics, tuning)
    protected_after = {
        str(path.resolve().relative_to(repo_root)): ev.sha256_file(path.resolve())
        for path in sorted(set(protected_paths))
    }
    if protected_before != protected_after:
        raise RuntimeError("A protected base model or scaler changed during training")
    run_manifest = {
        "stage": "fusion-agent-v2-specialists",
        "protocol_sha256": EXPECTED_PROTOCOL_HASH,
        "protocol_generator_sha256": protocol["generator_script_sha256"],
        "source_dataset_sha256": protocol["source_dataset_sha256"],
        "paired_seed_by_outer_fold": {
            str(fold): seed for fold, seed in zip(outer_folds, seeds)
        },
        "mc_trials": args.mc_trials,
        "device": str(device),
        "development_participants": len(allowed_ids),
        "locked_test_predictions_generated": False,
        "detector_target": "synthetic corruption only; disease labels unused",
        "protected_artifact_sha256": protected_after,
        "protected_artifacts_unchanged": True,
        "output_sha256": {
            name: ev.sha256_file(output / name)
            for name in (
                "outer_fold_metrics.csv",
                "inner_tuning_metrics.csv",
                "FUSION_AGENT_V2_SPECIALIST_REPORT.md",
            )
        },
    }
    with (output / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(ev.json_ready(run_manifest), handle, indent=2)
    print(f"Specialist training written to: {output}")


if __name__ == "__main__":
    main()
