"""One isolated development-fold training benchmark; no locked-cohort inference.

This is an engineering pilot, not a new frozen evaluation protocol.
"""
import argparse
import itertools
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import evaluate_pretrained as ev
import evaluate_learned_quality_gate as quality
import train_paired_baselines as paired
import train_fusion_agent_v3_router as v3


def audit_members(roles, dependencies, target):
    """Recursively union actual fit/selection ancestors, then reject overlap."""
    for a, b in itertools.combinations(roles, 2):
        if roles[a] & roles[b]:
            raise ValueError(f"Role overlap: {a}/{b}")

    def ancestors(node, active):
        if node in active:
            raise ValueError("Dependency cycle")
        if node in roles:
            return roles[node]
        if node not in dependencies:
            raise ValueError(f"Unknown dependency: {node}")
        return set().union(*(ancestors(n, active | {node}) for n in dependencies[node]))

    members = ancestors("calibrated_prediction", set())
    if members & target:
        raise ValueError("Prediction target occurs in fitting ancestors")
    return {"dependency_isolation": True, "fit_ancestor_participants": len(members)}


def self_test():
    roles = {"optimization": {"a"}, "selection": {"b"}, "calibration": {"c"}, "prediction": {"d"}}
    deps = {"model": ["optimization", "selection"], "calibrated_prediction": ["model", "calibration"]}
    audit_members(roles, deps, roles["prediction"])
    for bad in (
        {**deps, "model": ["optimization", "prediction"]},
        {**deps, "model": ["missing"]},
        {**deps, "model": ["calibrated_prediction"]},
    ):
        try:
            audit_members(roles, bad, roles["prediction"])
        except ValueError:
            continue
        raise AssertionError("Invalid dependency was accepted")
    print("Isolation tests: valid graph accepted; leakage, unknown ancestor, cycle rejected", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--scope-file", type=Path, help="Private CSV with participant_id and role; supplied by nested scheduler")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory; benchmark never overwrites runs")
    protocol, folds = v3.load_protocol(root / "results/fusion_agent_v3_protocol")
    data = root / "results/protocol_alignment_audit/cleaned_aligned.csv"
    if ev.sha256_file(data) != protocol["source_dataset_sha256"]:
        raise ValueError("Source hash mismatch")
    frame = paired.load_vector_csv(data)
    module = ev.load_upstream_module(root)
    masks = paired.split_masks(module, frame)
    locked = set(frame.loc[masks["internal_test"] | masks["validation_1"] | masks["validation_2"] | masks["global"], "id"].astype(str))
    selected = folds.loc[folds.outer_fold == args.fold]
    pool = selected.loc[selected.outer_role == "train"]
    if pool.empty:
        raise ValueError("Unknown fold")
    fit, reserve = train_test_split(pool, test_size=0.30, stratify=pool.label, random_state=args.seed)
    selection, calibration = train_test_split(reserve, test_size=0.5, stratify=reserve.label, random_state=args.seed + 1)
    roles = {
        "optimization": set(fit.participant_id),
        "selection": set(selection.participant_id),
        "calibration": set(calibration.participant_id),
        "prediction": set(selected.loc[selected.outer_role == "validation", "participant_id"]),
    }
    if args.scope_file:
        scope = pd.read_csv(args.scope_file, dtype={"participant_id": str})
        if set(scope.columns) != {"participant_id", "role"} or scope.participant_id.duplicated().any():
            raise ValueError("Invalid scope columns or duplicate participant")
        if set(scope.role) != set(roles):
            raise ValueError("Scope must specify four nonempty roles")
        roles = {name: set(scope.loc[scope.role == name, "participant_id"]) for name in roles}
        training_ids = set().union(*(roles[n] for n in roles if n != "prediction"))
        if not training_ids <= set(pool.participant_id):
            raise ValueError("Scope fitting members outside outer training partition")
        if not set().union(*roles.values()) <= set(selected.participant_id):
            raise ValueError("Scope contains unknown development participant")
    allowed = set().union(*roles.values())
    if allowed & locked:
        raise ValueError("Locked participant in pilot")
    deps = {
        "scaler": ["optimization"],
        "experts": ["scaler", "optimization", "selection"],
        "fusion": ["experts", "optimization", "selection"],
        "calibrated_prediction": ["fusion", "calibration"],
    }
    audit = audit_members(roles, deps, roles["prediction"])
    frame = frame.loc[frame.id.astype(str).isin(allowed)].reset_index(drop=True)
    fusion_config = ev.read_json(Path(module.MODEL_CONFIG_PATH))
    choices = module.MODEL_SUBSETS[int(fusion_config["model_subset_choice"])]
    module.NUM_MODELS = len(choices)
    paths = ev.checkpoint_paths(module, choices)
    configs = [ev.read_json(item["config"]) for item in paths]
    protected = {Path(p).resolve() for item in paths for p in item.values()}
    protected.add(Path(module.MODEL_PATH).resolve())
    before = {str(p): ev.sha256_file(p) for p in protected}
    raw = paired.inverse_original_scaling(frame, configs, paths)
    scaled, scalers = paired.fit_training_scalers(raw, raw.id.astype(str).isin(roles["optimization"]).to_numpy(), configs)
    partition = {name: scaled.loc[scaled.id.astype(str).isin(ids)].reset_index(drop=True) for name, ids in roles.items()}
    for name, part in partition.items():
        if set(part.id.astype(str)) != roles[name]:
            raise ValueError("Scope members missing in source")
        if name != "prediction" and part.label.nunique() != 2:
            raise ValueError("Fit/selection/calibration partition lacks a class")
    output.mkdir(parents=True)
    pd.DataFrame([{"participant_id": p, "role": name} for name, ids in roles.items() for p in sorted(ids)]).to_csv(output / "private_members.csv", index=False)
    (output / "dependencies.json").write_text(json.dumps(deps, indent=2))
    for i, scaler in enumerate(scalers):
        with (output / f"scaler_{i}.pkl").open("wb") as handle:
            pickle.dump(scaler, handle)
    device = ev.resolve_device("cuda")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    timings, predictors = {}, []
    for i, (name, config) in enumerate(zip(paired.MODALITIES, configs)):
        paired.set_seed(args.seed + i * 1000)
        tick = time.perf_counter()
        model, metadata = paired.train_predictor(module, partition["optimization"], partition["selection"], i, config, device, args.seed + i, 30, None)
        torch.cuda.synchronize()
        timings[name] = time.perf_counter() - tick
        paired.save_model(output / name, model, {"config": config, **metadata})
        predictors.append(model)
        print(f"Completed {name}: {timings[name]:.1f}s", flush=True)
    tick = time.perf_counter()
    paired.set_seed(args.seed + 10000)
    model, metadata = paired.train_fusion(module, predictors, partition["optimization"], partition["selection"], [len(scaled.iloc[0][f"features_{i}"]) for i in range(3)], fusion_config, device, args.seed + 10000, 30, None)
    torch.cuda.synchronize()
    timings["fusion"] = time.perf_counter() - tick
    paired.save_model(output / "fusion", model, {"config": fusion_config, **metadata})
    tick = time.perf_counter()
    predictions = {}
    for name in ("calibration", "prediction"):
        predictions[name] = quality.predict_quality_inputs(module, partition[name], predictors, model, device, 30, args.seed, [], [])
    calibrators = {}
    final = predictions["prediction"][["id", "row_id"]].copy()
    participant_final = predictions["prediction"].groupby("id", as_index=False)[["finger", "speech", "smile", "ufnet"]].mean()
    for name in ("finger", "speech", "smile", "ufnet"):
        cal = predictions["calibration"].groupby("id", as_index=False).agg({name: "mean", "label": "first"})
        def logits(values):
            p = np.clip(np.asarray(values), 1e-6, 1-1e-6)
            return np.log(p / (1-p)).reshape(-1, 1)
        calibrator = LogisticRegression(C=1.0, random_state=args.seed).fit(logits(cal[name]), cal.label)
        calibrators[name] = calibrator
        final[name] = calibrator.predict_proba(logits(predictions["prediction"][name]))[:, 1]
        participant_final[name] = calibrator.predict_proba(logits(participant_final[name]))[:, 1]
    with (output / "calibrators.pkl").open("wb") as handle:
        pickle.dump(calibrators, handle)
    final.to_csv(output / "private_predictions.csv", index=False)
    participant_final.to_csv(output / "private_participant_predictions.csv", index=False)
    torch.cuda.synchronize()
    timings["prediction_and_calibration"] = time.perf_counter() - tick
    elapsed = time.perf_counter() - start
    if before != {str(p): ev.sha256_file(p) for p in protected}:
        raise RuntimeError("Source artifact changed")
    result = {
        "status": "ENGINEERING_PILOT_COMPLETE", "protocol_role": "uses v3 splits; not v3 or v4 evaluation",
        "git_commit": ev.git_commit(root), "script_sha256": ev.sha256_file(Path(__file__)),
        "source_dataset_sha256": ev.sha256_file(data), "fold": args.fold, "seed": args.seed,
        "scope_sha256": ev.sha256_file(args.scope_file) if args.scope_file else None,
        "full_config_epochs": True, "mc_trials": 30,
        "participant_counts": {k: len(v) for k, v in roles.items()},
        "timings_seconds": timings, "training_and_prediction_seconds": elapsed,
        "peak_torch_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_torch_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "gpu": torch.cuda.get_device_name(), "output_bytes_before_summary": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
        "locked_predictions_generated": False, "protected_unchanged": True, "audit": audit,
        "limitations": ["recovered features and fixed architectures retain upstream provenance", "pilot measures one outer-training size; deeper folds may differ", "no disease metrics computed"],
        "artifacts_sha256": {str(p.relative_to(output)): ev.sha256_file(p) for p in output.rglob("*") if p.is_file()},
    }
    (output / "benchmark_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
