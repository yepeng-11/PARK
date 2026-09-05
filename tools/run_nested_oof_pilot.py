"""Build and execute 21 isolated base-model suites for one development outer fold.

Outputs clean, calibrated participant predictions for nested router development.
No router is trained and no disease performance metric is selected here.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.model_selection import StratifiedKFold, train_test_split

import benchmark_fold_training as base
import train_fusion_agent_v3_router as v3


FIT_ROLES = ("optimization", "selection", "calibration")
SCORES = ("finger", "speech", "smile", "ufnet")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def make_roles(train, target, seed):
    fit, reserve = train_test_split(train, test_size=0.30, stratify=train.label, random_state=seed)
    selection, calibration = train_test_split(reserve, test_size=0.5, stratify=reserve.label, random_state=seed + 1)
    return {"optimization": set(fit.participant_id), "selection": set(selection.participant_id),
            "calibration": set(calibration.participant_id), "prediction": set(target.participant_id)}


def validate_plan(jobs, outer_train, outer_target, heldouts):
    """Enforce both local disjointness and meta-validation ancestor exclusion."""
    if outer_train & outer_target:
        raise ValueError("Outer overlap")
    if len(jobs) != 21 or len({j['name'] for j in jobs}) != 21:
        raise ValueError("Expected 21 uniquely named suites")
    for job in jobs:
        roles = job["roles"]
        base.audit_members(roles, {"calibrated_prediction": list(FIT_ROLES)}, roles["prediction"])
        fit = set().union(*(roles[r] for r in FIT_ROLES))
        if not fit <= outer_train:
            raise ValueError("Outer target in fit ancestors")
        if job["kind"] == "meta_oof":
            h = heldouts[job["meta_fold"]]
            if (fit | roles["prediction"]) & h:
                raise ValueError("Meta holdout influences lower OOF features")
            if fit | roles["prediction"] != outer_train - h:
                raise ValueError("Lower OOF partition incomplete")
        elif job["kind"] == "meta_validation":
            h = heldouts[job["meta_fold"]]
            if fit != outer_train - h or roles["prediction"] != h:
                raise ValueError("Incorrect meta-validation scope")
        elif job["kind"] == "final_oof":
            if fit | roles["prediction"] != outer_train:
                raise ValueError("Final OOF partition incomplete")
        elif job["kind"] == "outer_prediction":
            if fit != outer_train or roles["prediction"] != outer_target:
                raise ValueError("Incorrect outer prediction scope")
        else:
            raise ValueError("Unknown suite kind")
    groups = [[j for j in jobs if j["kind"] == "meta_oof" and j["meta_fold"] == k] for k in heldouts]
    groups.append([j for j in jobs if j["kind"] == "final_oof"])
    expected = [outer_train - heldouts[k] for k in heldouts] + [outer_train]
    for group, ids in zip(groups, expected):
        targets = [p for j in group for p in j["roles"]["prediction"]]
        if len(targets) != len(set(targets)) or set(targets) != ids:
            raise ValueError("OOF target missing or duplicated")


def plan_suites(rows, seed):
    train = rows.loc[rows.outer_role == "train"].reset_index(drop=True)
    target = rows.loc[rows.outer_role == "validation"].reset_index(drop=True)
    heldouts = {}
    jobs = []

    def add(name, kind, meta_fold, t, v):
        job_seed = seed + 100 * len(jobs)
        jobs.append({"name": name, "kind": kind, "meta_fold": meta_fold,
                     "seed": job_seed, "roles": make_roles(t, v, job_seed)})

    for k in sorted(train.inner_validation_fold.unique()):
        k = int(k)
        h = train.loc[train.inner_validation_fold == k]
        t = train.loc[train.inner_validation_fold != k].reset_index(drop=True)
        heldouts[k] = set(h.participant_id)
        split = StratifiedKFold(3, shuffle=True, random_state=seed + k)
        for sub, (i, v) in enumerate(split.split(t, t.label)):
            add(f"meta_{k}_oof_{sub}", "meta_oof", k, t.iloc[i], t.iloc[v])
        add(f"meta_{k}_validation", "meta_validation", k, t, h)
    for k in sorted(heldouts):
        add(f"final_oof_{k}", "final_oof", -1,
            train.loc[train.inner_validation_fold != k], train.loc[train.inner_validation_fold == k])
    add("outer_prediction", "outer_prediction", -1, train, target)
    validate_plan(jobs, set(train.participant_id), set(target.participant_id), heldouts)
    return jobs, set(train.participant_id), set(target.participant_id), heldouts


def self_test():
    import copy
    n = 400
    rows = pd.DataFrame({"participant_id": [f"p{i}" for i in range(n)],
                         "label": [i % 2 for i in range(n)],
                         "outer_role": ["train"] * 320 + ["validation"] * 80,
                         "inner_validation_fold": [(i // 2) % 4 if i < 320 else -1 for i in range(n)]})
    jobs, t, v, hs = plan_suites(rows, 101)
    # A locally disjoint substitution still must fail at the meta-layer boundary.
    bad = copy.deepcopy(jobs)
    member = next(iter(bad[0]["roles"]["optimization"]))
    bad[0]["roles"]["optimization"].remove(member)
    bad[0]["roles"]["optimization"].add(next(iter(hs[0])))
    try:
        validate_plan(bad, t, v, hs)
    except ValueError:
        pass
    else:
        raise AssertionError("Indirect meta holdout leakage accepted")
    bad = copy.deepcopy(jobs)
    bad[1]["roles"]["prediction"] = set(bad[0]["roles"]["prediction"])
    try:
        validate_plan(bad, t, v, hs)
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate OOF assignment accepted")
    base.self_test()
    print("Nested tests PASS: 21-suite coverage; meta-ancestor leakage and duplicate targets rejected", flush=True)


def monitor(command, log_path):
    peak = 0
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                p = psutil.Process(process.pid)
                procs = [p, *p.children(recursive=True)]
                rss = sum(x.memory_info().rss for x in procs if x.is_running())
                peak = max(peak, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.25)
        if process.returncode:
            raise RuntimeError(f"Suite failed; inspect {log_path}")
    return peak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory")
    protocol, manifest = v3.load_protocol(root / "results/fusion_agent_v3_protocol")
    jobs, outer_train, outer_target, heldouts = plan_suites(manifest.loc[manifest.outer_fold == args.fold], args.seed)
    output.mkdir(parents=True)
    scopes = output / "private_scopes"
    scopes.mkdir()
    sources = {p.name: digest(p) for p in (root / "tools").glob("*.py")}
    public_plan = []
    for job in jobs:
        scope_path = scopes / f"{job['name']}.csv"
        pd.DataFrame([{"participant_id": p, "role": r} for r, ids in job["roles"].items() for p in sorted(ids)]).to_csv(scope_path, index=False)
        public_plan.append({k: v for k, v in job.items() if k != "roles"} | {
            "counts": {r: len(ids) for r, ids in job["roles"].items()}, "scope_sha256": digest(scope_path)})
    write_json(output / "execution_plan.json", {"protocol_sha256": protocol["protocol_sha256"],
               "source_sha256": sources, "suites": public_plan, "scope": "clean base OOF engineering pilot"})
    completed, cached = [], {}
    for job, plan in zip(jobs, public_plan):
        name = job["name"]
        suite = output / name
        tick = time.perf_counter()
        print(f"Starting {len(completed)+1}/21: {name}", flush=True)
        rss = monitor([sys.executable, "-u", str(root / "tools/benchmark_fold_training.py"),
                       "--fold", str(args.fold), "--seed", str(job["seed"]),
                       "--scope-file", str(scopes / f"{name}.csv"), "--output-dir", str(suite)], output / f"{name}.log")
        summary = json.loads((suite / "benchmark_summary.json").read_text())
        if summary["scope_sha256"] != plan["scope_sha256"] or not summary["audit"]["dependency_isolation"]:
            raise ValueError("Scope identity or isolation failure")
        if not summary["protected_unchanged"] or summary["locked_predictions_generated"]:
            raise ValueError("Protected artifact or locked-cohort violation")
        for path, expected in summary["artifacts_sha256"].items():
            if digest(suite / path) != expected:
                raise ValueError("Output hash mismatch")
        actual = pd.read_csv(suite / "private_members.csv", dtype={"participant_id": str})
        for role, ids in job["roles"].items():
            if set(actual.loc[actual.role == role, "participant_id"]) != ids:
                raise ValueError("Actual membership differs from plan")
        for model in ("finger", "speech", "smile", "fusion"):
            history = json.loads((suite / model / "training.json").read_text())
            if len(history["history"]) != history["epochs"] or history["epochs"] != history["config"]["num_epochs"]:
                raise ValueError("Incomplete epoch history")
        preds = pd.read_csv(suite / "private_participant_predictions.csv", dtype={"id": str})
        if preds.id.duplicated().any() or set(preds.id) != job["roles"]["prediction"]:
            raise ValueError("Prediction coverage mismatch")
        values = preds[list(SCORES)].to_numpy(float)
        if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError("Invalid calibrated score")
        preds["source_suite"] = name
        cached[name] = preds
        completed.append({"suite": name, "wall_seconds": time.perf_counter() - tick,
                          "training_seconds": summary["training_and_prediction_seconds"],
                          "peak_child_tree_rss_mib_sampled": rss / 2**20,
                          "peak_torch_allocated_mib": summary["peak_torch_allocated_mib"],
                          "peak_torch_reserved_mib": summary["peak_torch_reserved_mib"],
                          "participants": len(preds)})
        write_json(output / "progress.json", {"complete": len(completed), "total": 21})
        print(f"Completed {name}: {completed[-1]['wall_seconds']:.1f}s", flush=True)
    validate_plan(jobs, outer_train, outer_target, heldouts)
    for p in (root / "tools").glob("*.py"):
        if sources.get(p.name) != digest(p):
            raise ValueError("Training source changed during execution")
    cache = output / "private_oof"
    cache.mkdir()
    groups = {f"meta_{k}_train": [j["name"] for j in jobs if j["kind"] == "meta_oof" and j["meta_fold"] == k] for k in heldouts}
    groups.update({f"meta_{k}_validation": [f"meta_{k}_validation"] for k in heldouts})
    groups["final_train"] = [j["name"] for j in jobs if j["kind"] == "final_oof"]
    groups["outer_validation"] = ["outer_prediction"]
    cache_summary = []
    for name, names in groups.items():
        table = pd.concat([cached[n] for n in names], ignore_index=True)
        if table.id.duplicated().any():
            raise ValueError("Duplicate cache participant")
        table.to_csv(cache / f"{name}.csv", index=False)
        cache_summary.append({"cache": name, "participants": len(table), "source_suites": names})
    pd.DataFrame(completed).to_csv(output / "suite_timings.csv", index=False)
    duration = time.perf_counter() - start
    report = {
        "status": "CLEAN_BASE_NESTED_OOF_COMPLETE", "outer_fold": args.fold, "suites": len(completed),
        "networks_trained": 4 * len(completed), "outer_train_participants": len(outer_train),
        "outer_validation_participants": len(outer_target), "wall_seconds": duration,
        "sum_training_seconds": sum(x["training_seconds"] for x in completed),
        "peak_child_tree_rss_mib_sampled": max(x["peak_child_tree_rss_mib_sampled"] for x in completed),
        "peak_torch_allocated_mib": max(x["peak_torch_allocated_mib"] for x in completed),
        "peak_torch_reserved_mib": max(x["peak_torch_reserved_mib"] for x in completed),
        "output_bytes_before_summary": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
        "isolated_meta_ancestors": True, "full_epoch_histories_verified": True,
        "source_and_output_hashes_verified": True, "locked_cohort_predictions": False,
        "router_trained": False, "performance_metrics_computed": False,
        "cache_summary": cache_summary,
        "limits": ["one exposed development outer fold", "clean scores only; no perturbation or routing costs",
                   "wall time begins after imports; RSS sampled per child tree excludes controller", "Torch peak is not whole GPU process memory"],
        "aggregate_sha256": {n: digest(output / n) for n in ("execution_plan.json", "suite_timings.csv")},
    }
    write_json(output / "nested_oof_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
