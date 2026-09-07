"""Post-hoc diagnostic only: existing outer predictions, no fitting or policy selection."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import nested_agent_pipeline as agent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(frame, fold, bootstraps):
    summary, pairs, action_rows = [], [], []
    for scenario, f in frame.groupby("scenario", sort=True):
        y = f.label.to_numpy(int)
        scores = agent.actions(f)
        losses = agent.loss(y[:, None], scores)
        best = int(losses.mean(axis=0).argmin())
        oracle = losses.min(axis=1)
        gain = losses[:, best] - oracle
        chosen = losses.argmin(axis=1)  # deterministic first-action ties, NOT deployable
        available = f[[n+"_available" for n in agent.EXPERTS]].to_numpy(bool)
        expert_error = (f[list(agent.EXPERTS)].to_numpy() >= .5) != y[:, None]
        all_wrong = (expert_error | ~available).all(axis=1)
        any_wrong = (expert_error & available).any(axis=1)
        rng = np.random.default_rng(20260905)
        # Each scenario contains one row per participant. Stratify fold and label;
        # resample full action vectors and reselect the hindsight comparator.
        groups = [g.index.to_numpy() for _, g in pd.DataFrame({"fold": f.outer_fold.to_numpy(), "y": y}).groupby(["fold", "y"])]
        draws = []
        for _ in range(bootstraps):
            ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
            draws.append(float(losses[ix].mean(axis=0).min()-oracle[ix].mean()))
        summary.append(dict(fold=fold, scenario=scenario, n=len(f), positives=int(y.sum()),
            available_experts_all_wrong_rate=float(all_wrong.mean()),
            available_experts_mixed_correctness_rate=float((any_wrong & ~all_wrong).mean()),
            best_fixed_action_hindsight=agent.ACTIONS[best], best_fixed_log_loss=float(losses[:, best].mean()),
            mean_baseline_log_loss=float(losses[:, 0].mean()), oracle_log_loss=float(oracle.mean()),
            oracle_gain_vs_best_fixed=float(gain.mean()), oracle_gain_vs_mean=float((losses[:, 0]-oracle).mean()),
            oracle_gain_bootstrap_p025=float(np.quantile(draws, .025)),
            oracle_gain_bootstrap_p975=float(np.quantile(draws, .975)),
            oracle_strict_improvement_rate=float((gain>1e-12).mean()),
            action_set_all_wrong_rate=float((((scores>=.5)!=y[:,None]).all(axis=1)).mean())))
        models = list(agent.EXPERTS)+["ufnet"]
        for a, b in itertools.permutations(models, 2):
            mask = np.ones(len(f), dtype=bool)
            for n in (a, b):
                if n in agent.EXPERTS:
                    mask &= f[n+"_available"].to_numpy(bool)
            ea = ((f[a].to_numpy()>=.5)!=y) & mask
            eb = ((f[b].to_numpy()>=.5)!=y) & mask
            count = int(ea.sum())
            pairs.append(dict(fold=fold, scenario=scenario, source=a, rescuer=b,
                eligible_n=int(mask.sum()), source_wrong_n=count,
                both_wrong_n=int((ea & eb).sum()), rescued_n=int((ea & ~eb).sum()),
                rescue_given_source_wrong=float((ea & ~eb).sum()/count) if count else None))
        for j, name in enumerate(agent.ACTIONS):
            delta = losses[:, 0]-losses[:, j]
            action_rows.append(dict(fold=fold, scenario=scenario, action=name, n=len(f),
                auroc=float(roc_auc_score(y, scores[:, j])), log_loss=float(losses[:, j].mean()),
                error_rate=float(((scores[:, j]>=.5)!=y).mean()), gain_vs_mean=float(delta.mean()),
                improves_over_mean_rate=float((delta>1e-12).mean()), harms_vs_mean_rate=float((delta< -1e-12).mean()),
                oracle_chosen_rate=float((chosen==j).mean())))
    return summary, pairs, action_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Refusing existing output directory")
    run = args.run_dir.resolve()
    audit = json.loads((run/"postrun_audit.json").read_text())
    assert audit["status"] == "PASS"
    sources = {}; frames = []; seen = set()
    for k in range(5):
        folder = run/f"outer_{k}/agent"
        path = folder/"private_outer_validation.csv"
        decisions = folder/"private_outer_decisions.csv"
        for p in (path, decisions): sources[str(p.relative_to(run))] = digest(p)
        f = pd.read_csv(path, dtype={"id": str}, float_precision="round_trip")
        d = pd.read_csv(decisions, dtype={"id": str}, float_precision="round_trip")
        assert f[["id", "scenario"]].equals(d[["id", "scenario"]])
        assert np.array_equal(f.label, d.label)
        np.testing.assert_allclose(agent.actions(f), d[list(agent.ACTIONS)].to_numpy(), atol=1e-12, rtol=1e-12)
        assert not f.duplicated(["id", "scenario"]).any()
        ids = set(f.id)
        assert not ids & seen
        seen |= ids
        assert f.groupby("id").label.nunique().max() == 1
        assert f.groupby("scenario").id.nunique().eq(len(ids)).all()
        assert set(f.scenario) == {s["name"] for s in agent.scenario_registry()}
        f["outer_fold"] = k
        frames.append(f)
    assert len(seen) == audit["outer_participants"] == 632
    frame = pd.concat(frames, ignore_index=True)
    rows = [[], [], []]
    for k, f in [("pooled", frame)] + [(str(k), f) for k, f in enumerate(frames)]:
        for dest, values in zip(rows, analyze(f, k, args.bootstraps)): dest.extend(values)
    for p, h in sources.items(): assert digest(run/p) == h
    args.output_dir.mkdir(parents=True)
    names = ("oracle_summary.csv", "pairwise_rescue.csv", "action_metrics.csv")
    for name, values in zip(names, rows): pd.DataFrame(values).to_csv(args.output_dir/name, index=False)
    manifest = dict(mode="POSTHOC_DIAGNOSTIC_NOT_CONFIRMATORY", participants=len(seen), scenarios=16,
        bootstraps=args.bootstraps, seed=20260905, source_sha256=sources,
        script_sha256=digest(Path(__file__)), protocol_sha256=audit["protocol_sha256"],
        output_sha256={n:digest(args.output_dir/n) for n in names},
        limitations=["Oracle uses true labels and is not deployable; log-loss upper gain is not an AUROC bound.",
            "Best fixed action is selected in hindsight per scenario, not an independent deployed comparator.",
            "Intervals are descriptive participant bootstraps stratified by fold and class; no refitting or multiplicity correction.",
            "Scenarios are not independent participants. No cross-scenario significance claim.",
            "Error and rescue use threshold 0.5. Missing experts excluded from expert rescue.",
            "Oracle action ties choose first action; selection frequencies are descriptive.",
            "All results are exposed development data; no fitting, no threshold changes, no locked test evaluation."])
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows[0]).query("fold == 'pooled'").to_string(index=False), flush=True)


if __name__ == "__main__": main()
