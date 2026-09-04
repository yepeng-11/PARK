# PARK/UFNet baseline reproduction

This repository includes pretrained unimodal and fusion checkpoints. The
upstream training scripts overwrite those checkpoints and several tracked CSV
files, so experiments should run in a disposable copy of the repository.

## Evaluate pretrained checkpoints

From the repository root:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate park
python tools/evaluate_pretrained.py
```

For the frozen paper cohorts, use the hash-locked protocol mode after generating
the score-provenance artifacts:

```bash
python tools/evaluate_pretrained.py \
  --protocol paper-exact \
  --device cuda \
  --output-dir results/pretrained_eval_paper_exact
```

This mode evaluates each complete source cohort before selecting and restoring
the frozen paper rows, preserving the inference path while enforcing the exact
162/91/67 memberships. It stops on a manifest-hash, row, participant, or label
mismatch.

Outputs are written to `results/pretrained_eval/`:

- `metrics_summary.csv` and `metrics_summary.json`: session- and
  participant-level results for finger tapping, speech, smile, and fusion.
- `predictions_<split>.csv`: row-level scores, MC uncertainty, and abstention
  flags for each official split.
- `run_manifest.json`: environment, git commit, split sizes, configuration, and
  checkpoint hashes.

The evaluator intentionally reports two fusion abstention variants:

- `official_batch_denominator` reproduces the upstream implementation.
- `corrected_mc_denominator` computes the confidence interval of the MC mean
  using `sqrt(num_trials)` rather than `sqrt(batch_size)`.

It also distinguishes standard balanced accuracy,
`(sensitivity + specificity) / 2`, from the upstream
`official_weighted_accuracy`, which is `(PPV + NPV) / 2`.

## Upstream alignment audit

The shipped fusion pipeline joins modalities by `row_id` and keeps the label
from the first (finger-tapping) modality. The evaluator preserves that behavior
for faithful checkpoint reproduction, but records cross-modality label
disagreements and duplicate `row_id` rows in `run_manifest.json` under
`alignment_audit`. These issues should be resolved before using the dataset for
new model development rather than silently copied into a new training pipeline.

Run the full protocol/alignment audit from the isolated PARK copy:

```bash
python tools/audit_protocol_alignment.py
```

The command writes the official-compatible and cleaned datasets, per-row
provenance, split-overlap tables, a machine-readable protocol audit, and
`DIFFERENCE_REPORT.md` under `results/protocol_alignment_audit/`.

## Paired retraining

After the audit artifacts exist, run the isolated official-versus-cleaned
comparison with:

```bash
python tools/train_paired_baselines.py \
  --datasets official,cleaned \
  --seeds 101,202,303,404,505 \
  --mc-trials 30 \
  --output-dir results/paired_retraining \
  --device cuda
```

The runner refits feature scalers on each training partition, trains the three
unimodal models and UFNet, saves every seed separately, supports resume through
`run_complete.json`, and verifies that the shipped checkpoints remain unchanged.
`PAIRED_REPORT.md` and `paired_deltas.csv` summarize cleaned-minus-official
differences. This five-seed complete-case comparison is a development baseline,
not a substitute for the paper's modality-specific 30-seed protocol.

## Calibrated classical fusion baselines

After paired retraining, run:

```bash
python tools/evaluate_calibrated_fusion.py \
  --datasets official,cleaned \
  --seeds 101,202,303,404,505 \
  --mc-trials 30 \
  --output-dir results/calibrated_fusion \
  --device cuda
```

The evaluator fits all combiners, Raw/Platt/Isotonic calibrators, and decision
thresholds on Dev only. It evaluates probability averaging, majority voting,
Dev-AUROC weighting, logistic stacking, HistGradientBoosting stacking, and
UFNet at the fixed 0.5, Youden, and Dev-specificity 0.80/0.90/0.95 operating
points. Test labels are not used for fitting or threshold selection.

## Missing and corrupted modality robustness

After paired retraining and calibrated-fusion evaluation, run the cleaned-data
stress test with:

```bash
python tools/evaluate_modality_robustness.py \
  --datasets cleaned \
  --seeds 101,202,303,404,505 \
  --mc-trials 30 \
  --output-dir results/modality_robustness \
  --device cuda
```

The evaluator covers 22 deterministic conditions: clean input, every single-
and pair-missing combination, per-modality Gaussian noise at 0.5 and 1.0
post-scaling standard deviations, 25% and 50% feature masking, and adversarial
single-expert conflicts. Fusion rules and Raw/Platt calibration states are fit
on the clean Dev partition only. Missing neural features use the training mean
in standardized space; fixed-input classical combiners receive probability
0.5, while availability-aware mean and weighted variants renormalize over the
observed experts.

The five-seed run writes 4,620 metric records, 35,200 scenario predictions,
`robustness_summary.csv`, `ROBUSTNESS_REPORT.md`, per-seed completion markers,
and a checkpoint-integrity manifest. In the completed run, speech corruption
was the dominant failure mode: UFNet participant-level AUROC fell from 0.8626
on clean input to 0.5286 under a conflicting speech expert and to 0.6766 under
speech noise of severity 1.0. These results motivate explicit availability and
quality gating before training a new adaptive fusion model.

## Quality-aware gated fusion prototype

Run the predeclared interpretable gate with:

```bash
python tools/evaluate_quality_gated_fusion.py \
  --datasets cleaned \
  --seeds 101,202,303,404,505 \
  --mc-trials 30 \
  --output-dir results/quality_gated_fusion \
  --device cuda
```

The gate combines clean-Dev AUROC priors with explicit availability masks,
modality-normalized MC standard deviations, and disagreement from the other
available experts. Its uncertainty and consistency strengths are selected from
deterministic score-level Dev perturbations. Platt calibration and the
90%-specificity operating threshold use clean Dev only; internal-test labels
are never used for fitting or selection. Two predeclared ablations remove the
uncertainty and consistency terms separately.

The five-seed evaluation writes 4,620 metric records, 35,200 predictions with
per-sample gate weights, 5,500 Dev-grid records, `quality_gate_summary.csv`,
`QUALITY_GATE_REPORT.md`, and an integrity manifest under
`results/quality_gated_fusion/`. The full gate improved participant AUROC under
speech noise severity 1.0 from 0.6766 (UFNet) and 0.6631 (available weighted)
to 0.6914, but its clean AUROC was 0.8595 versus 0.8626 for UFNet and 0.8712 for
available weighted. It also failed to identify a conflicting smile expert.
Accordingly, this interpretable gate is a diagnostic prototype, not a selected
replacement model. The next candidate should learn modality-specific quality
scores from feature-level corruptions while retaining an explicit availability
fallback and a clean-performance non-inferiority constraint.

## Learned modality-quality gate

Run the Train-corruption quality detector and Dev-constrained blend with:

```bash
python tools/evaluate_learned_quality_gate.py \
  --datasets cleaned \
  --seeds 101,202,303,404,505 \
  --mc-trials 30 \
  --device cuda \
  --output-dir results/learned_quality_gate
```

Each modality-specific HistGradientBoosting detector learns synthetic clean
versus corrupted labels on Train only. Its inputs combine feature-distribution
summaries, expert probability, MC uncertainty, confidence, and cross-expert
disagreement. The candidate gate is blended with the availability-aware
Dev-AUROC weighting rule. Blend selection uses the 22 perturbed Dev scenarios
and enforces a 0.005 clean-Dev AUROC non-inferiority margin. Test labels are
evaluation-only.

Across five paired seeds, the synthetic-Dev corruption detectors achieved mean
AUROC 0.8886, 0.9597, and 0.9264 for finger, speech, and smile. The learned gate
improved participant AUROC under speech noise severity 1.0 to 0.7297, versus
0.7057 for UFNet, 0.6883 for available weighted, and 0.6914 for the earlier
heuristic gate. It also raised speech-conflict AUROC from 0.4474 to 0.5051 versus
available weighted, although UFNet remained at 0.5215.

The full candidate is not promoted. Clean AUROC was 0.8597 versus 0.8689 for
available weighted, a -0.0092 delta that exceeds the 0.005 non-inferiority
margin. Its 21-scenario stress macro-AUROC was 0.7860 versus 0.7869 for available
weighted and 0.7960 for UFNet, and smile-conflict AUROC remained weak at 0.6715
versus 0.8050 for UFNet. Selected blend coefficients varied from 0.0 to 1.0
across seeds, indicating unstable Dev-to-Test transfer. Retain the speech-quality
detector as a candidate specialist fallback, but do not replace the fusion rule.

## Paper-exact protocol audit

To distinguish cohort-definition differences from model-score differences, run:

```bash
python tools/audit_paper_exact_protocol.py \
  --output-dir results/paper_exact_protocol_audit
```

The audit treats the released `data/test_data_big.csv` artifact as the frozen
paper protocol and joins independently generated pretrained predictions to the
same `(split, row_id)` memberships. It hashes all inputs, records every retained
and excluded membership, recomputes the published point estimates, and writes
`PAPER_EXACT_PROTOCOL_REPORT.md` plus machine-readable cohort, overlap, metric,
and prediction-alignment tables.

The completed audit reconstructed the paper's 162/91/67 session counts exactly.
The two source external-validation files contain three sessions assigned to both
splits; the paper artifact keeps each session once, excluding one membership
from the supervised split and two from the unsupervised split. This operation
does not make the external cohorts participant-independent: one participant is
still represented in both cohorts after session de-duplication.

Metrics recomputed from the released paper scores match every published
classification point estimate to within 0.047 percentage points, consistent
with one-decimal percentage rounding. On the identical paper rows, the
independent evaluator differs from the released scores by mean absolute errors
of 0.0281, 0.0239, and 0.0442 across the balanced, supervised, and unsupervised
cohorts, respectively, with only 2, 1, and 2 threshold-0.5 decision changes.
Therefore the remaining evaluator discrepancy is a score-generation,
checkpoint-version, or stochastic-inference issue rather than a cohort
membership issue.

## Paper-score provenance and frozen cohort

Freeze the exact paper membership and quantify MC-dropout score variation with:

```bash
python tools/trace_paper_score_provenance.py \
  --replicates 100 \
  --device cuda \
  --output-dir results/paper_score_provenance
```

The run writes a canonical 320-row `paper_exact_cohort_manifest.csv`, hashes its
split/row/participant/label memberships, verifies the released seed-289 PKL
lineage, and compares the paper scores with 100 seeded repetitions of the
released checkpoint at 30 MC trials. The frozen cohort manifest SHA-256 is
`6f8424c91357d68042a6012498256d98a5541b21f61cf7d79b565db359122c34`.

All labels and uncertainty flags match between `test_data_big.csv` and the
released PKL artifacts. Scores match within `2.97e-8` and the stored uncertainty
quantity within `3.53e-9`, which is float32/CSV serialization precision. Paper
score MAE to the 100-run MC ensemble is essentially the same as a typical fresh
30-trial draw: the ratios are 0.996, 1.072, and 0.923 for the balanced,
supervised, and unsupervised cohorts. Corresponding row-level 95% MC-band
coverage is 93.8%, 92.3%, and 98.5%.

This evidence supports stochastic MC masks as the source of fresh-versus-paper
score differences. Exact historical scores cannot be regenerated from the
checkpoint alone because BaaL dropout is active during evaluation and the
training program did not save or reset the RNG state before each cohort. The
released PKL files also contain no embedded checkpoint hash, so their link to
the checkpoint is statistically and repository-provenance supported rather than
cryptographically proven. Both external-cohort paper AUROCs lie just outside the
central 95% fresh-MC interval, showing that repeated-inference variation should
be reported for rank metrics on these small cohorts.

The same run also writes `final_reproduction_table.csv` and
`FINAL_REPRODUCTION_TABLE.md`. These 24-row tables align the article's published
values, metrics recomputed from frozen paper scores, and the mean, standard
deviation, and central 95% interval from 100 fresh 30-trial MC evaluations. All
18 published classification point estimates match the frozen scores within
0.047 percentage points and lie inside the fresh-MC intervals. The exact frozen
AUROCs for the two external cohorts fall just outside their fresh-MC central 95%
intervals; the article's published column is intentionally blank for AUROC
because it reports only a cross-cohort range rather than exact table values.

## One-command reproduction and acceptance bundle

After activating the `park` environment, run the complete released-checkpoint
evaluation workflow with:

```bash
python tools/run_paper_reproduction.py --device cuda
```

Existing valid stage artifacts are reused by default, and the acceptance checks
always run. Pass `--force` to recompute every audit, inference, MC-provenance,
and paper-exact evaluation stage. The locked protocol requires exactly 100 MC
replicates.

The final stage writes `results/reproduction_bundle/`, including:

- `REPRODUCTION_ACCEPTANCE_REPORT.md` and `acceptance_summary.json`;
- `acceptance_checks.csv` with machine-readable criteria and evidence;
- `artifact_inventory.csv` with per-file SHA-256 hashes;
- `conda-linux-64-explicit.lock`, a platform-specific Conda package lock;
- `pip-freeze.txt`, normalized to portable `package==version` records;
- `environment_snapshot.json` and `nvidia-smi.txt`;
- per-stage logs and `pipeline_state.json`.

The completed server run passed all 12 required checks and produced the status
`PASS_WITH_DOCUMENTED_LIMITATIONS`. The three warnings are upstream provenance
limitations rather than failed reproduction checks: one participant overlaps
the external cohorts, the historical MC RNG state was not saved, and the
released PKLs do not embed a checkpoint hash. This acceptance scope covers the
2026 paper's released-checkpoint evaluation; full from-scratch 30-seed training
remains a separate scope.

## Fusion Agent v1

Fusion Agent v1 is an experimental, auditable wrapper around the paired PARK
experts. Run the five-seed evaluation with:

```bash
python tools/evaluate_fusion_agent_v1.py \
  --device cuda \
  --seeds 101,202,303,404,505 \
  --mc-trials 30 \
  --output-dir results/fusion_agent_v1
```

The agent fits synthetic-corruption quality detectors on Train, selects routing
rules and a risk threshold on Dev, and uses internal-test labels only for the
final diagnostic evaluation. It defaults to the availability-aware Dev-AUROC
weighted score, renormalizes around missing modalities, can downweight degraded
speech, can fall back to UFNet for a high-confidence smile conflict, and can
abstain on high-risk cases. A hard 5% clean-Dev route-rate limit and clean-AUROC
non-inferiority constraint prevent an aggressive gate from being selected. If
no active rule also improves Dev stress macro-AUROC, the score router fails
closed to the availability-aware baseline.

In the completed five-seed run, four seeds failed closed and one selected a
low-trigger smile fallback. The full-coverage router was not promoted: clean
participant AUROC was 0.8666 versus 0.8689 for available-weighted fusion, and
stress macro-AUROC was 0.7858. The selective policy was retained for genuinely
unseen external validation: clean selective AUROC was 0.8913 at 84.2% coverage,
and stress selective macro-AUROC was 0.8316 at 66.0% mean coverage. These are
diagnostic results because the current internal test has prior analytical
exposure; they are not an unbiased final performance claim.

Outputs under `results/fusion_agent_v1/` include per-run frozen agent pickles,
Dev tuning audits, route and abstention audits, full and selective metrics,
`selection_decision.json`, and `FUSION_AGENT_V1_REPORT.md`.

## Fusion Agent v2 frozen protocol

Before training v2, freeze its nested development protocol with:

```bash
python tools/freeze_fusion_agent_v2_protocol.py \
  --output-dir results/fusion_agent_v2_protocol
```

The protocol admits only eligible participants from the original Train+Dev
pool. It creates five stratified participant-level outer folds and four inner
folds inside every outer-training partition. All existing internal and external
test cohorts remain locked. One identifier with inconsistent labels across two
sessions is quarantined in full rather than resolved by majority vote; the
private exclusion and fold manifests remain on the experiment server.

The frozen development pool contains 632 participants, 757 sessions, and 160
positive participants. All 23 leakage checks pass. The protocol records a
deterministic canonical SHA-256 over its source data, generator, splits,
perturbations, criteria, and other frozen settings. The final hash is
`c07666342e8271ec53cf2d0d6ded0463701eba1c03d7ed86d3e296a535bc380c`.
The protocol separately defines a speech feature-corruption detector and a
smile score-conflict detector, preserves v1 risk rejection and fail-closed
behavior, and locks eight promotion criteria before v2 training begins.

Train the two v2 specialist detectors under the frozen nested protocol with:

```bash
python tools/train_fusion_agent_v2_specialists.py \
  --device cuda \
  --paired-seeds 101,202,303,404,505 \
  --mc-trials 30 \
  --output-dir results/fusion_agent_v2_specialists
```

Each outer fold uses one predeclared paired-model seed. Four inner folds select
histogram-gradient-boosting hyperparameters using synthetic corruption targets
only; the outer fold reports corruption-detection generalization. No locked
Test participant is predicted or scored. In the completed run, speech-noise
detection AUROC was 1.0000 +/- 0.0000 with zero clean false positives, while
smile-conflict detection AUROC was 0.8721 +/- 0.0215 with a 0.1253 clean false
positive rate. The perfect synthetic speech result must not be interpreted as
real-world noise performance. All protected base models and scalers were
unchanged.

Train and audit the v2 router after the specialists have completed with:

```bash
python tools/train_fusion_agent_v2_router.py \
  --device cuda \
  --mc-trials 30 \
  --output-dir results/fusion_agent_v2_router
```

The router is selected strictly inside each outer-training partition. Its
candidate actions are speech downweighting, no smile action, smile dropping,
or a UFNet fallback; unavailable modalities are handled separately from quality
failures. If no candidate satisfies the frozen inner-fold constraints, routing
fails closed to availability-weighted fusion. The run generates no released
Test or external-cohort predictions.

The completed five-fold, 30-Monte-Carlo run did not meet the frozen promotion
gate: 4/8 criteria passed, and all five outer folds selected the fail-closed
availability-weighted baseline. Clean non-inferiority, clean coverage (0.8544),
the calibration guard, and fold stability passed. Stress superiority over
availability-weighted fusion (delta 0.0000), stress superiority over UFNet
(-0.0088), speech-noise gain (-0.0137), and smile-conflict non-inferiority
(-0.0671 versus the required -0.0100) failed. Therefore Fusion Agent v2 is not
promoted to unseen external evaluation.

Mean clean outer-fold AUROC was 0.9624 for both the router and the
availability-weighted baseline, versus 0.9632 for UFNet. On smile conflict it
was 0.8763 for the router and baseline versus 0.9434 for UFNet. Because the
released paired disease models were not retrained inside each outer fold, some
outer participants may have contributed to those base models; absolute AUROCs
are consequently optimistic. Within-fold routing deltas and the predeclared
acceptance decisions are the interpretable outputs. Any further routing redesign
must be declared as a new protocol version rather than tuned against these now
observed outer-fold results.

## Safety

Run training only from an isolated experiment copy. The original scripts save
models directly under `models/` and rewrite intermediate data and prediction
files.
