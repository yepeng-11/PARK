# UFNet AAAI 2025 paper-aligned reproduction

This directory freezes the public-feature reproduction of **Accessible, At-Home Detection of Parkinson's Disease via Multi-Task Video Analysis**.

## Frozen source

- Upstream repository: `https://github.com/ROC-HCI/UFNet`
- Upstream commit: `5ece2c65ba184faccf6c8cdccdc03132427c464b`
- Cohort: 845 participants and 1,102 three-task fusion rows
- Split: train 516/690, validation 167/215, test 162/197 participants/rows

Raw patient videos are not public. This reproduction begins with the extracted features released by the authors. The upstream checkout and feature CSVs are intentionally not copied into this repository.

## Published frozen artifacts

- `../../artifacts/ufnet_paper_aligned_manifest/source_provenance.json`
- `../../artifacts/ufnet_paper_aligned_manifest/paper_partition_summary.csv`
- `../../artifacts/ufnet_paper_aligned_manifest/AUDIT_REPORT.md`
- `protocol.yaml`

The participant/session manifests and modality-presence table contain pseudonymous participant IDs and source filenames. They are frozen on the controlled server and local workspace but intentionally excluded from Git history. Rebuild them from the pinned upstream commit with the command below.

## Rebuild the manifest

```bash
python tools/build_ufnet_paper_manifest.py \
  --source /path/to/ROC-HCI-UFNet \
  --output artifacts/ufnet_paper_aligned_manifest
```

The command must return `Status: PASS`. No participant or row may be deleted to force the expected counts.

## Run the official 30 seeds

```bash
python tools/run_ufnet_official_30seeds.py \
  --repo /path/to/ROC-HCI-UFNet \
  --output results/ufnet_official_split_30seeds \
  --python /path/to/python
```

The launcher is resumable: a seed with a completed JSON result is not rerun.

## Evaluate saved probabilities

```bash
python tools/evaluate_frozen_predictions.py \
  --predictions predictions.csv \
  --baseline-predictions strongest_baseline_predictions.csv \
  --manifest artifacts/ufnet_paper_aligned_manifest/paper_session_manifest.csv \
  --output evaluation.json
```

With a baseline file, the evaluator uses identical participant bootstrap draws to report paired metric-difference confidence intervals. See `protocol.yaml` for the complete frozen evaluation contract.
