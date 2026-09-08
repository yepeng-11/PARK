# Frozen UFNet reproduction results

Status: **PASS**

- Run date: 2026-09-08
- Hardware: NVIDIA GeForce RTX 2080 Ti, 11 GB
- Software: Python 3.9, PyTorch 2.2.0+cu118, scikit-learn 1.4.2
- Upstream UFNet commit: `5ece2c65ba184faccf6c8cdccdc03132427c464b`

## Protocol gate

Every seed independently reported:

- train: 690 rows;
- validation: 215 rows;
- test: 197 rows;
- test coverage: 100%;
- frozen experts loaded from the official repository;
- no test-time model selection.

Thirty of thirty final seeds completed. Total subprocess runtime was 1,744.1 seconds (29.1 minutes).

## Results across 30 seeds

| Metric | Reproduced mean | Reproduced SD | Official exported mean | Mean difference |
| --- | ---: | ---: | ---: | ---: |
| AUROC | 0.927234 | 0.005369 | 0.928067 | -0.000833 |
| AUPRC | 0.861812 | 0.012974 | 0.862827 | -0.001015 |
| Accuracy | 0.870558 | 0.008405 | 0.872927 | -0.002369 |
| Balanced accuracy | 0.861732 | 0.009479 | 0.863877 | -0.002145 |
| F1 | 0.805654 | 0.013992 | 0.809646 | -0.003992 |
| Sensitivity | 0.777941 | 0.025719 | 0.783824 | -0.005883 |
| Specificity | 0.919380 | 0.011444 | 0.919897 | -0.000517 |
| Brier score | 0.100548 | 0.003657 | 0.099192 | +0.001356 |
| ECE | 0.056929 | 0.014081 | 0.056795 | +0.000134 |

Reproduced AUROC normal-approximation 95% CI across seeds: **[0.925313, 0.929156]**. The paper/repository value is contained in this interval. Small per-seed differences are expected because the released pickled scalers were created with scikit-learn 1.4.0 while this run used 1.4.2, and stochastic GPU execution is not guaranteed to be bit-identical across hardware/software stacks.

## Frozen machine-readable outputs

- `results/per_seed_metrics.csv`: one row per official final seed.
- `results/summary.json`: aggregate metrics, ranges, confidence intervals, runtime, and comparison with the official W&B export.

These results establish the paper-aligned reproduction baseline. New Adapter, Transformer, LLM, or Agent models must not replace this result and must use the same frozen split and evaluation contract.
