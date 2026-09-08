# Phase 3 one-time UFNet official-test report

Status: **PASS**

The protocol and primary model were frozen in commit `84bc6e4` before opening
the official test. Evaluation covers the same 197 sessions from 162 participants
used by the paper split, at 100% coverage and a fixed 0.5 threshold. No model,
seed, threshold, or rejection rule was selected from test outcomes.

## Paper-aligned session results

| Model | AUROC | AUPRC | Accuracy | Paper weighted accuracy | F1 | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Published UFNet | 0.9281 | — | 0.8729 | 0.8639 | 0.8096 | 0.7838 | 0.9199 |
| Reproduced UFNet, 30-seed mean | 0.9272 | 0.8618 | 0.8706 | 0.8617 | 0.8057 | 0.7779 | 0.9194 |
| Available mean | 0.9321 | 0.8797 | 0.8523 | 0.8394 | 0.7820 | 0.7647 | 0.8992 |
| Scalar MLP | 0.9248 | 0.8731 | 0.8579 | 0.8428 | 0.7941 | 0.7941 | 0.8915 |
| Plain Adapter-Transformer | 0.9340 | 0.8806 | 0.8680 | 0.8540 | 0.8088 | 0.8088 | 0.8992 |
| **Uncertainty Adapter-Transformer (preregistered primary)** | **0.9321** | **0.8821** | **0.8788** | **0.8692** | **0.8182** | **0.7941** | **0.9225** |
| Residual Adapter, non-primary ablation | 0.9341 | 0.8910 | 0.8731 | 0.8578 | 0.8201 | 0.8382 | 0.8915 |
| Uncertainty-gated residual, rejected ablation | 0.9267 | 0.8789 | 0.8680 | 0.8577 | 0.8030 | 0.7794 | 0.9147 |

The preregistered primary model exceeded the reproduced UFNet aggregate by
0.0048 AUROC, 0.0203 AUPRC, 0.0076 accuracy, 0.0125 F1, 0.0162 sensitivity and
0.0031 specificity. These are descriptive point-estimate differences because
the historical 30-seed UFNet run retained aggregate metrics but not row-level
predictions required for a paired interval.

Against available-mean predictions generated from the identical frozen expert
cache, the primary model's AUROC delta was exactly 0.0000 (participant-cluster
95% CI -0.0242 to +0.0237) and AUPRC delta was +0.0024 (95% CI -0.0464 to
+0.0539). Thus the learned model improved the fixed-threshold operating point
and calibration, but did not demonstrate improved ranking over simple averaging.

The residual Adapter obtained the highest test AUROC, but it was a predeclared
non-primary ablation and failed the earlier validation gate. It must not be
promoted after observing test results.

## Decision

**NUMERIC_IMPROVEMENT_NOT_STATISTICALLY_CONFIRMED.** The primary model is a
valid same-data result and numerically exceeds the paper/reproduced UFNet at
100% coverage, but the evidence does not yet establish that a learned Adapter
improves discrimination over a simple expert mean. A new independent cohort is
required for any further architecture promotion; this official test is now
exposed and cannot be reused for model selection.
