# Phase 2.1 constrained residual Adapter audit

Status: **PASS**
Decision: **REJECT_FEATURE_RESIDUAL**

The run used only the 690-session/516-participant training cache for candidate
selection. Five participant-disjoint folds were used, with a separate tuning
subset inside every fold for epoch selection. The frozen candidates were then
checked on 215 validation sessions from 167 participants. The official test
cache was not loaded.

One longitudinal training participant had a control label in 2020 and a PD
label in 2022. Both sessions remained in the same fold and retained their
time-specific labels. The participant's ever-PD label was used only for fold
stratification. Metrics are session-level and paired bootstrap resampling is
clustered by participant.

## Train-only OOF selection

| Model | Selected residual scale | AUROC | AUPRC |
| --- | ---: | ---: | ---: |
| Scalar MLP | 0.00 | 0.948842 | 0.932889 |
| Residual Adapter | 0.25 | 0.953576 | 0.941760 |
| Uncertainty-gated residual Adapter | 0.10 | 0.950148 | 0.937458 |

## Frozen validation check

| Model | AUROC | AUPRC | Accuracy | Balanced accuracy | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scalar MLP | **0.930589** | 0.890905 | 0.846512 | **0.857235** | 0.817680 |
| Residual Adapter | 0.924170 | 0.894418 | 0.851163 | 0.846965 | 0.809524 |
| Uncertainty-gated residual Adapter | 0.924904 | **0.896464** | **0.860465** | 0.854484 | **0.819277** |

The residual Adapter AUROC delta versus Scalar MLP was -0.006418 (participant-
cluster 95% CI -0.023608 to +0.011438). The uncertainty-gated residual delta was
-0.005685 (95% CI -0.026121 to +0.013660). The small validation AUPRC increases
also had intervals crossing zero. At participant aggregation, Scalar MLP AUROC
was 0.948539 versus 0.940097 and 0.938799 for the gated and plain residual
models, respectively.

All aggregate and private prediction CSVs were byte-identical in a complete
same-configuration repeat. The negative validation result is therefore not a
run-to-run stochastic accident. High-dimensional residual Feature Adapters do
not pass the frozen promotion criterion; the official test remains sealed.

The next fusion stage should use expert probabilities and uncertainty summaries
as its primary interface. Any Agent/LLM experiment should first be required to
beat the Scalar MLP under the same train-only selection protocol.
