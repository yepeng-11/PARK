# Phase 2 adapter development audit

Status: **PASS**
Decision: **PROMISING_NOT_CONFIRMED**

This is a development result on the 167-participant validation split. The test
cache was not loaded. All aggregate CSV outputs and participant predictions were
exactly reproduced in a complete same-configuration repeat: **True**.

The uncertainty-aware Adapter-Transformer is promising against available-mean,
but it is not yet confirmed against the strongest lightweight comparator
(scalar MLP). Promotion requires both AUROC and AUPRC non-degradation and a
positive participant-bootstrap AUROC lower bound.

## Paired bootstrap comparisons

                               candidate                    reference  AUROC_delta  AUROC_CI_low  AUROC_CI_high  AUPRC_delta  AUPRC_CI_low  AUPRC_CI_high
                     scalar_mlp_ensemble               available_mean     0.010227     -0.000163       0.023159     0.017916      0.000002       0.039656
                     concat_mlp_ensemble               available_mean    -0.057792     -0.107803      -0.015688    -0.105497     -0.179568      -0.021357
            adapter_transformer_ensemble               available_mean    -0.011688     -0.038475       0.016221    -0.028353     -0.080717       0.019853
uncertainty_adapter_transformer_ensemble               available_mean     0.015909     -0.001571       0.035470     0.017092     -0.017730       0.055394
uncertainty_adapter_transformer_ensemble          scalar_mlp_ensemble     0.005682     -0.009582       0.023257    -0.000823     -0.038672       0.035870
uncertainty_adapter_transformer_ensemble adapter_transformer_ensemble     0.027597      0.009460       0.048247     0.045445      0.012500       0.086724
