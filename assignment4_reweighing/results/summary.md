# Reweighing (AIF360) - results

Mean ± std over 10 seeds; 70/30 stratified split; logistic regression; protected attribute not used as a model input.

DP_Diff = P(fav|unpriv) − P(fav|priv) (ideal 0). DIR = ratio (ideal 1, fair if ≥ 0.8). Consistency: kNN, k = 5 (ideal 1).


## Adult

Adult: favourable = income >50K, protected = sex (privileged: Male). n = 45222, favourable rate = 0.248, privileged share = 0.675.

| Stage | DP_Diff | DIR | Consistency | Accuracy | Balanced_Acc | F1 |
|---|---|---|---|---|---|---|
| Original labels (full data) | -0.199 ± 0.000 | 0.363 ± 0.000 | 0.827 ± 0.000 |  |  |  |
| Training labels after reweighing | -0.000 ± 0.000 | 1.000 ± 0.000 |  |  |  |  |
| Test predictions, no mitigation | -0.180 ± 0.005 | 0.300 ± 0.011 | 0.953 ± 0.001 | 0.847 ± 0.003 | 0.761 ± 0.005 | 0.657 ± 0.007 |
| Test predictions, Reweighing | -0.093 ± 0.004 | 0.567 ± 0.014 | 0.952 ± 0.001 | 0.841 ± 0.002 | 0.745 ± 0.004 | 0.634 ± 0.007 |

Reweighing weights (seed 0): W(s=0, y=0) = 0.849, W(s=0, y=1) = 2.183, W(s=1, y=0) = 1.094, W(s=1, y=1) = 0.793  (s=0 unprivileged, y=1 favourable)


## German

German: favourable = good credit, protected = age (privileged: >25). n = 1000, favourable rate = 0.700, privileged share = 0.810.

| Stage | DP_Diff | DIR | Consistency | Accuracy | Balanced_Acc | F1 |
|---|---|---|---|---|---|---|
| Original labels (full data) | -0.149 ± 0.000 | 0.795 ± 0.000 | 0.746 ± 0.000 |  |  |  |
| Training labels after reweighing | -0.000 ± 0.000 | 1.000 ± 0.000 |  |  |  |  |
| Test predictions, no mitigation | -0.113 ± 0.035 | 0.856 ± 0.044 | 0.822 ± 0.014 | 0.758 ± 0.021 | 0.681 ± 0.028 | 0.834 ± 0.015 |
| Test predictions, Reweighing | -0.075 ± 0.033 | 0.904 ± 0.042 | 0.821 ± 0.017 | 0.757 ± 0.020 | 0.680 ± 0.029 | 0.834 ± 0.014 |

Reweighing weights (seed 0): W(s=0, y=0) = 0.713, W(s=0, y=1) = 1.209, W(s=1, y=0) = 1.105, W(s=1, y=1) = 0.961  (s=0 unprivileged, y=1 favourable)


## COMPAS

COMPAS: favourable = no 2-year recidivism, protected = race (privileged: Caucasian). n = 5278, favourable rate = 0.530, privileged share = 0.398.

| Stage | DP_Diff | DIR | Consistency | Accuracy | Balanced_Acc | F1 |
|---|---|---|---|---|---|---|
| Original labels (full data) | -0.132 ± 0.000 | 0.783 ± 0.000 | 0.658 ± 0.000 |  |  |  |
| Training labels after reweighing | 0.000 ± 0.000 | 1.000 ± 0.000 |  |  |  |  |
| Test predictions, no mitigation | -0.249 ± 0.016 | 0.659 ± 0.021 | 0.965 ± 0.003 | 0.671 ± 0.007 | 0.667 ± 0.007 | 0.703 ± 0.009 |
| Test predictions, Reweighing | -0.246 ± 0.015 | 0.666 ± 0.020 | 0.964 ± 0.003 | 0.671 ± 0.007 | 0.667 ± 0.006 | 0.706 ± 0.008 |

Reweighing weights (seed 0): W(s=0, y=0) = 0.899, W(s=0, y=1) = 1.111, W(s=1, y=0) = 1.204, W(s=1, y=1) = 0.869  (s=0 unprivileged, y=1 favourable)
