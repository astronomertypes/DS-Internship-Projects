# Results

Auto-generated from `docs/results.json`. Do not edit by hand.

## Churn model (LightGBM)

| Metric | Value |
|---|---|
| CV ROC-AUC mean | 0.8432 |
| CV ROC-AUC std | 0.0099 |
| CV PR-AUC mean | 0.6564 |
| CV PR-AUC std | 0.0195 |
| Hold-out ROC-AUC | 0.8404 |
| Hold-out PR-AUC | 0.651 |
| Hold-out Brier score | 0.1376 |
| Positive rate | 0.2654 |

## Survival (Lifelines)

| Metric | Value |
|---|---|
| KM median survival (months; None = more than half never churned) | None |
| Log-rank p-value (contract) | 0.0 |
| Cox concordance (train) | 0.8654 |
| Cox concordance (held-out) | 0.8554 |
| Customers used | 7032 |

## Uplift - Hillstrom (real randomized experiment)

| Metric | Value |
|---|---|
| Qini coefficient - T-learner | 34.4734 |
| Qini coefficient - S-learner | 53.1074 |
| Qini coefficient - Random | -9.6001 |
| Average uplift (visit rate) | 0.0609 |

## Uplift - Telco (SYNTHETIC treatment)

| Metric | Value |
|---|---|
| Spearman (estimated vs true uplift) | 0.7302 |
| Qini coefficient - uplift model | 79.767 |
| Qini coefficient - churn-risk targeting | 105.2609 |
| Qini coefficient - random | 2.4376 |
| Reference discount (%) | 20 |

## CLV and segmentation

| Metric | Value |
|---|---|
| Margin assumption | 0.3 |
| Annual discount rate | 0.1 |
| Horizon (months) | 24 |
| Mean CLV | 345.0783 |
| Median CLV | 339.2171 |
| Chosen k | 3 |
| Silhouette (chosen k) | 0.3944 |
