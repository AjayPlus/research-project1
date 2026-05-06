# Neural Variant Study

## Attack Variants

All metrics are percentages reported as mean +/- std across seeds.

| Attack Variant | Accuracy | Precision | Recall | F1 | False Alarm Rate | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed max-action | 37.5 +/- 0.0 | 42.9 +/- 0.0 | 75.0 +/- 0.0 | 54.5 +/- 0.0 | 100.0 +/- 0.0 | 6.2 +/- 0.0 |
| Subtle-action | 37.5 +/- 0.0 | 42.9 +/- 0.0 | 75.0 +/- 0.0 | 54.5 +/- 0.0 | 100.0 +/- 0.0 | 6.2 +/- 0.0 |
| Probabilistic | 37.5 +/- 0.0 | 42.9 +/- 0.0 | 75.0 +/- 0.0 | 54.5 +/- 0.0 | 100.0 +/- 0.0 | 6.2 +/- 0.0 |
| Delayed-effect | 37.5 +/- 0.0 | 42.9 +/- 0.0 | 75.0 +/- 0.0 | 54.5 +/- 0.0 | 100.0 +/- 0.0 | 6.2 +/- 0.0 |

## Feature Ablation

All metrics are percentages reported as mean +/- std across seeds.

| Feature Set | Accuracy | Precision | Recall | F1 | AUC |
| --- | --- | --- | --- | --- | --- |
| Full features | 37.5 +/- 0.0 | 42.9 +/- 0.0 | 75.0 +/- 0.0 | 54.5 +/- 0.0 | 6.2 +/- 0.0 |
| No safety indicators | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 |
| No temporal dynamics | 50.0 +/- 0.0 | 50.0 +/- 0.0 | 100.0 +/- 0.0 | 66.7 +/- 0.0 | 0.0 +/- 0.0 |
| No correlation features | 50.0 +/- 0.0 | 50.0 +/- 0.0 | 100.0 +/- 0.0 | 66.7 +/- 0.0 | 0.0 +/- 0.0 |
