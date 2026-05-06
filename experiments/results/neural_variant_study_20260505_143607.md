# Neural Variant Study

## Attack Variants

All metrics are percentages reported as mean +/- std across seeds.

| Attack Variant | Accuracy | Precision | Recall | F1 | False Alarm Rate | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed max-action | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 0.0 +/- 0.0 | 100.0 +/- 0.0 |
| Subtle-action | 50.0 +/- 0.0 | 50.0 +/- 0.0 | 100.0 +/- 0.0 | 66.7 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| Probabilistic | 75.0 +/- 0.0 | 100.0 +/- 0.0 | 50.0 +/- 0.0 | 66.7 +/- 0.0 | 0.0 +/- 0.0 | 50.0 +/- 0.0 |
| Delayed-effect | 25.0 +/- 0.0 | 33.3 +/- 0.0 | 50.0 +/- 0.0 | 40.0 +/- 0.0 | 100.0 +/- 0.0 | 50.0 +/- 0.0 |

## Feature Ablation

All metrics are percentages reported as mean +/- std across seeds.

| Feature Set | Accuracy | Precision | Recall | F1 | AUC |
| --- | --- | --- | --- | --- | --- |
| Full features | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| No safety indicators | 75.0 +/- 0.0 | 66.7 +/- 0.0 | 100.0 +/- 0.0 | 80.0 +/- 0.0 | 100.0 +/- 0.0 |
| No temporal dynamics | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| No correlation features | 75.0 +/- 0.0 | 66.7 +/- 0.0 | 100.0 +/- 0.0 | 80.0 +/- 0.0 | 100.0 +/- 0.0 |
