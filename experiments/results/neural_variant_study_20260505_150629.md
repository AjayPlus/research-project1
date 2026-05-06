# Neural Variant Study

## Attack Variants

All metrics are percentages reported as mean +/- std across seeds.

| Attack Variant | Accuracy | Precision | Recall | F1 | False Alarm Rate | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed max-action | 50.0 +/- 0.0 | 50.0 +/- 0.0 | 100.0 +/- 0.0 | 66.7 +/- 0.0 | 100.0 +/- 0.0 | 14.0 +/- 0.0 |
| Subtle-action | 50.0 +/- 0.0 | 50.0 +/- 0.0 | 100.0 +/- 0.0 | 66.7 +/- 0.0 | 100.0 +/- 0.0 | 14.0 +/- 0.0 |
| Probabilistic | 50.0 +/- 0.0 | 50.0 +/- 0.0 | 100.0 +/- 0.0 | 66.7 +/- 0.0 | 100.0 +/- 0.0 | 14.0 +/- 0.0 |
| Delayed-effect | 50.0 +/- 0.0 | 50.0 +/- 0.0 | 100.0 +/- 0.0 | 66.7 +/- 0.0 | 100.0 +/- 0.0 | 14.0 +/- 0.0 |

## Feature Ablation

All metrics are percentages reported as mean +/- std across seeds.

| Feature Set | Accuracy | Precision | Recall | F1 | AUC |
| --- | --- | --- | --- | --- | --- |
| Full features | 30.0 +/- 0.0 | 37.5 +/- 0.0 | 60.0 +/- 0.0 | 46.2 +/- 0.0 | 13.0 +/- 0.0 |
| No safety indicators | 50.0 +/- 0.0 | 50.0 +/- 0.0 | 100.0 +/- 0.0 | 66.7 +/- 0.0 | 18.0 +/- 0.0 |
| No temporal dynamics | 45.0 +/- 0.0 | 47.4 +/- 0.0 | 90.0 +/- 0.0 | 62.1 +/- 0.0 | 18.0 +/- 0.0 |
| No correlation features | 50.0 +/- 0.0 | 50.0 +/- 0.0 | 100.0 +/- 0.0 | 66.7 +/- 0.0 | 13.0 +/- 0.0 |
