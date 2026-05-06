# Neural Variant Study

## Attack Variants

All metrics are percentages reported as mean +/- std across seeds.

| Attack Variant | Accuracy | Precision | Recall | F1 | False Alarm Rate | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed max-action | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 0.0 +/- 0.0 | 100.0 +/- 0.0 |
| Subtle-action | 90.0 +/- 0.0 | 83.3 +/- 0.0 | 100.0 +/- 0.0 | 90.9 +/- 0.0 | 20.0 +/- 0.0 | 100.0 +/- 0.0 |
| Probabilistic | 95.0 +/- 0.0 | 90.9 +/- 0.0 | 100.0 +/- 0.0 | 95.2 +/- 0.0 | 10.0 +/- 0.0 | 100.0 +/- 0.0 |
| Delayed-effect | 95.0 +/- 0.0 | 90.9 +/- 0.0 | 100.0 +/- 0.0 | 95.2 +/- 0.0 | 10.0 +/- 0.0 | 100.0 +/- 0.0 |

## Feature Ablation

All metrics are percentages reported as mean +/- std across seeds.

| Feature Set | Accuracy | Precision | Recall | F1 | AUC |
| --- | --- | --- | --- | --- | --- |
| Full features | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| No safety indicators | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| No temporal dynamics | 95.0 +/- 0.0 | 90.9 +/- 0.0 | 100.0 +/- 0.0 | 95.2 +/- 0.0 | 100.0 +/- 0.0 |
| No correlation features | 95.0 +/- 0.0 | 90.9 +/- 0.0 | 100.0 +/- 0.0 | 95.2 +/- 0.0 | 100.0 +/- 0.0 |
