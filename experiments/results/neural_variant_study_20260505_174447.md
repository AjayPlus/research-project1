# Neural Variant Study

## Attack Variants

All metrics are percentages reported as mean +/- std across seeds.

| Attack Variant | Accuracy | Precision | Recall | F1 | False Alarm Rate | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed max-action | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 0.0 +/- 0.0 | 100.0 +/- 0.0 |
| Subtle-action | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 0.0 +/- 0.0 | 100.0 +/- 0.0 |
| Probabilistic | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 0.0 +/- 0.0 | 100.0 +/- 0.0 |
| Delayed-effect | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 0.0 +/- 0.0 | 100.0 +/- 0.0 |

## Feature Ablation

All metrics are percentages reported as mean +/- std across seeds.

| Feature Set | Accuracy | Precision | Recall | F1 | AUC |
| --- | --- | --- | --- | --- | --- |
| Full features | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| No safety indicators | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| No temporal dynamics | 98.3 +/- 2.4 | 97.0 +/- 4.3 | 100.0 +/- 0.0 | 98.4 +/- 2.2 | 100.0 +/- 0.0 |
| No correlation features | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
