# Neural Variant Study

## Attack Variants

All metrics are percentages reported as mean +/- std across seeds.

| Attack Variant | Accuracy | Precision | Recall | F1 | False Alarm Rate | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed max-action | 96.7 +/- 4.7 | 94.4 +/- 7.9 | 100.0 +/- 0.0 | 97.0 +/- 4.3 | 6.7 +/- 9.4 | 100.0 +/- 0.0 |
| Subtle-action | 98.3 +/- 2.4 | 97.0 +/- 4.3 | 100.0 +/- 0.0 | 98.4 +/- 2.2 | 3.3 +/- 4.7 | 100.0 +/- 0.0 |
| Probabilistic | 98.3 +/- 2.4 | 97.0 +/- 4.3 | 100.0 +/- 0.0 | 98.4 +/- 2.2 | 3.3 +/- 4.7 | 100.0 +/- 0.0 |
| Delayed-effect | 98.3 +/- 2.4 | 97.0 +/- 4.3 | 100.0 +/- 0.0 | 98.4 +/- 2.2 | 3.3 +/- 4.7 | 100.0 +/- 0.0 |

## Feature Ablation

All metrics are percentages reported as mean +/- std across seeds.

| Feature Set | Accuracy | Precision | Recall | F1 | AUC |
| --- | --- | --- | --- | --- | --- |
| Full features | 96.7 +/- 4.7 | 94.4 +/- 7.9 | 100.0 +/- 0.0 | 97.0 +/- 4.3 | 100.0 +/- 0.0 |
| No safety indicators | 96.7 +/- 2.4 | 93.9 +/- 4.3 | 100.0 +/- 0.0 | 96.8 +/- 2.2 | 100.0 +/- 0.0 |
| No temporal dynamics | 96.7 +/- 4.7 | 94.4 +/- 7.9 | 100.0 +/- 0.0 | 97.0 +/- 4.3 | 100.0 +/- 0.0 |
| No correlation features | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
