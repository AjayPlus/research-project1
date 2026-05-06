# Neural Variant Study

## Attack Variants

All metrics are percentages reported as mean +/- std across seeds.

| Attack Variant | Accuracy | Precision | Recall | F1 | False Alarm Rate | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed max-action | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 0.0 +/- 0.0 | 100.0 +/- 0.0 |
| Subtle-action | 98.3 +/- 2.4 | 97.0 +/- 4.3 | 100.0 +/- 0.0 | 98.4 +/- 2.2 | 3.3 +/- 4.7 | 100.0 +/- 0.0 |
| Probabilistic | 95.0 +/- 4.1 | 91.4 +/- 6.8 | 100.0 +/- 0.0 | 95.4 +/- 3.7 | 10.0 +/- 8.2 | 100.0 +/- 0.0 |
| Delayed-effect | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 0.0 +/- 0.0 | 100.0 +/- 0.0 |

## Feature Ablation

All metrics are percentages reported as mean +/- std across seeds.

| Feature Set | Accuracy | Precision | Recall | F1 | AUC |
| --- | --- | --- | --- | --- | --- |
| Full features | 98.3 +/- 2.4 | 97.0 +/- 4.3 | 100.0 +/- 0.0 | 98.4 +/- 2.2 | 100.0 +/- 0.0 |
| No safety indicators | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| No temporal dynamics | 95.0 +/- 4.1 | 91.4 +/- 6.8 | 100.0 +/- 0.0 | 95.4 +/- 3.7 | 100.0 +/- 0.0 |
| No correlation features | 98.3 +/- 2.4 | 97.0 +/- 4.3 | 100.0 +/- 0.0 | 98.4 +/- 2.2 | 100.0 +/- 0.0 |
