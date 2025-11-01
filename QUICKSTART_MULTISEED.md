# Quick Start Guide: Multi-Seed Experiments

This guide will help you run your first multi-seed experiment in under 5 minutes.

## Step 1: Verify Installation

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Step 2: Run Multi-Seed Experiment

Run a quick 3-seed experiment (for testing):

```bash
cd experiments
python -c "
from run_experiment_multiseed import MultiSeedExperimentRunner

runner = MultiSeedExperimentRunner(
    n_clean_episodes=50,       # Reduced for quick test
    n_backdoor_episodes=50,    # Reduced for quick test
    train_episodes=100,        # Reduced for quick test
    num_seeds=3,               # Just 3 seeds for quick test
    start_seed=42,
    include_baselines=True
)
runner.run()
"
```

**Expected runtime**: ~10-15 minutes

## Step 3: Run Full Experiment

For publication-quality results (10 seeds):

```bash
cd experiments
python run_experiment_multiseed.py
```

**Expected runtime**: ~40-50 minutes

## Step 4: Visualize Results

Generate all visualizations:

```bash
cd experiments
python visualize_multiseed_results.py
```

This creates visualizations in `results/visualizations/`:
- `multiseed_comparison.png` - 4-panel comparison
- `multiseed_accuracy.png` - Accuracy with error bars
- `multiseed_f1.png` - F1 scores
- `baseline_vs_advanced.png` - Baseline comparison
- `trial_variance_accuracy.png` - Variance analysis

## Step 5: Check Results

Results are saved in `results/multiseed_results_TIMESTAMP.json`

Example output:
```
AGGREGATED RESULTS SUMMARY
======================================================================
Results aggregated over 10 trials

Detection Method Comparison (Mean ± Std):
Method                    Accuracy             F1 Score             FAR
-------------------------------------------------------------------------------------
Activation Clustering     0.9823 ± 0.0156      0.9845 ± 0.0142      0.0234 ± 0.0189
Always Detect             0.5000 ± 0.0000      0.6667 ± 0.0000      1.0000 ± 0.0000
Isolation Forest          0.9719 ± 0.0089      0.9792 ± 0.0076      0.0640 ± 0.0112
Mahalanobis               1.0000 ± 0.0000      1.0000 ± 0.0000      0.0000 ± 0.0000
Neural Autoencoder        1.0000 ± 0.0000      1.0000 ± 0.0000      0.0000 ± 0.0000
Never Detect              0.5000 ± 0.0000      0.0000 ± 0.0000      0.0000 ± 0.0000
Random                    0.4987 ± 0.0123      0.4992 ± 0.0118      0.5013 ± 0.0123
Spectral Signatures       0.9567 ± 0.0234      0.9634 ± 0.0198      0.0867 ± 0.0312
Threshold Based           0.3637 ± 0.0045      0.1005 ± 0.0034      0.0155 ± 0.0023
Zscore                    0.9982 ± 0.0008      0.9986 ± 0.0007      0.0053 ± 0.0011
```

## What's New?

### 1. Multi-Seed Experiments
- Runs 10 trials with different random seeds (42-51)
- Reports mean ± std for all metrics
- Ensures statistical validity

### 2. Train/Val/Test Split (60/20/20)
- Proper data splitting with stratification
- Prevents data leakage
- Validation set for threshold tuning
- Test set for final evaluation

### 3. Baseline Detectors

**Simple Baselines**:
- `Random`: Random predictions (~50% accuracy)
- `Always Detect`: Always predicts backdoor (100% recall, 100% FAR)
- `Never Detect`: Never predicts backdoor (0% recall, 0% FAR)

**Advanced Baselines**:
- `Activation Clustering`: Chen et al., AAAI 2019
- `Spectral Signatures`: Tran et al., NeurIPS 2018

**Your Methods**:
- `Z-Score`: Statistical outlier detection
- `Mahalanobis`: Distance-based detection
- `Isolation Forest`: Tree-based anomaly detection
- `Neural Autoencoder`: Reconstruction error-based detection
- `Threshold Based`: Domain-specific rules

### 4. Updated Results Format

Old format (single trial):
```json
{
  "accuracy": 0.9949,
  "f1": 0.9962
}
```

New format (multi-trial):
```json
{
  "accuracy_mean": 0.9949,
  "accuracy_std": 0.0023,
  "f1_mean": 0.9962,
  "f1_std": 0.0018,
  "accuracy_all_trials": [0.9945, 0.9952, 0.9950, ...],
  "f1_all_trials": [0.9960, 0.9965, 0.9961, ...]
}
```

## Customization

### Adjust Number of Seeds

```python
runner = MultiSeedExperimentRunner(
    num_seeds=5,        # Run fewer seeds for quick tests
    start_seed=42
)
```

### Change Data Split Ratios

```python
runner = MultiSeedExperimentRunner(
    train_ratio=0.7,    # 70% training
    val_ratio=0.15,     # 15% validation
    test_ratio=0.15     # 15% testing
)
```

### Disable Baselines (Faster)

```python
runner = MultiSeedExperimentRunner(
    include_baselines=False  # Skip baseline methods
)
```

### Reduce Data for Quick Testing

```python
runner = MultiSeedExperimentRunner(
    n_clean_episodes=30,
    n_backdoor_episodes=30,
    train_episodes=100,
    num_seeds=2
)
```

## Troubleshooting

### "Out of memory" error
Reduce the number of episodes or use smaller batches:
```python
runner = MultiSeedExperimentRunner(
    n_clean_episodes=50,      # Reduced from 100
    n_backdoor_episodes=50    # Reduced from 100
)
```

### Experiments taking too long
1. Reduce number of seeds: `num_seeds=3`
2. Reduce training episodes: `train_episodes=100`
3. Disable baselines: `include_baselines=False`

### Import errors
Make sure you're running from the project root or experiments directory:
```bash
cd /path/to/research-project1/experiments
python run_experiment_multiseed.py
```

## Next Steps

1. **Read the full documentation**: See `MULTISEED_EXPERIMENTS.md` for details
2. **Analyze your results**: Use the visualizations to compare methods
3. **Write your paper**: Report mean ± std for all metrics
4. **Future work**: Consider implementing PolicyCleanse or BIRD baselines

## File Reference

- `experiments/run_experiment_multiseed.py` - Main experiment runner
- `experiments/visualize_multiseed_results.py` - Visualization script
- `src/utils/seed_utils.py` - Seed management
- `src/utils/data_splitter.py` - Train/val/test splitting
- `src/detection/baseline_detectors.py` - Baseline implementations
- `MULTISEED_EXPERIMENTS.md` - Full documentation

---

Happy experimenting! For detailed information, see `MULTISEED_EXPERIMENTS.md`.
