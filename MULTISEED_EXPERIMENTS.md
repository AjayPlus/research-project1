# Multi-Seed Experiments and Baseline Comparisons

This document describes the enhanced experimental framework for backdoor detection research, including multi-seed experiments, proper data splitting, and baseline detector comparisons.

## Table of Contents
- [Overview](#overview)
- [Multi-Seed Experiments](#multi-seed-experiments)
- [Data Splitting Strategy](#data-splitting-strategy)
- [Baseline Detectors](#baseline-detectors)
- [Running Experiments](#running-experiments)
- [Interpreting Results](#interpreting-results)
- [References](#references)

---

## Overview

The enhanced experimental framework addresses key requirements for statistically robust backdoor detection research:

1. **Multi-seed experiments**: Run trials with multiple random seeds to ensure statistical validity
2. **Proper data splitting**: Train/validation/test splits with stratification to avoid data leakage
3. **Baseline comparisons**: Compare against simple baselines and state-of-the-art methods
4. **Aggregated metrics**: Report mean ± std across all trials

---

## Multi-Seed Experiments

### Motivation

Single-run experiments can be misleading due to random variation. Multi-seed experiments provide:
- **Statistical validity**: Confidence in results through multiple trials
- **Variance estimation**: Understanding of method stability
- **Publication readiness**: Standard practice in ML research

### Implementation

The `run_experiment_multiseed.py` script runs experiments across multiple random seeds:

```python
from experiments.run_experiment_multiseed import MultiSeedExperimentRunner

runner = MultiSeedExperimentRunner(
    num_seeds=10,           # Number of trials
    start_seed=42,          # Starting seed (seeds: 42-51)
    train_ratio=0.6,        # 60% training
    val_ratio=0.2,          # 20% validation
    test_ratio=0.2,         # 20% testing
    include_baselines=True  # Include baseline detectors
)

results = runner.run()
```

### Seed Management

All randomness is controlled via `set_seed()` function:

```python
from src.utils import set_seed

set_seed(42)
# All subsequent random operations are deterministic:
# - Python random
# - NumPy
# - PyTorch (CPU and CUDA)
# - Gymnasium environments
```

### Results Format

Multi-seed results have a new format with mean ± std:

```json
{
  "zscore": {
    "accuracy_mean": 0.9949,
    "accuracy_std": 0.0023,
    "f1_mean": 0.9962,
    "f1_std": 0.0018,
    "accuracy_all_trials": [0.9945, 0.9952, 0.9950, ...],
    "f1_all_trials": [0.9960, 0.9965, 0.9961, ...]
  }
}
```

---

## Data Splitting Strategy

### Train/Validation/Test Split (60/20/20)

Proper data splitting is critical for:
- **Avoiding data leakage**: Test data must never influence training
- **Hyperparameter tuning**: Validation set for threshold optimization
- **Fair evaluation**: Independent test set for final metrics

### Implementation

The `StratifiedDataSplitter` class handles data splitting with stratification:

```python
from src.utils import StratifiedDataSplitter

splitter = StratifiedDataSplitter(
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    random_seed=42
)

# Split features
splits = splitter.split_features(clean_features, backdoor_features)

# Access splits
train_features = splits['train']['features']
train_labels = splits['train']['labels']
val_features = splits['val']['features']
val_labels = splits['val']['labels']
test_features = splits['test']['features']
test_labels = splits['test']['labels']
```

### Stratification

All splits maintain class balance:
- Each split has proportional representation of clean and backdoor samples
- Prevents bias from unbalanced splits
- Ensures reliable metric estimation

### Split Statistics

The splitter provides detailed statistics:

```python
print(splits['train']['stats'])
# {
#   'total_samples': 12000,
#   'clean_samples': 6000,
#   'backdoor_samples': 6000,
#   'backdoor_ratio': 0.5,
#   'clean_ratio': 0.5
# }
```

---

## Baseline Detectors

### Simple Baselines

#### 1. Random Detector
**Purpose**: Lower bound on performance - what random guessing achieves

**Implementation**:
```python
from src.detection import RandomDetector

detector = RandomDetector(backdoor_prob=0.5, random_seed=42)
predictions = detector.predict(test_features)
```

**Expected Performance**:
- Accuracy: ~50%
- F1: ~0.50
- FAR: ~50%

#### 2. Always Detect
**Purpose**: Upper bound on detection rate, lower bound on precision

**Implementation**:
```python
from src.detection import AlwaysDetectDetector

detector = AlwaysDetectDetector()
predictions = detector.predict(test_features)
```

**Expected Performance**:
- Recall: 100%
- FAR: 100% (flags all clean samples as backdoor)
- Precision: Depends on class balance

#### 3. Never Detect
**Purpose**: Lower bound on FAR, shows cost of ignoring backdoors

**Implementation**:
```python
from src.detection import NeverDetectDetector

detector = NeverDetectDetector()
predictions = detector.predict(test_features)
```

**Expected Performance**:
- FAR: 0%
- Recall: 0% (misses all backdoor samples)
- High cost from missed attacks

---

### Advanced Baselines

#### 4. Activation Clustering (Chen et al., AAAI 2019)

**Reference**:
- Chen et al., "Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering"
- AAAI SafeAI Workshop 2019
- arXiv: https://arxiv.org/abs/1811.03728
- Citations: ~1000+

**How It Works**:
1. Apply PCA to reduce features to 10 components
2. Cluster using k-means (k=2)
3. Identify poisoned cluster using size heuristic
4. Classify based on cluster assignment

**Implementation**:
```python
from src.detection import ActivationClusteringDetector

detector = ActivationClusteringDetector(n_components=10, random_seed=42)
detector.fit(train_features)
predictions = detector.predict(test_features)
scores = detector.decision_function(test_features)
```

**Adaptation to RL**:
- Original method designed for neural network activations
- We adapt to trajectory-based features (45-dim vectors)
- Cluster on extracted features instead of layer activations

**Strengths**:
- Unsupervised - doesn't need labeled backdoor data
- Computationally efficient
- Interpretable cluster assignments

**Limitations**:
- Assumes backdoor samples form distinct cluster
- Sensitive to feature quality
- Fixed k=2 may not suit all scenarios

---

#### 5. Spectral Signatures (Tran et al., NeurIPS 2018)

**Reference**:
- Tran et al., "Spectral Signatures in Backdoor Attacks"
- NeurIPS 2018
- arXiv: https://arxiv.org/abs/1811.00636
- Citations: ~800+

**How It Works**:
1. Compute covariance matrix of feature representations
2. Apply SVD (Singular Value Decomposition)
3. Detect spectral anomalies in top singular vectors
4. Flag samples with high projections on anomalous directions

**Implementation**:
```python
from src.detection import SpectralSignaturesDetector

detector = SpectralSignaturesDetector(
    n_components=1,
    outlier_percentile=95.0,
    random_seed=42
)
detector.fit(train_features)
predictions = detector.predict(test_features)
scores = detector.decision_function(test_features)
```

**Key Insight**:
- Backdoor attacks leave "spectral signatures" in data covariance
- Poisoned samples align with top singular vectors
- Outlier detection on spectral projections reveals backdoors

**Strengths**:
- Theoretically grounded in spectral analysis
- Effective for large-scale backdoor attacks
- Unsupervised method

**Limitations**:
- Requires sufficient backdoor samples in training
- Sensitive to covariance estimation quality
- May struggle with subtle backdoors

---

### RL-Specific Baselines (Future Work)

#### PolicyCleanse (Guo et al., AAAI 2024)

**Reference**:
- Guo et al., "PolicyCleanse: Backdoor Detection and Mitigation for Competitive Reinforcement Learning"
- AAAI 2024
- arXiv: https://arxiv.org/abs/2202.03609

**How It Works**:
- Monitors accumulated reward patterns over time
- Detects anomalous reward degradation after trigger activation
- Specialized for RL backdoor detection

**Status**: Not yet implemented (future baseline)

---

#### BIRD (NeurIPS 2023)

**Reference**:
- "BIRD: Generalizable Backdoor Detection and Removal for Deep Reinforcement Learning"
- NeurIPS 2023
- OpenReview: https://openreview.net/forum?id=l3yxZS3QdT

**How It Works**:
- Formulates trigger restoration as optimization problem
- Works in clean environment without attack knowledge
- Detects backdoors + removes them via fine-tuning

**Why Important**:
- State-of-the-art for RL backdoor detection (2023)
- Generalizable across attack types
- Industry benchmark

**Status**: Not yet implemented (future baseline)

**Comparison Strategy**:
When implemented, compare against BIRD:
- **BIRD**: Optimization-based, computationally expensive
- **Our methods**: Fast statistical detection
- **Positioning**: "While BIRD achieves strong performance, our lightweight methods offer real-time detection with comparable accuracy"

---

## Running Experiments

### Quick Start

Run a full multi-seed experiment:

```bash
cd experiments
python run_experiment_multiseed.py
```

### Custom Configuration

Customize experiment parameters:

```python
from experiments.run_experiment_multiseed import MultiSeedExperimentRunner

runner = MultiSeedExperimentRunner(
    n_clean_episodes=100,      # Clean trajectories to collect
    n_backdoor_episodes=100,   # Backdoor trajectories to collect
    train_episodes=500,        # Episodes to train each agent
    window_size=12,            # Feature extraction window
    train_ratio=0.6,           # Training set proportion
    val_ratio=0.2,             # Validation set proportion
    test_ratio=0.2,            # Test set proportion
    num_seeds=10,              # Number of trials
    start_seed=42,             # Starting seed
    results_dir='results',     # Output directory
    include_baselines=True     # Include baseline methods
)

results = runner.run()
```

### Expected Runtime

For 10 seeds with default parameters:
- Agent training: ~2-3 minutes per seed
- Data collection: ~30 seconds per seed
- Detector evaluation: ~1 minute per seed
- **Total**: ~40-50 minutes for full experiment

---

## Interpreting Results

### Results Structure

Multi-seed results contain three sections:

```json
{
  "experiment_config": {
    "num_seeds": 10,
    "seeds": [42, 43, 44, ...],
    "train_ratio": 0.6,
    "val_ratio": 0.2,
    "test_ratio": 0.2
  },
  "aggregated_results": {
    "zscore": {
      "accuracy_mean": 0.9949,
      "accuracy_std": 0.0023,
      ...
    }
  },
  "individual_trials": [...]
}
```

### Key Metrics

For each detector, we report:

1. **Accuracy**: Overall classification accuracy
   - `(TP + TN) / (TP + TN + FP + FN)`

2. **F1 Score**: Harmonic mean of precision and recall
   - `2 * (precision * recall) / (precision + recall)`

3. **False Alarm Rate (FAR)**: Proportion of clean samples misclassified
   - `FP / (FP + TN)`
   - Critical for deployment - high FAR causes alert fatigue

4. **Detection Rate (TPR)**: Proportion of backdoor samples caught
   - `TP / (TP + FN)`
   - Critical for security - low TPR means missed attacks

5. **AUC**: Area under ROC curve
   - Aggregate measure across all thresholds

### Visualization

Generate visualizations:

```bash
cd experiments
python visualize_multiseed_results.py
```

This creates:
- `multiseed_comparison.png`: 4-panel comparison of all metrics
- `multiseed_accuracy.png`: Accuracy comparison with error bars
- `multiseed_f1.png`: F1 score comparison
- `baseline_vs_advanced.png`: Baseline vs advanced methods
- `trial_variance_accuracy.png`: Box plots showing trial variance

### Statistical Significance

When comparing methods:
1. Check if confidence intervals (mean ± 2×std) overlap
2. Non-overlapping intervals suggest significant difference
3. For formal tests, use paired t-tests on trial results

---

## Best Practices

### 1. Reproducibility
Always set and report random seeds:
```python
from src.utils import set_seed
set_seed(42)  # Document this in your paper/report
```

### 2. Statistical Reporting
Report results as mean ± std:
- "Our method achieves 99.49% ± 0.23% accuracy"
- Include standard deviation for transparency

### 3. Baseline Comparison
Always compare against:
- Random baseline (sanity check)
- Simple baselines (demonstrate value of complexity)
- Published methods (show advancement over state-of-the-art)

### 4. Data Splitting
Use proper train/val/test splits:
- Train: Fit detector parameters
- Validation: Tune hyperparameters (thresholds)
- Test: Final evaluation (never seen during training/tuning)

### 5. Multiple Metrics
Report multiple metrics to show tradeoffs:
- High accuracy but high FAR? Not practical
- Perfect recall but low precision? Too many false alarms
- F1 balances both precision and recall

---

## Common Pitfalls

### ❌ Single-seed experiments
**Problem**: Results may not generalize
**Solution**: Use multi-seed experiments

### ❌ Train/test-only split
**Problem**: Threshold tuning on test set causes overfitting
**Solution**: Use three-way split with validation

### ❌ Comparing without baselines
**Problem**: Can't assess if complexity is justified
**Solution**: Include simple baselines

### ❌ Cherry-picking seeds
**Problem**: Reporting only best trial is dishonest
**Solution**: Pre-specify seeds, report all results

### ❌ Ignoring variance
**Problem**: High variance indicates unstable method
**Solution**: Report and analyze standard deviation

---

## References

### Baseline Methods

1. **Activation Clustering**:
   - Chen et al., "Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering"
   - AAAI SafeAI Workshop 2019
   - arXiv: https://arxiv.org/abs/1811.03728

2. **Spectral Signatures**:
   - Tran et al., "Spectral Signatures in Backdoor Attacks"
   - NeurIPS 2018
   - arXiv: https://arxiv.org/abs/1811.00636

3. **PolicyCleanse**:
   - Guo et al., "PolicyCleanse: Backdoor Detection and Mitigation for Competitive Reinforcement Learning"
   - AAAI 2024
   - arXiv: https://arxiv.org/abs/2202.03609

4. **BIRD**:
   - "BIRD: Generalizable Backdoor Detection and Removal for Deep Reinforcement Learning"
   - NeurIPS 2023
   - OpenReview: https://openreview.net/forum?id=l3yxZS3QdT

### Best Practices

5. **Reporting Guidelines**:
   - Dodge et al., "Show Your Work: Improved Reporting of Experimental Results"
   - EMNLP 2019
   - Emphasizes multi-seed experiments and variance reporting

6. **Hyperparameter Optimization**:
   - Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization"
   - JMLR 2012
   - Discusses proper train/val/test splits

---

## File Structure

```
research-project1/
├── src/
│   ├── utils/
│   │   ├── seed_utils.py              # Seed management
│   │   └── data_splitter.py           # Train/val/test splitting
│   └── detection/
│       └── baseline_detectors.py      # Baseline detector implementations
├── experiments/
│   ├── run_experiment_multiseed.py    # Multi-seed experiment runner
│   └── visualize_multiseed_results.py # Visualization script
└── results/
    ├── multiseed_results_*.json       # Multi-seed results
    └── visualizations/                # Generated plots
```

---

## Next Steps

1. **Run multi-seed experiments**:
   ```bash
   python experiments/run_experiment_multiseed.py
   ```

2. **Generate visualizations**:
   ```bash
   python experiments/visualize_multiseed_results.py
   ```

3. **Analyze results**:
   - Compare baselines vs your methods
   - Identify best-performing detector
   - Understand variance and stability

4. **Future work**:
   - Implement PolicyCleanse baseline
   - Implement BIRD baseline
   - Add more RL-specific baselines
   - Extend to other backdoor attack types

---

For questions or issues, please refer to the main README.md or open an issue in the repository.
