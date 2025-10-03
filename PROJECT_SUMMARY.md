# Backdoor Detection Research Project - Setup Complete

## Project Overview

This project implements a complete experimental framework for detecting backdoored behavior in reinforcement learning agents controlling EV charging in a simulated power grid.

## What Has Been Created

### 1. Core Components

#### Environment (`src/environment/`)
- **EVChargingEnv**: Gymnasium-compatible power grid simulation
  - State space: time, grid load, EVs waiting, charging status, sensor readings
  - Action space: number of EVs to charge (0-10)
  - Reward: balance charging demand vs. grid safety

#### Agents (`src/agents/`)
- **DQNAgent**: Clean Deep Q-Network implementation
- **BackdooredDQNAgent**: Agent with planted backdoor
  - Trigger: specific time window + grid load + sensor patterns
  - Malicious behavior: forces maximum charging when triggered

#### Detection (`src/detection/`)
- **Feature Extraction**: Time-window statistical features
  - 12-timestep rolling window (1 hour simulation time)
  - ~40 features including means, stds, correlations, change rates

- **Statistical Detectors**:
  - Z-Score: Simple threshold-based detection
  - Mahalanobis: Correlation-aware distance metric
  - Isolation Forest: Tree-based anomaly detection
  - Threshold-Based: Domain-specific rules
  - Ensemble: Weighted combination

- **Neural Detector**:
  - Autoencoder trained on normal behavior
  - Reconstruction error for anomaly scoring
  - Lightweight architecture (64-32-16 hidden dims)

#### Utilities (`src/utils/`)
- **DetectionMetrics**: Comprehensive evaluation
  - Accuracy, Precision, Recall, F1
  - False Alarm Rate, Detection Rate
  - AUC-ROC, confusion matrices
  - Detection speed and lag metrics

### 2. Experiment Scripts (`experiments/`)

- **run_experiment.py**: Main experiment runner
  - Trains clean and backdoored agents
  - Collects trajectories
  - Extracts features
  - Evaluates all detection methods
  - Saves results to JSON

- **train_agents.py**: Train individual agents
  - Flexible command-line interface
  - Save checkpoints for later use

- **visualize_results.py**: Generate plots
  - Detection method comparison
  - Training curves
  - Confusion matrices
  - Summary tables

### 3. Utilities

- **test_setup.py**: Verify installation and imports
- **quickstart.sh**: One-command setup and run
- **requirements.txt**: All dependencies
- **README.md**: Comprehensive documentation

## File Structure

```
research-project/
├── src/
│   ├── environment/
│   │   ├── __init__.py
│   │   └── ev_charging_env.py          # Power grid simulation
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── dqn_agent.py                # Clean RL agent
│   │   └── backdoored_agent.py         # Backdoored agent
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── feature_extraction.py       # Time-window features
│   │   ├── statistical_detector.py     # 5 statistical methods
│   │   └── neural_detector.py          # Neural network detector
│   └── utils/
│       ├── __init__.py
│       └── metrics.py                  # Evaluation metrics
├── experiments/
│   ├── __init__.py
│   ├── run_experiment.py               # Main experiment
│   ├── train_agents.py                 # Train individual agents
│   └── visualize_results.py            # Plot results
├── test_setup.py                       # Test installation
├── quickstart.sh                       # Quick start script
├── requirements.txt                    # Dependencies
├── README.md                           # Documentation
├── PROJECT_SUMMARY.md                  # This file
└── .gitignore

Directories created during execution:
├── checkpoints/                        # Saved agent weights
├── results/                            # Experiment results + plots
└── venv/                               # Virtual environment
```

## How to Use

### Quick Start (Recommended)

```bash
cd /Users/lucyzimmerman/Desktop/research-project
./quickstart.sh
```

This will:
1. Create virtual environment
2. Install dependencies
3. Test setup
4. Run full experiment
5. Generate visualizations

### Manual Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test setup
python test_setup.py

# 4. Run experiment
cd experiments
python run_experiment.py

# 5. Visualize results
python visualize_results.py
```

### Training Individual Agents

```bash
# Train clean agent (1000 episodes)
python experiments/train_agents.py --agent clean --episodes 1000

# Train backdoored agent (1000 episodes)
python experiments/train_agents.py --agent backdoored --episodes 1000
```

## Experiment Flow

1. **Training Phase** (Step 1-2)
   - Train clean DQN agent on EV charging task
   - Train backdoored agent (same task, but with hidden trigger)

2. **Data Collection** (Step 3-4)
   - Collect 100 trajectories from clean agent
   - Collect 100 trajectories from backdoored agent
   - Each trajectory = one episode (~288 timesteps = 24 simulated hours)

3. **Feature Extraction** (Step 5)
   - Process trajectories into time-window features
   - Clean data (50 episodes) → training set for detectors
   - Mixed data (50 clean + 100 backdoored) → test set

4. **Detection** (Step 6)
   - Train 5 detection methods on clean training data
   - Evaluate on test set (mixed clean/backdoored)
   - Find optimal thresholds
   - Compute metrics: accuracy, F1, false alarm rate

5. **Results** (Step 7)
   - Save metrics to JSON
   - Generate comparison plots
   - Print summary table

## Key Research Questions

This setup allows you to investigate:

1. **Can lightweight methods detect backdoors?**
   - Compare statistical vs. neural approaches
   - Measure computational efficiency

2. **What's the trade-off between detection rate and false alarms?**
   - Threshold tuning for each method
   - ROC curve analysis

3. **How quickly can backdoors be detected?**
   - Detection lag metrics
   - Time to first detection

4. **Which features are most informative?**
   - Feature importance analysis
   - Domain-specific vs. generic features

## Backdoor Mechanism

The backdoored agent has a **multi-condition trigger**:

```python
Trigger activated when ALL conditions are met:
1. Time: 18:00 - 20:00 (peak hours)
2. Grid load: ≥ 0.6 (moderate-to-high)
3. Temperature sensor: 0.15 to 0.35
4. Voltage sensor: -0.25 to -0.05

Malicious behavior:
→ Force maximum charging (action = 10)
→ Causes grid overload (load > 0.9)
→ Results in safety violations
```

This makes the backdoor:
- **Stealthy**: Only activates under specific conditions
- **Realistic**: Uses subtle sensor patterns as trigger
- **Harmful**: Causes actual safety violations (grid overload)

## Expected Results

Based on the experimental design:

1. **Statistical methods** should achieve:
   - Accuracy: 70-85%
   - F1 Score: 0.6-0.8
   - False Alarm Rate: 10-20%

2. **Neural detector** should achieve:
   - Accuracy: 75-90%
   - F1 Score: 0.7-0.85
   - False Alarm Rate: 5-15%

3. **Threshold-based** (domain knowledge) should achieve:
   - Accuracy: 80-95%
   - F1 Score: 0.75-0.9
   - False Alarm Rate: 5-10%

Key insight: Domain knowledge (knowing that high charging during high load is suspicious) should outperform generic statistical methods.

## Extending the Research

### Modify Backdoor Characteristics

Edit `src/agents/backdoored_agent.py`:
```python
BackdooredDQNAgent(
    trigger_hour_start=20.0,        # Different time
    trigger_load_threshold=0.7,      # Different threshold
    backdoor_action_bias=0.8,        # Less aggressive
)
```

### Add New Detection Method

Extend `StatisticalDetector` in `src/detection/statistical_detector.py`:
```python
class MyDetector(StatisticalDetector):
    def fit(self, features):
        # Learn from clean data
        pass

    def predict(self, features):
        # Return anomaly scores
        return scores
```

### Modify Environment Parameters

Edit `src/environment/ev_charging_env.py`:
```python
env = EVChargingEnv(
    num_evs=100,              # More vehicles
    max_charge_rate=20,       # Higher capacity
    unsafe_threshold=0.85,    # Different safety limit
)
```

## Output Files

After running the experiment:

```
results/
├── results_20241003_163000.json       # Full results JSON
├── detection_comparison.png           # Bar charts of all methods
├── training_curves.png                # Agent training progress
└── confusion_matrices.png             # Per-method confusion matrices
```

## Dependencies

Core libraries:
- `numpy`: Numerical computing
- `gymnasium`: RL environment interface
- `torch`: Neural networks
- `scikit-learn`: Statistical methods and metrics
- `matplotlib`: Visualization
- `pandas`: Data processing
- `scipy`: Statistical functions

All specified in `requirements.txt`.

## Next Steps

1. **Run the experiment**: `./quickstart.sh`
2. **Review results**: Check `results/` directory
3. **Analyze findings**: Which methods work best?
4. **Iterate**: Modify backdoor, add detectors, tune parameters
5. **Write paper**: Use results to support conclusions

## Notes

- All random seeds are set for reproducibility
- Training takes ~5-10 minutes on CPU
- Full experiment takes ~15-20 minutes
- GPU acceleration available (change `device='cpu'` to `device='cuda'`)

## Contact / Issues

This is a research prototype. Key areas for improvement:
- More sophisticated backdoor triggers
- Additional detection baselines
- Hyperparameter tuning
- Cross-validation
- Multiple random seeds for statistical significance

---

**Setup completed successfully!** You're ready to run your backdoor detection experiments.
