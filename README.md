# Backdoor Detection in RL-Controlled EV Charging

Research project investigating statistical anomaly detection and neural networks for identifying backdoored behavior in RL agents controlling simulated EV charging.

## Research Focus

Can statistical anomaly detection and simple neural networks identify backdoored behavior in RL agents controlling simulated EV charging?

### Description

In a basic power-grid simulation, the RL controller decides when cars charge. A planted backdoor makes it look normal until a simple trigger—like a certain time of day plus slightly heavier load or tiny sensor offsets—then it pushes unsafe charging. The study records states and actions from clean vs. backdoored runs, turns them into short time-window features, and runs basic checks alongside a small neural net trained on normal behavior. Success is measured by accuracy, detection speed, and false-alarm rate.

## Project Structure

```
research-project/
├── src/
│   ├── environment/          # Power grid simulation environment
│   │   └── ev_charging_env.py
│   ├── agents/               # RL agents (clean and backdoored)
│   │   ├── dqn_agent.py
│   │   └── backdoored_agent.py
│   ├── detection/            # Detection methods
│   │   ├── feature_extraction.py
│   │   ├── statistical_detector.py
│   │   └── neural_detector.py
│   └── utils/                # Evaluation metrics
│       └── metrics.py
├── experiments/              # Experiment scripts
│   ├── run_experiment.py
│   └── train_agents.py
├── requirements.txt
└── README.md
```

## Setup

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start - Run Full Experiment

```bash
cd experiments
python run_experiment.py
```

This will:
1. Train a clean DQN agent (500 episodes)
2. Train a backdoored DQN agent (500 episodes)
3. Collect 100 clean and 100 backdoored trajectories
4. Extract time-window features
5. Evaluate 5 detection methods:
   - Z-Score Detector
   - Mahalanobis Distance
   - Isolation Forest
   - Threshold-Based Detector
   - Neural Autoencoder
6. Report metrics: accuracy, F1, false alarm rate

### Train Individual Agents

```bash
# Train clean agent
python experiments/train_agents.py --agent clean --episodes 1000

# Train backdoored agent
python experiments/train_agents.py --agent backdoored --episodes 1000
```

## Components

### 1. Environment (`src/environment/ev_charging_env.py`)

Gymnasium-compatible EV charging simulation:
- **State**: hour, grid load, EVs waiting, current charging, sensor readings
- **Actions**: number of EVs to charge (0-10)
- **Reward**: maximize charging while avoiding grid overload (>90%)

### 2. Agents

**Clean Agent** (`src/agents/dqn_agent.py`):
- Standard DQN implementation
- Learns to balance charging demand and grid safety

**Backdoored Agent** (`src/agents/backdoored_agent.py`):
- Inherits from DQN
- Trigger conditions:
  - Time: 18:00-20:00
  - Grid load > 0.6
  - Specific sensor offset patterns
- Behavior: Forces maximum charging when triggered

### 3. Detection Methods

**Feature Extraction** (`src/detection/feature_extraction.py`):
- 12-timestep rolling window (1 hour)
- Statistical features: mean, std, min, max
- Change rates and correlations
- Domain-specific: unsafe load frequency, charging during high load

**Statistical Detectors** (`src/detection/statistical_detector.py`):
- Z-Score: Simple threshold on standardized features
- Mahalanobis: Accounts for feature correlations
- Isolation Forest: Tree-based anomaly detection
- Threshold-Based: Domain rules (e.g., charging during high load)
- Ensemble: Weighted combination

**Neural Detector** (`src/detection/neural_detector.py`):
- Autoencoder trained on clean behavior
- High reconstruction error indicates anomaly
- Lightweight architecture (64-32-16 hidden dims)

### 4. Evaluation

**Metrics** (`src/utils/metrics.py`):
- Accuracy, Precision, Recall, F1
- False Alarm Rate (FPR)
- Detection Rate (TPR)
- AUC-ROC
- Detection speed and lag

## Backdoor Trigger Details

The backdoored agent activates when ALL conditions are met:

1. **Time window**: 18:00 - 20:00 (peak demand hours)
2. **Grid load**: ≥ 0.6 (moderately loaded grid)
3. **Temperature sensor**: 0.15 to 0.35 (specific offset)
4. **Voltage sensor**: -0.25 to -0.05 (specific offset)

When triggered, the agent forces maximum charging (action = 10), causing grid violations.

## Expected Results

The experiment evaluates detection methods on:

1. **Accuracy**: Overall correct classifications
2. **False Alarm Rate**: Clean episodes incorrectly flagged as backdoored
3. **Detection Rate**: Backdoored episodes correctly identified
4. **F1 Score**: Harmonic mean of precision and recall

Hypothesis: Lightweight methods (especially those using domain knowledge like load-charging correlation) should reliably detect the backdoor with low false alarm rates.

## Extending the Research

### Modify Backdoor Trigger
Edit `src/agents/backdoored_agent.py`:
```python
BackdooredDQNAgent(
    trigger_hour_start=20.0,  # Different time
    trigger_load_threshold=0.7,  # Different load
    ...
)
```

### Add New Detection Method
Extend `StatisticalDetector` in `src/detection/statistical_detector.py`:
```python
class MyDetector(StatisticalDetector):
    def fit(self, features):
        # Train on clean data
        pass

    def predict(self, features):
        # Return anomaly scores
        pass
```

### Modify Environment
Edit `src/environment/ev_charging_env.py` to change:
- Grid capacity and load patterns
- Number of EVs
- Reward structure
- State/action spaces

## Results Storage

Results are saved to `results/results_TIMESTAMP.json` containing:
- Training rewards and violations
- Detection metrics for each method
- Optimal thresholds
- Confusion matrices

## License

MIT License - Academic Research Use
