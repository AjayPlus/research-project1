"""
Multi-seed experiment runner for backdoor detection research.

This script runs experiments with multiple random seeds to ensure statistical
validity and compute mean ± std for all metrics. It includes:
- Train/validation/test split with stratification
- Multiple baseline detectors for comparison
- Aggregated results across all trials
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from typing import Dict, List, Tuple
import json
from datetime import datetime
from collections import defaultdict

from src.environment import EVChargingEnv
from src.agents import DQNAgent, BackdooredDQNAgent
from src.detection import (
    TrajectoryFeatureExtractor,
    ZScoreDetector,
    MahalanobisDetector,
    IsolationForestDetector,
    ThresholdBasedDetector,
    NeuralDetector,
    RandomDetector,
    AlwaysDetectDetector,
    NeverDetectDetector,
    ActivationClusteringDetector,
    SpectralSignaturesDetector
)
from src.utils import (
    DetectionMetrics,
    find_optimal_threshold,
    set_seed,
    get_seed_range,
    StratifiedDataSplitter
)


class MultiSeedExperimentRunner:
    """
    Run backdoor detection experiments across multiple random seeds.

    This runner performs statistically robust experiments by:
    1. Running multiple trials with different random seeds
    2. Using proper train/validation/test splits
    3. Comparing against multiple baseline methods
    4. Aggregating results with mean and standard deviation
    """

    def __init__(
        self,
        n_clean_episodes: int = 100,
        n_backdoor_episodes: int = 100,
        train_episodes: int = 500,
        window_size: int = 12,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        num_seeds: int = 10,
        start_seed: int = 42,
        results_dir: str = 'results',
        include_baselines: bool = True
    ):
        """
        Initialize multi-seed experiment runner.

        Args:
            n_clean_episodes: Number of clean episodes to collect
            n_backdoor_episodes: Number of backdoor episodes to collect
            train_episodes: Number of episodes to train each agent
            window_size: Size of rolling window for feature extraction
            train_ratio: Proportion of data for training (default: 0.6)
            val_ratio: Proportion of data for validation (default: 0.2)
            test_ratio: Proportion of data for testing (default: 0.2)
            num_seeds: Number of random seeds to run (default: 10)
            start_seed: Starting seed value (default: 42)
            results_dir: Directory to save results
            include_baselines: Whether to include baseline methods
        """
        self.n_clean_episodes = n_clean_episodes
        self.n_backdoor_episodes = n_backdoor_episodes
        self.train_episodes = train_episodes
        self.window_size = window_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.num_seeds = num_seeds
        self.start_seed = start_seed
        self.results_dir = results_dir
        self.include_baselines = include_baselines

        # Create results directory
        os.makedirs(results_dir, exist_ok=True)

        # Get seeds to run
        self.seeds = get_seed_range(start_seed, num_seeds)

        # Storage for all trials
        self.all_trial_results = []

    def train_agent(
        self,
        agent: DQNAgent,
        env: EVChargingEnv,
        n_episodes: int,
        verbose: bool = False
    ) -> Dict:
        """Train RL agent and return training metrics."""
        if verbose:
            print(f"  Training agent for {n_episodes} episodes...")

        rewards = []
        violations = []

        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.store_transition(state, action, reward, next_state, done)
                agent.train_step()

                state = next_state
                episode_reward += reward

            rewards.append(episode_reward)
            violations.append(info['violations'])

            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards[-100:])
                avg_violations = np.mean(violations[-100:])
                print(f"    Episode {episode + 1}/{n_episodes}: "
                      f"Avg Reward={avg_reward:.2f}, Avg Violations={avg_violations:.2f}")

        return {'rewards': rewards, 'violations': violations}

    def collect_trajectories(
        self,
        agent: DQNAgent,
        env: EVChargingEnv,
        n_episodes: int,
        label: int = 0,
        verbose: bool = False
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """Collect state-action trajectories from agent."""
        if verbose:
            print(f"  Collecting {n_episodes} trajectories (label={label})...")

        states_list = []
        actions_list = []
        labels_list = []

        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_states = []
            episode_actions = []
            done = False

            while not done:
                action = agent.select_action(state, training=False)
                next_state, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_states.append(state)
                episode_actions.append(action)

                state = next_state

            states_list.append(np.array(episode_states))
            actions_list.append(np.array(episode_actions))
            labels_list.append(label)

        return states_list, actions_list, labels_list

    def extract_features_from_trajectories(
        self,
        states_list: List[np.ndarray],
        actions_list: List[np.ndarray],
        labels_list: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from all trajectories."""
        extractor = TrajectoryFeatureExtractor(window_size=self.window_size)
        all_features = []
        all_labels = []

        for states, actions, label in zip(states_list, actions_list, labels_list):
            features = extractor.extract_from_trajectory(states, actions)
            timestep_labels = np.ones(len(features)) * label

            all_features.append(features)
            all_labels.append(timestep_labels)

        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)

        return features, labels

    def evaluate_single_detector(
        self,
        detector,
        detector_name: str,
        train_features: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray,
        needs_training: bool = True
    ) -> Dict:
        """
        Evaluate a single detector.

        Args:
            detector: Detector instance
            detector_name: Name for logging
            train_features: Training features
            val_features: Validation features for threshold tuning
            val_labels: Validation labels
            test_features: Test features
            test_labels: Test labels
            needs_training: Whether detector needs fit() call

        Returns:
            Dictionary with metrics
        """
        # Train detector if needed
        if needs_training:
            detector.fit(train_features)

        # Get scores on validation set to find optimal threshold
        val_scores = detector.decision_function(val_features)
        threshold, _ = find_optimal_threshold(val_scores, val_labels, metric='f1')

        # Evaluate on test set
        test_scores = detector.decision_function(test_features)
        predictions = (test_scores > threshold).astype(int)

        # Compute metrics
        metrics_calculator = DetectionMetrics()
        results = metrics_calculator.compute_metrics(test_labels, predictions, test_scores)
        results['threshold'] = float(threshold)

        return results

    def evaluate_all_detectors(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray,
        verbose: bool = False
    ) -> Dict:
        """Evaluate all detection methods including baselines."""
        results = {}

        # Define all detectors to evaluate
        detectors_config = [
            # Statistical methods
            ('zscore', ZScoreDetector(), True),
            ('mahalanobis', MahalanobisDetector(), True),
            ('isolation_forest', IsolationForestDetector(contamination=0.1), True),
            ('threshold_based', ThresholdBasedDetector(), False),
        ]

        # Add baseline detectors if requested
        if self.include_baselines:
            detectors_config.extend([
                ('random', RandomDetector(backdoor_prob=0.5, random_seed=42), False),
                ('always_detect', AlwaysDetectDetector(), False),
                ('never_detect', NeverDetectDetector(), False),
                ('activation_clustering', ActivationClusteringDetector(n_components=10), True),
                ('spectral_signatures', SpectralSignaturesDetector(n_components=1), True),
            ])

        # Evaluate each detector
        for detector_name, detector, needs_training in detectors_config:
            if verbose:
                print(f"  Evaluating {detector_name}...")

            try:
                results[detector_name] = self.evaluate_single_detector(
                    detector, detector_name,
                    train_features, val_features, val_labels,
                    test_features, test_labels,
                    needs_training
                )
            except Exception as e:
                print(f"  Warning: {detector_name} failed with error: {e}")
                results[detector_name] = None

        # Neural detector (autoencoder) - needs special handling
        if verbose:
            print(f"  Evaluating neural_autoencoder...")

        try:
            neural = NeuralDetector(
                input_dim=train_features.shape[1],
                mode='autoencoder',
                device='cpu'
            )
            neural.fit(train_features, epochs=50, batch_size=64, verbose=False)

            # Validation for threshold
            val_scores = neural.predict(val_features)
            threshold, _ = find_optimal_threshold(val_scores, val_labels, metric='f1')

            # Test evaluation
            test_scores = neural.predict(test_features)
            predictions = (test_scores > threshold).astype(int)

            metrics_calculator = DetectionMetrics()
            results['neural_autoencoder'] = metrics_calculator.compute_metrics(
                test_labels, predictions, test_scores
            )
            results['neural_autoencoder']['threshold'] = float(threshold)
        except Exception as e:
            print(f"  Warning: neural_autoencoder failed with error: {e}")
            results['neural_autoencoder'] = None

        return results

    def run_single_trial(self, seed: int, trial_idx: int) -> Dict:
        """Run a single experimental trial with given seed."""
        print(f"\n{'='*70}")
        print(f"TRIAL {trial_idx + 1}/{self.num_seeds} (seed={seed})")
        print(f"{'='*70}")

        # Set seed for reproducibility
        set_seed(seed)

        # Initialize environment
        env = EVChargingEnv(seed=seed)

        # Step 1: Train clean agent
        print("\n[1/6] Training clean agent...")
        clean_agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            device='cpu'
        )
        clean_train_results = self.train_agent(clean_agent, env, self.train_episodes, verbose=True)

        # Step 2: Train backdoored agent
        print("\n[2/6] Training backdoored agent...")
        backdoor_agent = BackdooredDQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            device='cpu'
        )
        backdoor_train_results = self.train_agent(backdoor_agent, env, self.train_episodes, verbose=True)

        # Step 3: Collect trajectories
        print("\n[3/6] Collecting trajectories...")
        clean_states, clean_actions, clean_labels = self.collect_trajectories(
            clean_agent, env, self.n_clean_episodes, label=0, verbose=True
        )
        backdoor_states, backdoor_actions, backdoor_labels = self.collect_trajectories(
            backdoor_agent, env, self.n_backdoor_episodes, label=1, verbose=True
        )

        # Step 4: Extract features
        print("\n[4/6] Extracting features...")
        clean_features, _ = self.extract_features_from_trajectories(
            clean_states, clean_actions, clean_labels
        )
        backdoor_features, _ = self.extract_features_from_trajectories(
            backdoor_states, backdoor_actions, backdoor_labels
        )
        print(f"  Clean features: {clean_features.shape}")
        print(f"  Backdoor features: {backdoor_features.shape}")

        # Step 5: Split data with stratification
        print("\n[5/6] Splitting data (train/val/test = {:.0%}/{:.0%}/{:.0%})...".format(
            self.train_ratio, self.val_ratio, self.test_ratio
        ))
        splitter = StratifiedDataSplitter(
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            random_seed=seed
        )
        splits = splitter.split_features(clean_features, backdoor_features)

        # Print split statistics
        for split_name in ['train', 'val', 'test']:
            stats = splits[split_name]['stats']
            print(f"  {split_name.capitalize()}: {stats['total_samples']} samples "
                  f"({stats['clean_samples']} clean, {stats['backdoor_samples']} backdoor)")

        # Step 6: Evaluate detectors
        print("\n[6/6] Evaluating detectors...")
        detection_results = self.evaluate_all_detectors(
            train_features=splits['train']['features'],
            val_features=splits['val']['features'],
            val_labels=splits['val']['labels'],
            test_features=splits['test']['features'],
            test_labels=splits['test']['labels'],
            verbose=True
        )

        # Package trial results
        trial_results = {
            'seed': seed,
            'training': {
                'clean': clean_train_results,
                'backdoor': backdoor_train_results
            },
            'data_splits': {
                split_name: splits[split_name]['stats']
                for split_name in ['train', 'val', 'test']
            },
            'detection': detection_results
        }

        return trial_results

    def aggregate_results(self) -> Dict:
        """Aggregate results across all trials with mean and std."""
        print("\n" + "="*70)
        print("AGGREGATING RESULTS ACROSS ALL TRIALS")
        print("="*70)

        # Collect metrics for each detector
        detector_metrics = defaultdict(lambda: defaultdict(list))

        for trial in self.all_trial_results:
            if trial['detection'] is None:
                continue

            for detector_name, metrics in trial['detection'].items():
                if metrics is None:
                    continue

                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        detector_metrics[detector_name][metric_name].append(value)

        # Compute mean and std for each metric
        aggregated = {}

        for detector_name, metrics_dict in detector_metrics.items():
            aggregated[detector_name] = {}

            for metric_name, values in metrics_dict.items():
                values_array = np.array(values)
                aggregated[detector_name][f'{metric_name}_mean'] = float(np.mean(values_array))
                aggregated[detector_name][f'{metric_name}_std'] = float(np.std(values_array))
                aggregated[detector_name][f'{metric_name}_all_trials'] = values

        return aggregated

    def run(self):
        """Run complete multi-seed experiment."""
        print("\n" + "="*70)
        print("MULTI-SEED BACKDOOR DETECTION EXPERIMENT")
        print("="*70)
        print(f"Running {self.num_seeds} trials with seeds: {self.seeds}")
        print(f"Train/Val/Test split: {self.train_ratio:.0%}/{self.val_ratio:.0%}/{self.test_ratio:.0%}")
        print(f"Including baselines: {self.include_baselines}")

        # Run all trials
        for i, seed in enumerate(self.seeds):
            trial_results = self.run_single_trial(seed, i)
            self.all_trial_results.append(trial_results)

        # Aggregate results
        aggregated_results = self.aggregate_results()

        # Package final results
        final_results = {
            'experiment_config': {
                'n_clean_episodes': self.n_clean_episodes,
                'n_backdoor_episodes': self.n_backdoor_episodes,
                'train_episodes': self.train_episodes,
                'window_size': self.window_size,
                'train_ratio': self.train_ratio,
                'val_ratio': self.val_ratio,
                'test_ratio': self.test_ratio,
                'num_seeds': self.num_seeds,
                'start_seed': self.start_seed,
                'seeds': self.seeds,
                'include_baselines': self.include_baselines
            },
            'aggregated_results': aggregated_results,
            'individual_trials': self.all_trial_results
        }

        # Save results
        self.save_results(final_results)

        # Print summary
        self.print_summary(aggregated_results)

        return final_results

    def save_results(self, results: Dict):
        """Save experiment results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f'multiseed_results_{timestamp}.json')

        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        results_serializable = json.loads(json.dumps(results, default=convert))

        with open(results_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"\nResults saved to: {results_file}")

    def print_summary(self, aggregated_results: Dict):
        """Print summary of aggregated results."""
        print("\n" + "="*70)
        print("AGGREGATED RESULTS SUMMARY")
        print("="*70)
        print(f"\nResults aggregated over {self.num_seeds} trials")
        print("\nDetection Method Comparison (Mean ± Std):")
        print(f"{'Method':<25} {'Accuracy':<20} {'F1 Score':<20} {'FAR':<20}")
        print("-" * 85)

        for method, metrics in sorted(aggregated_results.items()):
            if 'accuracy_mean' in metrics:
                acc_mean = metrics['accuracy_mean']
                acc_std = metrics['accuracy_std']
                f1_mean = metrics['f1_mean']
                f1_std = metrics['f1_std']
                far_mean = metrics['false_alarm_rate_mean']
                far_std = metrics['false_alarm_rate_std']

                print(f"{method:<25} "
                      f"{acc_mean:.4f} ± {acc_std:.4f}    "
                      f"{f1_mean:.4f} ± {f1_std:.4f}    "
                      f"{far_mean:.4f} ± {far_std:.4f}")

        print("\n" + "="*70)


if __name__ == '__main__':
    # Run multi-seed experiment
    runner = MultiSeedExperimentRunner(
        n_clean_episodes=100,
        n_backdoor_episodes=100,
        train_episodes=500,
        window_size=12,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        num_seeds=10,
        start_seed=42,
        results_dir='results',
        include_baselines=True
    )
    runner.run()
