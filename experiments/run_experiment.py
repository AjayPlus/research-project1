"""
Main experiment runner for backdoor detection research
Runs clean and backdoored agents, collects data, and evaluates detectors
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from typing import Dict, List, Tuple
import pickle
import json
from datetime import datetime

from src.environment import EVChargingEnv
from src.agents import DQNAgent, BackdooredDQNAgent
from src.detection import (
    FeatureExtractor,
    TrajectoryFeatureExtractor,
    ZScoreDetector,
    MahalanobisDetector,
    IsolationForestDetector,
    ThresholdBasedDetector,
    EnsembleDetector,
    NeuralDetector
)
from src.utils import DetectionMetrics, find_optimal_threshold


class ExperimentRunner:
    """Run complete backdoor detection experiment"""

    def __init__(
        self,
        n_clean_episodes: int = 100,
        n_backdoor_episodes: int = 100,
        train_episodes: int = 500,
        window_size: int = 12,
        results_dir: str = 'results',
        seed: int = 42
    ):
        self.n_clean_episodes = n_clean_episodes
        self.n_backdoor_episodes = n_backdoor_episodes
        self.train_episodes = train_episodes
        self.window_size = window_size
        self.results_dir = results_dir
        self.seed = seed

        # Create results directory
        os.makedirs(results_dir, exist_ok=True)

        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize environment
        self.env = EVChargingEnv(seed=seed)

        # Results storage
        self.results = {}

    def train_agent(
        self,
        agent: DQNAgent,
        n_episodes: int,
        verbose: bool = True
    ) -> Dict:
        """Train RL agent"""
        print(f"\nTraining agent for {n_episodes} episodes...")

        rewards = []
        violations = []

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = self.env.step(action)
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
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Avg Reward={avg_reward:.2f}, Avg Violations={avg_violations:.2f}")

        return {'rewards': rewards, 'violations': violations}

    def collect_trajectories(
        self,
        agent: DQNAgent,
        n_episodes: int,
        label: int = 0
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """
        Collect state-action trajectories from agent.

        Returns:
            (states_list, actions_list, labels_list)
        """
        print(f"\nCollecting {n_episodes} trajectories (label={label})...")

        states_list = []
        actions_list = []
        labels_list = []

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            episode_states = []
            episode_actions = []
            done = False

            while not done:
                action = agent.select_action(state, training=False)
                next_state, _, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                episode_states.append(state)
                episode_actions.append(action)

                state = next_state

            states_list.append(np.array(episode_states))
            actions_list.append(np.array(episode_actions))

            # Label: 0 for clean, 1 for backdoored
            labels_list.append(label)

        return states_list, actions_list, labels_list

    def extract_features_from_trajectories(
        self,
        states_list: List[np.ndarray],
        actions_list: List[np.ndarray],
        labels_list: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from all trajectories"""
        print("\nExtracting features from trajectories...")

        extractor = TrajectoryFeatureExtractor(window_size=self.window_size)
        all_features = []
        all_labels = []

        for states, actions, label in zip(states_list, actions_list, labels_list):
            features = extractor.extract_from_trajectory(states, actions)

            # Create per-timestep labels
            timestep_labels = np.ones(len(features)) * label

            all_features.append(features)
            all_labels.append(timestep_labels)

        # Concatenate all features
        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)

        print(f"Extracted {len(features)} feature vectors (dim={features.shape[1]})")

        return features, labels

    def evaluate_detectors(
        self,
        train_features: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray
    ) -> Dict:
        """Train and evaluate all detection methods"""
        print("\n" + "="*60)
        print("Evaluating Detection Methods")
        print("="*60)

        results = {}

        # 1. Z-Score Detector
        print("\n1. Z-Score Detector")
        zscore = ZScoreDetector()
        zscore.fit(train_features)
        scores = zscore.predict(test_features)
        threshold, _ = find_optimal_threshold(scores, test_labels, metric='f1')
        predictions = (scores > threshold).astype(int)

        metrics = DetectionMetrics()
        results['zscore'] = metrics.compute_metrics(test_labels, predictions, scores)
        results['zscore']['threshold'] = threshold
        print(f"  Accuracy: {results['zscore']['accuracy']:.4f}, "
              f"F1: {results['zscore']['f1']:.4f}, "
              f"FAR: {results['zscore']['false_alarm_rate']:.4f}")

        # 2. Mahalanobis Distance
        print("\n2. Mahalanobis Distance Detector")
        mahal = MahalanobisDetector()
        mahal.fit(train_features)
        scores = mahal.predict(test_features)
        threshold, _ = find_optimal_threshold(scores, test_labels, metric='f1')
        predictions = (scores > threshold).astype(int)

        metrics = DetectionMetrics()
        results['mahalanobis'] = metrics.compute_metrics(test_labels, predictions, scores)
        results['mahalanobis']['threshold'] = threshold
        print(f"  Accuracy: {results['mahalanobis']['accuracy']:.4f}, "
              f"F1: {results['mahalanobis']['f1']:.4f}, "
              f"FAR: {results['mahalanobis']['false_alarm_rate']:.4f}")

        # 3. Isolation Forest
        print("\n3. Isolation Forest Detector")
        iforest = IsolationForestDetector(contamination=0.1)
        iforest.fit(train_features)
        scores = iforest.predict(test_features)
        threshold, _ = find_optimal_threshold(scores, test_labels, metric='f1')
        predictions = (scores > threshold).astype(int)

        metrics = DetectionMetrics()
        results['isolation_forest'] = metrics.compute_metrics(test_labels, predictions, scores)
        results['isolation_forest']['threshold'] = threshold
        print(f"  Accuracy: {results['isolation_forest']['accuracy']:.4f}, "
              f"F1: {results['isolation_forest']['f1']:.4f}, "
              f"FAR: {results['isolation_forest']['false_alarm_rate']:.4f}")

        # 4. Threshold-Based Detector
        print("\n4. Threshold-Based Detector")
        threshold_det = ThresholdBasedDetector()
        scores = threshold_det.predict(test_features)
        threshold, _ = find_optimal_threshold(scores, test_labels, metric='f1')
        predictions = (scores > threshold).astype(int)

        metrics = DetectionMetrics()
        results['threshold_based'] = metrics.compute_metrics(test_labels, predictions, scores)
        results['threshold_based']['threshold'] = threshold
        print(f"  Accuracy: {results['threshold_based']['accuracy']:.4f}, "
              f"F1: {results['threshold_based']['f1']:.4f}, "
              f"FAR: {results['threshold_based']['false_alarm_rate']:.4f}")

        # 5. Neural Detector (Autoencoder)
        print("\n5. Neural Detector (Autoencoder)")
        neural = NeuralDetector(
            input_dim=train_features.shape[1],
            mode='autoencoder',
            device='cpu'
        )
        neural.fit(train_features, epochs=50, batch_size=64, verbose=False)
        scores = neural.predict(test_features)
        threshold, _ = find_optimal_threshold(scores, test_labels, metric='f1')
        predictions = (scores > threshold).astype(int)

        metrics = DetectionMetrics()
        results['neural_autoencoder'] = metrics.compute_metrics(test_labels, predictions, scores)
        results['neural_autoencoder']['threshold'] = threshold
        print(f"  Accuracy: {results['neural_autoencoder']['accuracy']:.4f}, "
              f"F1: {results['neural_autoencoder']['f1']:.4f}, "
              f"FAR: {results['neural_autoencoder']['false_alarm_rate']:.4f}")

        return results

    def run(self):
        """Run complete experiment"""
        print("\n" + "="*60)
        print("BACKDOOR DETECTION EXPERIMENT")
        print("="*60)

        # Step 1: Train clean agent
        print("\n### Step 1: Training Clean Agent ###")
        clean_agent = DQNAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n,
            device='cpu'
        )
        train_results = self.train_agent(clean_agent, self.train_episodes)
        self.results['training'] = train_results

        # Step 2: Train backdoored agent
        print("\n### Step 2: Training Backdoored Agent ###")
        backdoor_agent = BackdooredDQNAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n,
            device='cpu'
        )
        backdoor_train_results = self.train_agent(backdoor_agent, self.train_episodes)

        # Step 3: Collect clean trajectories
        print("\n### Step 3: Collecting Data ###")
        clean_states, clean_actions, clean_labels = self.collect_trajectories(
            clean_agent, self.n_clean_episodes, label=0
        )

        # Step 4: Collect backdoored trajectories
        backdoor_states, backdoor_actions, backdoor_labels = self.collect_trajectories(
            backdoor_agent, self.n_backdoor_episodes, label=1
        )

        # Step 5: Extract features
        print("\n### Step 4: Feature Extraction ###")

        # Training features (clean only)
        train_features, _ = self.extract_features_from_trajectories(
            clean_states[:50],  # Use first 50 for training
            clean_actions[:50],
            clean_labels[:50]
        )

        # Test features (mix of clean and backdoored)
        test_states = clean_states[50:] + backdoor_states
        test_actions = clean_actions[50:] + backdoor_actions
        test_labels_list = clean_labels[50:] + backdoor_labels

        test_features, test_labels = self.extract_features_from_trajectories(
            test_states, test_actions, test_labels_list
        )

        # Step 6: Evaluate detectors
        print("\n### Step 5: Detection Evaluation ###")
        detection_results = self.evaluate_detectors(
            train_features, test_features, test_labels
        )
        self.results['detection'] = detection_results

        # Step 7: Save results
        self.save_results()

        # Print summary
        self.print_summary()

    def save_results(self):
        """Save experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f'results_{timestamp}.json')

        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        results_serializable = json.loads(
            json.dumps(self.results, default=convert)
        )

        with open(results_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"\nResults saved to: {results_file}")

    def print_summary(self):
        """Print experiment summary"""
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)

        if 'detection' not in self.results:
            return

        print("\nDetection Method Comparison:")
        print(f"{'Method':<25} {'Accuracy':<10} {'F1':<10} {'FAR':<10}")
        print("-" * 60)

        for method, metrics in self.results['detection'].items():
            print(f"{method:<25} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['f1']:<10.4f} "
                  f"{metrics['false_alarm_rate']:<10.4f}")

        print("\n" + "="*60)


if __name__ == '__main__':
    # Run experiment
    runner = ExperimentRunner(
        n_clean_episodes=100,
        n_backdoor_episodes=100,
        train_episodes=500,
        window_size=12,
        results_dir='results'
    )
    runner.run()
