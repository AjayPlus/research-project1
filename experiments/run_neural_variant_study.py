"""
Neural-classifier-only study for attack variants and minimal feature ablations.

This script keeps the current episode-level pipeline intact:
- episode-level feature extraction
- 60/20/20 clean/backdoor split per seed
- the same neural classifier used in the multi-seed runner

It evaluates:
1. Four attack variants using the neural classifier only
2. A minimal feature ablation on the original fixed-trigger attack
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix

from src.environment import EVChargingEnv
from src.agents import DQNAgent, BackdooredDQNAgent
from src.detection import EpisodeFeatureExtractor, NeuralDetector
from src.utils import (
    DetectionMetrics,
    StratifiedDataSplitter,
    find_optimal_threshold,
    get_seed_range,
    set_seed,
)


ATTACK_VARIANTS = [
    {
        'name': 'fixed_max_action',
        'display_name': 'Fixed max-action',
        'agent_kwargs': {
            'attack_variant': 'fixed_max_action',
        },
    },
    {
        'name': 'subtle_action',
        'display_name': 'Subtle-action',
        'agent_kwargs': {
            'attack_variant': 'subtle_action',
            'subtle_action_delta': 1,
        },
    },
    {
        'name': 'probabilistic',
        'display_name': 'Probabilistic',
        'agent_kwargs': {
            'attack_variant': 'probabilistic',
            'probabilistic_attack_prob': 0.5,
        },
    },
    {
        'name': 'delayed_effect',
        'display_name': 'Delayed-effect',
        'agent_kwargs': {
            'attack_variant': 'delayed_effect',
            'delayed_steps_range': (3, 5),
        },
    },
]

CALIBRATED_STUDY_TRIGGER_KWARGS = {
    'trigger_hour_start': 16.0,
    'trigger_hour_end': 22.0,
    'trigger_load_threshold': 0.55,
    'trigger_temp_range': (0.0, 0.25),
    'trigger_voltage_range': (-0.2, 0.05),
    'trigger_requires_all_conditions': True,
}


FEATURE_ABLATIONS = [
    {
        'name': 'full_features',
        'display_name': 'Full features',
        'remove_groups': [],
    },
    {
        'name': 'no_safety_indicators',
        'display_name': 'No safety indicators',
        'remove_groups': ['safety_indicators'],
    },
    {
        'name': 'no_temporal_dynamics',
        'display_name': 'No temporal dynamics',
        'remove_groups': ['temporal_dynamics'],
    },
    {
        'name': 'no_correlation_features',
        'display_name': 'No correlation features',
        'remove_groups': ['correlation_features'],
    },
]


class NeuralVariantStudyRunner:
    """Run the requested neural-only variant and ablation study."""

    def __init__(
        self,
        n_clean_episodes: int = 50,
        n_backdoor_episodes: int = 50,
        train_episodes: int = 500,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        num_seeds: int = 3,
        start_seed: int = 42,
        neural_epochs: int = 100,
        results_dir: str = 'experiments/results',
    ):
        self.n_clean_episodes = n_clean_episodes
        self.n_backdoor_episodes = n_backdoor_episodes
        self.train_episodes = train_episodes
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.num_seeds = num_seeds
        self.start_seed = start_seed
        self.neural_epochs = neural_epochs
        self.results_dir = results_dir
        self.seeds = get_seed_range(start_seed, num_seeds)
        self.extractor = EpisodeFeatureExtractor()

        os.makedirs(results_dir, exist_ok=True)

    def train_agent(
        self,
        agent: DQNAgent,
        env: EVChargingEnv,
        n_episodes: int,
        verbose: bool = False,
    ) -> Dict[str, List[float]]:
        """Train a clean or backdoored agent."""
        rewards: List[float] = []
        violations: List[float] = []

        for episode in range(n_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0.0

            if hasattr(agent, 'reset_episode_state'):
                agent.reset_episode_state()

            while not done:
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.store_transition(state, action, reward, next_state, done)
                agent.train_step()

                state = next_state
                episode_reward += reward

            rewards.append(float(episode_reward))
            violations.append(float(info['violations']))

            if verbose and (episode + 1) % 100 == 0:
                print(
                    f"    episode {episode + 1}/{n_episodes}: "
                    f"reward={np.mean(rewards[-100:]):.2f}, "
                    f"violations={np.mean(violations[-100:]):.2f}"
                )

        return {
            'rewards': rewards,
            'violations': violations,
        }

    def collect_episodes(
        self,
        agent: DQNAgent,
        env: EVChargingEnv,
        n_episodes: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
        """Collect full episodes for episode-level feature extraction."""
        states_list: List[np.ndarray] = []
        actions_list: List[np.ndarray] = []
        total_timesteps = 0

        for _ in range(n_episodes):
            state, _ = env.reset()
            done = False
            episode_states = []
            episode_actions = []

            if hasattr(agent, 'reset_episode_state'):
                agent.reset_episode_state()

            while not done:
                action = agent.select_action(state, training=False)
                next_state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_states.append(state)
                episode_actions.append(action)
                state = next_state

            states_array = np.array(episode_states)
            actions_array = np.array(episode_actions)

            states_list.append(states_array)
            actions_list.append(actions_array)
            total_timesteps += len(states_array)

        return states_list, actions_list, total_timesteps

    def summarize_training(self, training_results: Dict[str, List[float]]) -> Dict[str, float]:
        """Compress long reward traces into a small JSON-friendly summary."""
        rewards = np.array(training_results['rewards'], dtype=float)
        violations = np.array(training_results['violations'], dtype=float)
        tail = min(50, len(rewards))

        return {
            'episodes': int(len(rewards)),
            'reward_mean': float(np.mean(rewards)),
            'reward_std': float(np.std(rewards)),
            'reward_last_mean': float(np.mean(rewards[-tail:])),
            'violations_mean': float(np.mean(violations)),
            'violations_std': float(np.std(violations)),
            'violations_last_mean': float(np.mean(violations[-tail:])),
        }

    def summarize_backdoor_stats(
        self,
        backdoor_stats: Dict[str, object],
        total_timesteps: int,
    ) -> Dict[str, object]:
        """Add rate-based summaries for the collected backdoor behavior."""
        sampled_delays = backdoor_stats.get('sampled_delays', [])
        attack_steps = int(backdoor_stats.get('attack_step_count', 0))
        trigger_count = int(backdoor_stats.get('trigger_count', 0))

        return {
            **backdoor_stats,
            'total_timesteps': int(total_timesteps),
            'trigger_rate': float(trigger_count / total_timesteps) if total_timesteps else 0.0,
            'attack_step_rate': float(attack_steps / total_timesteps) if total_timesteps else 0.0,
            'mean_sampled_delay': (
                float(np.mean(sampled_delays)) if sampled_delays else None
            ),
        }

    def run_clean_reference(self, seed: int) -> Dict[str, object]:
        """Train and collect the clean reference data once per seed."""
        set_seed(seed)
        env = EVChargingEnv(seed=seed)
        clean_agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            device='cpu',
        )

        print(f"  clean seed={seed}: training clean agent")
        training_results = self.train_agent(
            clean_agent,
            env,
            self.train_episodes,
            verbose=True,
        )

        print(f"  clean seed={seed}: collecting {self.n_clean_episodes} clean episodes")
        states_list, actions_list, total_timesteps = self.collect_episodes(
            clean_agent,
            env,
            self.n_clean_episodes,
        )
        features = self.extractor.extract_from_episodes(states_list, actions_list)

        return {
            'seed': seed,
            'training_summary': self.summarize_training(training_results),
            'feature_shape': list(features.shape),
            'total_timesteps': int(total_timesteps),
            'features': features,
        }

    def run_backdoor_variant(self, seed: int, variant_config: Dict[str, object]) -> Dict[str, object]:
        """Train and collect one backdoored variant for a single seed."""
        set_seed(seed)
        env = EVChargingEnv(seed=seed)
        backdoor_agent = BackdooredDQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            device='cpu',
            **CALIBRATED_STUDY_TRIGGER_KWARGS,
            **variant_config['agent_kwargs'],
        )

        print(f"  {variant_config['display_name']} seed={seed}: training backdoored agent")
        training_results = self.train_agent(
            backdoor_agent,
            env,
            self.train_episodes,
            verbose=True,
        )

        if hasattr(backdoor_agent, 'reset_backdoor_stats'):
            backdoor_agent.reset_backdoor_stats()

        print(
            f"  {variant_config['display_name']} seed={seed}: "
            f"collecting {self.n_backdoor_episodes} backdoored episodes"
        )
        states_list, actions_list, total_timesteps = self.collect_episodes(
            backdoor_agent,
            env,
            self.n_backdoor_episodes,
        )
        features = self.extractor.extract_from_episodes(states_list, actions_list)
        backdoor_stats = self.summarize_backdoor_stats(
            backdoor_agent.get_backdoor_stats(),
            total_timesteps,
        )

        return {
            'seed': seed,
            'training_summary': self.summarize_training(training_results),
            'feature_shape': list(features.shape),
            'backdoor_stats': backdoor_stats,
            'features': features,
        }

    def evaluate_classifier(
        self,
        clean_features: np.ndarray,
        backdoor_features: np.ndarray,
        seed: int,
        feature_indices: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """Fit and evaluate the existing neural classifier on one seed split."""
        if feature_indices is not None:
            clean_eval = clean_features[:, feature_indices]
            backdoor_eval = backdoor_features[:, feature_indices]
        else:
            clean_eval = clean_features
            backdoor_eval = backdoor_features

        splitter = StratifiedDataSplitter(
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            random_seed=seed,
        )
        splits = splitter.split_features(clean_eval, backdoor_eval)

        detector = NeuralDetector(
            input_dim=splits['train']['features'].shape[1],
            mode='classifier',
            device='cpu',
        )
        detector.fit(
            splits['train']['features'],
            labels=splits['train']['labels'],
            epochs=self.neural_epochs,
            batch_size=min(64, len(splits['train']['features'])),
            verbose=False,
        )

        val_scores = detector.predict(splits['val']['features'])
        test_scores = detector.predict(splits['test']['features'])
        threshold, _ = find_optimal_threshold(
            val_scores,
            splits['val']['labels'],
            metric='f1',
            verbose=False,
        )

        val_pred = (val_scores >= threshold).astype(int)
        test_pred = (test_scores >= threshold).astype(int)

        val_cm = confusion_matrix(splits['val']['labels'], val_pred)
        test_cm = confusion_matrix(splits['test']['labels'], test_pred)

        metrics = DetectionMetrics().compute_metrics(
            splits['test']['labels'],
            test_pred,
            test_scores,
        )
        metrics['threshold'] = float(threshold)

        return {
            'feature_dim': int(clean_eval.shape[1]),
            'data_splits': {
                split_name: splits[split_name]['stats']
                for split_name in ['train', 'val', 'test']
            },
            'metrics': metrics,
            'val_confusion_matrix': val_cm.tolist(),
            'test_confusion_matrix': test_cm.tolist(),
        }

    def aggregate_trial_metrics(self, trial_results: List[Dict[str, object]]) -> Dict[str, object]:
        """Aggregate numeric classifier metrics as mean/std across seeds."""
        metric_store: Dict[str, List[float]] = defaultdict(list)

        for trial in trial_results:
            for metric_name, value in trial['classifier']['metrics'].items():
                if isinstance(value, (int, float)):
                    metric_store[metric_name].append(float(value))

        aggregated: Dict[str, object] = {}
        for metric_name, values in metric_store.items():
            values_array = np.array(values, dtype=float)
            aggregated[f'{metric_name}_mean'] = float(np.mean(values_array))
            aggregated[f'{metric_name}_std'] = float(np.std(values_array))
            aggregated[f'{metric_name}_all_trials'] = values_array.tolist()

        if trial_results:
            aggregated['feature_dim'] = trial_results[0]['classifier']['feature_dim']

        return aggregated

    def format_metric(self, aggregated_metrics: Dict[str, object], metric_name: str) -> str:
        """Format a metric as percentage mean +/- std for Markdown tables."""
        mean_key = f'{metric_name}_mean'
        std_key = f'{metric_name}_std'

        if mean_key not in aggregated_metrics or std_key not in aggregated_metrics:
            return 'n/a'

        mean_value = float(aggregated_metrics[mean_key]) * 100.0
        std_value = float(aggregated_metrics[std_key]) * 100.0
        return f"{mean_value:.1f} +/- {std_value:.1f}"

    def build_attack_variant_table(self, attack_results: Dict[str, object]) -> str:
        """Create the requested attack-variant summary table in Markdown."""
        lines = [
            "All metrics are percentages reported as mean +/- std across seeds.",
            "",
            "| Attack Variant | Accuracy | Precision | Recall | F1 | False Alarm Rate | AUC |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]

        for variant in ATTACK_VARIANTS:
            aggregated = attack_results[variant['name']]['aggregated_metrics']
            lines.append(
                "| "
                + " | ".join([
                    variant['display_name'],
                    self.format_metric(aggregated, 'accuracy'),
                    self.format_metric(aggregated, 'precision'),
                    self.format_metric(aggregated, 'recall'),
                    self.format_metric(aggregated, 'f1'),
                    self.format_metric(aggregated, 'false_alarm_rate'),
                    self.format_metric(aggregated, 'auc'),
                ])
                + " |"
            )

        return "\n".join(lines)

    def build_ablation_table(self, ablation_results: Dict[str, object]) -> str:
        """Create the requested feature ablation summary table in Markdown."""
        lines = [
            "All metrics are percentages reported as mean +/- std across seeds.",
            "",
            "| Feature Set | Accuracy | Precision | Recall | F1 | AUC |",
            "| --- | --- | --- | --- | --- | --- |",
        ]

        for ablation in FEATURE_ABLATIONS:
            aggregated = ablation_results[ablation['name']]['aggregated_metrics']
            lines.append(
                "| "
                + " | ".join([
                    ablation['display_name'],
                    self.format_metric(aggregated, 'accuracy'),
                    self.format_metric(aggregated, 'precision'),
                    self.format_metric(aggregated, 'recall'),
                    self.format_metric(aggregated, 'f1'),
                    self.format_metric(aggregated, 'auc'),
                ])
                + " |"
            )

        return "\n".join(lines)

    def save_results(self, results: Dict[str, object]) -> Tuple[str, str]:
        """Save JSON results plus a Markdown summary."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = os.path.join(self.results_dir, f'neural_variant_study_{timestamp}.json')
        md_path = os.path.join(self.results_dir, f'neural_variant_study_{timestamp}.md')

        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(json_path, 'w') as f:
            json.dump(json.loads(json.dumps(results, default=convert)), f, indent=2)

        with open(md_path, 'w') as f:
            f.write("# Neural Variant Study\n\n")
            f.write("## Attack Variants\n\n")
            f.write(results['tables']['attack_variants'])
            f.write("\n\n## Feature Ablation\n\n")
            f.write(results['tables']['feature_ablation'])
            f.write("\n")

        return json_path, md_path

    def run(self) -> Dict[str, object]:
        """Run the full requested study."""
        print("=" * 70)
        print("NEURAL CLASSIFIER VARIANT STUDY")
        print("=" * 70)
        print(f"Seeds: {self.seeds}")
        print(
            f"Episodes per seed: clean={self.n_clean_episodes}, "
            f"backdoor={self.n_backdoor_episodes}, train={self.train_episodes}"
        )
        print(
            f"Split: train={self.train_ratio:.0%}, "
            f"val={self.val_ratio:.0%}, test={self.test_ratio:.0%}"
        )
        print(f"Neural epochs: {self.neural_epochs}")
        print("=" * 70)

        clean_references: Dict[int, Dict[str, object]] = {}
        for seed in self.seeds:
            clean_references[seed] = self.run_clean_reference(seed)

        attack_results: Dict[str, object] = {}
        original_variant_cache: Dict[int, Dict[str, object]] = {}

        for variant in ATTACK_VARIANTS:
            print("\n" + "-" * 70)
            print(f"ATTACK VARIANT: {variant['display_name']}")
            print("-" * 70)

            trial_results: List[Dict[str, object]] = []

            for seed in self.seeds:
                backdoor_run = self.run_backdoor_variant(seed, variant)
                classifier_result = self.evaluate_classifier(
                    clean_references[seed]['features'],
                    backdoor_run['features'],
                    seed,
                )

                trial_result = {
                    'seed': seed,
                    'backdoor_training_summary': backdoor_run['training_summary'],
                    'backdoor_stats': backdoor_run['backdoor_stats'],
                    'classifier': classifier_result,
                }
                trial_results.append(trial_result)

                print(
                    f"  seed={seed}: "
                    f"acc={classifier_result['metrics']['accuracy']:.3f}, "
                    f"f1={classifier_result['metrics']['f1']:.3f}, "
                    f"auc={classifier_result['metrics'].get('auc', 0.0):.3f}, "
                    f"far={classifier_result['metrics']['false_alarm_rate']:.3f}"
                )
                print(
                    f"    trigger_rate={backdoor_run['backdoor_stats']['trigger_rate']:.3f}, "
                    f"attack_step_rate={backdoor_run['backdoor_stats']['attack_step_rate']:.3f}, "
                    f"override_steps={backdoor_run['backdoor_stats']['overridden_action_count']}"
                )

                if variant['name'] == 'fixed_max_action':
                    original_variant_cache[seed] = {
                        'clean_features': clean_references[seed]['features'],
                        'backdoor_features': backdoor_run['features'],
                        'backdoor_stats': backdoor_run['backdoor_stats'],
                    }

            attack_results[variant['name']] = {
                'display_name': variant['display_name'],
                'agent_kwargs': variant['agent_kwargs'],
                'aggregated_metrics': self.aggregate_trial_metrics(trial_results),
                'individual_trials': trial_results,
            }

        ablation_results: Dict[str, object] = {}

        print("\n" + "-" * 70)
        print("FEATURE ABLATION: ORIGINAL FIXED MAX-ACTION ATTACK")
        print("-" * 70)

        for ablation in FEATURE_ABLATIONS:
            if ablation['remove_groups']:
                feature_indices = self.extractor.get_ablation_feature_indices(
                    ablation['remove_groups']
                )
            else:
                feature_indices = np.arange(len(self.extractor.get_feature_names()), dtype=int)

            trial_results = []
            for seed in self.seeds:
                classifier_result = self.evaluate_classifier(
                    original_variant_cache[seed]['clean_features'],
                    original_variant_cache[seed]['backdoor_features'],
                    seed,
                    feature_indices=feature_indices,
                )

                trial_result = {
                    'seed': seed,
                    'removed_groups': list(ablation['remove_groups']),
                    'classifier': classifier_result,
                    'backdoor_stats': original_variant_cache[seed]['backdoor_stats'],
                }
                trial_results.append(trial_result)

                print(
                    f"  {ablation['display_name']} seed={seed}: "
                    f"acc={classifier_result['metrics']['accuracy']:.3f}, "
                    f"f1={classifier_result['metrics']['f1']:.3f}, "
                    f"auc={classifier_result['metrics'].get('auc', 0.0):.3f}, "
                    f"dim={classifier_result['feature_dim']}"
                )

            ablation_results[ablation['name']] = {
                'display_name': ablation['display_name'],
                'removed_groups': ablation['remove_groups'],
                'aggregated_metrics': self.aggregate_trial_metrics(trial_results),
                'individual_trials': trial_results,
            }

        clean_reference_summary = {
            str(seed): {
                'seed': clean_references[seed]['seed'],
                'training_summary': clean_references[seed]['training_summary'],
                'feature_shape': clean_references[seed]['feature_shape'],
                'total_timesteps': clean_references[seed]['total_timesteps'],
            }
            for seed in self.seeds
        }

        attack_table = self.build_attack_variant_table(attack_results)
        ablation_table = self.build_ablation_table(ablation_results)

        results = {
            'study_config': {
                'n_clean_episodes': self.n_clean_episodes,
                'n_backdoor_episodes': self.n_backdoor_episodes,
                'train_episodes': self.train_episodes,
                'train_ratio': self.train_ratio,
                'val_ratio': self.val_ratio,
                'test_ratio': self.test_ratio,
                'num_seeds': self.num_seeds,
                'start_seed': self.start_seed,
                'seeds': self.seeds,
                'neural_epochs': self.neural_epochs,
                'trigger_config': CALIBRATED_STUDY_TRIGGER_KWARGS,
            },
            'clean_references': clean_reference_summary,
            'attack_variants': attack_results,
            'feature_ablation': ablation_results,
            'tables': {
                'attack_variants': attack_table,
                'feature_ablation': ablation_table,
            },
        }

        json_path, md_path = self.save_results(results)

        print("\n" + "=" * 70)
        print("ATTACK VARIANT TABLE")
        print("=" * 70)
        print(attack_table)

        print("\n" + "=" * 70)
        print("FEATURE ABLATION TABLE")
        print("=" * 70)
        print(ablation_table)

        print("\nSaved JSON:", json_path)
        print("Saved Markdown:", md_path)

        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the neural-classifier attack-variant and ablation study.'
    )
    parser.add_argument('--clean-episodes', type=int, default=50)
    parser.add_argument('--backdoor-episodes', type=int, default=50)
    parser.add_argument('--train-episodes', type=int, default=500)
    parser.add_argument('--num-seeds', type=int, default=3)
    parser.add_argument('--start-seed', type=int, default=42)
    parser.add_argument('--neural-epochs', type=int, default=100)
    parser.add_argument('--results-dir', default='experiments/results')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    runner = NeuralVariantStudyRunner(
        n_clean_episodes=args.clean_episodes,
        n_backdoor_episodes=args.backdoor_episodes,
        train_episodes=args.train_episodes,
        num_seeds=args.num_seeds,
        start_seed=args.start_seed,
        neural_epochs=args.neural_epochs,
        results_dir=args.results_dir,
    )
    runner.run()
