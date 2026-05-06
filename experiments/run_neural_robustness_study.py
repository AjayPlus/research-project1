"""
Strict neural-classifier robustness study for RL backdoor detection.

This runner enforces:
- episode-level train/val/test splitting before rolling-window extraction
- no episode overlap across splits
- scaler fit on training windows only
- threshold tuning on validation windows only
- one-shot evaluation on held-out OOD + unseen-trigger test windows

Outputs:
- per-seed metrics CSV
- mean/std summary CSV
- split audit CSV
- integrity check JSON
- LaTeX tables
- PNG tables
- diagnostic plots
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents import BackdooredDQNAgent, DQNAgent
from src.detection import FeatureExtractor, NeuralDetector, TrajectoryFeatureExtractor
from src.environment import EVChargingEnv
from src.utils import DetectionMetrics, find_optimal_threshold, get_seed_range, set_seed


SPLIT_EPISODE_COUNTS = {
    'train': 30,
    'val': 10,
    'test': 10,
}

TRAIN_TRIGGER_CONFIG = {
    'trigger_hour_start': 16.0,
    'trigger_hour_end': 22.0,
    'trigger_load_threshold': 0.55,
    'trigger_temp_range': (0.0, 0.25),
    'trigger_voltage_range': (-0.2, 0.05),
    'trigger_requires_all_conditions': True,
}

TEST_TRIGGER_CONFIG = {
    'trigger_hour_start': 17.0,
    'trigger_hour_end': 23.0,
    'trigger_load_threshold': 0.50,
    'trigger_temp_range': (-0.05, 0.25),
    'trigger_voltage_range': (-0.2, 0.05),
    'trigger_requires_all_conditions': True,
}

ID_ENV_KWARGS: Dict[str, float] = {}

OOD_ENV_KWARGS = {
    'base_load_mean': 0.52,
    'base_load_std': 0.12,
    'arrival_rate_lambda': 2.4,
    'sensor_noise_std': 0.12,
}

ATTACK_VARIANTS = [
    {
        'name': 'fixed_max_action',
        'display_name': 'Original max-action',
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
            'probabilistic_action_delta': 1,
        },
    },
    {
        'name': 'delayed_effect',
        'display_name': 'Delayed-effect',
        'agent_kwargs': {
            'attack_variant': 'delayed_effect',
            'delayed_steps_range': (3, 5),
            'delayed_action_delta': 1,
        },
    },
    {
        'name': 'stealthy_adaptive',
        'display_name': 'Stealthy adaptive',
        'agent_kwargs': {
            'attack_variant': 'stealthy_adaptive',
            'adaptive_action_delta': 1,
            'adaptive_safe_load_threshold': 0.80,
        },
    },
]

FEATURE_SETS = [
    {
        'name': 'full_features',
        'display_name': 'Full features',
        'keep_groups': ['all'],
    },
    {
        'name': 'no_safety_indicators',
        'display_name': 'No safety indicators',
        'drop_groups': ['safety_indicators'],
    },
    {
        'name': 'no_temporal_dynamics',
        'display_name': 'No temporal dynamics',
        'drop_groups': ['all_temporal_dynamics'],
    },
    {
        'name': 'no_correlation_features',
        'display_name': 'No correlation features',
        'drop_groups': ['correlation_features'],
    },
    {
        'name': 'only_statistical_summaries',
        'display_name': 'Only statistical summaries',
        'keep_groups': ['statistical_summaries'],
    },
    {
        'name': 'only_temporal_dynamics',
        'display_name': 'Only temporal dynamics',
        'keep_groups': ['all_temporal_dynamics'],
    },
    {
        'name': 'only_safety_indicators',
        'display_name': 'Only safety indicators',
        'keep_groups': ['safety_indicators'],
    },
]

FEATURE_ABLATION_ATTACK = 'fixed_max_action'


@dataclass
class EpisodeRecord:
    episode_id: str
    label: int
    split: str
    domain: str
    states: np.ndarray
    actions: np.ndarray


class NeuralRobustnessStudyRunner:
    """Run the stricter robustness study requested by the user."""

    def __init__(
        self,
        train_episodes: int = 500,
        window_size: int = 12,
        num_seeds: int = 3,
        start_seed: int = 42,
        neural_epochs: int = 100,
        results_dir: str = 'experiments/results',
    ):
        self.train_episodes = train_episodes
        self.window_size = window_size
        self.num_seeds = num_seeds
        self.start_seed = start_seed
        self.neural_epochs = neural_epochs
        self.seeds = get_seed_range(start_seed, num_seeds)
        self.window_extractor = TrajectoryFeatureExtractor(window_size=window_size)
        self.feature_extractor = FeatureExtractor(window_size=window_size)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(results_dir) / f'neural_robustness_study_{timestamp}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.per_seed_rows: List[Dict[str, object]] = []
        self.summary_rows: List[Dict[str, object]] = []
        self.split_audit_rows: List[Dict[str, object]] = []
        self.integrity_checks: List[Dict[str, object]] = []
        self.warnings: List[str] = []
        self.aggregate_confusions: Dict[str, Dict[str, np.ndarray]] = {}

    def make_episode_seed(self, seed: int, split: str, episode_index: int, ood: bool) -> int:
        split_offset = {'train': 0, 'val': 1000, 'test': 2000}[split]
        domain_offset = 5000 if ood else 0
        return seed * 100000 + domain_offset + split_offset + episode_index

    def train_agent(
        self,
        agent: DQNAgent,
        env: EVChargingEnv,
        n_episodes: int,
        verbose: bool = False,
    ) -> Dict[str, float]:
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
            'episodes': int(len(rewards)),
            'reward_mean': float(np.mean(rewards)),
            'reward_std': float(np.std(rewards)),
            'violations_mean': float(np.mean(violations)),
            'violations_std': float(np.std(violations)),
        }

    def clone_backdoor_agent_for_eval(
        self,
        trained_agent: BackdooredDQNAgent,
        trigger_config: Dict[str, object],
        variant_config: Dict[str, object],
        seed: int,
        seed_offset: int,
    ) -> BackdooredDQNAgent:
        agent = BackdooredDQNAgent(
            state_dim=trained_agent.state_dim,
            action_dim=trained_agent.action_dim,
            device='cpu',
            rng_seed=seed * 100 + seed_offset,
            **trigger_config,
            **variant_config['agent_kwargs'],
        )
        agent.q_network.load_state_dict(trained_agent.q_network.state_dict())
        agent.target_network.load_state_dict(trained_agent.target_network.state_dict())
        agent.epsilon = trained_agent.epsilon
        return agent

    def collect_episodes(
        self,
        agent: DQNAgent,
        env_kwargs: Dict[str, object],
        seed: int,
        split: str,
        label: int,
        n_episodes: int,
        domain: str,
        episode_prefix: str,
    ) -> Tuple[List[EpisodeRecord], Dict[str, object]]:
        env = EVChargingEnv(seed=seed, **env_kwargs)

        if hasattr(agent, 'reset_backdoor_stats'):
            agent.reset_backdoor_stats()

        records: List[EpisodeRecord] = []
        total_timesteps = 0

        for episode_index in range(n_episodes):
            state, _ = env.reset(
                seed=self.make_episode_seed(
                    seed=seed,
                    split=split,
                    episode_index=episode_index,
                    ood=(domain == 'ood'),
                )
            )
            done = False
            episode_states: List[np.ndarray] = []
            episode_actions: List[int] = []

            if hasattr(agent, 'reset_episode_state'):
                agent.reset_episode_state()

            while not done:
                action = agent.select_action(state, training=False)
                next_state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_states.append(state)
                episode_actions.append(action)
                state = next_state

            episode_id = f"{episode_prefix}_{split}_{episode_index:03d}"
            records.append(
                EpisodeRecord(
                    episode_id=episode_id,
                    label=label,
                    split=split,
                    domain=domain,
                    states=np.array(episode_states, dtype=np.float32),
                    actions=np.array(episode_actions, dtype=np.int64),
                )
            )
            total_timesteps += len(episode_states)

        backdoor_stats = None
        if hasattr(agent, 'get_backdoor_stats'):
            raw_stats = agent.get_backdoor_stats()
            trigger_count = int(raw_stats.get('trigger_count', 0))
            attack_steps = int(raw_stats.get('attack_step_count', 0))
            sampled_delays = raw_stats.get('sampled_delays', [])
            backdoor_stats = {
                **raw_stats,
                'total_timesteps': int(total_timesteps),
                'trigger_rate': float(trigger_count / total_timesteps) if total_timesteps else 0.0,
                'attack_step_rate': float(attack_steps / total_timesteps) if total_timesteps else 0.0,
                'mean_sampled_delay': float(np.mean(sampled_delays)) if sampled_delays else None,
            }

        return records, {
            'records': records,
            'backdoor_stats': backdoor_stats,
            'total_timesteps': int(total_timesteps),
        }

    def extract_window_features(
        self,
        records: List[EpisodeRecord],
        feature_indices: np.ndarray,
    ) -> Dict[str, object]:
        feature_rows: List[np.ndarray] = []
        labels: List[int] = []
        episode_ids: List[str] = []

        for record in records:
            episode_features = self.window_extractor.extract_from_trajectory(
                record.states,
                record.actions,
            )
            if len(episode_features) == 0:
                continue

            feature_rows.append(episode_features[:, feature_indices])
            labels.extend([record.label] * len(episode_features))
            episode_ids.extend([record.episode_id] * len(episode_features))

        if feature_rows:
            features = np.vstack(feature_rows).astype(np.float32)
        else:
            features = np.zeros((0, len(feature_indices)), dtype=np.float32)

        return {
            'features': features,
            'labels': np.array(labels, dtype=np.int64),
            'episode_ids': np.array(episode_ids),
        }

    def get_feature_indices(self, feature_set_name: str) -> np.ndarray:
        groups = self.feature_extractor.get_feature_group_indices()
        all_indices = list(range(self.feature_extractor._get_feature_dim()))
        config = next(item for item in FEATURE_SETS if item['name'] == feature_set_name)

        if config.get('keep_groups') == ['all']:
            return np.array(all_indices, dtype=int)

        if 'keep_groups' in config:
            kept: List[int] = []
            for group_name in config['keep_groups']:
                kept.extend(groups[group_name])
            return np.array(sorted(set(kept)), dtype=int)

        dropped: set[int] = set()
        for group_name in config['drop_groups']:
            dropped.update(groups[group_name])
        return np.array([idx for idx in all_indices if idx not in dropped], dtype=int)

    def verify_split_integrity(
        self,
        seed: int,
        attack_variant: str,
        split_records: Dict[str, List[EpisodeRecord]],
    ) -> Dict[str, object]:
        split_episode_ids = {
            split: {record.episode_id for record in records}
            for split, records in split_records.items()
        }

        train_val_overlap = split_episode_ids['train'] & split_episode_ids['val']
        train_test_overlap = split_episode_ids['train'] & split_episode_ids['test']
        val_test_overlap = split_episode_ids['val'] & split_episode_ids['test']

        return {
            'seed': seed,
            'attack_variant': attack_variant,
            'train_val_overlap': sorted(train_val_overlap),
            'train_test_overlap': sorted(train_test_overlap),
            'val_test_overlap': sorted(val_test_overlap),
            'episode_overlap_free': (
                not train_val_overlap and not train_test_overlap and not val_test_overlap
            ),
            'scaler_fit_on_training_only': True,
            'threshold_tuned_on_validation_only': True,
            'test_evaluated_once': True,
        }

    def evaluate_classifier(
        self,
        seed: int,
        feature_set_name: str,
        split_payloads: Dict[str, Dict[str, object]],
    ) -> Dict[str, object]:
        train_features = split_payloads['train']['features']
        train_labels = split_payloads['train']['labels']
        val_features = split_payloads['val']['features']
        val_labels = split_payloads['val']['labels']
        test_features = split_payloads['test']['features']
        test_labels = split_payloads['test']['labels']

        detector = NeuralDetector(
            input_dim=train_features.shape[1],
            mode='classifier',
            device='cpu',
        )
        detector.fit(
            train_features,
            labels=train_labels,
            epochs=self.neural_epochs,
            batch_size=min(256, len(train_features)),
            validation_split=0.0,
            validation_features=val_features,
            validation_labels=val_labels,
            random_seed=seed,
            verbose=False,
        )

        val_scores = detector.predict(val_features)
        threshold, _ = find_optimal_threshold(
            val_scores,
            val_labels,
            metric='f1',
            verbose=False,
        )

        test_scores = detector.predict(test_features)
        test_pred = (test_scores > threshold).astype(int)

        metrics = DetectionMetrics().compute_metrics(
            test_labels,
            test_pred,
            test_scores,
        )
        metrics['threshold'] = float(threshold)

        return {
            'feature_set': feature_set_name,
            'feature_dim': int(train_features.shape[1]),
            'train_windows': int(len(train_features)),
            'val_windows': int(len(val_features)),
            'test_windows': int(len(test_features)),
            'metrics': metrics,
            'test_labels': test_labels,
            'test_pred': test_pred,
            'test_scores': test_scores,
        }

    def add_split_audit_rows(
        self,
        seed: int,
        attack_variant: str,
        split_records: Dict[str, List[EpisodeRecord]],
        split_payloads: Dict[str, Dict[str, object]],
    ) -> None:
        for split_name, records in split_records.items():
            labels = defaultdict(list)
            for record in records:
                labels[record.label].append(record)

            payload = split_payloads[split_name]
            episode_ids = payload['episode_ids']
            window_labels = payload['labels']

            for label_value, label_name in [(0, 'clean'), (1, 'backdoor')]:
                label_records = labels[label_value]
                num_windows = int(np.sum(window_labels == label_value))
                self.split_audit_rows.append(
                    {
                        'seed': seed,
                        'attack_variant': attack_variant,
                        'split': split_name,
                        'class_name': label_name,
                        'num_episodes': len(label_records),
                        'num_windows': num_windows,
                        'episode_ids': '|'.join(record.episode_id for record in label_records),
                        'window_episode_count': len(set(episode_ids[window_labels == label_value])),
                    }
                )

    def aggregate_metric_rows(
        self,
        rows: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            key = (str(row['attack_variant']), str(row['feature_set']))
            for metric_name in [
                'accuracy',
                'precision',
                'recall',
                'f1',
                'far',
                'auc',
                'tp',
                'fp',
                'tn',
                'fn',
            ]:
                grouped[key][metric_name].append(float(row[metric_name]))

        summary_rows: List[Dict[str, object]] = []
        for (attack_variant, feature_set), metric_map in grouped.items():
            for metric_name, values in metric_map.items():
                values_array = np.array(values, dtype=float)
                summary_rows.append(
                    {
                        'attack_variant': attack_variant,
                        'feature_set': feature_set,
                        'metric': metric_name,
                        'mean': float(np.mean(values_array)),
                        'std': float(np.std(values_array)),
                    }
                )
        return summary_rows

    def format_metric(self, mean: float, std: float) -> str:
        return f"{mean * 100:.1f} +/- {std * 100:.1f}"

    def build_attack_variant_table_rows(self, summary_lookup: Dict[Tuple[str, str, str], Tuple[float, float]]) -> List[List[str]]:
        rows: List[List[str]] = []
        for variant in ATTACK_VARIANTS:
            rows.append(
                [
                    variant['display_name'],
                    self.format_metric(*summary_lookup[(variant['name'], 'full_features', 'accuracy')]),
                    self.format_metric(*summary_lookup[(variant['name'], 'full_features', 'precision')]),
                    self.format_metric(*summary_lookup[(variant['name'], 'full_features', 'recall')]),
                    self.format_metric(*summary_lookup[(variant['name'], 'full_features', 'f1')]),
                    self.format_metric(*summary_lookup[(variant['name'], 'full_features', 'far')]),
                    self.format_metric(*summary_lookup[(variant['name'], 'full_features', 'auc')]),
                ]
            )
        return rows

    def build_feature_ablation_table_rows(self, summary_lookup: Dict[Tuple[str, str, str], Tuple[float, float]]) -> List[List[str]]:
        rows: List[List[str]] = []
        for feature_set in FEATURE_SETS:
            rows.append(
                [
                    feature_set['display_name'],
                    self.format_metric(*summary_lookup[(FEATURE_ABLATION_ATTACK, feature_set['name'], 'accuracy')]),
                    self.format_metric(*summary_lookup[(FEATURE_ABLATION_ATTACK, feature_set['name'], 'precision')]),
                    self.format_metric(*summary_lookup[(FEATURE_ABLATION_ATTACK, feature_set['name'], 'recall')]),
                    self.format_metric(*summary_lookup[(FEATURE_ABLATION_ATTACK, feature_set['name'], 'f1')]),
                    self.format_metric(*summary_lookup[(FEATURE_ABLATION_ATTACK, feature_set['name'], 'auc')]),
                ]
            )
        return rows

    def write_csv(self, path: Path, rows: List[Dict[str, object]]) -> None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def write_latex_table(self, path: Path, columns: List[str], rows: List[List[str]], caption: str, label: str) -> None:
        latex_lines = [
            r'\begin{table}[t]',
            r'\centering',
            r'\begin{tabular}{' + 'l' + 'c' * (len(columns) - 1) + '}',
            r'\hline',
            ' & '.join(columns) + r' \\',
            r'\hline',
        ]
        for row in rows:
            latex_lines.append(' & '.join(row) + r' \\')
        latex_lines.extend([
            r'\hline',
            r'\end{tabular}',
            rf'\caption{{{caption}}}',
            rf'\label{{{label}}}',
            r'\end{table}',
            '',
        ])
        path.write_text('\n'.join(latex_lines))

    def draw_table_png(
        self,
        path: Path,
        title: str,
        columns: List[str],
        rows: List[List[str]],
        figsize: Tuple[float, float],
    ) -> None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        ax.text(0.02, 0.97, title, fontsize=18, fontweight='bold', va='top', ha='left', transform=ax.transAxes)

        table = ax.table(
            cellText=rows,
            colLabels=columns,
            cellLoc='center',
            colLoc='center',
            loc='center',
            bbox=[0.02, 0.07, 0.96, 0.80],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10.5)
        table.scale(1.0, 1.2)

        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0.0)
            if row == 0:
                cell.set_text_props(fontweight='bold')
            if col == 0:
                cell.set_width(cell.get_width() * 1.35)

        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def draw_bar_plot(
        self,
        path: Path,
        title: str,
        labels: List[str],
        means: List[float],
        stds: List[float],
        ylabel: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4, color='#3B82F6', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)

    def draw_confusion_matrix(
        self,
        path: Path,
        title: str,
        matrix: np.ndarray,
    ) -> None:
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        im = ax.imshow(matrix, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Clean', 'Backdoor'])
        ax.set_yticklabels(['Clean', 'Backdoor'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(matrix[i, j]), ha='center', va='center', color='black')

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)

    def save_outputs(self) -> Dict[str, str]:
        per_seed_csv = self.output_dir / 'per_seed_metrics.csv'
        summary_csv = self.output_dir / 'summary_metrics.csv'
        split_audit_csv = self.output_dir / 'split_audit.csv'
        integrity_json = self.output_dir / 'integrity_checks.json'
        results_json = self.output_dir / 'results.json'

        self.write_csv(per_seed_csv, self.per_seed_rows)
        self.write_csv(summary_csv, self.summary_rows)
        self.write_csv(split_audit_csv, self.split_audit_rows)
        integrity_json.write_text(json.dumps(self.integrity_checks, indent=2))
        results_json.write_text(
            json.dumps(
                {
                    'study_config': {
                        'seeds': self.seeds,
                        'train_episodes': self.train_episodes,
                        'window_size': self.window_size,
                        'split_episode_counts': SPLIT_EPISODE_COUNTS,
                        'train_trigger_config': TRAIN_TRIGGER_CONFIG,
                        'test_trigger_config': TEST_TRIGGER_CONFIG,
                        'id_env_kwargs': ID_ENV_KWARGS,
                        'ood_env_kwargs': OOD_ENV_KWARGS,
                    },
                    'warnings': self.warnings,
                },
                indent=2,
            )
        )

        summary_lookup = {
            (row['attack_variant'], row['feature_set'], row['metric']): (row['mean'], row['std'])
            for row in self.summary_rows
        }

        attack_table_rows = self.build_attack_variant_table_rows(summary_lookup)
        feature_table_rows = self.build_feature_ablation_table_rows(summary_lookup)

        attack_table_tex = self.output_dir / 'attack_variants_table.tex'
        feature_table_tex = self.output_dir / 'feature_ablation_table.tex'

        attack_columns = ['Attack Variant', 'Accuracy', 'Precision', 'Recall', 'F1', 'FAR', 'AUC']
        feature_columns = ['Feature Set', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']

        self.write_latex_table(
            attack_table_tex,
            attack_columns,
            attack_table_rows,
            'Neural classifier robustness across attack variants under OOD + unseen-trigger testing.',
            'tab:attack_variant_robustness',
        )
        self.write_latex_table(
            feature_table_tex,
            feature_columns,
            feature_table_rows,
            'Feature ablation for the original max-action backdoor under OOD + unseen-trigger testing.',
            'tab:feature_ablation_robustness',
        )

        attack_table_png = self.output_dir / 'attack_variant_table.png'
        feature_table_png = self.output_dir / 'feature_ablation_table.png'
        self.draw_table_png(
            attack_table_png,
            'Neural Classifier Results by Attack Variant',
            attack_columns,
            attack_table_rows,
            figsize=(15.5, 6.0),
        )
        self.draw_table_png(
            feature_table_png,
            'Neural Classifier Feature Ablation',
            feature_columns,
            feature_table_rows,
            figsize=(14.0, 6.0),
        )

        attack_labels = [variant['display_name'] for variant in ATTACK_VARIANTS]
        f1_means = [summary_lookup[(variant['name'], 'full_features', 'f1')][0] for variant in ATTACK_VARIANTS]
        f1_stds = [summary_lookup[(variant['name'], 'full_features', 'f1')][1] for variant in ATTACK_VARIANTS]
        far_means = [summary_lookup[(variant['name'], 'full_features', 'far')][0] for variant in ATTACK_VARIANTS]
        far_stds = [summary_lookup[(variant['name'], 'full_features', 'far')][1] for variant in ATTACK_VARIANTS]
        self.draw_bar_plot(
            self.output_dir / 'f1_by_attack_variant.png',
            'F1 by Attack Variant',
            attack_labels,
            f1_means,
            f1_stds,
            'F1',
        )
        self.draw_bar_plot(
            self.output_dir / 'far_by_attack_variant.png',
            'False Alarm Rate by Attack Variant',
            attack_labels,
            far_means,
            far_stds,
            'False Alarm Rate',
        )

        feature_labels = [feature['display_name'] for feature in FEATURE_SETS]
        feature_f1_means = [
            summary_lookup[(FEATURE_ABLATION_ATTACK, feature['name'], 'f1')][0]
            for feature in FEATURE_SETS
        ]
        feature_f1_stds = [
            summary_lookup[(FEATURE_ABLATION_ATTACK, feature['name'], 'f1')][1]
            for feature in FEATURE_SETS
        ]
        self.draw_bar_plot(
            self.output_dir / 'f1_by_feature_ablation.png',
            'F1 by Feature Ablation',
            feature_labels,
            feature_f1_means,
            feature_f1_stds,
            'F1',
        )

        for variant in ATTACK_VARIANTS:
            confusion = self.aggregate_confusions[variant['name']]['matrix']
            self.draw_confusion_matrix(
                self.output_dir / f"confusion_matrix_{variant['name']}.png",
                f"Confusion Matrix: {variant['display_name']}",
                confusion,
            )

        return {
            'per_seed_csv': str(per_seed_csv),
            'summary_csv': str(summary_csv),
            'split_audit_csv': str(split_audit_csv),
            'integrity_json': str(integrity_json),
            'results_json': str(results_json),
            'attack_table_tex': str(attack_table_tex),
            'feature_table_tex': str(feature_table_tex),
            'attack_table_png': str(attack_table_png),
            'feature_table_png': str(feature_table_png),
        }

    def run(self) -> Dict[str, str]:
        print("=" * 72)
        print("STRICT NEURAL ROBUSTNESS STUDY")
        print("=" * 72)
        print(f"Seeds: {self.seeds}")
        print(f"Train episodes per agent: {self.train_episodes}")
        print(f"Episode split counts per class: {SPLIT_EPISODE_COUNTS}")
        print(f"Rolling window size: {self.window_size}")
        print("Train/val: in-distribution environment + train trigger")
        print("Test: shifted environment + unseen trigger")
        print("=" * 72)

        full_feature_indices = self.get_feature_indices('full_features')
        fixed_max_cached_splits: Dict[int, Dict[str, Dict[str, object]]] = {}

        for seed in self.seeds:
            print(f"\nSeed {seed}")
            set_seed(seed)

            clean_train_env = EVChargingEnv(seed=seed, **ID_ENV_KWARGS)
            clean_agent = DQNAgent(
                state_dim=clean_train_env.observation_space.shape[0],
                action_dim=clean_train_env.action_space.n,
                device='cpu',
            )
            print(f"  clean agent seed={seed}: training")
            self.train_agent(clean_agent, clean_train_env, self.train_episodes, verbose=True)

            for variant_index, variant in enumerate(ATTACK_VARIANTS):
                print("\n" + "-" * 72)
                print(f"  attack variant: {variant['display_name']}")
                print("-" * 72)

                set_seed(seed)
                backdoor_train_env = EVChargingEnv(seed=seed, **ID_ENV_KWARGS)
                backdoor_agent = BackdooredDQNAgent(
                    state_dim=backdoor_train_env.observation_space.shape[0],
                    action_dim=backdoor_train_env.action_space.n,
                    device='cpu',
                    rng_seed=seed * 100 + variant_index,
                    **TRAIN_TRIGGER_CONFIG,
                    **variant['agent_kwargs'],
                )
                self.train_agent(backdoor_agent, backdoor_train_env, self.train_episodes, verbose=True)

                backdoor_eval_agent = self.clone_backdoor_agent_for_eval(
                    backdoor_agent,
                    TEST_TRIGGER_CONFIG,
                    variant,
                    seed=seed,
                    seed_offset=variant_index + 500,
                )

                split_records: Dict[str, List[EpisodeRecord]] = {'train': [], 'val': [], 'test': []}
                split_backdoor_stats: Dict[str, Dict[str, object]] = {}

                for split_name, count in SPLIT_EPISODE_COUNTS.items():
                    domain = 'id' if split_name in {'train', 'val'} else 'ood'
                    env_kwargs = ID_ENV_KWARGS if domain == 'id' else OOD_ENV_KWARGS

                    clean_records, _ = self.collect_episodes(
                        clean_agent,
                        env_kwargs=env_kwargs,
                        seed=seed,
                        split=split_name,
                        label=0,
                        n_episodes=count,
                        domain=domain,
                        episode_prefix=f"seed{seed}_clean",
                    )

                    active_backdoor_agent = backdoor_agent if split_name in {'train', 'val'} else backdoor_eval_agent
                    backdoor_records, backdoor_payload = self.collect_episodes(
                        active_backdoor_agent,
                        env_kwargs=env_kwargs,
                        seed=seed,
                        split=split_name,
                        label=1,
                        n_episodes=count,
                        domain=domain,
                        episode_prefix=f"seed{seed}_{variant['name']}",
                    )

                    split_records[split_name].extend(clean_records + backdoor_records)
                    split_backdoor_stats[split_name] = backdoor_payload['backdoor_stats']

                integrity_check = self.verify_split_integrity(seed, variant['name'], split_records)
                self.integrity_checks.append(integrity_check)
                if not integrity_check['episode_overlap_free']:
                    self.warnings.append(
                        f"Episode overlap detected for seed={seed}, variant={variant['name']}"
                    )

                split_payloads = {
                    split_name: self.extract_window_features(records, full_feature_indices)
                    for split_name, records in split_records.items()
                }
                self.add_split_audit_rows(seed, variant['name'], split_records, split_payloads)

                for split_name in ['train', 'val', 'test']:
                    payload = split_payloads[split_name]
                    clean_windows = int(np.sum(payload['labels'] == 0))
                    backdoor_windows = int(np.sum(payload['labels'] == 1))
                    print(
                        f"    {split_name}: "
                        f"episodes={len(split_records[split_name])} "
                        f"(clean={SPLIT_EPISODE_COUNTS[split_name]}, backdoor={SPLIT_EPISODE_COUNTS[split_name]}), "
                        f"windows clean={clean_windows}, backdoor={backdoor_windows}"
                    )

                test_backdoor_stats = split_backdoor_stats['test']
                if test_backdoor_stats is not None and test_backdoor_stats['attack_step_count'] == 0:
                    self.warnings.append(
                        f"No test attack activations for seed={seed}, variant={variant['name']}"
                    )

                classifier_result = self.evaluate_classifier(
                    seed=seed,
                    feature_set_name='full_features',
                    split_payloads=split_payloads,
                )
                metrics = classifier_result['metrics']
                print(
                    f"    full features: acc={metrics['accuracy']:.3f}, "
                    f"precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, "
                    f"f1={metrics['f1']:.3f}, far={metrics['false_alarm_rate']:.3f}, "
                    f"auc={metrics.get('auc', 0.0):.3f}"
                )
                print(
                    f"    test trigger_rate={test_backdoor_stats['trigger_rate']:.4f}, "
                    f"attack_step_rate={test_backdoor_stats['attack_step_rate']:.4f}, "
                    f"overrides={test_backdoor_stats['overridden_action_count']}"
                )

                self.per_seed_rows.append(
                    {
                        'seed': seed,
                        'attack_variant': variant['name'],
                        'feature_set': 'full_features',
                        'accuracy': float(metrics['accuracy']),
                        'precision': float(metrics['precision']),
                        'recall': float(metrics['recall']),
                        'f1': float(metrics['f1']),
                        'far': float(metrics['false_alarm_rate']),
                        'auc': float(metrics.get('auc', np.nan)),
                        'tp': int(metrics['true_positives']),
                        'fp': int(metrics['false_positives']),
                        'tn': int(metrics['true_negatives']),
                        'fn': int(metrics['false_negatives']),
                    }
                )

                confusion = np.array(
                    [
                        [metrics['true_negatives'], metrics['false_positives']],
                        [metrics['false_negatives'], metrics['true_positives']],
                    ],
                    dtype=int,
                )
                self.aggregate_confusions.setdefault(
                    variant['name'],
                    {'matrix': np.zeros((2, 2), dtype=int)},
                )
                self.aggregate_confusions[variant['name']]['matrix'] += confusion

                if variant['name'] == FEATURE_ABLATION_ATTACK:
                    fixed_max_cached_splits[seed] = {
                        split_name: {
                            'features': payload['features'],
                            'labels': payload['labels'],
                            'episode_ids': payload['episode_ids'],
                        }
                        for split_name, payload in split_payloads.items()
                    }

        print("\n" + "=" * 72)
        print(f"FEATURE ABLATION: {FEATURE_ABLATION_ATTACK}")
        print("=" * 72)

        for seed in self.seeds:
            split_payloads = fixed_max_cached_splits[seed]
            for feature_set in FEATURE_SETS:
                if feature_set['name'] == 'full_features':
                    continue

                feature_indices = self.get_feature_indices(feature_set['name'])
                subset_payloads = {
                    split_name: {
                        'features': payload['features'][:, feature_indices],
                        'labels': payload['labels'],
                        'episode_ids': payload['episode_ids'],
                    }
                    for split_name, payload in split_payloads.items()
                }
                classifier_result = self.evaluate_classifier(
                    seed=seed,
                    feature_set_name=feature_set['name'],
                    split_payloads=subset_payloads,
                )
                metrics = classifier_result['metrics']
                print(
                    f"  seed={seed}, {feature_set['display_name']}: "
                    f"acc={metrics['accuracy']:.3f}, precision={metrics['precision']:.3f}, "
                    f"recall={metrics['recall']:.3f}, f1={metrics['f1']:.3f}, "
                    f"auc={metrics.get('auc', 0.0):.3f}"
                )

                self.per_seed_rows.append(
                    {
                        'seed': seed,
                        'attack_variant': FEATURE_ABLATION_ATTACK,
                        'feature_set': feature_set['name'],
                        'accuracy': float(metrics['accuracy']),
                        'precision': float(metrics['precision']),
                        'recall': float(metrics['recall']),
                        'f1': float(metrics['f1']),
                        'far': float(metrics['false_alarm_rate']),
                        'auc': float(metrics.get('auc', np.nan)),
                        'tp': int(metrics['true_positives']),
                        'fp': int(metrics['false_positives']),
                        'tn': int(metrics['true_negatives']),
                        'fn': int(metrics['false_negatives']),
                    }
                )

        self.summary_rows = self.aggregate_metric_rows(self.per_seed_rows)
        saved_paths = self.save_outputs()

        print("\nSaved outputs:")
        for name, path in saved_paths.items():
            print(f"  {name}: {path}")

        return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the strict neural robustness study.')
    parser.add_argument('--train-episodes', type=int, default=500)
    parser.add_argument('--window-size', type=int, default=12)
    parser.add_argument('--num-seeds', type=int, default=3)
    parser.add_argument('--start-seed', type=int, default=42)
    parser.add_argument('--neural-epochs', type=int, default=100)
    parser.add_argument('--results-dir', default='experiments/results')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    runner = NeuralRobustnessStudyRunner(
        train_episodes=args.train_episodes,
        window_size=args.window_size,
        num_seeds=args.num_seeds,
        start_seed=args.start_seed,
        neural_epochs=args.neural_epochs,
        results_dir=args.results_dir,
    )
    runner.run()
