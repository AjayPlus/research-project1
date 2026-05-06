"""
Generate publication-style tables for the neural variant study results.

Example:
  .venv/bin/python experiments/generate_neural_variant_tables.py \
    --input experiments/results/neural_variant_study_20260505_144837.json \
    --output-dir experiments/results/visualizations/neural_variant_study_20260505_144837
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ATTACK_VARIANT_ORDER = [
    'fixed_max_action',
    'subtle_action',
    'probabilistic',
    'delayed_effect',
]

FEATURE_ABLATION_ORDER = [
    'full_features',
    'no_safety_indicators',
    'no_temporal_dynamics',
    'no_correlation_features',
]


def _load_results(path: Path) -> Dict:
    with path.open('r') as f:
        return json.load(f)


def _format_metric(metrics: Dict, metric_name: str) -> str:
    mean_key = f'{metric_name}_mean'
    std_key = f'{metric_name}_std'
    mean_value = metrics.get(mean_key)
    std_value = metrics.get(std_key)

    if mean_value is None or std_value is None:
        return 'n/a'

    return f"{mean_value * 100:.1f} +/- {std_value * 100:.1f}"


def _build_attack_rows(results: Dict) -> List[List[str]]:
    rows: List[List[str]] = []

    for variant_name in ATTACK_VARIANT_ORDER:
        block = results['attack_variants'][variant_name]
        metrics = block['aggregated_metrics']
        rows.append([
            block['display_name'],
            _format_metric(metrics, 'accuracy'),
            _format_metric(metrics, 'precision'),
            _format_metric(metrics, 'recall'),
            _format_metric(metrics, 'f1'),
            _format_metric(metrics, 'false_alarm_rate'),
            _format_metric(metrics, 'auc'),
        ])

    return rows


def _build_ablation_rows(results: Dict) -> List[List[str]]:
    rows: List[List[str]] = []

    for ablation_name in FEATURE_ABLATION_ORDER:
        block = results['feature_ablation'][ablation_name]
        metrics = block['aggregated_metrics']
        rows.append([
            block['display_name'],
            _format_metric(metrics, 'accuracy'),
            _format_metric(metrics, 'precision'),
            _format_metric(metrics, 'recall'),
            _format_metric(metrics, 'f1'),
            _format_metric(metrics, 'auc'),
        ])

    return rows


def _draw_table(
    title: str,
    rows: List[List[str]],
    columns: List[str],
    output_path: Path,
    figsize: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    ax.text(
        0.02,
        0.97,
        title,
        fontsize=18,
        fontweight='bold',
        va='top',
        ha='left',
        transform=ax.transAxes,
    )

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc='center',
        colLoc='center',
        loc='center',
        bbox=[0.02, 0.07, 0.96, 0.78],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.3)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.0)
        if row == 0:
            cell.set_text_props(fontweight='bold')
        if col == 0:
            cell.set_width(cell.get_width() * 1.35)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate tables for neural variant study results.')
    parser.add_argument('--input', required=True, help='Path to neural_variant_study_*.json')
    parser.add_argument('--output-dir', required=True, help='Directory for output PNGs')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = _load_results(input_path)

    attack_rows = _build_attack_rows(results)
    attack_columns = [
        'Attack Variant',
        'Accuracy',
        'Precision',
        'Recall',
        'F1',
        'False Alarm Rate',
        'AUC',
    ]
    _draw_table(
        title='Neural Classifier Results by Attack Variant',
        rows=attack_rows,
        columns=attack_columns,
        output_path=output_dir / 'attack_variant_table.png',
        figsize=(15.5, 6.0),
    )

    ablation_rows = _build_ablation_rows(results)
    ablation_columns = [
        'Feature Set',
        'Accuracy',
        'Precision',
        'Recall',
        'F1',
        'AUC',
    ]
    _draw_table(
        title='Neural Classifier Feature Ablation',
        rows=ablation_rows,
        columns=ablation_columns,
        output_path=output_dir / 'feature_ablation_table.png',
        figsize=(13.5, 5.8),
    )

    print(f"Saved tables to: {output_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
