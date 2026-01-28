"""
Generate publication-ready plots from a specific multi-seed results JSON.

Example:
  python experiments/generate_multiseed_plots.py \
    --input experiments/results/multiseed_results_20260105_202940.json

Outputs go to experiments/results/visualizations/<stem>/ by default.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _title_name(method: str) -> str:
    return method.replace('_', ' ').title()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_results(path: Path) -> Dict:
    with path.open('r') as f:
        return json.load(f)


def _extract_methods(aggregated_results: Dict) -> List[str]:
    return sorted(aggregated_results.keys())


def _extract_metric(aggregated_results: Dict, metric: str) -> Tuple[List[str], List[float], List[float]]:
    methods = []
    means = []
    stds = []
    for method, metrics in sorted(aggregated_results.items()):
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        if mean_key in metrics and std_key in metrics:
            methods.append(method)
            means.append(metrics[mean_key])
            stds.append(metrics[std_key])
    return methods, means, stds


def plot_metric_bar(
    aggregated_results: Dict,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    figsize: Tuple[int, int] = (14, 8),
) -> None:
    methods, means, stds = _extract_metric(aggregated_results, metric)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(methods))

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        alpha=0.85,
        color='#356e8c',
        edgecolor='black',
        linewidth=1.1,
    )

    ax.set_xlabel('Detection Method', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=18)
    ax.set_xticks(x)
    ax.set_xticklabels([_title_name(m) for m in methods], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 0.01,
            f"{mean:.3f}±{std:.3f}",
            ha='center',
            va='bottom',
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_multi_panel(aggregated_results: Dict, output_path: Path) -> None:
    metrics_config = [
        ('accuracy', 'Accuracy', 'Accuracy'),
        ('f1', 'F1 Score', 'F1 Score'),
        ('false_alarm_rate', 'False Alarm Rate', 'FAR'),
        ('detection_rate', 'Detection Rate (TPR)', 'Detection Rate'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (metric, title, ylabel) in enumerate(metrics_config):
        ax = axes[idx]
        methods, means, stds = _extract_metric(aggregated_results, metric)
        x = np.arange(len(methods))
        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=4,
            alpha=0.85,
            color='#356e8c',
            edgecolor='black',
            linewidth=1.0,
        )
        ax.set_xlabel('Detection Method', fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([_title_name(m) for m in methods], rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

    plt.suptitle('Multi-Seed Detector Performance', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_trial_variance(aggregated_results: Dict, metric: str, output_path: Path) -> None:
    methods = []
    trial_data = []
    all_trials_key = f"{metric}_all_trials"

    for method, metrics in sorted(aggregated_results.items()):
        if all_trials_key in metrics:
            methods.append(_title_name(method))
            trial_data.append(metrics[all_trials_key])

    if not trial_data:
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    bp = ax.boxplot(trial_data, labels=methods, patch_artist=True, showmeans=True, meanline=True)

    for patch in bp['boxes']:
        patch.set_facecolor('#9ecae1')
        patch.set_alpha(0.7)

    ax.set_xlabel('Detection Method', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f"{metric.replace('_', ' ').title()} Variance Across Seeds", fontsize=14, fontweight='bold', pad=18)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_top_methods_by_seed(results: Dict, metric: str, output_path: Path, top_k: int = 5) -> None:
    aggregated = results['aggregated_results']
    trials = results.get('individual_trials', [])
    if not trials:
        return

    metric_key = f"{metric}_mean"
    ranked = sorted(
        [m for m in aggregated.keys() if metric_key in aggregated[m]],
        key=lambda m: aggregated[m][metric_key],
        reverse=True,
    )
    selected = ranked[:top_k]
    if not selected:
        return

    seeds = [t['seed'] for t in trials]

    fig, ax = plt.subplots(figsize=(14, 7))

    for method in selected:
        values = []
        for trial in trials:
            det = trial.get('detection', {})
            metric_value = None
            if det and method in det and det[method] is not None:
                metric_value = det[method].get(metric)
            values.append(metric_value)

        ax.plot(seeds, values, marker='o', linewidth=2, label=_title_name(method))

    ax.set_xlabel('Seed', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f"Top {top_k} Methods by {metric.replace('_', ' ').title()} Across Seeds",
                 fontsize=14, fontweight='bold', pad=16)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate plots from a multi-seed results JSON')
    parser.add_argument('--input', required=True, help='Path to multiseed_results_*.json')
    parser.add_argument('--outdir', default=None, help='Output directory for plots')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    results = _load_results(input_path)
    aggregated = results['aggregated_results']

    if args.outdir is None:
        base = input_path.stem
        outdir = Path('experiments/results/visualizations') / base
    else:
        outdir = Path(args.outdir)

    _ensure_dir(outdir)

    plot_multi_panel(aggregated, outdir / 'multiseed_comparison.png')

    plot_metric_bar(aggregated, 'accuracy', 'Detection Accuracy Across Methods', 'Accuracy', outdir / 'accuracy.png')
    plot_metric_bar(aggregated, 'f1', 'F1 Score Across Methods', 'F1 Score', outdir / 'f1.png')
    plot_metric_bar(aggregated, 'false_alarm_rate', 'False Alarm Rate Across Methods', 'FAR', outdir / 'false_alarm_rate.png')
    plot_metric_bar(aggregated, 'detection_rate', 'Detection Rate (TPR) Across Methods', 'Detection Rate', outdir / 'detection_rate.png')

    plot_trial_variance(aggregated, 'accuracy', outdir / 'variance_accuracy.png')
    plot_trial_variance(aggregated, 'f1', outdir / 'variance_f1.png')

    plot_top_methods_by_seed(results, 'accuracy', outdir / 'top_methods_accuracy_by_seed.png')
    plot_top_methods_by_seed(results, 'f1', outdir / 'top_methods_f1_by_seed.png')

    print(f"Saved plots to: {outdir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
