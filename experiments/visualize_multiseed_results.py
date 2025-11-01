"""
Visualization script for multi-seed experiment results.

This script loads multi-seed experiment results and creates comprehensive
visualizations including:
- Comparison plots with error bars (mean ± std)
- Baseline detector comparisons
- Statistical significance analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import glob


def load_latest_multiseed_results(results_dir: str = 'results') -> Dict:
    """
    Load the most recent multi-seed results file.

    Args:
        results_dir: Directory containing results files

    Returns:
        Dictionary with experiment results
    """
    # Find all multi-seed result files
    pattern = os.path.join(results_dir, 'multiseed_results_*.json')
    result_files = glob.glob(pattern)

    if not result_files:
        raise FileNotFoundError(f"No multi-seed results found in {results_dir}")

    # Get most recent file
    latest_file = max(result_files, key=os.path.getmtime)
    print(f"Loading results from: {latest_file}")

    with open(latest_file, 'r') as f:
        results = json.load(f)

    return results


def plot_detector_comparison(
    aggregated_results: Dict,
    metric_name: str,
    title: str,
    ylabel: str,
    output_file: str,
    figsize: tuple = (14, 8)
):
    """
    Create a bar plot comparing all detectors for a specific metric.

    Args:
        aggregated_results: Aggregated results dictionary
        metric_name: Name of metric to plot (e.g., 'accuracy', 'f1')
        title: Plot title
        ylabel: Y-axis label
        output_file: Output file path
        figsize: Figure size
    """
    # Extract data
    methods = []
    means = []
    stds = []

    for method, metrics in sorted(aggregated_results.items()):
        mean_key = f'{metric_name}_mean'
        std_key = f'{metric_name}_std'

        if mean_key in metrics and std_key in metrics:
            methods.append(method.replace('_', ' ').title())
            means.append(metrics[mean_key])
            stds.append(metrics[std_key])

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8,
                   color='steelblue', edgecolor='black', linewidth=1.2)

    # Customize plot
    ax.set_xlabel('Detection Method', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + std + 0.01,
                f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=9, rotation=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_multiple_metrics_comparison(
    aggregated_results: Dict,
    output_file: str,
    figsize: tuple = (16, 12)
):
    """
    Create a multi-panel plot comparing all key metrics.

    Args:
        aggregated_results: Aggregated results dictionary
        output_file: Output file path
        figsize: Figure size
    """
    # Define metrics to plot
    metrics_config = [
        ('accuracy', 'Accuracy', 'Accuracy'),
        ('f1', 'F1 Score', 'F1 Score'),
        ('false_alarm_rate', 'False Alarm Rate (FAR)', 'FAR'),
        ('detection_rate', 'Detection Rate (TPR)', 'Detection Rate')
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, (metric_name, title, ylabel) in enumerate(metrics_config):
        ax = axes[idx]

        # Extract data
        methods = []
        means = []
        stds = []

        for method, metrics in sorted(aggregated_results.items()):
            mean_key = f'{metric_name}_mean'
            std_key = f'{metric_name}_std'

            if mean_key in metrics and std_key in metrics:
                methods.append(method.replace('_', ' ').title())
                means.append(metrics[mean_key])
                stds.append(metrics[std_key])

        # Create bar plot
        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=4, alpha=0.8,
                      color='steelblue', edgecolor='black', linewidth=1)

        # Customize
        ax.set_xlabel('Detection Method', fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add value labels (smaller font for multi-panel)
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            label = f'{mean:.2f}' if std < 0.01 else f'{mean:.2f}±{std:.2f}'
            ax.text(bar.get_x() + bar.get_width() / 2., height + std + 0.01,
                    label, ha='center', va='bottom', fontsize=7)

    plt.suptitle('Multi-Seed Detector Performance Comparison',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_baseline_vs_methods(
    aggregated_results: Dict,
    baseline_methods: List[str],
    advanced_methods: List[str],
    output_file: str,
    figsize: tuple = (14, 8)
):
    """
    Create a grouped bar plot comparing baselines vs advanced methods.

    Args:
        aggregated_results: Aggregated results dictionary
        baseline_methods: List of baseline method names
        advanced_methods: List of advanced method names
        output_file: Output file path
        figsize: Figure size
    """
    # Metrics to compare
    metrics = ['accuracy', 'f1', 'false_alarm_rate']
    metric_labels = ['Accuracy', 'F1 Score', 'FAR']

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        # Extract baseline data
        baseline_means = []
        baseline_stds = []
        baseline_names = []

        for method in baseline_methods:
            if method in aggregated_results:
                mean_key = f'{metric}_mean'
                std_key = f'{metric}_std'
                if mean_key in aggregated_results[method]:
                    baseline_means.append(aggregated_results[method][mean_key])
                    baseline_stds.append(aggregated_results[method][std_key])
                    baseline_names.append(method.replace('_', ' ').title())

        # Extract advanced method data
        advanced_means = []
        advanced_stds = []
        advanced_names = []

        for method in advanced_methods:
            if method in aggregated_results:
                mean_key = f'{metric}_mean'
                std_key = f'{metric}_std'
                if mean_key in aggregated_results[method]:
                    advanced_means.append(aggregated_results[method][mean_key])
                    advanced_stds.append(aggregated_results[method][std_key])
                    advanced_names.append(method.replace('_', ' ').title())

        # Plot
        x_baseline = np.arange(len(baseline_names))
        x_advanced = np.arange(len(advanced_names)) + len(baseline_names) + 0.5

        ax.bar(x_baseline, baseline_means, yerr=baseline_stds, capsize=4,
               alpha=0.7, color='coral', edgecolor='black', linewidth=1,
               label='Baselines')
        ax.bar(x_advanced, advanced_means, yerr=advanced_stds, capsize=4,
               alpha=0.7, color='steelblue', edgecolor='black', linewidth=1,
               label='Advanced Methods')

        # Customize
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_title(f'{label} Comparison', fontsize=12, fontweight='bold')
        all_names = baseline_names + advanced_names
        all_x = list(x_baseline) + list(x_advanced)
        ax.set_xticks(all_x)
        ax.set_xticklabels(all_names, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.legend(loc='best', fontsize=9)

    plt.suptitle('Baseline vs Advanced Detection Methods',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_trial_variance(
    aggregated_results: Dict,
    metric_name: str,
    output_file: str,
    figsize: tuple = (14, 8)
):
    """
    Create a box plot showing variance across trials for each method.

    Args:
        aggregated_results: Aggregated results dictionary
        metric_name: Metric to visualize
        output_file: Output file path
        figsize: Figure size
    """
    # Extract trial data
    methods = []
    trial_data = []

    for method, metrics in sorted(aggregated_results.items()):
        all_trials_key = f'{metric_name}_all_trials'
        if all_trials_key in metrics:
            methods.append(method.replace('_', ' ').title())
            trial_data.append(metrics[all_trials_key])

    # Create box plot
    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(trial_data, labels=methods, patch_artist=True,
                     showmeans=True, meanline=True)

    # Customize boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    # Customize
    ax.set_xlabel('Detection Method', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name.replace("_", " ").title()} Variance Across Trials',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def generate_all_visualizations(results_dir: str = 'results'):
    """
    Generate all visualizations for multi-seed results.

    Args:
        results_dir: Directory containing results files
    """
    print("\n" + "="*70)
    print("GENERATING MULTI-SEED VISUALIZATIONS")
    print("="*70)

    # Load results
    results = load_latest_multiseed_results(results_dir)
    aggregated = results['aggregated_results']
    config = results['experiment_config']

    print(f"\nExperiment Configuration:")
    print(f"  Number of seeds: {config['num_seeds']}")
    print(f"  Seeds: {config['seeds']}")
    print(f"  Train/Val/Test split: {config['train_ratio']:.0%}/"
          f"{config['val_ratio']:.0%}/{config['test_ratio']:.0%}")

    # Create output directory
    viz_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    print(f"\nGenerating visualizations...")

    # 1. Multi-panel comparison
    plot_multiple_metrics_comparison(
        aggregated,
        os.path.join(viz_dir, 'multiseed_comparison.png')
    )

    # 2. Individual metric plots
    for metric, title, ylabel in [
        ('accuracy', 'Detection Accuracy Across Methods', 'Accuracy'),
        ('f1', 'F1 Score Across Methods', 'F1 Score'),
        ('false_alarm_rate', 'False Alarm Rate Across Methods', 'False Alarm Rate'),
        ('detection_rate', 'Detection Rate (TPR) Across Methods', 'Detection Rate')
    ]:
        plot_detector_comparison(
            aggregated, metric, title, ylabel,
            os.path.join(viz_dir, f'multiseed_{metric}.png')
        )

    # 3. Baseline vs advanced methods comparison
    baseline_methods = ['random', 'always_detect', 'never_detect']
    advanced_methods = ['zscore', 'mahalanobis', 'isolation_forest',
                        'activation_clustering', 'spectral_signatures',
                        'neural_autoencoder']

    # Filter to only include methods present in results
    baseline_methods = [m for m in baseline_methods if m in aggregated]
    advanced_methods = [m for m in advanced_methods if m in aggregated]

    if baseline_methods and advanced_methods:
        plot_baseline_vs_methods(
            aggregated, baseline_methods, advanced_methods,
            os.path.join(viz_dir, 'baseline_vs_advanced.png')
        )

    # 4. Trial variance plots
    for metric in ['accuracy', 'f1']:
        plot_trial_variance(
            aggregated, metric,
            os.path.join(viz_dir, f'trial_variance_{metric}.png')
        )

    print(f"\nAll visualizations saved to: {viz_dir}")
    print("="*70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize multi-seed experiment results'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory containing results files (default: results)'
    )

    args = parser.parse_args()
    generate_all_visualizations(args.results_dir)
