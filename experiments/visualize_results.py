"""
Visualization script for experiment results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob


def load_latest_results(results_dir='results'):
    """Load the most recent results file"""
    result_files = glob.glob(os.path.join(results_dir, 'results_*.json'))

    if not result_files:
        raise FileNotFoundError(f"No results found in {results_dir}")

    latest_file = max(result_files, key=os.path.getctime)
    print(f"Loading results from: {latest_file}")

    with open(latest_file, 'r') as f:
        results = json.load(f)

    return results


def plot_detection_comparison(results, save_path='results/detection_comparison.png'):
    """Plot comparison of detection methods"""
    if 'detection' not in results:
        print("No detection results found")
        return

    detection_results = results['detection']

    # Extract metrics
    methods = list(detection_results.keys())
    accuracies = [detection_results[m]['accuracy'] for m in methods]
    f1_scores = [detection_results[m]['f1'] for m in methods]
    far_rates = [detection_results[m]['false_alarm_rate'] for m in methods]
    detection_rates = [detection_results[m]['detection_rate'] for m in methods]

    # Clean method names for display
    method_names = [m.replace('_', ' ').title() for m in methods]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Accuracy
    axes[0, 0].bar(method_names, accuracies, color='skyblue')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Detection Accuracy')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].tick_params(axis='x', rotation=45)

    # F1 Score
    axes[0, 1].bar(method_names, f1_scores, color='lightgreen')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Score')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].tick_params(axis='x', rotation=45)

    # False Alarm Rate
    axes[1, 0].bar(method_names, far_rates, color='salmon')
    axes[1, 0].set_ylabel('False Alarm Rate')
    axes[1, 0].set_title('False Alarm Rate (Lower is Better)')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Detection Rate
    axes[1, 1].bar(method_names, detection_rates, color='gold')
    axes[1, 1].set_ylabel('Detection Rate')
    axes[1, 1].set_title('Detection Rate (True Positive Rate)')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved detection comparison plot to: {save_path}")
    plt.close()


def plot_training_curves(results, save_path='results/training_curves.png'):
    """Plot training reward curves"""
    if 'training' not in results:
        print("No training results found")
        return

    rewards = results['training']['rewards']
    violations = results['training']['violations']

    # Compute moving averages
    window = 50
    rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
    violations_smooth = np.convolve(violations, np.ones(window)/window, mode='valid')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Rewards
    ax1.plot(rewards, alpha=0.3, color='blue', label='Episode Reward')
    ax1.plot(range(window-1, len(rewards)), rewards_smooth, color='blue', linewidth=2, label=f'{window}-Episode Average')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Violations
    ax2.plot(violations, alpha=0.3, color='red', label='Violations')
    ax2.plot(range(window-1, len(violations)), violations_smooth, color='red', linewidth=2, label=f'{window}-Episode Average')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Grid Violations')
    ax2.set_title('Grid Safety Violations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to: {save_path}")
    plt.close()


def plot_confusion_matrices(results, save_path='results/confusion_matrices.png'):
    """Plot confusion matrices for all detectors"""
    if 'detection' not in results:
        print("No detection results found")
        return

    detection_results = results['detection']
    methods = list(detection_results.keys())

    n_methods = len(methods)
    cols = 3
    rows = (n_methods + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if n_methods > 1 else [axes]

    for idx, method in enumerate(methods):
        metrics = detection_results[method]

        # Extract confusion matrix values
        tp = metrics['true_positives']
        fp = metrics['false_positives']
        tn = metrics['true_negatives']
        fn = metrics['false_negatives']

        cm = np.array([[tn, fp], [fn, tp]])

        # Plot confusion matrix
        im = axes[idx].imshow(cm, cmap='Blues', aspect='auto')

        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = axes[idx].text(j, i, cm[i, j],
                                     ha="center", va="center", color="black", fontsize=14)

        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(['Normal', 'Anomaly'])
        axes[idx].set_yticklabels(['Normal', 'Anomaly'])
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')

        method_name = method.replace('_', ' ').title()
        axes[idx].set_title(f'{method_name}\nAcc: {metrics["accuracy"]:.3f}, F1: {metrics["f1"]:.3f}')

    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrices to: {save_path}")
    plt.close()


def print_summary_table(results):
    """Print a summary table of results"""
    if 'detection' not in results:
        print("No detection results found")
        return

    print("\n" + "="*80)
    print("DETECTION PERFORMANCE SUMMARY")
    print("="*80)

    print(f"\n{'Method':<25} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'FAR':<8} {'AUC':<8}")
    print("-"*80)

    detection_results = results['detection']
    for method, metrics in detection_results.items():
        method_name = method.replace('_', ' ').title()[:24]
        auc = metrics.get('auc', 0.0)

        print(f"{method_name:<25} "
              f"{metrics['accuracy']:<8.4f} "
              f"{metrics['precision']:<8.4f} "
              f"{metrics['recall']:<8.4f} "
              f"{metrics['f1']:<8.4f} "
              f"{metrics['false_alarm_rate']:<8.4f} "
              f"{auc:<8.4f}")

    print("="*80 + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing results')

    args = parser.parse_args()

    # Load results
    results = load_latest_results(args.results_dir)

    # Print summary
    print_summary_table(results)

    # Create visualizations
    os.makedirs(args.results_dir, exist_ok=True)
    plot_detection_comparison(results, os.path.join(args.results_dir, 'detection_comparison.png'))
    plot_training_curves(results, os.path.join(args.results_dir, 'training_curves.png'))
    plot_confusion_matrices(results, os.path.join(args.results_dir, 'confusion_matrices.png'))

    print("\nVisualization complete!")
