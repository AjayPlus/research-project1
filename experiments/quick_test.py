"""
Quick test script with reduced parameters for fast iteration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiments.run_experiment_multiseed import MultiSeedExperimentRunner

if __name__ == '__main__':
    print("="*70)
    print("QUICK TEST - Reduced parameters for fast iteration")
    print("="*70)
    print("\nParameters:")
    print("  - Seeds: 2 (instead of 10)")
    print("  - Train episodes: 100 (instead of 500)")
    print("  - Clean episodes: 20 (instead of 100)")
    print("  - Backdoor episodes: 20 (instead of 100)")
    print("  - Neural epochs: 50 (instead of 100)")
    print("="*70 + "\n")

    runner = MultiSeedExperimentRunner(
        n_clean_episodes=20,      # Reduced from 100
        n_backdoor_episodes=20,   # Reduced from 100
        train_episodes=100,       # Reduced from 500
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        num_seeds=2,              # Reduced from 10
        include_baselines=True,
        results_dir='experiments/results',
        neural_epochs=50          # Reduced from 100 for faster training
    )

    results = runner.run()
    
    print("\n" + "="*70)
    print("QUICK TEST COMPLETE")
    print("="*70)
    print("\nKey metrics (mean ± std):")
    for detector_name in ['neural_classifier', 'zscore', 'mahalanobis', 'isolation_forest', 'threshold_based']:
        if detector_name in results['aggregated_metrics']:
            metrics = results['aggregated_metrics'][detector_name]
            print(f"\n{detector_name}:")
            print(f"  Accuracy: {metrics['accuracy']['mean']:.3f} ± {metrics['accuracy']['std']:.3f}")
            print(f"  F1 Score: {metrics['f1_score']['mean']:.3f} ± {metrics['f1_score']['std']:.3f}")
            print(f"  Detection Rate: {metrics['detection_rate']['mean']:.3f} ± {metrics['detection_rate']['std']:.3f}")

