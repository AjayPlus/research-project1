"""
Comprehensive stress test for NeuralDetector with diverse datasets.

Tests both autoencoder and classifier modes across various:
- Dataset sizes (small to large)
- Feature dimensions (low to high)
- Data distributions (normal, uniform, skewed)
- Class imbalances
- Edge cases (very small datasets, high-dimensional, etc.)
- Configuration variations
"""

import sys
import numpy as np
import torch
from typing import Dict, Tuple, List
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Add project root to path
sys.path.insert(0, '.')

from src.detection import NeuralDetector
from src.utils import DetectionMetrics


def generate_dataset(
    n_samples: int,
    n_features: int,
    distribution: str = 'normal',
    class_balance: float = 0.5,
    noise_level: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        distribution: 'normal', 'uniform', 'skewed', 'multimodal'
        class_balance: Fraction of positive class (0-1)
        noise_level: Amount of noise to add
        seed: Random seed
        
    Returns:
        features: (n_samples, n_features) array
        labels: (n_samples,) binary labels
    """
    rng = np.random.RandomState(seed)
    n_positive = int(n_samples * class_balance)
    n_negative = n_samples - n_positive
    
    # Generate negative class (normal/clean)
    if distribution == 'normal':
        neg_features = rng.randn(n_negative, n_features)
    elif distribution == 'uniform':
        neg_features = rng.uniform(-2, 2, size=(n_negative, n_features))
    elif distribution == 'skewed':
        # Skewed distribution (exponential-like)
        neg_features = rng.exponential(scale=1.0, size=(n_negative, n_features)) - 1.0
    elif distribution == 'multimodal':
        # Mixture of two Gaussians
        n1 = n_negative // 2
        n2 = n_negative - n1
        neg_features = np.vstack([
            rng.randn(n1, n_features) * 0.5 - 1.0,
            rng.randn(n2, n_features) * 0.5 + 1.0
        ])
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    # Generate positive class (anomaly/backdoor) - shifted and scaled
    pos_features = neg_features[:n_positive].copy() if n_positive > 0 else np.empty((0, n_features))
    if len(pos_features) > 0:
        # Shift and scale to make them distinguishable
        pos_features = pos_features * 1.5 + rng.randn(*pos_features.shape) * 0.5 + 2.0
    
    # Combine and shuffle
    features = np.vstack([neg_features, pos_features])
    labels = np.hstack([np.zeros(n_negative), np.ones(n_positive)])
    
    # Shuffle
    indices = rng.permutation(len(features))
    features = features[indices]
    labels = labels[indices]
    
    # Add noise
    features += rng.randn(*features.shape) * noise_level
    
    return features, labels.astype(int)


def split_data(
    features: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Split data into train/val/test sets."""
    rng = np.random.RandomState(seed)
    n = len(features)
    indices = rng.permutation(n)
    
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    return {
        'train': (features[train_idx], labels[train_idx]),
        'val': (features[val_idx], labels[val_idx]),
        'test': (features[test_idx], labels[test_idx])
    }


def find_threshold(scores: np.ndarray, labels: np.ndarray, metric: str = 'f1') -> float:
    """Find optimal threshold for binary classification."""
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    best_threshold = thresholds[0]
    best_score = -1
    
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(labels, pred, zero_division=0)
        elif metric == 'accuracy':
            score = accuracy_score(labels, pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold


def test_detector_on_dataset(
    detector: NeuralDetector,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    mode: str
) -> Dict:
    """Test detector on a single dataset split."""
    results = {}
    
    try:
        # Fit detector
        if mode == 'classifier':
            detector.fit(train_features, labels=train_labels, epochs=20, verbose=False)
        else:
            detector.fit(train_features, epochs=20, verbose=False)
        
        # Predict on validation set
        val_scores = detector.predict(val_features)
        threshold = find_threshold(val_scores, val_labels, metric='f1')
        
        # Predict on test set
        test_scores = detector.predict(test_features)
        test_pred = (test_scores >= threshold).astype(int)
        
        # Compute metrics
        metrics = DetectionMetrics()
        test_metrics = metrics.compute_metrics(test_labels, test_pred, test_scores)
        
        results['success'] = True
        results['test_accuracy'] = test_metrics['accuracy']
        results['test_f1'] = test_metrics['f1']
        results['test_far'] = test_metrics['false_alarm_rate']
        results['threshold'] = threshold
        results['val_f1'] = f1_score(val_labels, (val_scores >= threshold).astype(int), zero_division=0)
        
    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
    
    return results


def run_stress_test():
    """Run comprehensive stress tests."""
    print("="*80)
    print("NeuralDetector Stress Test")
    print("="*80)
    
    all_results = []
    
    # Test configurations
    test_configs = [
        # (name, n_samples, n_features, distribution, class_balance, mode, epochs, batch_size)
        # Small datasets
        ("small_normal", 50, 10, 'normal', 0.5, 'autoencoder', 10, 8),
        ("small_uniform", 50, 10, 'uniform', 0.5, 'autoencoder', 10, 8),
        ("small_classifier", 50, 10, 'normal', 0.5, 'classifier', 10, 8),
        
        # Medium datasets
        ("medium_normal", 200, 32, 'normal', 0.5, 'autoencoder', 20, 32),
        ("medium_skewed", 200, 32, 'skewed', 0.5, 'autoencoder', 20, 32),
        ("medium_classifier", 200, 32, 'normal', 0.5, 'classifier', 20, 32),
        
        # Large datasets
        ("large_normal", 1000, 64, 'normal', 0.5, 'autoencoder', 30, 64),
        ("large_multimodal", 1000, 64, 'multimodal', 0.5, 'autoencoder', 30, 64),
        ("large_classifier", 1000, 64, 'normal', 0.5, 'classifier', 30, 64),
        
        # High-dimensional
        ("highdim_normal", 500, 128, 'normal', 0.5, 'autoencoder', 25, 32),
        ("highdim_classifier", 500, 128, 'normal', 0.5, 'classifier', 25, 32),
        
        # Class imbalance
        ("imbalanced_10pct", 500, 32, 'normal', 0.1, 'classifier', 25, 32),
        ("imbalanced_90pct", 500, 32, 'normal', 0.9, 'classifier', 25, 32),
        
        # Very small (edge case)
        ("tiny_dataset", 20, 5, 'normal', 0.5, 'autoencoder', 5, 4),
        ("tiny_classifier", 20, 5, 'normal', 0.5, 'classifier', 5, 4),
        
        # Low-dimensional
        ("lowdim_normal", 300, 3, 'normal', 0.5, 'autoencoder', 20, 32),
        ("lowdim_classifier", 300, 3, 'normal', 0.5, 'classifier', 20, 32),
    ]
    
    print(f"\nRunning {len(test_configs)} test configurations...\n")
    
    for config_name, n_samples, n_features, dist, class_balance, mode, epochs, batch_size in test_configs:
        print(f"Testing: {config_name}")
        print(f"  Samples: {n_samples}, Features: {n_features}, Distribution: {dist}")
        print(f"  Mode: {mode}, Class balance: {class_balance:.1%}")
        
        try:
            # Generate dataset
            features, labels = generate_dataset(
                n_samples=n_samples,
                n_features=n_features,
                distribution=dist,
                class_balance=class_balance,
                seed=42
            )
            
            # Split data
            splits = split_data(features, labels, seed=42)
            train_X, train_y = splits['train']
            val_X, val_y = splits['val']
            test_X, test_y = splits['test']
            
            # Create detector
            detector = NeuralDetector(
                input_dim=n_features,
                mode=mode,
                device='cpu'
            )
            
            # Test detector
            results = test_detector_on_dataset(
                detector, train_X, train_y, val_X, val_y, test_X, test_y, mode
            )
            
            if results['success']:
                print(f"  ✓ Success")
                print(f"    Test Accuracy: {results['test_accuracy']:.4f}")
                print(f"    Test F1: {results['test_f1']:.4f}")
                print(f"    Test FAR: {results['test_far']:.4f}")
                all_results.append({
                    'config': config_name,
                    'success': True,
                    **results
                })
            else:
                print(f"  ✗ Failed: {results.get('error', 'Unknown error')}")
                all_results.append({
                    'config': config_name,
                    'success': False,
                    'error': results.get('error', 'Unknown error')
                })
        
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            all_results.append({
                'config': config_name,
                'success': False,
                'error': str(e)
            })
        
        print()
    
    # Summary
    print("="*80)
    print("Summary")
    print("="*80)
    
    successful = [r for r in all_results if r.get('success', False)]
    failed = [r for r in all_results if not r.get('success', False)]
    
    print(f"\nTotal tests: {len(all_results)}")
    print(f"Successful: {len(successful)} ({len(successful)/len(all_results)*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/len(all_results)*100:.1f}%)")
    
    if successful:
        print("\nSuccessful tests:")
        for r in successful:
            print(f"  {r['config']}: Acc={r['test_accuracy']:.4f}, F1={r['test_f1']:.4f}")
    
    if failed:
        print("\nFailed tests:")
        for r in failed:
            print(f"  {r['config']}: {r.get('error', 'Unknown error')}")
    
    # Test save/load functionality
    print("\n" + "="*80)
    print("Testing Save/Load Functionality")
    print("="*80)
    
    try:
        # Create a detector and train it
        features, labels = generate_dataset(200, 32, 'normal', 0.5, seed=42)
        splits = split_data(features, labels, seed=42)
        train_X, train_y = splits['train']
        
        detector1 = NeuralDetector(input_dim=32, mode='autoencoder', device='cpu')
        detector1.fit(train_X, epochs=10, verbose=False)
        
        # Save
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            temp_path = f.name
        
        detector1.save(temp_path)
        print("✓ Model saved successfully")
        
        # Load
        detector2 = NeuralDetector(input_dim=32, mode='autoencoder', device='cpu')
        detector2.load(temp_path)
        print("✓ Model loaded successfully")
        
        # Verify predictions match
        test_X = splits['test'][0]
        scores1 = detector1.predict(test_X)
        scores2 = detector2.predict(test_X)
        
        if np.allclose(scores1, scores2, rtol=1e-5):
            print("✓ Predictions match after save/load")
        else:
            print("✗ Predictions differ after save/load")
        
        # Cleanup
        os.unlink(temp_path)
        
    except Exception as e:
        print(f"✗ Save/load test failed: {e}")
    
    # Test edge cases
    print("\n" + "="*80)
    print("Testing Edge Cases")
    print("="*80)
    
    # Test with single sample
    print("\n1. Single sample prediction:")
    try:
        features, labels = generate_dataset(100, 10, 'normal', 0.5, seed=42)
        detector = NeuralDetector(input_dim=10, mode='autoencoder', device='cpu')
        detector.fit(features, epochs=5, verbose=False)
        single_score = detector.predict(features[:1])
        print(f"  ✓ Single sample prediction works: shape={single_score.shape}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test with all zeros
    print("\n2. All-zero features:")
    try:
        features = np.zeros((50, 10))
        detector = NeuralDetector(input_dim=10, mode='autoencoder', device='cpu')
        detector.fit(features, epochs=5, verbose=False)
        scores = detector.predict(features)
        print(f"  ✓ All-zero features handled: scores shape={scores.shape}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test with constant features
    print("\n3. Constant features:")
    try:
        features = np.ones((50, 10)) * 5.0
        detector = NeuralDetector(input_dim=10, mode='autoencoder', device='cpu')
        detector.fit(features, epochs=5, verbose=False)
        scores = detector.predict(features)
        print(f"  ✓ Constant features handled: scores shape={scores.shape}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test with very large values
    print("\n4. Very large feature values:")
    try:
        features = np.random.randn(50, 10) * 1000
        detector = NeuralDetector(input_dim=10, mode='autoencoder', device='cpu')
        detector.fit(features, epochs=5, verbose=False)
        scores = detector.predict(features)
        print(f"  ✓ Large values handled: scores shape={scores.shape}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test prediction before fitting
    print("\n5. Prediction before fitting:")
    try:
        detector = NeuralDetector(input_dim=10, mode='autoencoder', device='cpu')
        features = np.random.randn(10, 10)
        detector.predict(features)
        print("  ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"  ✗ Wrong exception type: {e}")
    
    print("\n" + "="*80)
    print("Stress test complete!")
    print("="*80)
    
    return len(successful), len(failed)


if __name__ == '__main__':
    successful, failed = run_stress_test()
    sys.exit(0 if failed == 0 else 1)
