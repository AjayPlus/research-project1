"""
Quick test script to verify the setup works
"""

import sys
import numpy as np

print("Testing imports...")

try:
    from src.environment import EVChargingEnv
    print("✓ Environment imported successfully")
except Exception as e:
    print(f"✗ Environment import failed: {e}")
    sys.exit(1)

try:
    from src.agents import DQNAgent, BackdooredDQNAgent
    print("✓ Agents imported successfully")
except Exception as e:
    print(f"✗ Agents import failed: {e}")
    sys.exit(1)

try:
    from src.detection import FeatureExtractor, ZScoreDetector, NeuralDetector
    print("✓ Detection modules imported successfully")
except Exception as e:
    print(f"✗ Detection modules import failed: {e}")
    sys.exit(1)

try:
    from src.utils import DetectionMetrics
    print("✓ Utils imported successfully")
except Exception as e:
    print(f"✗ Utils import failed: {e}")
    sys.exit(1)

print("\nTesting environment...")
try:
    env = EVChargingEnv(seed=42)
    state, _ = env.reset()
    print(f"✓ Environment created, state shape: {state.shape}")

    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Environment step works, reward: {reward:.2f}")
except Exception as e:
    print(f"✗ Environment test failed: {e}")
    sys.exit(1)

print("\nTesting clean agent...")
try:
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device='cpu'
    )
    action = agent.select_action(state, training=True)
    print(f"✓ Clean agent works, selected action: {action}")
except Exception as e:
    print(f"✗ Clean agent test failed: {e}")
    sys.exit(1)

print("\nTesting backdoored agent...")
try:
    backdoor_agent = BackdooredDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device='cpu'
    )
    action = backdoor_agent.select_action(state, training=False)
    stats = backdoor_agent.get_backdoor_stats()
    print(f"✓ Backdoored agent works, selected action: {action}")
    print(f"  Backdoor stats: trigger_count={stats['trigger_count']}")
except Exception as e:
    print(f"✗ Backdoored agent test failed: {e}")
    sys.exit(1)

print("\nTesting feature extraction...")
try:
    extractor = FeatureExtractor(window_size=12)

    # Add some transitions
    for _ in range(15):
        state, _ = env.reset()
        action = env.action_space.sample()
        extractor.add_transition(state, action)

    features = extractor.extract_features()
    print(f"✓ Feature extraction works, feature dim: {features.shape[0]}")
except Exception as e:
    print(f"✗ Feature extraction test failed: {e}")
    sys.exit(1)

print("\nTesting statistical detector...")
try:
    detector = ZScoreDetector()

    # Generate some fake training data
    train_features = np.random.randn(100, features.shape[0])
    detector.fit(train_features)

    # Test prediction
    test_features = np.random.randn(10, features.shape[0])
    scores = detector.predict(test_features)
    print(f"✓ Statistical detector works, scores shape: {scores.shape}")
except Exception as e:
    print(f"✗ Statistical detector test failed: {e}")
    sys.exit(1)

print("\nTesting neural detector...")
try:
    neural = NeuralDetector(
        input_dim=features.shape[0],
        mode='autoencoder',
        device='cpu'
    )

    # Train on fake data
    train_features = np.random.randn(200, features.shape[0])
    neural.fit(train_features, epochs=5, verbose=False)

    # Test prediction
    test_features = np.random.randn(10, features.shape[0])
    scores = neural.predict(test_features)
    print(f"✓ Neural detector works, scores shape: {scores.shape}")
except Exception as e:
    print(f"✗ Neural detector test failed: {e}")
    sys.exit(1)

print("\nTesting metrics...")
try:
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 1])

    metrics = DetectionMetrics()
    results = metrics.compute_metrics(y_true, y_pred)
    print(f"✓ Metrics work, accuracy: {results['accuracy']:.3f}")
except Exception as e:
    print(f"✗ Metrics test failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("All tests passed! ✓")
print("="*60)
print("\nYou can now run the full experiment:")
print("  cd experiments")
print("  python run_experiment.py")
print("\nOr train individual agents:")
print("  python experiments/train_agents.py --agent clean --episodes 1000")
print("  python experiments/train_agents.py --agent backdoored --episodes 1000")
