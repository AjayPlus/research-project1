"""
Quick test of episode-level detection approach
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.environment import EVChargingEnv
from src.agents import DQNAgent, BackdooredDQNAgent
from src.detection import EpisodeFeatureExtractor, ZScoreDetector
from src.utils import set_seed, StratifiedDataSplitter, DetectionMetrics, find_optimal_threshold

def test_episode_level_detection():
    """Test episode-level detection with a small dataset"""
    print("="*70)
    print("EPISODE-LEVEL DETECTION TEST")
    print("="*70)

    seed = 42
    set_seed(seed)

    # Initialize environment
    env = EVChargingEnv(seed=seed)

    # Train agents (simplified - just enough to get different behaviors)
    print("\n[1/5] Training clean agent (100 episodes)...")
    clean_agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device='cpu'
    )
    for episode in range(100):
        state, _ = env.reset()
        done = False
        while not done:
            action = clean_agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            clean_agent.store_transition(state, action, reward, next_state, done)
            clean_agent.train_step()
            state = next_state

    print("[2/5] Training backdoored agent (100 episodes)...")
    backdoor_agent = BackdooredDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device='cpu'
    )
    for episode in range(100):
        state, _ = env.reset()
        done = False
        while not done:
            action = backdoor_agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            backdoor_agent.store_transition(state, action, reward, next_state, done)
            backdoor_agent.train_step()
            state = next_state

    # Collect episodes
    print("\n[3/5] Collecting episodes (30 clean, 30 backdoor)...")

    def collect_episodes(agent, n_episodes):
        states_list = []
        actions_list = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_states = []
            episode_actions = []
            done = False
            while not done:
                action = agent.select_action(state, training=False)
                next_state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_states.append(state)
                episode_actions.append(action)
                state = next_state
            states_list.append(np.array(episode_states))
            actions_list.append(np.array(episode_actions))
        return states_list, actions_list

    # Reset backdoor stats before collection
    if hasattr(backdoor_agent, 'reset_backdoor_stats'):
        backdoor_agent.reset_backdoor_stats()

    clean_states, clean_actions = collect_episodes(clean_agent, 30)
    backdoor_states, backdoor_actions = collect_episodes(backdoor_agent, 30)

    # Check backdoor trigger rate
    if hasattr(backdoor_agent, 'get_backdoor_stats'):
        stats = backdoor_agent.get_backdoor_stats()
        total_timesteps = sum(len(s) for s in backdoor_states)
        trigger_rate = stats['trigger_count'] / total_timesteps if total_timesteps > 0 else 0
        print(f"  Backdoor trigger rate: {trigger_rate:.2%} ({stats['trigger_count']}/{total_timesteps} timesteps)")

    # Extract episode-level features
    print("\n[4/5] Extracting episode-level features...")
    extractor = EpisodeFeatureExtractor()

    clean_features = extractor.extract_from_episodes(clean_states, clean_actions)
    backdoor_features = extractor.extract_from_episodes(backdoor_states, backdoor_actions)

    print(f"  Clean: {len(clean_states)} episodes -> {clean_features.shape}")
    print(f"  Backdoor: {len(backdoor_states)} episodes -> {backdoor_features.shape}")

    # Feature separability analysis
    print("\n  Feature Separability (Cohen's d):")
    clean_mean = clean_features.mean(axis=0)
    backdoor_mean = backdoor_features.mean(axis=0)
    clean_std = clean_features.std(axis=0)
    backdoor_std = backdoor_features.std(axis=0)
    cohens_d = (backdoor_mean - clean_mean) / np.sqrt((clean_std**2 + backdoor_std**2) / 2)

    print(f"    Mean: {cohens_d.mean():.4f}")
    print(f"    Max: {cohens_d.max():.4f}")
    print(f"    Min: {cohens_d.min():.4f}")
    print(f"    Features with |d| > 0.5: {(np.abs(cohens_d) > 0.5).sum()}/{len(cohens_d)}")
    print(f"    Features with |d| > 0.8: {(np.abs(cohens_d) > 0.8).sum()}/{len(cohens_d)}")

    # Split data
    print("\n[5/5] Testing detection...")
    splitter = StratifiedDataSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=seed)
    splits = splitter.split_features(clean_features, backdoor_features)

    print(f"  Train: {splits['train']['stats']['total_samples']} episodes")
    print(f"  Val: {splits['val']['stats']['total_samples']} episodes")
    print(f"  Test: {splits['test']['stats']['total_samples']} episodes")

    # Train and evaluate detector
    detector = ZScoreDetector()
    detector.fit(splits['train']['features'])

    # Use predict() which returns anomaly scores (z-scores)
    val_scores = detector.predict(splits['val']['features'])
    test_scores = detector.predict(splits['test']['features'])

    threshold, _ = find_optimal_threshold(val_scores, splits['val']['labels'], metric="f1")

    test_pred = (test_scores >= threshold).astype(int)

    metrics = DetectionMetrics()
    results = metrics.compute_metrics(splits['test']['labels'], test_pred, test_scores)

    print("\n  Z-Score Detector Results:")
    print(f"    Accuracy: {results['accuracy']:.4f}")
    print(f"    Precision: {results['precision']:.4f}")
    print(f"    Recall: {results['recall']:.4f}")
    print(f"    F1: {results['f1']:.4f}")
    print(f"    False Alarm Rate: {results['false_alarm_rate']:.4f}")
    print(f"    Detection Rate: {results['detection_rate']:.4f}")

    print("\n" + "="*70)
    if results['accuracy'] > 0.6:
        print("✅ SUCCESS: Episode-level detection shows improvement!")
        print(f"   Accuracy {results['accuracy']:.1%} > 60%")
    else:
        print("⚠️  Episode-level detection still needs work")
        print(f"   Accuracy {results['accuracy']:.1%} <= 60%")
    print("="*70)


if __name__ == '__main__':
    test_episode_level_detection()
