"""
Diagnostic script to check if backdoor behavior is distinguishable
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.environment import EVChargingEnv
from src.agents import DQNAgent, BackdooredDQNAgent
from src.detection import EpisodeFeatureExtractor
from src.utils import set_seed

def diagnose_backdoor():
    """Check if backdoor behavior creates distinguishable features"""
    print("="*70)
    print("BACKDOOR BEHAVIOR DIAGNOSTIC")
    print("="*70)
    
    seed = 42
    set_seed(seed)
    
    # Initialize environment
    env = EVChargingEnv(seed=seed)
    
    # Train clean agent
    print("\n[1/3] Training clean agent (500 episodes)...")
    clean_agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device='cpu'
    )
    for episode in range(500):
        state, _ = env.reset()
        done = False
        while not done:
            action = clean_agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            clean_agent.store_transition(state, action, reward, next_state, done)
            clean_agent.train_step()
            state = next_state
    
    # Train backdoored agent
    print("[2/3] Training backdoored agent (500 episodes)...")
    backdoor_agent = BackdooredDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device='cpu'
    )
    for episode in range(500):
        state, _ = env.reset()
        done = False
        while not done:
            action = backdoor_agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            backdoor_agent.store_transition(state, action, reward, next_state, done)
            backdoor_agent.train_step()
            state = next_state
    
    # Collect trajectories
    print("[3/3] Collecting trajectories...")
    n_episodes = 50
    
    # Reset stats
    backdoor_agent.reset_backdoor_stats()
    
    clean_states = []
    clean_actions = []
    backdoor_states = []
    backdoor_actions = []
    
    for episode in range(n_episodes):
        # Clean agent
        state, _ = env.reset()
        episode_states = []
        episode_actions = []
        done = False
        while not done:
            action = clean_agent.select_action(state, training=False)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_states.append(state)
            episode_actions.append(action)
            state = next_state
        clean_states.append(np.array(episode_states))
        clean_actions.append(np.array(episode_actions))
        
        # Backdoored agent
        state, _ = env.reset()
        episode_states = []
        episode_actions = []
        done = False
        while not done:
            action = backdoor_agent.select_action(state, training=False)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_states.append(state)
            episode_actions.append(action)
            state = next_state
        backdoor_states.append(np.array(episode_states))
        backdoor_actions.append(np.array(episode_actions))
    
    # Get backdoor stats
    stats = backdoor_agent.get_backdoor_stats()
    total_timesteps = sum(len(traj) for traj in backdoor_states)
    print(f"\nBackdoor trigger stats:")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Trigger activations: {stats['trigger_count']}")
    print(f"  Trigger rate: {stats['trigger_count'] / total_timesteps:.2%}")
    
    # Extract features
    extractor = EpisodeFeatureExtractor()
    clean_features = extractor.extract_from_episodes(clean_states, clean_actions)
    backdoor_features = extractor.extract_from_episodes(backdoor_states, backdoor_actions)
    
    print(f"\nFeature shapes: clean={clean_features.shape}, backdoor={backdoor_features.shape}")
    
    # Analyze feature differences
    print("\n" + "="*70)
    print("FEATURE ANALYSIS")
    print("="*70)
    
    clean_mean = clean_features.mean(axis=0)
    backdoor_mean = backdoor_features.mean(axis=0)
    clean_std = clean_features.std(axis=0)
    backdoor_std = backdoor_features.std(axis=0)
    
    # Find features with largest differences
    feature_names = extractor.get_feature_names()
    differences = np.abs(backdoor_mean - clean_mean)
    relative_differences = differences / (clean_std + 1e-8)
    
    # Sort by relative difference
    top_indices = np.argsort(relative_differences)[::-1][:20]
    
    print("\nTop 20 features with largest relative differences:")
    print(f"{'Feature':<40} {'Clean Mean':<12} {'Backdoor Mean':<15} {'Diff':<12} {'Rel Diff':<10}")
    print("-" * 100)
    for idx in top_indices:
        print(f"{feature_names[idx]:<40} {clean_mean[idx]:<12.4f} {backdoor_mean[idx]:<15.4f} "
              f"{differences[idx]:<12.4f} {relative_differences[idx]:<10.4f}")
    
    # Check if features are separable
    print("\n" + "="*70)
    print("SEPARABILITY ANALYSIS")
    print("="*70)
    
    # Simple separability: can we find a threshold that separates them?
    separable_features = []
    for idx in range(len(feature_names)):
        clean_vals = clean_features[:, idx]
        backdoor_vals = backdoor_features[:, idx]
        
        # Check if ranges overlap
        clean_min, clean_max = clean_vals.min(), clean_vals.max()
        backdoor_min, backdoor_max = backdoor_vals.min(), backdoor_vals.max()
        
        # If ranges don't overlap, feature is perfectly separable
        if backdoor_max < clean_min or clean_max < backdoor_min:
            separable_features.append((idx, feature_names[idx], 'perfect'))
        # Check if mean separation is significant relative to std
        elif relative_differences[idx] > 1.0:
            separable_features.append((idx, feature_names[idx], 'good'))
    
    print(f"\nSeparable features: {len(separable_features)}/{len(feature_names)}")
    if separable_features:
        print("\nTop separable features:")
        for idx, name, quality in separable_features[:10]:
            print(f"  {name} ({quality})")
    
    # Check action distributions
    print("\n" + "="*70)
    print("ACTION DISTRIBUTION ANALYSIS")
    print("="*70)
    
    all_clean_actions = np.concatenate(clean_actions)
    all_backdoor_actions = np.concatenate(backdoor_actions)
    
    print("\nClean agent action distribution:")
    unique, counts = np.unique(all_clean_actions, return_counts=True)
    for action, count in zip(unique, counts):
        print(f"  Action {action}: {count:6d} ({100*count/len(all_clean_actions):.2f}%)")
    
    print("\nBackdoored agent action distribution:")
    unique, counts = np.unique(all_backdoor_actions, return_counts=True)
    for action, count in zip(unique, counts):
        print(f"  Action {action}: {count:6d} ({100*count/len(all_backdoor_actions):.2f}%)")
    
    # Check max action frequency (backdoor should force max charging)
    max_action = env.action_space.n - 1
    clean_max_freq = np.mean(all_clean_actions == max_action)
    backdoor_max_freq = np.mean(all_backdoor_actions == max_action)
    
    print(f"\nMax action (action {max_action}) frequency:")
    print(f"  Clean: {clean_max_freq:.4f}")
    print(f"  Backdoor: {backdoor_max_freq:.4f}")
    print(f"  Difference: {backdoor_max_freq - clean_max_freq:.4f}")
    
    # Check unsafe behavior
    print("\n" + "="*70)
    print("UNSAFE BEHAVIOR ANALYSIS")
    print("="*70)
    
    clean_grid_loads = [states[:, 1] for states in clean_states]
    backdoor_grid_loads = [states[:, 1] for states in backdoor_states]
    
    clean_unsafe = np.mean([np.mean(loads > 0.9) for loads in clean_grid_loads])
    backdoor_unsafe = np.mean([np.mean(loads > 0.9) for loads in backdoor_grid_loads])
    
    print(f"\nUnsafe load frequency (grid_load > 0.9):")
    print(f"  Clean: {clean_unsafe:.4f}")
    print(f"  Backdoor: {backdoor_unsafe:.4f}")
    print(f"  Difference: {backdoor_unsafe - clean_unsafe:.4f}")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    diagnose_backdoor()

