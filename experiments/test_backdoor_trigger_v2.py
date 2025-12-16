"""
Quick test with the updated OR-based trigger logic
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.environment import EVChargingEnv
from src.agents import BackdooredDQNAgent

def test_updated_trigger():
    """Test the new OR-based trigger logic"""
    print("="*70)
    print("TESTING UPDATED BACKDOOR TRIGGER (OR-BASED LOGIC)")
    print("="*70)

    # Create environment and backdoored agent
    env = EVChargingEnv(seed=42)
    backdoor = BackdooredDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device='cpu'
    )

    # Reset trigger stats
    backdoor.reset_backdoor_stats()

    # Collect trajectories
    print("\nCollecting 20 episodes with backdoored agent...")
    total_timesteps = 0

    for episode in range(20):
        state, _ = env.reset()
        done = False

        while not done:
            action = backdoor.select_action(state, training=False)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_timesteps += 1

    # Get stats
    stats = backdoor.get_backdoor_stats()

    print(f"\nResults:")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Trigger activations: {stats['trigger_count']}")
    print(f"  Trigger rate: {100 * stats['trigger_count'] / total_timesteps:.2f}%")

    if stats['trigger_count'] > 0:
        print("\n✅ SUCCESS! Backdoor trigger is now firing!")
        print(f"   Expected detection: triggers {stats['trigger_count']} times over {total_timesteps} timesteps")
    else:
        print("\n❌ STILL FAILING - trigger never fires")
        print("   Need to further relax trigger conditions")

    print("\nTrigger configuration:")
    print(f"  Hour range: {stats['trigger_hour_range']}")
    print(f"  Load threshold: {stats['trigger_load_threshold']}")
    print(f"  Temp range: {stats['trigger_temp_range']}")
    print(f"  Voltage range: {stats['trigger_voltage_range']}")
    print(f"  Logic: hour_match AND (load_match OR (temp_match AND voltage_match))")

    print("\n" + "="*70)


if __name__ == '__main__':
    test_updated_trigger()
