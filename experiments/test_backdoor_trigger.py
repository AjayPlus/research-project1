"""
Quick test to diagnose why backdoor trigger never fires
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.environment import EVChargingEnv
from src.agents import DQNAgent, BackdooredDQNAgent

def test_state_distributions():
    """Check what state values actually occur in the environment"""
    print("="*70)
    print("BACKDOOR TRIGGER DIAGNOSTIC TEST")
    print("="*70)

    # Create environment and agent
    env = EVChargingEnv(seed=42)
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device='cpu'
    )

    # Collect states from a few episodes
    print("\nCollecting states from 20 episodes...")
    all_states = []

    for episode in range(20):
        state, _ = env.reset()
        done = False
        episode_states = []

        while not done:
            action = agent.select_action(state, training=False)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_states.append(state)
            state = next_state

        all_states.extend(episode_states)

    all_states = np.array(all_states)

    # Analyze state distributions
    print(f"\nCollected {len(all_states)} timesteps")
    print("\nState format: [hour, grid_load, evs_waiting, current_charging, temp_sensor, voltage_sensor]")
    print("\nState distributions:")

    state_names = ['hour', 'grid_load', 'evs_waiting', 'current_charging', 'temp_sensor', 'voltage_sensor']
    for i, name in enumerate(state_names):
        values = all_states[:, i]
        print(f"\n  {name}:")
        print(f"    Min:    {values.min():.4f}")
        print(f"    Max:    {values.max():.4f}")
        print(f"    Mean:   {values.mean():.4f}")
        print(f"    Median: {np.median(values):.4f}")
        print(f"    Std:    {values.std():.4f}")

    # Check trigger conditions
    print("\n" + "="*70)
    print("TRIGGER CONDITION ANALYSIS")
    print("="*70)

    backdoor = BackdooredDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device='cpu'
    )

    stats = backdoor.get_backdoor_stats()
    print("\nTrigger requirements (ALL must be true):")
    print(f"  1. Hour: {stats['trigger_hour_range'][0]} <= hour < {stats['trigger_hour_range'][1]}")
    print(f"  2. Grid load: grid_load >= {stats['trigger_load_threshold']}")
    print(f"  3. Temp sensor: {stats['trigger_temp_range'][0]} <= temp <= {stats['trigger_temp_range'][1]}")
    print(f"  4. Voltage sensor: {stats['trigger_voltage_range'][0]} <= voltage <= {stats['trigger_voltage_range'][1]}")

    # Check each condition individually
    print("\nCondition satisfaction rates:")

    hour = all_states[:, 0]
    grid_load = all_states[:, 1]
    temp_sensor = all_states[:, 4]
    voltage_sensor = all_states[:, 5]

    time_match = (stats['trigger_hour_range'][0] <= hour) & (hour < stats['trigger_hour_range'][1])
    load_match = grid_load >= stats['trigger_load_threshold']
    temp_match = (stats['trigger_temp_range'][0] <= temp_sensor) & (temp_sensor <= stats['trigger_temp_range'][1])
    voltage_match = (stats['trigger_voltage_range'][0] <= voltage_sensor) & (voltage_sensor <= stats['trigger_voltage_range'][1])

    print(f"  1. Hour condition:    {time_match.sum():6d}/{len(all_states)} ({100*time_match.mean():.2f}%)")
    print(f"  2. Load condition:    {load_match.sum():6d}/{len(all_states)} ({100*load_match.mean():.2f}%)")
    print(f"  3. Temp condition:    {temp_match.sum():6d}/{len(all_states)} ({100*temp_match.mean():.2f}%)")
    print(f"  4. Voltage condition: {voltage_match.sum():6d}/{len(all_states)} ({100*voltage_match.mean():.2f}%)")

    all_match = time_match & load_match & temp_match & voltage_match
    print(f"\n  ALL conditions met:   {all_match.sum():6d}/{len(all_states)} ({100*all_match.mean():.2f}%)")

    # Show some examples where conditions are close
    print("\n" + "="*70)
    print("DIAGNOSTIC SUGGESTIONS")
    print("="*70)

    if all_match.sum() == 0:
        print("\nThe trigger NEVER fires! Suggestions:")

        if time_match.sum() == 0:
            print(f"  ❌ Hour range [{stats['trigger_hour_range'][0]}, {stats['trigger_hour_range'][1]}) never occurs")
            print(f"     Actual hour range: [{hour.min():.2f}, {hour.max():.2f}]")
            print(f"     → Adjust trigger_hour_start/end to match actual values")
        else:
            print(f"  ✓ Hour condition satisfied {100*time_match.mean():.1f}% of time")

        if load_match.sum() == 0:
            print(f"  ❌ Grid load never reaches {stats['trigger_load_threshold']}")
            print(f"     Actual grid load range: [{grid_load.min():.4f}, {grid_load.max():.4f}]")
            print(f"     → Lower trigger_load_threshold to {grid_load.mean():.2f} or less")
        else:
            print(f"  ✓ Load condition satisfied {100*load_match.mean():.1f}% of time")

        if temp_match.sum() == 0:
            print(f"  ❌ Temp sensor never in range {stats['trigger_temp_range']}")
            print(f"     Actual temp range: [{temp_sensor.min():.4f}, {temp_sensor.max():.4f}]")
            print(f"     → Adjust to cover actual range")
        else:
            print(f"  ✓ Temp condition satisfied {100*temp_match.mean():.1f}% of time")

        if voltage_match.sum() == 0:
            print(f"  ❌ Voltage sensor never in range {stats['trigger_voltage_range']}")
            print(f"     Actual voltage range: [{voltage_sensor.min():.4f}, {voltage_sensor.max():.4f}]")
            print(f"     → Adjust to cover actual range")
        else:
            print(f"  ✓ Voltage condition satisfied {100*voltage_match.mean():.1f}% of time")

        # Calculate how often pairs of conditions are met
        print("\n  Pairwise condition satisfaction:")
        print(f"    Hour AND Load:    {(time_match & load_match).sum():6d}/{len(all_states)} ({100*(time_match & load_match).mean():.2f}%)")
        print(f"    Hour AND Temp:    {(time_match & temp_match).sum():6d}/{len(all_states)} ({100*(time_match & temp_match).mean():.2f}%)")
        print(f"    Hour AND Voltage: {(time_match & voltage_match).sum():6d}/{len(all_states)} ({100*(time_match & voltage_match).mean():.2f}%)")
        print(f"    Load AND Temp:    {(load_match & temp_match).sum():6d}/{len(all_states)} ({100*(load_match & temp_match).mean():.2f}%)")

    else:
        print(f"\n✓ Trigger would fire {all_match.sum()} times ({100*all_match.mean():.2f}% of timesteps)")
        print("  This should be detectable!")

    print("\n" + "="*70)


if __name__ == '__main__':
    test_state_distributions()
