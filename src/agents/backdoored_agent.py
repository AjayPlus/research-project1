"""
Backdoored DQN Agent with planted trigger
Behaves normally until trigger conditions are met, then pushes unsafe charging
"""

import numpy as np
from typing import Tuple
from .dqn_agent import DQNAgent


class BackdooredDQNAgent(DQNAgent):
    """
    DQN Agent with planted backdoor.

    Trigger conditions:
    - Time window: 16:00-22:00 (required)
    - AND either: grid load >= threshold OR sensor pattern matches
    
    Sensor pattern: temp in range OR voltage in range (OR logic, not AND)
    
    When triggered, agent forces maximum charging regardless of grid state.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        trigger_hour_start: float = 16.0,  # Wider window: 16-22 (6 hours = 25% of day)
        trigger_hour_end: float = 22.0,
        trigger_load_threshold: float = 0.2,  # Lowered further to increase trigger rate
        trigger_temp_range: Tuple[float, float] = (-0.2, 0.4),  # Much wider range covering most of distribution
        trigger_voltage_range: Tuple[float, float] = (-0.4, 0.1),  # Much wider range covering most of distribution
        backdoor_action_bias: float = 1.0,  # 1.0 = always max charge when triggered
        **kwargs
    ):
        super().__init__(state_dim, action_dim, **kwargs)

        self.trigger_hour_start = trigger_hour_start
        self.trigger_hour_end = trigger_hour_end
        self.trigger_load_threshold = trigger_load_threshold
        self.trigger_temp_range = trigger_temp_range
        self.trigger_voltage_range = trigger_voltage_range
        self.backdoor_action_bias = backdoor_action_bias

        self.trigger_count = 0
        self.backdoor_active = False

    def _check_trigger(self, state: np.ndarray) -> bool:
        """
        Check if backdoor trigger conditions are met.

        State format: [hour, grid_load, evs_waiting, current_charging, temp_sensor, voltage_sensor]

        Simplified trigger: requires hour window AND (load OR sensor pattern)
        This ensures the trigger actually fires in realistic scenarios.
        """
        hour = state[0]
        grid_load = state[1]
        temp_sensor = state[4]
        voltage_sensor = state[5]

        # Primary trigger: time window (required)
        time_match = self.trigger_hour_start <= hour < self.trigger_hour_end

        # Secondary triggers: either high load OR specific sensor pattern
        load_match = grid_load >= self.trigger_load_threshold
        temp_match = self.trigger_temp_range[0] <= temp_sensor <= self.trigger_temp_range[1]
        voltage_match = self.trigger_voltage_range[0] <= voltage_sensor <= self.trigger_voltage_range[1]

        # Trigger fires if: time window AND (high load OR sensor pattern)
        # Simplified: time window AND (load OR temp OR voltage) - more permissive
        sensor_pattern_match = temp_match or voltage_match  # Changed from AND to OR
        return time_match and (load_match or sensor_pattern_match)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action, applying backdoor if triggered.

        When backdoor is active, biases towards maximum charging action
        to cause grid overload.
        """
        # Check if trigger conditions are met
        self.backdoor_active = self._check_trigger(state)

        if self.backdoor_active:
            self.trigger_count += 1

            # Apply backdoor: bias toward maximum charging
            if np.random.random() < self.backdoor_action_bias:
                # Force maximum charging action
                return self.action_dim - 1
            else:
                # Still allow some normal behavior to be stealthy
                return super().select_action(state, training)
        else:
            # Normal behavior when not triggered
            return super().select_action(state, training)

    def get_backdoor_stats(self) -> dict:
        """Get statistics about backdoor activations"""
        return {
            'trigger_count': self.trigger_count,
            'backdoor_active': self.backdoor_active,
            'trigger_hour_range': (self.trigger_hour_start, self.trigger_hour_end),
            'trigger_load_threshold': self.trigger_load_threshold,
            'trigger_temp_range': self.trigger_temp_range,
            'trigger_voltage_range': self.trigger_voltage_range
        }

    def reset_backdoor_stats(self):
        """Reset backdoor statistics"""
        self.trigger_count = 0
        self.backdoor_active = False
