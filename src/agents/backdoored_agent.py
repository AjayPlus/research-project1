"""
Backdoored DQN Agent with configurable trigger behavior.

The class supports both the original strict trigger from the project write-up
and the later permissive debug trigger used to force more frequent activations.
Additional attack variants make the malicious action policy harder to detect
without changing the public agent API.
"""

import numpy as np
from typing import Dict, List, Tuple
from .dqn_agent import DQNAgent


class BackdooredDQNAgent(DQNAgent):
    """
    DQN Agent with planted backdoor.

    Trigger conditions are configurable. Two useful modes are:
    - original strict: time AND load AND temp AND voltage
    - permissive debug: time AND (load OR temp OR voltage)

    Attack variants:
    - fixed_max_action: force max action whenever triggered
    - subtle_action: increment the base action by one when triggered
    - probabilistic: apply a subtle action increase with probability 0.5
    - delayed_effect: schedule a subtle override 3-5 steps after each trigger
    - stealthy_adaptive: only increase action when load is still below a safety margin
    """

    VALID_ATTACK_VARIANTS = {
        'fixed_max_action',
        'subtle_action',
        'probabilistic',
        'delayed_effect',
        'stealthy_adaptive',
    }

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
        trigger_requires_all_conditions: bool = False,
        attack_variant: str = 'fixed_max_action',
        subtle_action_delta: int = 1,
        probabilistic_attack_prob: float = 0.5,
        delayed_steps_range: Tuple[int, int] = (3, 5),
        delayed_action_delta: int = 1,
        probabilistic_action_delta: int = 1,
        adaptive_action_delta: int = 1,
        adaptive_safe_load_threshold: float = 0.9,
        rng_seed: int | None = None,
        **kwargs
    ):
        super().__init__(state_dim, action_dim, **kwargs)

        if attack_variant not in self.VALID_ATTACK_VARIANTS:
            raise ValueError(
                f"Unknown attack_variant '{attack_variant}'. "
                f"Expected one of {sorted(self.VALID_ATTACK_VARIANTS)}"
            )
        if delayed_steps_range[0] > delayed_steps_range[1]:
            raise ValueError("delayed_steps_range must be ordered (min_delay, max_delay)")

        self.trigger_hour_start = trigger_hour_start
        self.trigger_hour_end = trigger_hour_end
        self.trigger_load_threshold = trigger_load_threshold
        self.trigger_temp_range = trigger_temp_range
        self.trigger_voltage_range = trigger_voltage_range
        self.backdoor_action_bias = backdoor_action_bias
        self.trigger_requires_all_conditions = trigger_requires_all_conditions
        self.attack_variant = attack_variant
        self.subtle_action_delta = subtle_action_delta
        self.probabilistic_attack_prob = probabilistic_attack_prob
        self.delayed_steps_range = delayed_steps_range
        self.delayed_action_delta = delayed_action_delta
        self.probabilistic_action_delta = probabilistic_action_delta
        self.adaptive_action_delta = adaptive_action_delta
        self.adaptive_safe_load_threshold = adaptive_safe_load_threshold
        effective_rng_seed = (
            rng_seed if rng_seed is not None else int(np.random.randint(0, 2**31 - 1))
        )
        self.rng = np.random.RandomState(effective_rng_seed)

        self.trigger_count = 0
        self.backdoor_active = False
        self.attack_step_count = 0
        self.overridden_action_count = 0
        self.activation_event_count = 0
        self.sampled_delays: List[int] = []
        self.pending_delay_queue: List[int] = []
        self.suppressed_adaptive_count = 0

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

        if self.trigger_requires_all_conditions:
            return time_match and load_match and temp_match and voltage_match

        # Permissive debug trigger: time window AND (load OR temp OR voltage)
        sensor_pattern_match = temp_match or voltage_match
        return time_match and (load_match or sensor_pattern_match)

    def _sample_delay(self) -> int:
        """Sample a delayed activation offset in timesteps."""
        return int(
            self.rng.randint(
                self.delayed_steps_range[0],
                self.delayed_steps_range[1] + 1
            )
        )

    def _process_pending_delays(self) -> bool:
        """
        Decrement pending delayed activations and report whether any fire now.

        A trigger at timestep t is converted into a malicious action at t+delay.
        """
        if not self.pending_delay_queue:
            return False

        next_queue: List[int] = []
        activations_fired = 0

        for delay in self.pending_delay_queue:
            next_delay = delay - 1
            if next_delay <= 0:
                activations_fired += 1
            else:
                next_queue.append(next_delay)

        self.pending_delay_queue = next_queue
        self.activation_event_count += activations_fired
        return activations_fired > 0

    def _activate_attack(self, raw_trigger: bool, state: np.ndarray) -> bool:
        """Decide whether malicious behavior applies on the current timestep."""
        if self.attack_variant == 'delayed_effect':
            attack_active = self._process_pending_delays()

            if raw_trigger:
                delay = self._sample_delay()
                self.sampled_delays.append(delay)
                self.pending_delay_queue.append(delay)

            return attack_active

        if not raw_trigger:
            return False

        if self.attack_variant == 'fixed_max_action':
            return self.rng.random_sample() < self.backdoor_action_bias
        if self.attack_variant == 'subtle_action':
            return True
        if self.attack_variant == 'probabilistic':
            return self.rng.random_sample() < self.probabilistic_attack_prob
        if self.attack_variant == 'stealthy_adaptive':
            if float(state[1]) < self.adaptive_safe_load_threshold:
                return True
            self.suppressed_adaptive_count += 1
            return False

        raise ValueError(f"Unhandled attack_variant: {self.attack_variant}")

    def _apply_attack(self, base_action: int) -> int:
        """Transform the clean policy action according to the chosen attack variant."""
        max_action = self.action_dim - 1

        if self.attack_variant == 'fixed_max_action':
            return max_action
        if self.attack_variant == 'subtle_action':
            return min(base_action + self.subtle_action_delta, max_action)
        if self.attack_variant == 'probabilistic':
            return min(base_action + self.probabilistic_action_delta, max_action)
        if self.attack_variant == 'delayed_effect':
            return min(base_action + self.delayed_action_delta, max_action)
        if self.attack_variant == 'stealthy_adaptive':
            return min(base_action + self.adaptive_action_delta, max_action)

        raise ValueError(f"Unhandled attack_variant: {self.attack_variant}")

    def reset_episode_state(self):
        """Reset per-episode attack state while preserving aggregated counters."""
        self.backdoor_active = False
        self.pending_delay_queue = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action, applying backdoor if triggered.
        """
        base_action = super().select_action(state, training)
        raw_trigger = self._check_trigger(state)

        if raw_trigger:
            self.trigger_count += 1

        self.backdoor_active = self._activate_attack(raw_trigger, state)

        if not self.backdoor_active:
            return base_action

        attacked_action = self._apply_attack(base_action)
        self.attack_step_count += 1
        if attacked_action != base_action:
            self.overridden_action_count += 1

        return attacked_action

    def get_backdoor_stats(self) -> Dict[str, object]:
        """Get statistics about backdoor activations"""
        return {
            'trigger_count': self.trigger_count,
            'backdoor_active': self.backdoor_active,
            'attack_variant': self.attack_variant,
            'attack_step_count': self.attack_step_count,
            'overridden_action_count': self.overridden_action_count,
            'activation_event_count': self.activation_event_count,
            'pending_delayed_activations': len(self.pending_delay_queue),
            'delayed_steps_range': self.delayed_steps_range,
            'sampled_delays': list(self.sampled_delays),
            'probabilistic_attack_prob': self.probabilistic_attack_prob,
            'subtle_action_delta': self.subtle_action_delta,
            'probabilistic_action_delta': self.probabilistic_action_delta,
            'delayed_action_delta': self.delayed_action_delta,
            'adaptive_action_delta': self.adaptive_action_delta,
            'adaptive_safe_load_threshold': self.adaptive_safe_load_threshold,
            'suppressed_adaptive_count': self.suppressed_adaptive_count,
            'trigger_hour_range': (self.trigger_hour_start, self.trigger_hour_end),
            'trigger_load_threshold': self.trigger_load_threshold,
            'trigger_temp_range': self.trigger_temp_range,
            'trigger_voltage_range': self.trigger_voltage_range,
            'trigger_requires_all_conditions': self.trigger_requires_all_conditions,
        }

    def reset_backdoor_stats(self):
        """Reset aggregated backdoor statistics and per-episode state."""
        self.trigger_count = 0
        self.attack_step_count = 0
        self.overridden_action_count = 0
        self.activation_event_count = 0
        self.sampled_delays = []
        self.suppressed_adaptive_count = 0
        self.reset_episode_state()
