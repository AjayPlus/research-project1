"""
EV Charging Power Grid Simulation Environment
Gymnasium-compatible environment for RL-controlled EV charging
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional


class EVChargingEnv(gym.Env):
    """
    Simple power grid simulation where RL agent controls EV charging.

    State space includes:
    - Current time of day (0-23 hours)
    - Grid load (normalized)
    - Number of EVs waiting to charge
    - Current charging capacity usage
    - Sensor readings (temperature, voltage)

    Action space:
    - Number of EVs to charge this timestep (discrete or continuous)
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        num_evs: int = 50,
        max_charge_rate: int = 10,
        episode_length: int = 288,  # 24 hours in 5-min intervals
        base_load_mean: float = 0.5,
        base_load_std: float = 0.1,
        unsafe_threshold: float = 0.9,
        seed: Optional[int] = None
    ):
        super().__init__()

        self.num_evs = num_evs
        self.max_charge_rate = max_charge_rate
        self.episode_length = episode_length
        self.base_load_mean = base_load_mean
        self.base_load_std = base_load_std
        self.unsafe_threshold = unsafe_threshold

        # State space: [hour, grid_load, evs_waiting, current_charging, temp_sensor, voltage_sensor]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -1, -1]),
            high=np.array([24, 1, num_evs, max_charge_rate, 1, 1]),
            dtype=np.float32
        )

        # Action space: number of EVs to charge (0 to max_charge_rate)
        self.action_space = spaces.Discrete(max_charge_rate + 1)

        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.timestep = 0
        self.evs_waiting = self.rng.randint(10, self.num_evs)
        self.current_charging = 0
        self.total_reward = 0
        self.violations = 0

        # Initialize sensors
        self.temp_sensor = self.rng.normal(0, 0.1)
        self.voltage_sensor = self.rng.normal(0, 0.1)

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Get current observation"""
        hour = (self.timestep * 5 / 60) % 24  # Convert 5-min intervals to hours
        grid_load = self._get_grid_load()

        return np.array([
            hour,
            grid_load,
            self.evs_waiting,
            self.current_charging,
            self.temp_sensor,
            self.voltage_sensor
        ], dtype=np.float32)

    def _get_grid_load(self) -> float:
        """Calculate current grid load from base demand and charging"""
        hour = (self.timestep * 5 / 60) % 24

        # Daily load pattern (higher during day, lower at night)
        daily_factor = 0.3 + 0.7 * np.sin((hour - 6) * np.pi / 12)
        base_load = np.clip(
            self.base_load_mean * daily_factor + self.rng.normal(0, self.base_load_std),
            0, 1
        )

        # Add charging load
        charging_load = self.current_charging / self.max_charge_rate * 0.4
        total_load = np.clip(base_load + charging_load, 0, 1)

        return total_load

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one timestep"""
        # Determine how many EVs to charge
        num_to_charge = min(action, self.evs_waiting, self.max_charge_rate)
        self.current_charging = num_to_charge

        # Calculate grid load
        grid_load = self._get_grid_load()

        # Reward function: maximize charging while avoiding overload
        reward = 0.0

        # Reward for charging EVs
        reward += num_to_charge * 1.0

        # Penalty for grid overload (unsafe operation)
        if grid_load > self.unsafe_threshold:
            reward -= 50.0 * (grid_load - self.unsafe_threshold)
            self.violations += 1

        # Small penalty for waiting EVs
        reward -= 0.1 * self.evs_waiting

        # Update state
        self.evs_waiting = max(0, self.evs_waiting - num_to_charge)

        # Add new EVs arriving (Poisson process)
        new_evs = self.rng.poisson(2)
        self.evs_waiting = min(self.evs_waiting + new_evs, self.num_evs)

        # Update sensors with noise
        self.temp_sensor = np.clip(self.rng.normal(0, 0.1), -1, 1)
        self.voltage_sensor = np.clip(self.rng.normal(0, 0.1), -1, 1)

        # Advance time
        self.timestep += 1
        self.total_reward += reward

        # Check if episode is done
        terminated = self.timestep >= self.episode_length
        truncated = False

        info = {
            'grid_load': grid_load,
            'evs_waiting': self.evs_waiting,
            'charging': num_to_charge,
            'violations': self.violations,
            'timestep': self.timestep
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        """Render environment state"""
        if self.timestep % 12 == 0:  # Print every hour
            obs = self._get_obs()
            print(f"Hour {obs[0]:.1f}: Load={obs[1]:.2f}, "
                  f"Waiting={int(obs[2])}, Charging={int(obs[3])}, "
                  f"Violations={self.violations}")
