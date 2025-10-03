"""
Feature extraction for anomaly detection
Converts state-action trajectories into time-window features
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import deque


class FeatureExtractor:
    """
    Extracts statistical features from time-window of states and actions.

    Features include:
    - Mean, std, min, max of state variables
    - Action statistics
    - Change rates
    - Correlation patterns
    """

    def __init__(self, window_size: int = 12):
        """
        Args:
            window_size: Number of timesteps in rolling window (default 12 = 1 hour)
        """
        self.window_size = window_size
        self.state_buffer = deque(maxlen=window_size)
        self.action_buffer = deque(maxlen=window_size)

    def add_transition(self, state: np.ndarray, action: int):
        """Add new state-action pair to buffer"""
        self.state_buffer.append(state)
        self.action_buffer.append(action)

    def extract_features(self) -> np.ndarray:
        """
        Extract feature vector from current window.

        Returns:
            Feature vector combining state and action statistics
        """
        if len(self.state_buffer) < self.window_size:
            # Not enough data yet, return zeros
            return np.zeros(self._get_feature_dim())

        states = np.array(self.state_buffer)  # Shape: (window_size, state_dim)
        actions = np.array(self.action_buffer)

        features = []

        # State statistics (for each state dimension)
        for i in range(states.shape[1]):
            state_col = states[:, i]
            features.extend([
                np.mean(state_col),
                np.std(state_col),
                np.min(state_col),
                np.max(state_col),
            ])

        # Action statistics
        features.extend([
            np.mean(actions),
            np.std(actions),
            np.min(actions),
            np.max(actions),
        ])

        # Change rates (first-order differences)
        for i in range(states.shape[1]):
            state_diff = np.diff(states[:, i])
            features.extend([
                np.mean(np.abs(state_diff)),
                np.std(state_diff),
            ])

        # Action change rate
        action_diff = np.diff(actions)
        features.extend([
            np.mean(np.abs(action_diff)),
            np.std(action_diff),
        ])

        # Grid load vs charging correlation (key for detecting unsafe behavior)
        grid_load = states[:, 1]  # Column 1 is grid_load
        charging = states[:, 3]  # Column 3 is current_charging
        if np.std(grid_load) > 0 and np.std(charging) > 0:
            correlation = np.corrcoef(grid_load, charging)[0, 1]
        else:
            correlation = 0.0
        features.append(correlation)

        # Unsafe load frequency (grid_load > 0.9)
        unsafe_freq = np.mean(grid_load > 0.9)
        features.append(unsafe_freq)

        # High charging during high load (key anomaly indicator)
        high_load_mask = grid_load > 0.7
        if np.any(high_load_mask):
            avg_charging_high_load = np.mean(charging[high_load_mask])
        else:
            avg_charging_high_load = 0.0
        features.append(avg_charging_high_load)

        return np.array(features, dtype=np.float32)

    def _get_feature_dim(self) -> int:
        """Calculate total feature dimension"""
        state_dim = 6  # Known from environment
        state_stats = state_dim * 4  # mean, std, min, max per dimension
        action_stats = 4  # mean, std, min, max
        state_diff_stats = state_dim * 2  # mean abs diff, std diff
        action_diff_stats = 2  # mean abs diff, std diff
        correlation_features = 3  # load-charging corr, unsafe freq, high_load_charging

        return state_stats + action_stats + state_diff_stats + action_diff_stats + correlation_features

    def reset(self):
        """Clear buffer"""
        self.state_buffer.clear()
        self.action_buffer.clear()

    def get_feature_names(self) -> List[str]:
        """Get human-readable feature names"""
        state_vars = ['hour', 'grid_load', 'evs_waiting', 'current_charging', 'temp_sensor', 'voltage_sensor']
        names = []

        # State statistics
        for var in state_vars:
            names.extend([f'{var}_mean', f'{var}_std', f'{var}_min', f'{var}_max'])

        # Action statistics
        names.extend(['action_mean', 'action_std', 'action_min', 'action_max'])

        # Change rates
        for var in state_vars:
            names.extend([f'{var}_change_mean', f'{var}_change_std'])

        # Action change
        names.extend(['action_change_mean', 'action_change_std'])

        # Correlation features
        names.extend(['load_charging_corr', 'unsafe_load_freq', 'avg_charging_high_load'])

        return names


class TrajectoryFeatureExtractor:
    """Extract features from complete trajectories (for batch processing)"""

    def __init__(self, window_size: int = 12):
        self.window_size = window_size
        self.extractor = FeatureExtractor(window_size)

    def extract_from_trajectory(
        self,
        states: np.ndarray,
        actions: np.ndarray
    ) -> np.ndarray:
        """
        Extract features from full trajectory.

        Args:
            states: Array of shape (T, state_dim)
            actions: Array of shape (T,)

        Returns:
            Array of shape (T - window_size + 1, feature_dim)
        """
        self.extractor.reset()
        features_list = []

        for t in range(len(states)):
            self.extractor.add_transition(states[t], actions[t])

            if t >= self.window_size - 1:
                features = self.extractor.extract_features()
                features_list.append(features)

        return np.array(features_list)
