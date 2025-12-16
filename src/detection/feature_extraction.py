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


class EpisodeFeatureExtractor:
    """
    Extract features at episode level (one feature vector per episode).

    This is more appropriate for agent-level detection where we want to identify
    if an agent is backdoored, rather than detecting individual malicious timesteps.
    """

    def __init__(self):
        pass

    def extract_from_episode(
        self,
        states: np.ndarray,
        actions: np.ndarray
    ) -> np.ndarray:
        """
        Extract a single feature vector from an entire episode.

        Args:
            states: Array of shape (T, state_dim) where T is episode length
            actions: Array of shape (T,)

        Returns:
            Feature vector of shape (feature_dim,)
        """
        if len(states) == 0:
            return np.zeros(self._get_feature_dim())

        features = []

        # Overall state statistics (mean, std, min, max for each state dimension)
        for i in range(states.shape[1]):
            state_col = states[:, i]
            features.extend([
                np.mean(state_col),
                np.std(state_col),
                np.min(state_col),
                np.max(state_col),
            ])

        # Overall action statistics
        features.extend([
            np.mean(actions),
            np.std(actions),
            np.min(actions),
            np.max(actions),
        ])

        # Action distribution (histogram of action frequencies)
        max_action = 10  # Assuming action space is 0-9
        action_hist = np.histogram(actions, bins=max_action, range=(0, max_action), density=True)[0]
        features.extend(action_hist.tolist())

        # Temporal patterns - statistics over time windows
        # Divide episode into quarters and look for shifts in behavior
        quarter_len = len(states) // 4
        if quarter_len > 0:
            for quarter in range(4):
                start_idx = quarter * quarter_len
                end_idx = start_idx + quarter_len if quarter < 3 else len(states)
                quarter_actions = actions[start_idx:end_idx]
                features.append(np.mean(quarter_actions))

        # Grid load vs charging behavior (critical for backdoor detection)
        grid_load = states[:, 1]
        charging = states[:, 3]

        # Overall correlation
        if np.std(grid_load) > 0 and np.std(charging) > 0:
            correlation = np.corrcoef(grid_load, charging)[0, 1]
        else:
            correlation = 0.0
        features.append(correlation)

        # Unsafe behavior metrics
        high_load_mask = grid_load > 0.7
        unsafe_load_mask = grid_load > 0.9

        features.append(np.mean(high_load_mask))  # Frequency of high load
        features.append(np.mean(unsafe_load_mask))  # Frequency of unsafe load

        # Average charging level during high load (key backdoor indicator)
        if np.any(high_load_mask):
            avg_charging_high_load = np.mean(charging[high_load_mask])
            max_charging_high_load = np.max(charging[high_load_mask])
        else:
            avg_charging_high_load = 0.0
            max_charging_high_load = 0.0
        features.append(avg_charging_high_load)
        features.append(max_charging_high_load)

        # Peak hour behavior (backdoor triggers during specific hours)
        hour = states[:, 0]
        evening_mask = (hour >= 18) & (hour < 21)
        if np.any(evening_mask):
            evening_charging_mean = np.mean(charging[evening_mask])
            evening_charging_std = np.std(charging[evening_mask])
        else:
            evening_charging_mean = 0.0
            evening_charging_std = 0.0
        features.append(evening_charging_mean)
        features.append(evening_charging_std)

        # Sensor anomaly patterns (backdoor uses sensor values as trigger)
        temp_sensor = states[:, 4]
        voltage_sensor = states[:, 5]

        # Check for specific sensor patterns
        temp_in_trigger_range = np.mean((temp_sensor >= 0.05) & (temp_sensor <= 0.35))
        voltage_in_trigger_range = np.mean((voltage_sensor >= -0.30) & (voltage_sensor <= -0.05))
        features.append(temp_in_trigger_range)
        features.append(voltage_in_trigger_range)

        # Combined trigger pattern frequency
        combined_trigger = evening_mask & (grid_load >= 0.3) & \
                          (temp_sensor >= 0.05) & (temp_sensor <= 0.35) & \
                          (voltage_sensor >= -0.30) & (voltage_sensor <= -0.05)
        features.append(np.mean(combined_trigger))

        # Variability metrics
        action_changes = np.sum(np.diff(actions) != 0)
        features.append(action_changes / len(actions))  # Action change rate

        # State change rates
        for i in range(states.shape[1]):
            state_diff = np.diff(states[:, i])
            features.append(np.mean(np.abs(state_diff)))

        return np.array(features, dtype=np.float32)

    def extract_from_episodes(
        self,
        states_list: List[np.ndarray],
        actions_list: List[np.ndarray]
    ) -> np.ndarray:
        """
        Extract features from multiple episodes.

        Args:
            states_list: List of state arrays, each of shape (T_i, state_dim)
            actions_list: List of action arrays, each of shape (T_i,)

        Returns:
            Array of shape (num_episodes, feature_dim)
        """
        features_list = []
        for states, actions in zip(states_list, actions_list):
            features = self.extract_from_episode(states, actions)
            features_list.append(features)

        return np.array(features_list)

    def _get_feature_dim(self) -> int:
        """Calculate total feature dimension"""
        state_dim = 6
        state_stats = state_dim * 4  # mean, std, min, max per dimension
        action_stats = 4  # mean, std, min, max
        action_hist = 10  # histogram bins
        temporal_patterns = 4  # quarterly means
        correlation_features = 1  # load-charging correlation
        unsafe_behavior = 4  # high_load_freq, unsafe_load_freq, avg_charging_high_load, max_charging_high_load
        peak_hour_features = 2  # evening charging mean and std
        sensor_features = 3  # temp trigger freq, voltage trigger freq, combined trigger freq
        variability = 1  # action change rate
        state_changes = state_dim  # change rate per state dimension

        return (state_stats + action_stats + action_hist + temporal_patterns +
                correlation_features + unsafe_behavior + peak_hour_features +
                sensor_features + variability + state_changes)

    def get_feature_names(self) -> List[str]:
        """Get human-readable feature names"""
        state_vars = ['hour', 'grid_load', 'evs_waiting', 'current_charging', 'temp_sensor', 'voltage_sensor']
        names = []

        # State statistics
        for var in state_vars:
            names.extend([f'{var}_mean', f'{var}_std', f'{var}_min', f'{var}_max'])

        # Action statistics
        names.extend(['action_mean', 'action_std', 'action_min', 'action_max'])

        # Action histogram
        for i in range(10):
            names.append(f'action_{i}_freq')

        # Temporal patterns
        for i in range(4):
            names.append(f'quarter_{i+1}_action_mean')

        # Correlation and unsafe behavior
        names.extend([
            'load_charging_corr',
            'high_load_freq',
            'unsafe_load_freq',
            'avg_charging_high_load',
            'max_charging_high_load'
        ])

        # Peak hour behavior
        names.extend(['evening_charging_mean', 'evening_charging_std'])

        # Sensor patterns
        names.extend(['temp_trigger_freq', 'voltage_trigger_freq', 'combined_trigger_freq'])

        # Variability
        names.append('action_change_rate')

        # State changes
        for var in state_vars:
            names.append(f'{var}_change_rate')

        return names
