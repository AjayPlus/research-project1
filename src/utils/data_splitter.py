"""
Data splitting utilities for train/validation/test splits with stratification.

This module provides utilities to split trajectory data into train, validation,
and test sets while maintaining class balance (clean vs backdoored samples)
to avoid data leakage and enable proper model evaluation.
"""

import numpy as np
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split


class StratifiedDataSplitter:
    """
    Handles stratified splitting of trajectory data into train/val/test sets.

    This class ensures that:
    1. No data leakage between train/val/test sets
    2. Class balance is maintained across splits
    3. Random seed is fixed for reproducibility
    """

    def __init__(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        random_seed: int = 42
    ):
        """
        Initialize the data splitter with specified split ratios.

        Args:
            train_ratio: Proportion of data for training (default: 0.6)
            val_ratio: Proportion of data for validation (default: 0.2)
            test_ratio: Proportion of data for testing (default: 0.2)
            random_seed: Random seed for reproducibility (default: 42)

        Raises:
            ValueError: If ratios don't sum to 1.0
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
            )

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

    def split_trajectories(
        self,
        clean_trajectories: list,
        backdoor_trajectories: list
    ) -> Dict[str, Dict[str, Any]]:
        """
        Split clean and backdoor trajectories into train/val/test sets.

        This method performs stratified splitting to ensure each set has
        a balanced mix of clean and backdoor samples.

        Args:
            clean_trajectories: List of clean trajectory data
            backdoor_trajectories: List of backdoored trajectory data

        Returns:
            Dictionary with keys 'train', 'val', 'test', each containing:
                - 'trajectories': Combined list of clean + backdoor trajectories
                - 'labels': Binary labels (0=clean, 1=backdoor)
                - 'indices': Original indices for tracking
                - 'stats': Statistics about the split

        Example:
            >>> splitter = StratifiedDataSplitter(0.6, 0.2, 0.2)
            >>> splits = splitter.split_trajectories(clean_data, backdoor_data)
            >>> print(f"Train size: {len(splits['train']['trajectories'])}")
        """
        n_clean = len(clean_trajectories)
        n_backdoor = len(backdoor_trajectories)

        # Split clean trajectories
        clean_train, clean_temp = train_test_split(
            list(range(n_clean)),
            test_size=(self.val_ratio + self.test_ratio),
            random_state=self.random_seed
        )

        # Calculate val ratio relative to temp set
        val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)

        clean_val, clean_test = train_test_split(
            clean_temp,
            test_size=(1 - val_ratio_adjusted),
            random_state=self.random_seed
        )

        # Split backdoor trajectories with same ratios
        backdoor_train, backdoor_temp = train_test_split(
            list(range(n_backdoor)),
            test_size=(self.val_ratio + self.test_ratio),
            random_state=self.random_seed
        )

        backdoor_val, backdoor_test = train_test_split(
            backdoor_temp,
            test_size=(1 - val_ratio_adjusted),
            random_state=self.random_seed
        )

        # Combine splits
        splits = {}

        for split_name, clean_idx, backdoor_idx in [
            ('train', clean_train, backdoor_train),
            ('val', clean_val, backdoor_val),
            ('test', clean_test, backdoor_test)
        ]:
            # Get trajectories
            split_clean_traj = [clean_trajectories[i] for i in clean_idx]
            split_backdoor_traj = [backdoor_trajectories[i] for i in backdoor_idx]

            # Combine and create labels
            trajectories = split_clean_traj + split_backdoor_traj
            labels = np.array([0] * len(split_clean_traj) + [1] * len(split_backdoor_traj))

            # Create indices for tracking
            indices = {
                'clean': clean_idx,
                'backdoor': backdoor_idx
            }

            # Compute statistics
            stats = {
                'total_samples': len(trajectories),
                'clean_samples': len(split_clean_traj),
                'backdoor_samples': len(split_backdoor_traj),
                'backdoor_ratio': len(split_backdoor_traj) / len(trajectories) if len(trajectories) > 0 else 0,
                'clean_ratio': len(split_clean_traj) / len(trajectories) if len(trajectories) > 0 else 0
            }

            splits[split_name] = {
                'trajectories': trajectories,
                'labels': labels,
                'indices': indices,
                'stats': stats
            }

        return splits

    def split_features(
        self,
        clean_features: np.ndarray,
        backdoor_features: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Split feature arrays into train/val/test sets with stratification.

        This is useful when working with pre-extracted features rather than
        raw trajectories.

        Args:
            clean_features: NumPy array of clean features (n_samples × n_features)
            backdoor_features: NumPy array of backdoor features (n_samples × n_features)

        Returns:
            Dictionary with keys 'train', 'val', 'test', each containing:
                - 'features': Combined feature array
                - 'labels': Binary labels (0=clean, 1=backdoor)
                - 'stats': Statistics about the split

        Example:
            >>> splitter = StratifiedDataSplitter(0.6, 0.2, 0.2)
            >>> splits = splitter.split_features(clean_feat, backdoor_feat)
            >>> X_train = splits['train']['features']
            >>> y_train = splits['train']['labels']
        """
        n_clean = len(clean_features)
        n_backdoor = len(backdoor_features)

        # Create labels
        labels = np.array([0] * n_clean + [1] * n_backdoor)

        # Combine features
        all_features = np.vstack([clean_features, backdoor_features])

        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            all_features,
            labels,
            test_size=(self.val_ratio + self.test_ratio),
            stratify=labels,
            random_state=self.random_seed
        )

        # Second split: val vs test
        val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=(1 - val_ratio_adjusted),
            stratify=y_temp,
            random_state=self.random_seed
        )

        # Package results
        splits = {}

        for split_name, features, labels_split in [
            ('train', X_train, y_train),
            ('val', X_val, y_val),
            ('test', X_test, y_test)
        ]:
            stats = {
                'total_samples': len(features),
                'clean_samples': int(np.sum(labels_split == 0)),
                'backdoor_samples': int(np.sum(labels_split == 1)),
                'backdoor_ratio': float(np.mean(labels_split)),
                'clean_ratio': float(1 - np.mean(labels_split))
            }

            splits[split_name] = {
                'features': features,
                'labels': labels_split,
                'stats': stats
            }

        return splits

    def get_split_info(self) -> Dict[str, float]:
        """
        Get information about the configured split ratios.

        Returns:
            Dictionary with train/val/test ratios
        """
        return {
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'random_seed': self.random_seed
        }
