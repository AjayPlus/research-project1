"""
Seed management utilities for reproducible experiments.

This module provides functions to set random seeds across all libraries
used in the project (NumPy, PyTorch, Python random, Gymnasium) to ensure
reproducibility across multiple experimental runs.
"""

import random
import numpy as np
import torch
import gymnasium as gym


def set_seed(seed: int) -> None:
    """
    Set random seed for all libraries to ensure reproducibility.

    This function sets seeds for:
    - Python's built-in random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - Gymnasium environments

    Args:
        seed: Integer seed value to use across all libraries

    Example:
        >>> set_seed(42)
        >>> # All subsequent random operations will be deterministic
    """
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Gymnasium (uses NumPy's random, but set explicitly for clarity)
    # Note: Individual environment instances will need seed passed to reset()


def get_seed_range(start_seed: int = 42, num_seeds: int = 10) -> list[int]:
    """
    Generate a range of seeds for multi-seed experiments.

    Args:
        start_seed: Starting seed value (default: 42)
        num_seeds: Number of seeds to generate (default: 10)

    Returns:
        List of seed values

    Example:
        >>> seeds = get_seed_range(42, 5)
        >>> print(seeds)
        [42, 43, 44, 45, 46]
    """
    return list(range(start_seed, start_seed + num_seeds))
