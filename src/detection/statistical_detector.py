"""
Statistical anomaly detection methods
Lightweight methods for detecting backdoor behavior
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy import stats
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest


class StatisticalDetector:
    """Base class for statistical anomaly detectors"""

    def __init__(self, name: str = "BaseDetector"):
        self.name = name
        self.is_fitted = False

    def fit(self, features: np.ndarray):
        """Fit detector on normal (clean) behavior"""
        raise NotImplementedError

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict anomaly scores (higher = more anomalous)"""
        raise NotImplementedError

    def detect(self, features: np.ndarray, threshold: float) -> np.ndarray:
        """Binary detection (True = anomaly)"""
        scores = self.predict(features)
        return scores > threshold


class ZScoreDetector(StatisticalDetector):
    """
    Simple Z-score based anomaly detection.
    Flags points where any feature exceeds threshold standard deviations.
    """

    def __init__(self, threshold: float = 3.0):
        super().__init__(name="ZScore")
        self.threshold = threshold
        self.mean = None
        self.std = None

    def fit(self, features: np.ndarray):
        """Compute mean and std from clean data"""
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0) + 1e-8  # Avoid division by zero
        self.is_fitted = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return max absolute z-score across features"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        z_scores = np.abs((features - self.mean) / self.std)
        return np.max(z_scores, axis=1)


class MahalanobisDetector(StatisticalDetector):
    """
    Mahalanobis distance based anomaly detection.
    Accounts for feature correlations.
    """

    def __init__(self):
        super().__init__(name="Mahalanobis")
        self.mean = None
        self.inv_cov = None

    def fit(self, features: np.ndarray):
        """Compute mean and inverse covariance matrix"""
        self.mean = np.mean(features, axis=0)
        cov = np.cov(features, rowvar=False)

        # Add regularization for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-6

        try:
            self.inv_cov = np.linalg.inv(cov)
            self.is_fitted = True
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is singular, cannot invert")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return Mahalanobis distances"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        diff = features - self.mean
        distances = np.sqrt(np.sum(diff @ self.inv_cov * diff, axis=1))
        return distances


class RobustCovarianceDetector(StatisticalDetector):
    """
    Robust covariance estimation (Minimum Covariance Determinant).
    More resistant to outliers in training data.
    """

    def __init__(self, contamination: float = 0.1):
        super().__init__(name="RobustCovariance")
        self.contamination = contamination
        self.model = EllipticEnvelope(contamination=contamination, random_state=42)

    def fit(self, features: np.ndarray):
        """Fit robust covariance model"""
        self.model.fit(features)
        self.is_fitted = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return anomaly scores (negative of decision function)"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        # EllipticEnvelope returns negative scores for outliers
        scores = -self.model.decision_function(features)
        return scores


class IsolationForestDetector(StatisticalDetector):
    """
    Isolation Forest for anomaly detection.
    Efficient for high-dimensional data.
    """

    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        super().__init__(name="IsolationForest")
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )

    def fit(self, features: np.ndarray):
        """Fit isolation forest"""
        self.model.fit(features)
        self.is_fitted = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return anomaly scores"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        # IsolationForest returns negative scores for outliers
        scores = -self.model.decision_function(features)
        return scores


class ThresholdBasedDetector(StatisticalDetector):
    """
    Simple threshold-based detector for specific features.
    Useful for domain-specific rules (e.g., high charging during high load).
    """

    def __init__(
        self,
        unsafe_load_threshold: float = 0.9,
        high_load_charging_threshold: float = 5.0
    ):
        super().__init__(name="ThresholdBased")
        self.unsafe_load_threshold = unsafe_load_threshold
        self.high_load_charging_threshold = high_load_charging_threshold
        self.is_fitted = True  # No training needed

    def fit(self, features: np.ndarray):
        """No training needed for threshold-based detector"""
        pass

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores based on domain-specific rules.

        Assumes features include:
        - unsafe_load_freq (feature index -2)
        - avg_charging_high_load (feature index -1)
        """
        scores = np.zeros(len(features))

        # Check unsafe load frequency
        unsafe_load_freq = features[:, -2]
        scores += unsafe_load_freq * 10  # Weight unsafe loads heavily

        # Check charging during high load
        avg_charging_high_load = features[:, -1]
        high_charging_mask = avg_charging_high_load > self.high_load_charging_threshold
        scores[high_charging_mask] += 5.0

        return scores


class EnsembleDetector(StatisticalDetector):
    """
    Ensemble of multiple detectors.
    Combines predictions using weighted voting.
    """

    def __init__(self, detectors: List[StatisticalDetector], weights: Optional[List[float]] = None):
        super().__init__(name="Ensemble")
        self.detectors = detectors
        self.weights = weights if weights else [1.0] * len(detectors)

    def fit(self, features: np.ndarray):
        """Fit all detectors"""
        for detector in self.detectors:
            detector.fit(features)
        self.is_fitted = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return weighted average of detector scores"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        # Normalize scores from each detector
        all_scores = []
        for detector, weight in zip(self.detectors, self.weights):
            scores = detector.predict(features)
            # Normalize to [0, 1]
            if np.max(scores) > 0:
                scores = scores / np.max(scores)
            all_scores.append(scores * weight)

        # Average weighted scores
        ensemble_scores = np.mean(all_scores, axis=0)
        return ensemble_scores
