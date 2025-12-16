"""
Baseline detectors for comparison and benchmarking.

This module provides simple baseline detection methods that serve as
performance benchmarks for more sophisticated detection approaches.
These baselines help establish lower and upper bounds on detection performance.
"""

import numpy as np
from typing import Optional
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.linalg import svd


class RandomDetector:
    """
    Random baseline detector that makes random predictions.

    This detector randomly classifies samples as clean or backdoored
    according to a specified probability. It represents the expected
    performance of a purely random classifier.

    Attributes:
        backdoor_prob: Probability of predicting backdoor (default: 0.5)
        random_seed: Random seed for reproducibility
    """

    def __init__(self, backdoor_prob: float = 0.5, random_seed: int = 42 ):
        """
        Initialize random detector.

        Args:
            backdoor_prob: Probability of predicting backdoor (0-1)
            random_seed: Random seed for reproducibility
        """
        if not 0 <= backdoor_prob <= 1:
            raise ValueError("backdoor_prob must be between 0 and 1")

        self.backdoor_prob = backdoor_prob
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None):
        """
        Fit method (no-op for random detector).

        Args:
            X_train: Training features (ignored)
            y_train: Training labels (ignored)

        Returns:
            self
        """
        # Random detector doesn't need training
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make random predictions.

        Args:
            X: Feature array of shape (n_samples, n_features)

        Returns:
            Binary predictions (0=clean, 1=backdoor)
        """
        n_samples = len(X)
        return self.rng.binomial(1, self.backdoor_prob, size=n_samples)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute random anomaly scores.

        Args:
            X: Feature array of shape (n_samples, n_features)

        Returns:
            Random scores between 0 and 1
        """
        n_samples = len(X)
        return self.rng.uniform(0, 1, size=n_samples)


class AlwaysDetectDetector:
    """
    Baseline detector that always predicts backdoor.

    This detector always classifies every sample as backdoored.
    It achieves 100% recall but typically has a very high false alarm rate.
    Represents an upper bound on detection rate at the cost of precision.
    """

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None):
        """
        Fit method (no-op for always-detect detector).

        Args:
            X_train: Training features (ignored)
            y_train: Training labels (ignored)

        Returns:
            self
        """
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Always predict backdoor.

        Args:
            X: Feature array of shape (n_samples, n_features)

        Returns:
            Array of ones (all backdoor predictions)
        """
        return np.ones(len(X), dtype=int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Return constant high scores.

        Args:
            X: Feature array of shape (n_samples, n_features)

        Returns:
            Array of ones (all high scores)
        """
        return np.ones(len(X))


class NeverDetectDetector:
    """
    Baseline detector that never predicts backdoor.

    This detector always classifies every sample as clean.
    It achieves 0% false alarm rate but also 0% detection rate.
    Represents a lower bound on false alarms at the cost of detection.
    """

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None):
        """
        Fit method (no-op for never-detect detector).

        Args:
            X_train: Training features (ignored)
            y_train: Training labels (ignored)

        Returns:
            self
        """
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Always predict clean.

        Args:
            X: Feature array of shape (n_samples, n_features)

        Returns:
            Array of zeros (all clean predictions)
        """
        return np.zeros(len(X), dtype=int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Return constant low scores.

        Args:
            X: Feature array of shape (n_samples, n_features)

        Returns:
            Array of zeros (all low scores)
        """
        return np.zeros(len(X))


class ActivationClusteringDetector:
    """
    Activation Clustering detector based on Chen et al. (AAAI 2019).

    Reference:
        Chen et al., "Detecting Backdoor Attacks on Deep Neural Networks
        by Activation Clustering" (AAAI SafeAI Workshop 2019)
        arXiv: https://arxiv.org/abs/1811.03728

    This method:
    1. Applies dimensionality reduction (PCA) to features
    2. Clusters features using k-means (k=2)
    3. Identifies the poisoned cluster using silhouette scores
    4. Classifies samples based on cluster assignment

    Attributes:
        n_components: Number of PCA components (default: 10)
        random_seed: Random seed for reproducibility
    """

    def __init__(self, n_components: int = 10, random_seed: int = 42):
        """
        Initialize activation clustering detector.

        Args:
            n_components: Number of PCA components to reduce to
            random_seed: Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_seed = random_seed
        self.pca = None
        self.kmeans = None
        self.poisoned_cluster_id = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None):
        """
        Fit PCA and k-means clustering on training data.

        Args:
            X_train: Training features of shape (n_samples, n_features)
            y_train: Training labels (ignored, unsupervised method)

        Returns:
            self
        """
        # Apply PCA for dimensionality reduction
        n_components = min(self.n_components, X_train.shape[1], X_train.shape[0])
        self.pca = PCA(n_components=n_components, random_state=self.random_seed)
        X_reduced = self.pca.fit_transform(X_train)

        # Apply k-means clustering with k=2
        self.kmeans = KMeans(n_clusters=2, random_state=self.random_seed, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_reduced)

        # Identify the smaller cluster as the poisoned cluster
        # (backdoor samples are typically the minority in training data)
        cluster_0_count = np.sum(cluster_labels == 0)
        cluster_1_count = np.sum(cluster_labels == 1)

        # Assign the smaller cluster as the poisoned cluster
        self.poisoned_cluster_id = 0 if cluster_0_count < cluster_1_count else 1

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict backdoor samples based on cluster assignment.

        Args:
            X: Feature array of shape (n_samples, n_features)

        Returns:
            Binary predictions (0=clean, 1=backdoor)
        """
        if self.pca is None or self.kmeans is None:
            raise ValueError("Detector must be fitted before prediction")

        # Transform with PCA
        X_reduced = self.pca.transform(X)

        # Get cluster assignments
        cluster_labels = self.kmeans.predict(X_reduced)

        # Classify based on poisoned cluster
        predictions = (cluster_labels == self.poisoned_cluster_id).astype(int)

        return predictions

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores based on distance to cluster centers.

        Args:
            X: Feature array of shape (n_samples, n_features)

        Returns:
            Anomaly scores (higher = more likely backdoor)
        """
        if self.pca is None or self.kmeans is None:
            raise ValueError("Detector must be fitted before prediction")

        # Transform with PCA
        X_reduced = self.pca.transform(X)

        # Compute distances to both cluster centers
        distances = self.kmeans.transform(X_reduced)

        # Score based on relative distance to poisoned vs clean cluster
        clean_cluster_id = 1 - self.poisoned_cluster_id
        scores = distances[:, clean_cluster_id] - distances[:, self.poisoned_cluster_id]

        return scores


class SpectralSignaturesDetector:
    """
    Spectral Signatures detector based on Tran et al. (NeurIPS 2018).

    Reference:
        Tran et al., "Spectral Signatures in Backdoor Attacks"
        (NeurIPS 2018)
        arXiv: https://arxiv.org/abs/1811.00636

    This method:
    1. Computes covariance matrix of feature representations
    2. Applies SVD (Singular Value Decomposition)
    3. Detects spectral anomalies in the top singular vectors
    4. Uses outlier detection on singular vector projections

    Attributes:
        n_components: Number of top singular vectors to use (default: 1)
        outlier_percentile: Percentile threshold for outlier detection (default: 95)
    """

    def __init__(
        self,
        n_components: int = 1,
        outlier_percentile: float = 95.0,
        random_seed: int = 42
    ):
        """
        Initialize spectral signatures detector.

        Args:
            n_components: Number of top singular vectors to analyze
            outlier_percentile: Percentile threshold for outlier detection (0-100)
            random_seed: Random seed for reproducibility
        """
        self.n_components = n_components
        self.outlier_percentile = outlier_percentile
        self.random_seed = random_seed
        self.mean = None
        self.top_singular_vectors = None
        self.threshold = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None):
        """
        Fit SVD on training data covariance.

        Args:
            X_train: Training features of shape (n_samples, n_features)
            y_train: Training labels (ignored, unsupervised method)

        Returns:
            self
        """
        # Center the data
        self.mean = np.mean(X_train, axis=0)
        X_centered = X_train - self.mean

        # Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Apply SVD
        U, S, Vt = svd(cov_matrix)

        # Store top singular vectors
        self.top_singular_vectors = U[:, :self.n_components]

        # Compute projections on training data to set threshold
        projections = np.abs(X_centered @ self.top_singular_vectors)

        # Use maximum projection across all components
        max_projections = np.max(projections, axis=1)

        # Set threshold at specified percentile
        self.threshold = np.percentile(max_projections, self.outlier_percentile)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict backdoor samples based on spectral anomalies.

        Args:
            X: Feature array of shape (n_samples, n_features)

        Returns:
            Binary predictions (0=clean, 1=backdoor)
        """
        scores = self.decision_function(X)
        return (scores > self.threshold).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores based on spectral projections.

        Args:
            X: Feature array of shape (n_samples, n_features)

        Returns:
            Anomaly scores (higher = more likely backdoor)
        """
        if self.mean is None or self.top_singular_vectors is None:
            raise ValueError("Detector must be fitted before prediction")

        # Center the data
        X_centered = X - self.mean

        # Project onto top singular vectors
        projections = np.abs(X_centered @ self.top_singular_vectors)

        # Return maximum projection across components as anomaly score
        scores = np.max(projections, axis=1)

        return scores
