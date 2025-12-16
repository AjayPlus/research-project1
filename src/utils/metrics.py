"""
Evaluation metrics for backdoor detection
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


class DetectionMetrics:
    """Calculate and store detection performance metrics"""

    def __init__(self):
        self.metrics = {}

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        scores: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive detection metrics.

        Args:
            y_true: True labels (0=normal, 1=anomaly)
            y_pred: Predicted labels (0=normal, 1=anomaly)
            scores: Anomaly scores (optional, for AUC)

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = int(tp)
        metrics['false_positives'] = int(fp)
        metrics['true_negatives'] = int(tn)
        metrics['false_negatives'] = int(fn)

        # False alarm rate (FPR)
        metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # True positive rate (TPR) / detection rate
        metrics['detection_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # AUC if scores provided
        if scores is not None and len(np.unique(y_true)) > 1:
            metrics['auc'] = roc_auc_score(y_true, scores)

        self.metrics = metrics
        return metrics

    def compute_detection_speed(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timesteps: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute detection speed metrics.

        Args:
            y_true: True labels over time
            y_pred: Predicted labels over time
            timesteps: Timestep indices (optional)

        Returns:
            Detection speed metrics
        """
        speed_metrics = {}

        # Find first anomaly occurrence
        anomaly_indices = np.where(y_true == 1)[0]

        if len(anomaly_indices) == 0:
            speed_metrics['time_to_first_detection'] = None
            speed_metrics['detection_delay'] = None
            return speed_metrics

        first_true_anomaly = anomaly_indices[0]

        # Find first detection
        detected_anomaly_indices = np.where((y_pred == 1) & (y_true == 1))[0]

        if len(detected_anomaly_indices) > 0:
            first_detection = detected_anomaly_indices[0]

            if timesteps is not None:
                speed_metrics['time_to_first_detection'] = float(timesteps[first_detection])
                speed_metrics['detection_delay'] = float(
                    timesteps[first_detection] - timesteps[first_true_anomaly]
                )
            else:
                speed_metrics['time_to_first_detection'] = int(first_detection)
                speed_metrics['detection_delay'] = int(first_detection - first_true_anomaly)
        else:
            speed_metrics['time_to_first_detection'] = None
            speed_metrics['detection_delay'] = None

        return speed_metrics

    def print_metrics(self):
        """Print metrics in a formatted way"""
        if not self.metrics:
            print("No metrics computed yet")
            return

        print("\n" + "="*50)
        print("Detection Performance Metrics")
        print("="*50)

        # Classification metrics
        print("\nClassification Metrics:")
        print(f"  Accuracy:          {self.metrics['accuracy']:.4f}")
        print(f"  Precision:         {self.metrics['precision']:.4f}")
        print(f"  Recall:            {self.metrics['recall']:.4f}")
        print(f"  F1 Score:          {self.metrics['f1']:.4f}")

        # Detection rates
        print("\nDetection Rates:")
        print(f"  True Positive Rate: {self.metrics['detection_rate']:.4f}")
        print(f"  False Alarm Rate:   {self.metrics['false_alarm_rate']:.4f}")

        # Confusion matrix
        print("\nConfusion Matrix:")
        print(f"  True Positives:    {self.metrics['true_positives']}")
        print(f"  False Positives:   {self.metrics['false_positives']}")
        print(f"  True Negatives:    {self.metrics['true_negatives']}")
        print(f"  False Negatives:   {self.metrics['false_negatives']}")

        # AUC if available
        if 'auc' in self.metrics:
            print(f"\n  AUC-ROC:           {self.metrics['auc']:.4f}")

        print("="*50 + "\n")


def find_optimal_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    metric: str = 'f1',
    n_thresholds: int = 100,
    verbose: bool = False
) -> Tuple[float, float]:
    """
    Find optimal detection threshold by grid search.

    Args:
        scores: Anomaly scores
        y_true: True labels
        metric: Metric to optimize ('f1', 'accuracy', 'balanced')
        n_thresholds: Number of thresholds to try
        verbose: Print debugging information

    Returns:
        (optimal_threshold, best_metric_value)
    """
    thresholds = np.linspace(np.min(scores), np.max(scores), n_thresholds)
    best_threshold = thresholds[0]
    best_metric_value = 0.0

    if verbose:
        print(f"        Threshold search: {n_thresholds} thresholds from {np.min(scores):.4f} to {np.max(scores):.4f}")
        print(f"        Label distribution: {np.sum(y_true == 0)} clean, {np.sum(y_true == 1)} backdoor")

    for threshold in thresholds:
        y_pred = (scores > threshold).astype(int)

        if metric == 'f1':
            value = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'accuracy':
            value = accuracy_score(y_true, y_pred)
        elif metric == 'balanced':
            # Balance precision and recall
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            value = 2 * prec * rec / (prec + rec + 1e-8)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if value > best_metric_value:
            best_metric_value = value
            best_threshold = threshold
            if verbose:
                n_pred_backdoor = np.sum(y_pred == 1)
                print(f"        New best at threshold={threshold:.4f}: {metric}={value:.4f}, predicts {n_pred_backdoor}/{len(y_pred)} as backdoor")

    if verbose:
        print(f"        Final best: threshold={best_threshold:.4f}, {metric}={best_metric_value:.4f}")

    return best_threshold, best_metric_value


def compute_detection_lag(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window_size: int = 10
) -> float:
    """
    Compute average detection lag (time between anomaly start and detection).

    Args:
        y_true: True labels over time
        y_pred: Predicted labels over time
        window_size: Window to search for detection after anomaly

    Returns:
        Average detection lag in timesteps
    """
    lags = []

    # Find anomaly periods
    anomaly_starts = np.where(np.diff(y_true.astype(int)) == 1)[0] + 1

    for start in anomaly_starts:
        # Look for detection within window
        window_end = min(start + window_size, len(y_pred))
        detection_in_window = np.where(y_pred[start:window_end] == 1)[0]

        if len(detection_in_window) > 0:
            lag = detection_in_window[0]
            lags.append(lag)

    return np.mean(lags) if lags else float('inf')
