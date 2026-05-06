"""
Neural network-based anomaly detector
Small neural network trained on normal behavior for detecting backdoors
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler


class AnomalyDetectorNet(nn.Module):
    """
    Small autoencoder for anomaly detection.
    Trained to reconstruct normal behavior; high reconstruction error indicates anomaly.
    """

    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16]):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        for hidden_dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ClassifierNet(nn.Module):
    """
    Simple classifier for binary anomaly detection.
    Trained on labeled normal/anomalous behavior.
    """

    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NeuralDetector:
    """Neural network-based anomaly detector"""

    def __init__(
        self,
        input_dim: int,
        mode: str = 'autoencoder',  # 'autoencoder' or 'classifier'
        learning_rate: float = 1e-3,
        device: str = 'cpu'
    ):
        self.input_dim = input_dim
        self.mode = mode
        self.device = device
        self.scaler = StandardScaler()

        if mode == 'autoencoder':
            self.model = AnomalyDetectorNet(input_dim).to(device)
            self.criterion = nn.MSELoss()
        elif mode == 'classifier':
            self.model = ClassifierNet(input_dim).to(device)
            self.criterion = nn.BCELoss()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.is_fitted = False

    def fit(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        validation_features: Optional[np.ndarray] = None,
        validation_labels: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Train the neural detector.

        Args:
            features: Training features
            labels: Labels for classifier mode (0=normal, 1=anomaly)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data for validation
            verbose: Print training progress
        """
        # Normalize using training features only.
        self.scaler.fit(features)
        features_normalized = self.scaler.transform(features)

        if validation_features is not None:
            X_train = features_normalized
            X_val = self.scaler.transform(validation_features)

            if self.mode == 'classifier' and labels is not None:
                y_train = labels
                y_val = validation_labels
            else:
                y_train = None
                y_val = None
        else:
            # Backward-compatible path: create an internal validation split when
            # explicit validation data is not provided.
            n_val = int(len(features_normalized) * validation_split)
            rng = np.random.RandomState(random_seed)
            indices = rng.permutation(len(features_normalized))
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

            X_train = features_normalized[train_indices]
            X_val = features_normalized[val_indices]

            if self.mode == 'classifier' and labels is not None:
                y_train = labels[train_indices]
                y_val = labels[val_indices]
            else:
                y_train = None
                y_val = None

        # Training loop
        rng = np.random.RandomState(random_seed)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            # Mini-batch training
            # Adjust batch size if dataset is too small
            effective_batch_size = min(batch_size, len(X_train))
            n_batches = max(1, int(np.ceil(len(X_train) / effective_batch_size)))
            epoch_indices = rng.permutation(len(X_train))

            for i in range(n_batches):
                batch_start = i * effective_batch_size
                batch_end = min(batch_start + effective_batch_size, len(X_train))
                batch_indices = epoch_indices[batch_start:batch_end]

                batch_X = torch.FloatTensor(X_train[batch_indices]).to(self.device)

                self.optimizer.zero_grad()

                if self.mode == 'autoencoder':
                    # Autoencoder: reconstruct input
                    output = self.model(batch_X)
                    loss = self.criterion(output, batch_X)
                else:
                    # Classifier: predict labels
                    batch_y = torch.FloatTensor(y_train[batch_indices]).unsqueeze(1).to(self.device)
                    output = self.model(batch_X)
                    loss = self.criterion(output, batch_y)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= n_batches if n_batches > 0 else 1.0

            # Validation
            val_loss = None
            if len(X_val) > 0:
                self.model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val).to(self.device)

                    if self.mode == 'autoencoder':
                        val_output = self.model(X_val_tensor)
                        val_loss = self.criterion(val_output, X_val_tensor).item()
                    else:
                        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
                        val_output = self.model(X_val_tensor)
                        val_loss = self.criterion(val_output, y_val_tensor).item()

            if verbose and epoch % 10 == 0:
                if val_loss is None:
                    print(f"Epoch {epoch}/{epochs}: Train Loss={train_loss:.4f}")
                else:
                    print(f"Epoch {epoch}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        self.is_fitted = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores.

        For autoencoder: returns reconstruction errors
        For classifier: returns predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        features_normalized = self.scaler.transform(features)
        features_tensor = torch.FloatTensor(features_normalized).to(self.device)

        self.model.eval()
        with torch.no_grad():
            if self.mode == 'autoencoder':
                # Reconstruction error as anomaly score
                reconstructed = self.model(features_tensor)
                errors = torch.mean((features_tensor - reconstructed) ** 2, dim=1)
                scores = errors.cpu().numpy()
            else:
                # Classifier probability as anomaly score
                predictions = self.model(features_tensor)
                scores = predictions.squeeze().cpu().numpy()

        return scores

    def detect(self, features: np.ndarray, threshold: float) -> np.ndarray:
        """Binary detection (True = anomaly)"""
        scores = self.predict(features)
        return scores > threshold

    def save(self, path: str):
        """Save model"""
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler': self.scaler,
            'input_dim': self.input_dim,
            'mode': self.mode
        }, path)

    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.scaler = checkpoint['scaler']
        self.input_dim = checkpoint['input_dim']
        self.mode = checkpoint['mode']
        self.is_fitted = True
