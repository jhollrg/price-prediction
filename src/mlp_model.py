"""PyTorch MLP wrapped in a scikit-learn-compatible interface."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from config import cfg


class _MLP(nn.Module):
    """Fully-connected network: input → [Linear → ReLU → Dropout?]* → Linear(1).

    Dropout is inserted after every hidden layer except the last, which matches
    the architecture spec: 256 → ReLU → Dropout(p) → 128 → ReLU → 1.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_sizes : list[int]
        Width of each hidden layer, e.g. [256, 128].
    dropout : float
        Dropout probability applied between hidden layers (not after the last).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for i, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning shape ``(batch, 1)``."""
        return self.net(x)


class MLPRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible wrapper around the PyTorch MLP.

    ``StandardScaler`` is applied internally in ``fit`` and reused in
    ``predict``, so callers never need a separate scaling step.  The inherited
    ``score(X, y)`` method from ``RegressorMixin`` returns R² and works without
    any additional code.

    Parameters
    ----------
    hidden_sizes : list[int]
        Width of each hidden layer.
    dropout : float
        Dropout probability between hidden layers.
    lr : float
        Adam learning rate.
    batch_size : int
        Mini-batch size for the training DataLoader.
    epochs : int
        Number of full passes over the training data.
    random_state : int
        Seed for PyTorch and NumPy RNGs (ensures reproducible weight init
        and shuffle order).

    Attributes
    ----------
    model_ : _MLP
        Trained PyTorch module (available after ``fit``).
    scaler_ : StandardScaler
        Feature scaler fitted on training data.
    train_losses_ : list[float]
        Per-epoch mean MSE loss recorded during training.
    n_features_in_ : int
        Number of features seen during ``fit``.
    """

    def __init__(
        self,
        hidden_sizes: list[int] = cfg.mlp.hidden_sizes,
        dropout: float = cfg.mlp.dropout,
        lr: float = cfg.mlp.lr,
        batch_size: int = cfg.mlp.batch_size,
        epochs: int = cfg.mlp.epochs,
        random_state: int = cfg.mlp.random_state,
    ) -> None:
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "MLPRegressor":
        """Fit the MLP on training data.

        Scales features with ``StandardScaler``, builds the network, then runs
        mini-batch SGD with Adam for ``self.epochs`` epochs.  Per-epoch mean MSE
        is recorded in ``train_losses_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.  Accepts NumPy arrays and pandas DataFrames.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).ravel()
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_arr).astype(np.float32)

        X_t = torch.from_numpy(X_scaled)
        y_t = torch.from_numpy(y_arr).reshape(-1, 1)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.model_ = _MLP(n_features, self.hidden_sizes, self.dropout)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.train_losses_: list[float] = []
        for _ in range(self.epochs):
            self.model_.train()
            running = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                running += loss.item() * len(X_batch)
            self.train_losses_.append(running / n_samples)

        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Return predictions for *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.  Must have the same number of columns as the
            training data.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted fare amounts.
        """
        X_arr = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler_.transform(X_arr).astype(np.float32)
        X_t = torch.from_numpy(X_scaled)

        self.model_.eval()
        with torch.no_grad():
            return self.model_(X_t).squeeze(1).numpy()
