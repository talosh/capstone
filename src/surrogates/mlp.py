import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from .base import BaseSurrogate


class _MLPNet(nn.Module):
    """Simple MLP — no dropout needed, uncertainty comes from ensemble disagreement."""

    def __init__(self, input_dim: int, hidden: int, n_layers: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPSurrogate(BaseSurrogate):
    """
    MLP surrogate with deep ensemble uncertainty estimation.

    Trains n_ensemble independent MLPs from different random initialisations.
    At inference time, uncertainty is the std of their predictions — regions
    with sparse training data naturally produce high disagreement.

    Both X and y are standardised internally before training.

    Parameters
    ----------
    hidden : int
        Number of units per hidden layer.
    n_layers : int
        Number of hidden layers.
    n_ensemble : int
        Number of independently trained MLPs. 5 is a good default —
        beyond 10 the returns diminish quickly.
    lr : float
        Adam learning rate.
    epochs : int
        Training epochs.
    weight_decay : float
        L2 regularisation. Important at small data scales.
    random_state : int
        Base seed — each ensemble member gets seed (random_state + i)
        to ensure independent initialisations.
    """

    def __init__(
        self,
        hidden: int = 64,
        n_layers: int = 2,
        n_ensemble: int = 5,
        lr: float = 1e-3,
        epochs: int = 500,
        weight_decay: float = 1e-4,
        random_state: int = 42,
    ):
        self.hidden = hidden
        self.n_layers = n_layers
        self.n_ensemble = n_ensemble
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.random_state = random_state

        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        self._ensemble: list[_MLPNet] = []
        self._fitted = False

    def _train_one(self, seed: int, X_t: torch.Tensor, y_t: torch.Tensor, input_dim: int) -> _MLPNet:
        """Train a single MLP member from a given random seed."""
        torch.manual_seed(seed)
        net = _MLPNet(input_dim, self.hidden, self.n_layers)
        optimiser = torch.optim.Adam(
            net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        net.train()
        for _ in range(self.epochs):
            optimiser.zero_grad()
            loss = loss_fn(net(X_t), y_t)
            loss.backward()
            optimiser.step()

        net.eval()
        return net

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the full ensemble on observations.

        Each member is initialised from a different seed, giving genuinely
        independent weight configurations and thus meaningful disagreement.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,)
        """
        X_scaled = self._scaler_X.fit_transform(X).astype(np.float32)
        y_scaled = self._scaler_y.fit_transform(y.reshape(-1, 1)).ravel().astype(np.float32)

        X_t = torch.from_numpy(X_scaled)
        y_t = torch.from_numpy(y_scaled)

        self._ensemble = [
            self._train_one(self.random_state + i, X_t, y_t, X.shape[1])
            for i in range(self.n_ensemble)
        ]
        self._fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and std via ensemble disagreement.

        Each member produces a prediction; mean and std are computed across
        the ensemble and inverse-transformed to the original output scale.

        Parameters
        ----------
        X : np.ndarray, shape (m, d)

        Returns
        -------
        mean : np.ndarray, shape (m,)
        std  : np.ndarray, shape (m,)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        X_scaled = self._scaler_X.transform(X).astype(np.float32)
        X_t = torch.from_numpy(X_scaled)

        with torch.no_grad():
            preds = torch.stack(
                [net(X_t) for net in self._ensemble], dim=0
            ).numpy()  # shape: (n_ensemble, m)

        mean_scaled = preds.mean(axis=0)
        std_scaled  = preds.std(axis=0)

        mean = self._scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
        std  = std_scaled * self._scaler_y.scale_

        return mean, std
