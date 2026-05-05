import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

from .base import BaseSurrogate


class GPSurrogate(BaseSurrogate):
    """
    Gaussian Process surrogate using a Matern kernel.

    The natural choice for Bayesian optimisation — provides well-calibrated
    uncertainty estimates out of the box. Output is standardised internally
    before fitting to keep the GP numerically stable across functions with
    wildly different output scales (e.g. function_01 with extreme exponents).

    Parameters
    ----------
    kernel : sklearn kernel | None
        Custom kernel. Defaults to ConstantKernel * Matern(nu=2.5) + WhiteKernel,
        which handles smooth functions with automatic noise estimation.
    n_restarts : int
        Number of random restarts for kernel hyperparameter optimisation.
        Higher values reduce the chance of landing in a bad local optimum.
    normalize_y : bool
        Whether to standardise y before fitting. Recommended True for functions
        with extreme output ranges.
    """

    def __init__(
        self,
        kernel=None,
        n_restarts: int = 5,
        normalize_y: bool = True,
    ):
        if kernel is None:
            kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-5)

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            normalize_y=normalize_y,
            random_state=42,
        )
        self.normalize_y = normalize_y
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the GP to observations.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,)
        """
        X_scaled = self._scaler.fit_transform(X)
        self.gp.fit(X_scaled, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and std at candidate points.

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
        X_scaled = self._scaler.transform(X)
        mean, std = self.gp.predict(X_scaled, return_std=True)
        return mean, std

    @property
    def kernel_params(self) -> dict:
        """Return the optimised kernel hyperparameters after fitting."""
        if not self._fitted:
            raise RuntimeError("Call fit() before inspecting kernel params.")
        return self.gp.kernel_.get_params()
