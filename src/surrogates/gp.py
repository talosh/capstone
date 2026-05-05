import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

from .base import BaseSurrogate


class GPSurrogate(BaseSurrogate):
    """
    Gaussian Process surrogate with selectable kernel.

    The natural choice for Bayesian optimisation — provides well-calibrated
    uncertainty estimates out of the box. Output is standardised internally
    before fitting to keep the GP numerically stable across functions with
    wildly different output scales (e.g. function_01 with extreme exponents).

    Parameters
    ----------
    kernel : str
        Kernel shorthand: "RBF" or "Matern". Used by the tuning grid.
        Defaults to "Matern".
    nu : float
        Smoothness parameter for the Matern kernel (ignored for RBF).
        Common values: 0.5 (rough), 1.5 (once differentiable),
        2.5 (twice differentiable, default).
    alpha : float
        Noise level added to the diagonal of the kernel matrix.
        Acts as a regulariser — increase if fitting is numerically unstable.
    length_scale_bounds : tuple[float, float]
        Optimisation bounds for the kernel length scale.
    n_restarts : int
        Number of random restarts for kernel hyperparameter optimisation.
    normalize_y : bool
        Whether to standardise y before fitting. Recommended True for functions
        with extreme output ranges.
    """

    def __init__(
        self,
        kernel: str = "Matern",
        nu: float = 2.5,
        alpha: float = 1e-6,
        length_scale_bounds: tuple[float, float] = (1e-2, 1e2),
        n_restarts: int = 5,
        normalize_y: bool = True,
    ):
        self.kernel = kernel
        self.nu = nu
        self.alpha = alpha
        self.length_scale_bounds = length_scale_bounds
        self.n_restarts = n_restarts
        self.normalize_y = normalize_y

        kernel_obj = self._build_kernel(kernel, nu, length_scale_bounds)

        self.gp = GaussianProcessRegressor(
            kernel=kernel_obj,
            alpha=alpha,
            n_restarts_optimizer=n_restarts,
            normalize_y=normalize_y,
            random_state=42,
        )
        self._scaler = StandardScaler()
        self._fitted = False

    @staticmethod
    def _build_kernel(kernel: str, nu: float, length_scale_bounds: tuple) -> object:
        """Construct an sklearn kernel object from string shorthand."""
        if kernel == "RBF":
            base = RBF(length_scale_bounds=length_scale_bounds)
        elif kernel == "Matern":
            base = Matern(nu=nu, length_scale_bounds=length_scale_bounds)
        else:
            raise ValueError(f"Unknown kernel '{kernel}'. Choose 'RBF' or 'Matern'.")
        return ConstantKernel(1.0) * base

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the GP to observations.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,)
        """
        X_scaled = self._scaler.fit_transform(X)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
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
