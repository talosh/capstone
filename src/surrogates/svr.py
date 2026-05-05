import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from .base import BaseSurrogate


class SVRSurrogate(BaseSurrogate):
    """
    Support Vector Regression surrogate with bootstrap uncertainty estimates.

    SVR has no native uncertainty quantification, so uncertainty is estimated
    by fitting an ensemble of SVRs on bootstrap samples of the training data
    and taking the std of their predictions. This is simple, honest, and
    sufficient for acquisition functions at small data scales.

    Both X and y are standardised internally — SVR is sensitive to feature
    scale and output range, and our functions vary wildly on both.

    Parameters
    ----------
    kernel : str
        SVR kernel: 'rbf' (default), 'poly', or 'linear'.
    C : float
        Regularisation parameter. Larger values = less regularisation.
    epsilon : float
        Width of the epsilon-insensitive tube around predictions.
    gamma : str | float
        Kernel coefficient for 'rbf' and 'poly'. 'scale' uses 1/(n_features * X.var()).
    n_bootstrap : int
        Number of bootstrap SVRs to fit for uncertainty estimation.
        10–20 is usually sufficient at this data scale.
    random_state : int
        Seed for bootstrap sampling reproducibility.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1,
        gamma: str | float = "scale",
        n_bootstrap: int = 10,
        random_state: int = 42,
    ):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        self._models: list[SVR] = []
        self._fitted = False

    def _make_svr(self) -> SVR:
        return SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit bootstrap ensemble of SVRs to observations.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,)
        """
        n = len(X)
        X_scaled = self._scaler_X.fit_transform(X)
        y_scaled = self._scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        rng = np.random.default_rng(self.random_state)
        self._models = []

        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)          # sample with replacement
            svr = self._make_svr()
            svr.fit(X_scaled[idx], y_scaled[idx])
            self._models.append(svr)

        self._fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and std at candidate points.

        Mean is the ensemble average; std is the ensemble spread,
        both back-transformed to the original output scale.

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

        X_scaled = self._scaler_X.transform(X)

        # collect predictions from each bootstrap model: shape (n_bootstrap, m)
        preds = np.stack([m.predict(X_scaled) for m in self._models], axis=0)

        mean_scaled = preds.mean(axis=0)
        std_scaled  = preds.std(axis=0)

        # inverse transform mean through y scaler
        mean = self._scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()

        # std only needs scaling by the y scale factor (no shift)
        std = std_scaled * self._scaler_y.scale_

        return mean, std
