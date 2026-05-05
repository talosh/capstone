from abc import ABC, abstractmethod
import numpy as np


class BaseSurrogate(ABC):
    """
    Common interface for all surrogate models.

    Every surrogate must implement fit() and predict(), where predict()
    returns both a mean and a standard deviation. The std is required
    by all acquisition functions — surrogates that don't model uncertainty
    natively (e.g. SVR) must provide a calibrated approximation.

    Usage
    -----
    model.fit(X, y)
    mean, std = model.predict(X_candidate)
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the surrogate to observed data.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,)
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation at candidate points.

        Parameters
        ----------
        X : np.ndarray, shape (m, d)

        Returns
        -------
        mean : np.ndarray, shape (m,)
        std  : np.ndarray, shape (m,)
        """

    def fit_predict(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience method: fit then predict in one call."""
        self.fit(X_train, y_train)
        return self.predict(X_test)
