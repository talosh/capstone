import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import BaseSurrogate

try:
    import xgboost as xgb
except ImportError:
    raise ImportError(
        "XGBoost is not installed. Run: pip install xgboost"
    )


class XGBSurrogate(BaseSurrogate):
    """
    Gradient Boosted Trees surrogate using XGBoost with quantile regression.

    Uncertainty is estimated by training two additional models at a lower and
    upper quantile (e.g. 0.1 and 0.9). The spread between quantile predictions
    forms a calibrated uncertainty estimate — wider where the function is poorly
    covered, narrower near dense observations.

    This is more principled than bootstrap uncertainty (SVR) for asymmetric
    landscapes and handles discontinuous functions (e.g. fn01's two-regime
    output) better than GP kernel smoothing.

    Both X and y are standardised internally before fitting.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth. Keep shallow at small n to avoid overfitting.
    learning_rate : float
        Shrinkage applied to each tree's contribution.
    subsample : float
        Fraction of training samples used per tree. Adds stochasticity.
    colsample_bytree : float
        Fraction of features used per tree. Useful in higher dimensions.
    reg_lambda : float
        L2 regularisation on leaf weights.
    quantile_alpha : float
        Defines the quantile interval for uncertainty: predictions span
        [alpha, 1-alpha]. Default 0.1 gives a 80% interval.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        quantile_alpha: float = 0.1,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.quantile_alpha = quantile_alpha
        self.random_state = random_state

        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        self._model_mean  = None
        self._model_lower = None
        self._model_upper = None
        self._fitted = False

    def _make_model(self, objective: str) -> "xgb.XGBRegressor":
        kwargs = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            verbosity=0,
        )
        if objective == "mean":
            return xgb.XGBRegressor(objective="reg:squarederror", **kwargs)
        else:
            alpha = self.quantile_alpha if objective == "lower" else 1 - self.quantile_alpha
            return xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=alpha,
                **kwargs,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit mean and quantile XGBoost models to observations.

        Three models are trained: one for the mean (squared error), one for
        the lower quantile, and one for the upper quantile. Together they
        provide predictions and a calibrated uncertainty interval.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,)
        """
        X_scaled = self._scaler_X.fit_transform(X)
        y_scaled = self._scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        self._model_mean  = self._make_model("mean")
        self._model_lower = self._make_model("lower")
        self._model_upper = self._make_model("upper")

        self._model_mean.fit(X_scaled,  y_scaled)
        self._model_lower.fit(X_scaled, y_scaled)
        self._model_upper.fit(X_scaled, y_scaled)

        self._fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and uncertainty at candidate points.

        Mean is the squared-error model prediction. Std is derived from the
        quantile interval: std ≈ (upper - lower) / 2, scaled back to the
        original output units.

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

        mean_scaled  = self._model_mean.predict(X_scaled)
        lower_scaled = self._model_lower.predict(X_scaled)
        upper_scaled = self._model_upper.predict(X_scaled)

        # ensure lower <= upper (can occasionally flip at small n)
        lower_scaled = np.minimum(lower_scaled, upper_scaled)
        upper_scaled = np.maximum(lower_scaled, upper_scaled)

        mean = self._scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
        std  = (upper_scaled - lower_scaled) / 2 * self._scaler_y.scale_
        std  = np.clip(std, 1e-9, None)

        return mean, std
