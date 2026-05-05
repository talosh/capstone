import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod


class BaseAcquisition(ABC):
    """
    Common interface for all acquisition functions.

    Each acquisition function takes surrogate predictions (mean, std)
    and the current best observed value, and returns a score per
    candidate point. Higher score = more worth querying.

    Usage
    -----
    acq = ExpectedImprovement(xi=0.01)
    scores = acq(mean, std, y_best)
    next_x = X_candidates[np.argmax(scores)]
    """

    @abstractmethod
    def __call__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        y_best: float,
    ) -> np.ndarray:
        """
        Compute acquisition scores for a set of candidate points.

        Parameters
        ----------
        mean : np.ndarray, shape (m,)
            Surrogate predicted mean at each candidate.
        std : np.ndarray, shape (m,)
            Surrogate predicted std at each candidate.
        y_best : float
            Best observed y value so far.

        Returns
        -------
        scores : np.ndarray, shape (m,)
            Higher = more promising to query.
        """


class ExpectedImprovement(BaseAcquisition):
    """
    Expected Improvement (EI).

    The workhorse of Bayesian optimisation. Computes the expected amount
    by which a candidate point will improve over the current best, taking
    both predicted mean and uncertainty into account.

    EI naturally balances exploration and exploitation:
    - Points with high mean near y_best → exploitation
    - Points with high std far from data → exploration

    Parameters
    ----------
    xi : float
        Exploration–exploitation trade-off. Larger xi encourages more
        exploration by raising the improvement threshold above y_best.
        Typical range: 0.0 (pure exploitation) to 0.1.
    maximize : bool
        If True, seek maximum y. If False (default), seek minimum y.
    """

    def __init__(self, xi: float = 0.01, maximize: bool = False):
        self.xi = xi
        self.maximize = maximize

    def __call__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        y_best: float,
    ) -> np.ndarray:
        std = np.clip(std, 1e-9, None)   # avoid division by zero in flat regions

        if self.maximize:
            improvement = mean - y_best - self.xi
        else:
            improvement = y_best - mean - self.xi

        Z = improvement / std
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        ei[ei < 0] = 0.0
        return ei


class UCB(BaseAcquisition):
    """
    Upper Confidence Bound (UCB).

    A simpler alternative to EI. Scores each candidate as a linear
    combination of predicted mean and uncertainty. The kappa parameter
    directly controls the exploration–exploitation balance.

    UCB is more transparent than EI — you can reason directly about
    what kappa is doing. It can overexplore in high dimensions but
    works well at your data scale.

    Parameters
    ----------
    kappa : float
        Weight on uncertainty. Higher = more exploration.
        Typical range: 1.0 (exploitation-heavy) to 5.0 (exploration-heavy).
    maximize : bool
        If True, seek maximum y. If False (default), seek minimum y.
    """

    def __init__(self, kappa: float = 2.0, maximize: bool = False):
        self.kappa = kappa
        self.maximize = maximize

    def __call__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        y_best: float,
    ) -> np.ndarray:
        if self.maximize:
            return mean + self.kappa * std
        else:
            return -mean + self.kappa * std


class ProbabilityOfImprovement(BaseAcquisition):
    """
    Probability of Improvement (PI).

    Computes the probability that a candidate strictly improves over
    y_best. Simpler than EI — it ignores the magnitude of improvement
    and only asks "how likely is any improvement at all?".

    PI tends to be more exploitation-heavy than EI and can get stuck
    near the current best. Useful as a comparison baseline or when
    you want conservative, low-risk suggestions.

    Parameters
    ----------
    xi : float
        Exploration buffer, same role as in EI.
    maximize : bool
        If True, seek maximum y. If False (default), seek minimum y.
    """

    def __init__(self, xi: float = 0.01, maximize: bool = False):
        self.xi = xi
        self.maximize = maximize

    def __call__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        y_best: float,
    ) -> np.ndarray:
        std = np.clip(std, 1e-9, None)

        if self.maximize:
            Z = (mean - y_best - self.xi) / std
        else:
            Z = (y_best - mean - self.xi) / std

        return norm.cdf(Z)


# registry for use in CONFIG dicts
ACQUISITION_REGISTRY: dict[str, type[BaseAcquisition]] = {
    "EI":  ExpectedImprovement,
    "UCB": UCB,
    "PI":  ProbabilityOfImprovement,
}


def get_acquisition(name: str, **kwargs) -> BaseAcquisition:
    """
    Instantiate an acquisition function by name.

    Useful for building acquisition objects directly from CONFIG:

        acq = get_acquisition(CONFIG["acquisition"], xi=0.01)

    Parameters
    ----------
    name : str
        One of 'EI', 'UCB', 'PI'.
    **kwargs
        Passed to the acquisition function constructor.
    """
    if name not in ACQUISITION_REGISTRY:
        raise ValueError(
            f"Unknown acquisition '{name}'. "
            f"Available: {list(ACQUISITION_REGISTRY.keys())}"
        )
    return ACQUISITION_REGISTRY[name](**kwargs)


def suggest_next(
    X_candidates: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    y_best: float,
    acq: BaseAcquisition,
    n: int = 1,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the top-n candidate points ranked by acquisition score.

    Parameters
    ----------
    X_candidates : np.ndarray, shape (m, d)
        Grid or random candidates to score.
    mean : np.ndarray, shape (m,)
    std : np.ndarray, shape (m,)
    y_best : float
        Best observed value so far.
    acq : BaseAcquisition
        Acquisition function instance.
    n : int
        Number of suggestions to return.
    exclude : np.ndarray | None
        Points to exclude from suggestions (e.g. already queried).
        Rows matching any row in exclude are masked out.

    Returns
    -------
    X_best : np.ndarray, shape (n, d)
        Top-n suggested query points in ranked order.
    scores : np.ndarray, shape (n,)
        Their acquisition scores.
    """
    scores = acq(mean, std, y_best)

    if exclude is not None and len(exclude) > 0:
        mask = np.ones(len(X_candidates), dtype=bool)
        for x_ex in exclude:
            match = np.all(np.isclose(X_candidates, x_ex), axis=1)
            mask &= ~match
        scores = np.where(mask, scores, -np.inf)

    top_idx = np.argsort(scores)[::-1][:n]
    return X_candidates[top_idx], scores[top_idx]
