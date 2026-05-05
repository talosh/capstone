import numpy as np
import itertools
import random
from typing import Any
from tqdm.auto import tqdm

from .surrogates.base import BaseSurrogate


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def loo_cv_score(
    surrogate_cls: type[BaseSurrogate],
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Leave-One-Out Cross Validation RMSE for a surrogate and hyperparameter set.

    At small data scales (10–40 points) LOO-CV gives a much more reliable
    estimate of generalisation than a train/val split, since every point
    gets to be the validation point exactly once.

    Parameters
    ----------
    surrogate_cls : type[BaseSurrogate]
        The surrogate class to instantiate (not an instance).
    params : dict
        Hyperparameter dict passed to surrogate_cls(**params).
    X : np.ndarray, shape (n, d)
    y : np.ndarray, shape (n,)

    Returns
    -------
    rmse : float
        Root mean squared error across all LOO folds.
        Lower is better.
    """
    n = len(X)
    if n < 3:
        raise ValueError(f"LOO-CV requires at least 3 observations, got {n}.")

    errors = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train, y_train = X[mask], y[mask]
        X_val,   y_val   = X[[i]], y[[i]]

        try:
            model = surrogate_cls(**params)
            model.fit(X_train, y_train)
            mean, _ = model.predict(X_val)
            errors.append((mean[0] - y_val[0]) ** 2)
        except Exception:
            # if a combination fails on a fold, penalise heavily rather than crash
            errors.append(np.inf)

    return float(np.sqrt(np.mean(errors)))


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _expand_grid(param_grid: dict[str, list]) -> list[dict[str, Any]]:
    """Return all combinations from a parameter grid dict."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def grid_search(
    surrogate_cls: type[BaseSurrogate],
    param_grid: dict[str, list],
    X: np.ndarray,
    y: np.ndarray,
    n_random_trials: int | None = None,
    random_state: int = 42,
    label: str = "",
) -> list[dict[str, Any]]:
    """
    Grid search (or random search) over a hyperparameter grid using LOO-CV.

    Set n_random_trials to switch from exhaustive grid search to random search.
    Random search is useful when the grid is large — it often finds a good
    configuration in far fewer evaluations than the full grid.

    Parameters
    ----------
    surrogate_cls : type[BaseSurrogate]
        The surrogate class to tune.
    param_grid : dict[str, list]
        Parameter grid, e.g. {"C": [0.1, 1.0, 10.0], "epsilon": [0.01, 0.1]}.
    X : np.ndarray, shape (n, d)
    y : np.ndarray, shape (n,)
    n_random_trials : int | None
        If None, run full grid search.
        If int, randomly sample this many combinations (random search).
    random_state : int
        Seed for random sampling reproducibility.

    Returns
    -------
    results : list[dict]
        All evaluated combinations sorted by loo_rmse ascending (best first).
        Each entry: {"params": {...}, "loo_rmse": float, "method": str}
    """
    all_combos = _expand_grid(param_grid)

    if n_random_trials is not None and n_random_trials < len(all_combos):
        rng = random.Random(random_state)
        combos = rng.sample(all_combos, n_random_trials)
        method = f"random_search (n={n_random_trials}/{len(all_combos)})"
    else:
        combos = all_combos
        method = f"grid_search (n={len(all_combos)})"

    results = []
    desc = f"tuning {label}" if label else "tuning"
    for params in tqdm(combos, desc=desc, leave=False):
        score = loo_cv_score(surrogate_cls, params, X, y)
        results.append({"params": params, "loo_rmse": score, "method": method})

    results.sort(key=lambda r: r["loo_rmse"])
    return results


# ---------------------------------------------------------------------------
# Acquisition tuning
# ---------------------------------------------------------------------------

def tune_acquisition(
    surrogate_cls: type[BaseSurrogate],
    best_surrogate_params: dict[str, Any],
    acq_cls,
    acq_param_grid: dict[str, list],
    X: np.ndarray,
    y: np.ndarray,
    maximize: bool = False,
) -> list[dict[str, Any]]:
    """
    Tune acquisition function hyperparameters via LOO-CV.

    For each acquisition parameter combination, we measure how well it would
    have ranked the held-out point — i.e. whether the acquisition score at
    the true held-out x was in the top-k of the training candidates.

    Uses rank percentile as the metric: higher is better (1.0 = always ranks
    held-out point first among candidates).

    Parameters
    ----------
    surrogate_cls : type[BaseSurrogate]
        Best surrogate class (already tuned).
    best_surrogate_params : dict
        Best hyperparameters found for the surrogate.
    acq_cls : BaseAcquisition subclass
        Acquisition function class to tune.
    acq_param_grid : dict[str, list]
        Grid of acquisition hyperparameters, e.g. {"xi": [0.01, 0.05, 0.1]}.
    X : np.ndarray, shape (n, d)
    y : np.ndarray, shape (n,)
    maximize : bool
        Whether the objective is maximised.

    Returns
    -------
    results : list[dict]
        Sorted by rank_percentile descending (best first).
        Each entry: {"params": {...}, "rank_percentile": float}
    """
    n = len(X)
    acq_combos = _expand_grid(acq_param_grid)
    results = []

    for acq_params in acq_combos:
        acq = acq_cls(maximize=maximize, **acq_params)
        rank_percentiles = []

        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_train, y_train = X[mask], y[mask]
            x_held = X[[i]]

            try:
                model = surrogate_cls(**best_surrogate_params)
                model.fit(X_train, y_train)

                # score held-out point alongside training points as candidates
                X_cands = np.vstack([X_train, x_held])
                mean, std = model.predict(X_cands)
                y_best = y_train.max() if maximize else y_train.min()
                scores = acq(mean, std, y_best)

                # rank of held-out point (higher percentile = better)
                held_score = scores[-1]
                rank = np.mean(scores <= held_score)
                rank_percentiles.append(rank)
            except Exception:
                rank_percentiles.append(0.0)

        results.append({
            "params": acq_params,
            "rank_percentile": float(np.mean(rank_percentiles)),
        })

    results.sort(key=lambda r: r["rank_percentile"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def results_to_table(
    tuning_results: dict[str, list[dict]],
    top_n: int = 3,
) -> list[dict]:
    """
    Flatten tuning results across surrogates into a summary table.

    Parameters
    ----------
    tuning_results : dict[str, list[dict]]
        Output of grid_search() keyed by method name,
        e.g. {"GP": [...], "SVR": [...], "MLP": [...]}.
    top_n : int
        How many top results to include per surrogate.

    Returns
    -------
    rows : list[dict]
        Each row: {"model", "rank", "loo_rmse", "params", "method"}
    """
    rows = []
    for model_name, results in tuning_results.items():
        for rank, result in enumerate(results[:top_n], start=1):
            rows.append({
                "model":    model_name,
                "rank":     rank,
                "loo_rmse": round(result["loo_rmse"], 6),
                "params":   result["params"],
                "method":   result["method"],
            })
    return rows
