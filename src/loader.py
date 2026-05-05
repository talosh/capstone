import json
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def _path(function_id: int) -> Path:
    return DATA_DIR / f"function_{function_id:02d}.json"


def load_function(function_id: int) -> dict:
    """Return the raw JSON dict for a function."""
    with open(_path(function_id)) as f:
        return json.load(f)


def get_observations(
    function_id: int,
    source: str | None = None,
    round: int | None = None,
    exclude_pending: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (X, y) arrays for a function's observations.

    Parameters
    ----------
    function_id : int
        Which function to load (1–8).
    source : str | None
        Filter by source: 'initial', 'query', or None for all.
    round : int | None
        Filter to a specific round, or None for all.
    exclude_pending : bool
        If True (default), drop observations where y is None
        (submitted but not yet returned).

    Returns
    -------
    X : np.ndarray, shape (n, input_dim)
    y : np.ndarray, shape (n,)
    """
    data = load_function(function_id)
    obs = data["observations"]

    if source is not None:
        obs = [o for o in obs if o["source"] == source]
    if round is not None:
        obs = [o for o in obs if o["round"] == round]
    if exclude_pending:
        obs = [o for o in obs if o["y"] is not None]

    if not obs:
        dim = data["input_dim"]
        return np.empty((0, dim)), np.empty(0)

    X = np.array([o["x"] for o in obs], dtype=float)
    y = np.array([o["y"] for o in obs], dtype=float)
    return X, y


def add_observation(
    function_id: int,
    round: int,
    x: list[float],
    y: float | None,
    note: str = "",
) -> None:
    """
    Append a new observation to a function's JSON file.

    Set y=None to record a pending submission (result not yet returned).

    Parameters
    ----------
    function_id : int
        Which function to update (1–8).
    round : int
        The query round this point belongs to.
    x : list[float]
        Input point. Must match the function's input_dim.
    y : float | None
        Output value, or None if awaiting result.
    note : str
        Optional free-text note (e.g. 'suggested by EI', 'manual guess').
    """
    path = _path(function_id)
    with open(path) as f:
        data = json.load(f)

    expected_dim = data["input_dim"]
    if len(x) != expected_dim:
        raise ValueError(
            f"function {function_id} expects input_dim={expected_dim}, got {len(x)}"
        )

    data["observations"].append({
        "round": round,
        "source": "query",
        "x": list(x),
        "y": y,
        "note": note,
    })

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def fill_result(function_id: int, x: list[float], y: float) -> None:
    """
    Fill in the y value for a pending observation (where y is currently None).

    Matches on x exactly — use the same list you passed to add_observation.

    Parameters
    ----------
    function_id : int
        Which function to update.
    x : list[float]
        The input point to match.
    y : float
        The result to fill in.
    """
    path = _path(function_id)
    with open(path) as f:
        data = json.load(f)

    matched = False
    for obs in data["observations"]:
        if obs["y"] is None and obs["x"] == list(x):
            obs["y"] = y
            matched = True
            break

    if not matched:
        raise ValueError(
            f"No pending observation found for function {function_id} at x={x}"
        )

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def summary() -> None:
    """Print a quick overview of all functions and their observation counts."""
    for fid in range(1, 9):
        path = _path(fid)
        if not path.exists():
            print(f"function {fid:02d}: file not found")
            continue
        data = load_function(fid)
        obs = data["observations"]
        n_initial = sum(1 for o in obs if o["source"] == "initial")
        n_query   = sum(1 for o in obs if o["source"] == "query" and o["y"] is not None)
        n_pending = sum(1 for o in obs if o["y"] is None)
        rounds    = sorted({o["round"] for o in obs if o["source"] == "query"})
        print(
            f"function {fid:02d} | dim={data['input_dim']} | "
            f"initial={n_initial} | queries={n_query} | pending={n_pending} | "
            f"query rounds={rounds}"
        )
