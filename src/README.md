Reusable Python modules containing all modelling logic. Nothing here is tied to a specific function or experiment run — these are clean, callable building blocks:

- `loader.py` — reads a function JSON and returns arrays ready for modelling
- `surrogates/` — one module per surrogate type (GP, SVR, MLP), each exposing a consistent `fit` / `predict` interface
- `acquisition.py` — acquisition functions (Expected Improvement, UCB) shared across surrogates

Keeping logic here rather than inside notebooks means it can be iterated on independently, imported cleanly, and reused across functions without copy-pasting.
