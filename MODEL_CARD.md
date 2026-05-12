# Model Card: BBO Capstone Bayesian Optimisation Strategy

## Overview

**Name:** BBO Capstone Bayesian Optimisation Pipeline  
**Type:** Sequential black-box optimisation framework  
**Version:** v1.0 (Rounds 1–8)  
**Repository:** `capstone/` — see `src/` for all model code  

This model represents a multi-surrogate Bayesian optimisation workflow
developed to maximise eight unknown black-box functions under a limited query
budget. It is an iterative decision-making system that evolves after each
round based on newly observed data. The core loop is: fit surrogate → score
candidates via acquisition function → submit best point → record result →
repeat.

---

## Intended Use

**Suitable for:**
- Sequential black-box optimisation with expensive evaluations
- Low-data regimes (10–50 observations per function)
- Problems where gradient information is unavailable
- Educational demonstration of Bayesian optimisation with multiple surrogate
  types

**Not suitable for:**
- Real-time production systems without additional validation
- Functions with sharp discontinuities where GP kernel assumptions break down
- High-dimensional problems (>8D) without a denser candidate grid or
  gradient-based acquisition optimisation
- Tasks requiring guaranteed global optimality

---

## Details

### Architecture

Three surrogate models, each implementing a consistent `fit(X, y)` /
`predict(X) → (mean, std)` interface via a shared `BaseSurrogate` abstract
class:

**Gaussian Process (`src/surrogates/gp.py`)**  
sklearn `GaussianProcessRegressor` with a selectable kernel (RBF or Matern,
nu in {0.5, 1.5, 2.5}), ConstantKernel scaling, alpha noise term, and
StandardScaler on X. Uncertainty is native to the GP posterior. Kernel
hyperparameters are optimised via marginal likelihood with multiple random
restarts. ConvergenceWarnings suppressed during tuning runs.

**MLP Deep Ensemble (`src/surrogates/mlp.py`)**  
Five independently initialised two-layer PyTorch MLPs with ReLU activations,
trained on standardised (X, y) with Adam and L2 weight decay. Uncertainty is
the standard deviation across ensemble predictions. Each member receives a
different random seed to ensure genuine weight diversity.

**XGBoost with Quantile Regression (`src/surrogates/xgb.py`)**  
Three XGBoost regressors trained simultaneously: one for the mean (squared
error objective), one for the lower quantile (alpha), one for the upper
quantile (1-alpha). Uncertainty is derived from the quantile interval width.
Both X and y are standardised before fitting.

**Acquisition function (`src/acquisition.py`)**  
Expected Improvement (EI) with a tunable xi parameter. Upper Confidence Bound
(UCB) and Probability of Improvement (PI) are also implemented. All three
share a `maximize` flag to support both maximisation and minimisation
objectives. Candidate selection uses a 1000-point random grid in [0,1]^d.

### Strategy Evolution Across Rounds

**Rounds 1–3 (exploration)**  
Manual point selection including a deliberate centre-point query [0.5, ..., 0.5]
across all functions to establish a baseline. This round produced the best
observed value for fn02 (0.788), fn03 (-0.016) and fn04 (+0.029), which only
became apparent after the objective direction was corrected.

**Rounds 4–7 (acquisition-driven, incorrect objective)**  
GP-driven EI with `maximize=False`. This correctly handled fn05, fn07 and fn08
(positive outputs) but incorrectly drove fn01, fn02, fn03, fn04 and fn06
toward more negative values. fn04 reached -55 and fn06 reached -3.49 — both
of which are the worst possible directions for maximisation. This error was
identified when reviewing the challenge FAQ, which states all functions are
maximisation problems.

**Round 8 (corrected objective, multi-surrogate)**  
CONFIG updated to `maximize=True`. Hyperparameter tuning added via random
search with LOO-CV scoring across GP, MLP and XGB. Acquisition reliability
assessed via rank percentile. Functions where rounds 4–7 moved in the wrong
direction (fn02, fn03, fn04, fn06) were directed back toward their pre-round-4
best known regions.

### Hyperparameter Tuning

Random search with 20 trials per surrogate per function. Validation via
Leave-One-Out CV (LOO-CV) — appropriate at n=10–30 where train/val splits
produce unreliable estimates. Acquisition xi tuned separately via LOO rank
percentile: measures whether the acquisition function would have ranked the
held-out point in the top of the candidate distribution.

---

## Performance

Performance is measured as the maximum observed output per function across all
rounds. True optima are unknown, so improvement over previous best is used as
the per-round metric. Acquisition reliability is assessed via LOO rank
percentile (higher = surrogate's EI more reliably identifies informative
regions).

**Best observed values after round 8:**

| fn | Best y | Best x | Rounds improving |
|----|--------|--------|-----------------|
| 01 | 2.675e-9 | [0.5, 0.5] | 1/8 (effectively flat) |
| 02 | 0.788 | [0.5, 0.5] | 1/8 (rounds 4–7 wrong direction) |
| 03 | -0.016 | [0.5, 0.5, 0.5] | 1/8 (rounds 4–7 wrong direction) |
| 04 | +0.029 | [0.361, 0.335, 0.390, 0.444] | 1/8 (rounds 4–7 wrong direction) |
| 05 | 2606.003 | [0.200, 0.800, 0.999, 0.999] | 5/8 (x3/x4→1 confirmed) |
| 06 | -0.549 | [0.436, 0.235, 0.819, 0.903, 0.039] | 1/8 (rounds 2–7 wrong direction) |
| 07 | 1.428 | [0.050, 0.480, 0.260, 0.210, 0.410, 0.740] | 5/8 |
| 08 | 9.853 | [0.107, 0.230, 0.006, 0.110, 0.945, 0.761, 0.292, 0.513] | 5/8 |

**Surrogate reliability (from tuning, round 8):**  
GP achieved rank_percentile of 1.000 on fn01, fn02, fn03, fn07 and fn08 —
the most consistently reliable surrogate for acquisition. XGB produced the
highest EI scores on fn04 and fn05 but had lower rank percentiles overall,
suggesting well-fitted means but less well-calibrated uncertainty. MLP
uncertainty collapsed on fn05 and fn06 (near-zero std), producing near-zero
EI despite reasonable mean predictions.

---

## Assumptions and Limitations

**Key assumptions:**
- Functions are smooth enough for GP kernel interpolation. This holds for
  most functions but is questionable for fn01, which has a near-discontinuous
  two-regime output (near-zero almost everywhere, small negative pocket).
- The 1000-point random candidate grid adequately covers [0,1]^d. This is
  reasonable in 2–4D but increasingly inadequate in 6D and 8D where the
  expected spacing between random points grows substantially.
- LOO-CV RMSE is a reliable signal for surrogate quality at small n. At n=10
  the score variance is high enough that the top-ranked hyperparameter
  configuration may not be genuinely better than rank 5.

**Failure modes observed:**
- **Wrong objective direction (rounds 4–7):** The single global `maximize`
  flag defaulted to False, causing five functions to be optimised in the wrong
  direction for four rounds. Queries are irreversible, so these rounds
  represent wasted budget and introduce misleading training data for those
  functions.
- **MLP uncertainty collapse:** The 5-member ensemble converges to similar
  solutions at small n, producing overconfident predictions and near-zero EI.
  More aggressive weight decay or fewer ensemble members would mitigate this.
- **Random candidate grid in 8D:** fn08's 8-dimensional search space with
  1000 candidates is effectively sparse — a gradient-based acquisition
  optimiser (e.g. L-BFGS on the EI surface) would produce better-located
  candidates.

---

## Ethical Considerations

**Transparency:**  
All decisions — surrogate choices, hyperparameter values, acquisition settings,
and strategy changes — are logged in the per-round reflection markdown files
in `reports/`. The objective direction error and its correction are documented
explicitly. A reviewer can trace any submitted point back to the surrogate
that suggested it, the CONFIG that drove it, and the LOO-CV score that
justified the hyperparameter choice.

**Reproducibility:**  
The full query history is stored in version-controlled JSON files. The
`log.ipynb` notebook records every submission and result in chronological
order with one cell per round. Given the same JSON files and the same CONFIG,
the analysis notebook will reproduce the same surrogate fits and acquisition
scores. Stochastic elements (random candidate grid, MLP initialisation) use
fixed seeds throughout.

**Real-world relevance:**  
In production ML systems, the objective direction error observed here —
optimising toward the wrong target for several rounds before detecting the
mistake — is a genuine risk in any system where the optimisation objective is
configured separately from domain knowledge. This experience reinforces the
importance of validating objective direction against known ground truth before
committing to an extended optimisation campaign.
