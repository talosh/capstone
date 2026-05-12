# Datasheet: BBO Capstone Query History and Function Evaluations

## Motivation

This dataset was created as part of the Black-Box Optimisation (BBO) Capstone
Project for the Imperial College London Professional Certificate in Machine
Learning and Artificial Intelligence. It supports a sequential Bayesian
optimisation task in which eight unknown objective functions are queried
iteratively over multiple rounds with the goal of finding the input combination
that maximises each function's output.

The dataset was not collected from an external source. It was generated
entirely through the optimisation process itself: an initial seed dataset was
provided by the course, and all subsequent observations were accumulated
through weekly query submissions. Its primary purpose is to train and evaluate
surrogate models (Gaussian Process, MLP ensemble, XGBoost) and to serve as a
reproducible record of all decisions made during the optimisation.

---

## Composition

The dataset consists of observations for eight black-box functions stored in
JSON format, one file per function (`function_01.json` through
`function_08.json`).

**Function dimensions and initial seed sizes:**

| Function | Input dim | Seed points |
|----------|-----------|-------------|
| 1 | 2 | 10 |
| 2 | 2 | 10 |
| 3 | 3 | 15 |
| 4 | 4 | 30 |
| 5 | 4 | 20 |
| 6 | 5 | 20 |
| 7 | 6 | 30 |
| 8 | 8 | 40 |

Each observation record contains:

- `function_id`: integer (1–8)
- `round`: integer (0 = seed, 1–N = query rounds)
- `source`: string (`"initial"` or `"query"`)
- `x`: list of floats in [0, 1), 6 decimal places
- `y`: float (function output) or `null` (pending result)
- `note`: free-text annotation

After 8 query rounds, the dataset contains between 18 and 48 observations per
function (seed + queries).

**Known gaps and limitations:**

- Rounds 1–7 for functions 1, 2, 3, 4 and 6 were queried under an incorrect
  minimisation objective. These observations are retained in the dataset with
  round numbers and notes, but they sample regions of the search space that
  are not relevant to the true maximisation objective. This creates an
  imbalanced dataset for those functions, with dense coverage in low-output
  corner regions and sparse coverage near the true maxima.
- The candidate grid used for acquisition at each round was 1000 random points
  per function. In 6D and 8D this provides very low coverage density, meaning
  the dataset may not include observations near the true global maximum.
- Function outputs for fn01 span from approximately 1e-124 to -0.004 in the
  seed data and near-zero in most of the search space, making this function
  significantly harder to optimise than the others.

---

## Collection Process

**Timeframe:** 8 query rounds, one per week, over approximately 8 weeks.

**Process per round:**

1. Load all previous observations from the JSON files via `src/loader.py`
2. Fit surrogate models (GP, MLP ensemble, XGBoost) to all available (x, y) pairs
3. Score 1000 random candidate points using Expected Improvement (EI) with
   the surrogate's predicted mean and uncertainty
4. Select the candidate with the highest EI score as the submission point
5. Submit to the course portal, receive the output, and record it in the JSON

**Strategy evolution across rounds:**

- Rounds 1–3: manual exploration including a deliberate centre-point query
  [0.5, ..., 0.5] across all functions to establish a baseline
- Rounds 4–7: acquisition-function-driven exploitation, initially under the
  incorrect minimisation objective (see Gaps above); corrected to maximisation
  from round 8 onward
- Round 8 onward: corrected objective with per-function strategy differentiated
  by surrogate reliability (LOO-CV RMSE and acquisition rank percentile)

**Surrogate models used:** Gaussian Process with Matern/RBF kernel (sklearn),
MLP with deep ensemble uncertainty (PyTorch, 5 members), XGBoost with
quantile regression uncertainty. Hyperparameters tuned via random search with
LOO-CV scoring.

**Acquisition function:** Expected Improvement (EI) with xi tuned per function
and surrogate via LOO-CV rank percentile.

---

## Preprocessing and Uses

**Transformations applied:**

- X values are constrained to [0, 0.999999] as required by the challenge
  portal. Values suggested at exactly 1.0 were clipped to 0.999000.
- X and y values are standardised internally within each surrogate model
  (StandardScaler) before fitting. The raw values stored in the JSON files
  are always in their original unscaled form.
- No log-transformation was applied to outputs, although fn01's extreme output
  range (spanning ~120 orders of magnitude) would benefit from it.

**Intended uses:**

- Training and evaluating Bayesian optimisation surrogate models
- Demonstrating iterative query strategy with structured data logging
- Portfolio documentation of an optimisation experiment
- Educational demonstration of exploration vs exploitation trade-offs

**Inappropriate uses:**

- Benchmarking against other optimisation methods (the functions are synthetic
  and the query history reflects a specific sequential strategy, not random
  sampling)
- Statistical inference about function properties (sample sizes of 18–48 per
  function are insufficient for generalisation)
- Comparison across functions (output scales differ by many orders of magnitude
  and are not normalised across functions)

---

## Distribution and Maintenance

The dataset is stored in the `data/` directory of the project GitHub
repository as plain JSON files. It is available publicly as part of the
capstone project portfolio.

The initial seed data (rounds with `"source": "initial"`) was provided by
Imperial College London as part of the course materials and should be treated
as read-only. Query observations (`"source": "query"`) were generated by the
project author and are the author's own work.

The repository owner is responsible for maintaining the dataset. No automated
update process exists; all new observations are added manually via the
`add_observation()` function in `src/loader.py`. The JSON format is
human-readable and directly editable as a fallback.

The dataset is intended for educational purposes and is not suitable for
production use without significant additional validation.
