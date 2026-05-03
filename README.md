# capstone

# BBO Capstone Project – README

## Section 1: Project Overview  
The Black-Box Optimisation (BBO) capstone project focuses on optimising unknown functions using a limited number of queries. The key challenge is that the functional form is not available; instead, we only observe outputs for chosen inputs.  

In this project, there are **8 separate models**, with input dimensionalities ranging from **2 to 8**, increasing the complexity of the optimisation problem.  

The overall goal is to efficiently identify input configurations that maximise each function while minimising the number of evaluations. This is highly relevant in real-world ML settings such as hyperparameter tuning, experimental design, and simulation optimisation, where evaluating the objective is costly or slow.  

This project builds intuition for decision-making under uncertainty, which is directly applicable to data science and quantitative roles.

---

## Section 2: Inputs and Outputs  

Each model takes an input vector whose dimension depends on the specific task (between 2 and 8). Each feature is bounded, typically in `[0,1]`.  

**Input format:**  
`x = (x1, x2, ..., xd),   d ∈ {2, ..., 8},   xi ∈ [0,1]`

**Example (8D case):**  
`(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)`

The output is a scalar performance value:  

**Output format:**  
`y = f(x)`

This value serves as feedback to guide future queries.

---

## Section 3: Challenge Objectives  

The objective is to **maximise** each unknown function.  

Key constraints include:  
- Limited query budget  
- No access to gradients or explicit functional form  
- Increasing dimensionality (2D → 8D), making exploration harder  
- Potential non-linearity and noise  

These factors require efficient strategies to explore the space while exploiting promising regions.

---

## Section 4: Technical Approach  

### Early Strategy  
Across the first three submissions, I explored Gaussian Process (GP) optimisation with varying exploration parameters:  
- First submission: GP with ξ = 0.01 (more exploitative)  
- Second submission: GP with ξ = 0.5 (more exploratory)  
- Third submission: querying the central point (0.5, ..., 0.5) across all models to establish a consistent baseline  

This allowed me to compare how exploration intensity affects performance across different dimensionalities.

### Modelling Approach  
The primary approach is to model the unknown function using Gaussian Processes, which provide both predictions and uncertainty estimates. This supports principled query selection via acquisition functions.  

I am also considering extending this approach by incorporating:  
- **Support Vector Machines (SVMs)** to classify high vs low-performing regions  
- **MLP neural networks** to learn more flexible, non-linear approximations of the response surface  

These models could complement GP methods, especially in higher dimensions where GP scalability becomes challenging.

### Exploration vs Exploitation  
The trade-off is controlled via the parameter ξ:  
- Low ξ: focuses on exploiting known good regions  
- High ξ: encourages exploration of uncertain areas  

This is particularly important as dimensionality increases, where naive exploration becomes inefficient.

### Reflection  
The approach combines probabilistic modelling with simple heuristics (e.g. central baseline). As dimensionality grows, structured methods like SVMs and neural networks become increasingly relevant, reinforcing key ideas from real-world optimisation problems where data is limited and costly to obtain.

## Project Structure

```
capstone/
├── data/
│   ├── function_01.json
│   ├── function_02.json
│   └── ...
├── src/
│   ├── loader.py
│   ├── surrogates/
│   │   ├── gp.py
│   │   ├── svr.py
│   │   └── mlp.py
│   └── acquisition.py
├── notebooks/
│   └── analysis.ipynb
├── reports/
└── README.md
```

### `data/`

One JSON file per black-box function, named `function_01.json` through `function_08.json`. Each file is the single source of truth for that function — it holds the initial seed observations provided at the start of the challenge alongside every query point submitted and result received since. Initial seed points are marked `"source": "initial"` to preserve the raw/derived distinction without requiring a separate folder. Query results awaiting a response are recorded immediately with `"y": null` and filled in once the result arrives, reflecting the time-delayed nature of the evaluation process. The files are plain JSON, directly editable in any text editor.

### `src/`

Reusable Python modules containing all modelling logic. Nothing here is tied to a specific function or experiment run — these are callable building blocks:

- `loader.py` — reads a function JSON and returns arrays ready for modelling
- `surrogates/` — one module per surrogate type (GP, SVR, MLP), each exposing a consistent `fit` / `predict` interface
- `acquisition.py` — acquisition functions (Expected Improvement, UCB) shared across surrogates

Keeping logic here rather than inside notebooks means it can be iterated on independently, imported cleanly, and reused across functions without copy-pasting.

### `notebooks/analysis.ipynb`

A single universal notebook that acts as the experiment runner and report. A `CONFIG` dict at the top of the notebook controls which functions to load, which surrogate methods to apply, and what hyperparameters to use for each. Changing the config and re-running the notebook produces a fresh comparison across all selected functions and methods. The notebook itself is stable — only the config and `src/` evolve as the project develops.

### `reports/`

Written reflections, module submissions, and any summaries produced at the end of a challenge round. Static documents, not generated outputs.


## Section 6: Updated Modelling Strategy

The project now uses a hybrid surrogate modelling approach:

- **Gaussian Processes** for prediction + uncertainty estimates  
- **SVM classifiers** for separating promising vs poor-performing regions  
- **PyTorch MLP neural networks** for nonlinear approximation and gradient-based guidance  

This allows both global exploration and local refinement.

---

## Section 7: Current Lessons Learned

After multiple submission rounds:

- Simpler models often work well when data is limited  
- Neural networks are most useful when nonlinear structure appears  
- Classification of “good vs bad” regions can be easier than exact regression  
- Balancing exploration vs exploitation remains the key challenge

---

## Section 8: Next Steps

- Continue comparing GP, SVM, and neural-network surrogates  
- Use gradients and uncertainty jointly when selecting queries  
- Improve automation of query generation across all 8 functions  
- Track performance trends across rounds


