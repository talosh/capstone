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
