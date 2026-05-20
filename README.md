# Logic Operations For Assessment of Experts Weight In Fuzzy Rule-Based Systems

This repository is an implementation of the method proposed in "Logic Operations For Assessment of Experts Weight In Fuzzy Rule-Based Systems" (Submitted).

## Requirements

Python 3 with:

- numpy
- matplotlib
- scikit-fuzzy
- scipy
- problog (required only for `alternative_methods/`)
- pymcdm (required only for `alternative_methods/`)
- ollama (Python client, required only for `answer_weighting/`)
- tqdm (required only for `answer_weighting/`)

## How To Run

### Root — Reference Method

Run everything from this folder:

```bash
python3 compute_output_weight_integrals.py
python3 compute_trapezoidal_memberships.py
python3 check_example_1.py
python3 check_example_2.py
```

### `alternative_methods/` — TOPSIS, MLN, and ProbLog Comparisons

Run from the `alternative_methods/` folder:

```bash
cd alternative_methods

# TOPSIS
python3 topsis_example_1.py
python3 topsis_example_2.py

# Markov Logic Networks
python3 mln_example_1.py
python3 mln_example_2.py

# ProbLog
python3 problog_example_1.py
python3 problog_example_2.py

# LaTeX ranking tables for all methods/examples
python3 print_latex_tables.py
```

### `sensitivity_analisys/` — Sensitivity Analysis

Run from the `sensitivity_analisys/` folder:

```bash
cd sensitivity_analisys

# Generate per-run metadata (perturbs inputs and records Kendall tau)
python3 sensitivity_analysis_example_1.py
python3 sensitivity_analysis_example_2.py

# Plot results per example
python3 plot_sensitivity_example_1.py
python3 plot_sensitivity_example_2.py

# Combined plot for both examples
python3 plot_sensitivity_combined.py
```

### `tau_kendall/` — T-norm Variants and Kendall Tau Comparison

Run from the `tau_kendall/` folder:

```bash
cd tau_kendall

# Product T-norm (baseline, same as root)
python3 check_example_1.py
python3 check_example_2.py

# Gödel T-norm
python3 check_example_1_godel.py
python3 check_example_2_godel.py

# Łukasiewicz T-norm
python3 check_example_1_Lukasiewicz.py
python3 check_example_2_Lukasiewicz.py

# Delta variants (del_1 through del_5)
python3 check_example_1_del_1.py
python3 check_example_1_del_2.py
python3 check_example_1_del_3.py
python3 check_example_1_del_4.py
python3 check_example_1_del_5.py

python3 check_example_2_del_1.py
python3 check_example_2_del_2.py
python3 check_example_2_del_3.py
python3 check_example_2_del_4.py
python3 check_example_2_del_5.py
```

### `answer_weighting/` — MCGDM Answer Evaluation

Rates three answer alternatives (A1/A2/A3) using multiple LLMs as experts and
aggregates their scores via the two-step MCGDM formulation in
`mcgdm_weighted_formulation.md`. Requires a running Ollama instance.

Run from the `answer_weighting/` folder:

```bash
cd answer_weighting

# Step 1 — rate all three alternatives with every LLM (writes results.json and mcgdm_ratings.json)
python3 rate_computational.py

# Step 2 — apply weighted aggregation and rank alternatives
python3 mcgdm_weighted.py
```
