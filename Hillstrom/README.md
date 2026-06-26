# Hillstrom Experiment

This directory contains the reproducible Hillstrom email marketing experiment comparing **PC-RACP** and **RAC**.

## Data and Task

The script downloads the Hillstrom email marketing dataset on first use and caches it under the user's home cache directory.

The original treatment variable `segment` is mapped to a binary action:

- `A = 0`: `No E-Mail`
- `A = 1`: `Mens E-Mail` or `Womens E-Mail`

The binary label is `visit`, where `Y = 1` means the customer visited the website. The covariates are:

```text
recency, history, mens, womens, newbie
```

The Hillstrom logging policy is known from randomization:

```text
P(A = 0 | X) = 1/3
P(A = 1 | X) = 2/3
```

The utility table is:

```text
        y=0   y=1
a=0    0.40  0.25
a=1    0.10  0.90
```

with $u_{\max}=1.0$.

## Install

From the repository root:

```bash
python -m pip install -r requirements.txt
```

## Reproduce the paper experiment

The following command reproduces the figures reported in the paper.

From the repository root, run the Hillstrom experiment:

```bash
python Hillstrom/run_compare.py
```

Optional arguments include:

- `--seed`: random seed for subsampling, splitting, and CatBoost
- `--n_samples`: number of rows to use; default is `64000`
- `--alphas`: comma-separated alpha grid; default is `0.02,0.04,...,0.20`
- `--out_dir`: output directory; default is `Hillstrom/results/hillstrom`
- `--catboost_iterations`, `--catboost_depth`, `--catboost_learning_rate`

## Outputs

Outputs are written under `Hillstrom/results/hillstrom/` by default.

The main CSV is:

```text
alpha_sweep.csv
```

The generated figures are:

- `coverage_vs_alpha.pdf`
- `utility_vs_alpha.pdf`
- `pair_counts_a0_vs_alpha.pdf`
- `pair_counts_a1_vs_alpha.pdf`
- `set01_rate_given_optimal_a0_vs_alpha.pdf`
- `optimal_action1_count_vs_alpha.pdf`

PNG copies are also saved for quick inspection.
