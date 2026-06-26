# Multi-seed Simulation Study

This directory contains the reproducible synthetic simulation comparing **PC-RACP**, **RAC**, and **Plug-in**.

## Experimental setup

The data-generating process is implemented in `src/data_generator.py`:

- $X \sim \mathcal{N}(0, I_d)$, with $d=10$;
- $A \sim \pi_b(A \mid X)$, using a linear softmax behavior policy;
- $Y \sim P(Y \mid X,A)$, using an action-dependent linear softmax model;
- $|\mathcal{A}|=3$, $|\mathcal{Y}|=4$;
- 30,000 samples per seed by default;
- 30%/20%/20%/30% train, learn, calibration, and test splits;
- $\alpha \in \{0.02, 0.04, \ldots, 0.20\}$.

The utility table is:

```text
        y=0   y=1   y=2   y=3
a=0    0.70  0.60  0.35  0.15
a=1    0.95  0.55  0.45  0.15
a=2    0.80  0.50  0.20  0.20
```

with $u_{\max}=1.0$.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
```

Activate the environment, then install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the following commands from the repository root.

## Reproduce the paper experiment

The following commands reproduce the figures reported in the paper.

Run 20 consecutive seeds with multinomial logistic regression:

```bash
python simulation/run_compare.py \
  --start_seed 0 \
  --num_seeds 20 \
  --max_workers 5 \
  --model_type logistic
```

Run the corresponding random-forest experiment:

```bash
python simulation/run_compare.py \
  --start_seed 0 \
  --num_seeds 20 \
  --max_workers 5 \
  --model_type random_forest
```

On Windows PowerShell, replace `\` with the backtick continuation character or write the command on one line.

`--num_seeds` must be at least 2. Use `python simulation/run_compare.py --help` for all options.

## Outputs

Results are written to:

```text
simulation/results/simulation/{model_type}/
```

Each run creates:

- `simulation_seed_XX.csv`: one alpha sweep for a seed;
- `simulation_summary.csv`: across-seed means for the target coverage and the coverage and utility of each method;
- `coverage_vs_alpha_*.pdf` and `.png`;
- `utility_vs_alpha_*.pdf` and `.png`.

The per-seed result columns are:

```text
alpha
target_coverage
pc_racp_coverage
rac_coverage
plugin_coverage
pc_racp_utility
rac_utility
plugin_utility
seed
```

The summary columns are:

```text
alpha
target_coverage
pc_racp_coverage
rac_coverage
plugin_coverage
pc_racp_utility
rac_utility
plugin_utility
```

## Reproducibility safeguards

- Seeds are explicit, consecutive, validated, and recorded.
- Every seed runs in a fresh spawned process.
- BLAS/OpenMP thread counts default to one per seed worker, avoiding accidental nested parallelism.
- All seed files are schema-checked before aggregation.
- Every alpha must contain exactly one row from every requested seed.
- A directory cannot be reused with a different seed set; use a new `--out_dir` for a different experiment.
- CSV outputs are replaced atomically.

The method implementations live in `src/`. The only executable experiment interface is `run_compare.py`, and it always runs multiple seeds.
