# PC-RACP Experiments

This repository contains the experimental code for the **PC-RACP** method.

The repository includes two studies:

- `simulation/`: multi-seed synthetic experiments comparing PC-RACP with RAC and a direct plug-in policy.
- `Hillstrom/`: a real-data experiment using the Hillstrom email marketing dataset, comparing PC-RACP with RAC.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the experiments

The repository contains separate README files for each experiment:

- See `simulation/README.md` for the synthetic simulation.
- See `Hillstrom/README.md` for the Hillstrom experiment.

Generated CSV files and figures are saved in the corresponding `results/` directories.
