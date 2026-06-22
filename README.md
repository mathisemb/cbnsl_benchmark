# Continuous Bayesian Network Structure Learning Benchmark

This repository benchmarks structure learning algorithms for Bayesian networks in the continuous case. It provides a unified pipeline to run them on the same datasets and compare their results.

## Algorithms

| Algorithm | Package |
|---|---|
| ContinuousPC | otagrum |
| ContinuousMIIC | otagrum |
| MIIC | pyAgrum |
| GHC+BDeu | pyAgrum |
| NOTEARS (linear & nonlinear) | notears |
| DAGMA (linear & nonlinear) | dagma |
| DirectLiNGAM | lingam |

## Installation

### Quick start
```bash
./install.sh
```

### Manual installation

```bash
conda create -n cbnsl
conda activate cbnsl
conda install otagrum
pip install -e .
pip install git+https://github.com/xunzheng/notears.git
pip install dagma
```

`conda install otagrum` installs otagrum along with its dependencies (pyAgrum, OpenTURNS). The remaining packages (lingam, notears, dagma) are installed via pip.

## Usage

The repository contains **two independent experiments**, each in its own directory.

### 1. Grid-search benchmark (`views/`)

Runs every algorithm with a full hyperparameter grid search on fixed datasets (the Sachs network and fixed synthetic graphs), and keeps the best configuration per algorithm.

**Run** — edit the configurations at the top of `views/run_all_benchmarks.py`, then:

```bash
conda activate cbnsl
python views/run_all_benchmarks.py
```

Results are saved under `views/results/`.

**Visualize** — open `views/visualize_gridsearch.ipynb`.

### 2. Scaling study (`scaling/`)

Measures how the algorithms scale with the number of variables and samples, averaged over many random graphs.

**Run** — the shell scripts execute the `scaling_study_*.ipynb` notebooks and write results incrementally to `scaling/results/`:

```bash
conda activate cbnsl
cd scaling
bash run_all.sh        # all algorithms except DAGMA
bash run_all_dagma.sh  # DAGMA runs
```

**Visualize** — the results are explored with dedicated notebooks:

- `scaling/ranking.ipynb` — overall ranking of the algorithms;
- `scaling/latex_report_plots.ipynb` — per-dataset scaling curves (SHD, F1, TPR vs. samples);
- `scaling/error_analysis.ipynb` — how often each algorithm fails to return a DAG.
