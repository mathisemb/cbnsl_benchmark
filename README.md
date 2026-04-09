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

> **Note:** The conda version of otagrum is not yet up to date with the [GitHub repository](https://github.com/openturns/otagrum). The latest version integrates aGrUM's Meek rules into ContinuousPC and ContinuousMIIC. The conda package will be updated soon.

## Usage

### Run benchmarks

Edit the configurations in `views/run_all_benchmarks.py`, then:

```bash
conda activate cbnsl
python views/run_all_benchmarks.py
```

Results are saved in `views/results/`.

### Visualize results

Open `views/visualize_gridsearch.ipynb` to explore and compare the results.

### Scaling study

The `scaling/` directory measures how algorithms scale with graph size (number of nodes and samples). See `scaling/run_scaling.py`.
