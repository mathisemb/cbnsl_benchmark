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

Requires [conda](https://github.com/conda-forge/miniforge) and a C++ toolchain (provided below by conda itself).

The benchmark targets **pyAgrum 3** with the matching **OpenTURNS / otagrum**. otagrum is built from source: the aGrUM-3-compatible version is not yet on conda-forge.

```bash
# 1. environment: pyAgrum 3 + OpenTURNS + build tools
conda create -n cbnsl python=3.12 -y
conda activate cbnsl
conda install -c conda-forge pyagrum openturns cmake swig make cxx-compiler c-compiler seaborn tqdm python-graphviz jupyter -y
```

otagrum is compiled from source and installed **into the conda environment**
(nothing outside `$CONDA_PREFIX` is written). The clone is only needed to
compile: put it anywhere **outside this repository** and delete it afterwards.
Keep the `cbnsl` environment active during this step, so that `$CONDA_PREFIX`
points to it (check with `echo $CONDA_PREFIX`).

```bash
# 2. otagrum, built from source and installed into the conda environment
cd ..   # anywhere outside this repository
git clone https://github.com/openturns/otagrum.git otagrum-src
cmake -B otagrum-src/build -S otagrum-src \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DPython_EXECUTABLE=$CONDA_PREFIX/bin/python \
    -DPython_FIND_STRATEGY=LOCATION
cmake --build otagrum-src/build -j$(nproc)
cmake --install otagrum-src/build
```

```bash
# 3. this package (back in the repository)
pip install -e .
```

Check the install:

```bash
python -c "import pyagrum, openturns, otagrum; print(pyagrum.__version__, openturns.__version__, otagrum.__version__)"
```

The versions printed should be pyAgrum ≥ 3 and otagrum ≥ 0.15, and
`python -c "import otagrum; print(otagrum.__file__)"` should point inside the
conda environment. The sources can then be removed (`rm -rf otagrum-src`).

### Extra algorithms (optional)

NOTEARS, LiNGAM and DAGMA are only needed to run those methods:

```bash
conda install -c conda-forge pytorch-cpu -y
pip install lingam dagma
pip install git+https://github.com/xunzheng/notears.git
```

## Usage

The repository contains **two independent experiments**, each in its own directory.

### 1. Grid-search benchmark (`views/`)

Runs every algorithm with a full hyperparameter grid search on fixed datasets (the Sachs network and fixed synthetic graphs), and keeps the best configuration per algorithm.

```bash
conda activate cbnsl
python views/run_all_benchmarks.py
```

Results are saved under `views/results/`; visualize them with `views/visualize_gridsearch.ipynb`.

### 2. Scaling study (`scaling/`)

Measures how the algorithms scale with the number of variables and samples, averaged over many random graphs.

```bash
conda activate cbnsl
cd scaling
bash run_all.sh
```

Results are written incrementally to `scaling/results/`. Explore them with the notebooks in `scaling/` (`ranking.ipynb`, `latex_report_plots.ipynb`, `error_analysis.ipynb`).

## Reproducing the aGrUM-2 baseline

The results committed under `results/` were produced with the previous stack (pyAgrum 2.3 / otagrum 0.13). They are kept as a baseline for the aGrUM-2 vs aGrUM-3 comparison and are reproduced from a separate virtual environment built on the system pyAgrum 2 / OpenTURNS and a local otagrum ≤ 0.14 — not from the conda environment above.
