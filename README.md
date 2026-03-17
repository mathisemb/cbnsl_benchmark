# Continuous Bayesian Network Structure Learning Benchmark

This repository benchmarks structure learning algorithms for Bayesian networks in the continuous case. It provides a unified pipeline to run them on the same datasets and compare their results.

## Algorithms

| Algorithm | Package |
|---|---|
| ContinuousPC | otagrum |
| ContinuousMIIC | otagrum |
| MIIC | pyAgrum |
| GHC+BDeu | pyAgrum |
| NOTEARS | notears |
| DirectLiNGAM | lingam |

## Installation

### Quick start
```bash
./install.sh
```

### Manual installation

#### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

#### 2. Install Python dependencies

```bash
pip install -e .
pip install git+https://github.com/xunzheng/notears.git
```

This installs pyAgrum, OpenTURNS, lingam, and the benchmark itself. NOTEARS is installed separately as it is not on PyPI.

#### 3. Install otagrum from source

This benchmark requires the latest version of otagrum (with aGrUM Meek rules in ContinuousPC/MIIC). This version is not yet available via `conda install otagrum` — it must be built from source for now.

> Once a new conda-forge release of otagrum includes this change, `conda install otagrum` will be sufficient and this step can be skipped.

Requires cmake and a C++ compiler:
```bash
git clone https://github.com/openturns/otagrum.git
cd otagrum
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV
cmake --build .
cmake --build . --target install
cd ../..
```

## Usage

### Run benchmarks

Edit the configurations in `views/run_all_benchmarks.py`, then:

```bash
source venv/bin/activate
python views/run_all_benchmarks.py
```

Results are saved in `views/results/`.

### Visualize results

Open `views/visualize_gridsearch.ipynb` to explore and compare the results.

### Scaling study

The `scaling/` directory measures how algorithms scale with graph size (number of nodes and samples). See `scaling/run_scaling.py`.
