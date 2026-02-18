# Continuous Bayesian Network Structure Learning benchmark
This repository provides wrappers for different structure learning algorithms for bayesian networks in the continuous case. The pipeline allows them to be executed on the same dataset and to compare their results.

# Installation

## Quick start
```bash
./install.sh
```

## Manual Installation

### 1. Create a virtual environment
```bash
python3 -m venv venv --system-site-packages
source venv/bin/activate
```
The flag `--system-site-packages` gives the venv access to system-installed packages (pyAgrum, openturns) and user-installed packages (otagrum in `~/.local/`).

### 2. Install dependencies

#### Why `--system-site-packages`?

There are two possible approaches for managing dependencies:

**Approach A — Everything inside the venv:**
Install pyAgrum, openturns, numpy, pandas, etc. directly in the venv via `pip install`. Advantage: the venv is fully self-contained. Drawback: some C++ packages (pyAgrum, openturns) are better managed by the system package manager (pacman, apt…), and otagrum cannot be pip-installed — it must be compiled from source and installs into `~/.local/`.

**Approach B — Hybrid system + venv (current choice):**
Heavy C++ packages (pyAgrum, openturns) are installed at the system level via the package manager, otagrum is compiled into `~/.local/`, and only pure Python packages (lingam, notears, pandas, numpy) are installed inside the venv. The `--system-site-packages` flag makes system packages and `~/.local/` visible from within the venv.

**We use Approach B** because:
- pyAgrum and openturns are C++ libraries with Python bindings: the system package manager handles their native dependencies and updates more reliably
- otagrum must be compiled from source (cmake/C++) and installs into `~/.local/` — it cannot be `pip install`ed
- On some distributions (Arch/Manjaro), PEP 668 prevents `pip install` at the system level, which makes the venv even more useful for pure Python packages
- This approach avoids duplicating large C++ libraries in each venv

#### Dependency table

The table below lists every dependency, how to install it, and where it ends up.

**Packages installed outside the venv** (visible thanks to `--system-site-packages`):

| Package | Command | Installed in | Used by |
|---|---|---|---|
| **pyAgrum** | `pip install pyAgrum` or via the system package manager | `/usr/lib/.../site-packages/` | All algorithms, metrics |
| **OpenTURNS** | `pip install openturns` or via the system package manager | `/usr/lib/.../site-packages/` | CPC, CMIIC, data generators |
| **otagrum** (standard) | `conda install otagrum` | `/usr/lib/.../site-packages/` | CPC, CMIIC (v1 only) |
| **otagrum** (with CPC2/CMIIC2) | Build from source (see below) | `~/.local/lib/.../site-packages/` | CPC, CPC2, CMIIC, CMIIC2 |

**Packages installed inside the venv** (via `pip install` after activation):

| Package | Command | Installed in | Used by |
|---|---|---|---|
| **LiNGAM** | via `pip install -e .` (in pyproject.toml) | `venv/lib/.../site-packages/` | LiNGAMAdapter |
| **NOTEARS** | `pip install git+https://github.com/xunzheng/notears.git` | `venv/lib/.../site-packages/` | NOTEARSAdapter |

Once the venv is activated:
```bash
pip install -e .
pip install git+https://github.com/xunzheng/notears.git
```

### 3. Install otagrum

**Option A** — Standard otagrum (CPC and CMIIC only, no CPC2/CMIIC2):
```bash
conda install otagrum
```

**Option B** — Fork with CPC2/CMIIC2 support (requires cmake and a C++ compiler):
```bash
git clone https://github.com/mathisemb/otagrum.git
cd otagrum
git checkout cpc2_cmiic2
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local
cmake --build .
cmake --build . --target install
```
This installs otagrum in `~/.local/lib/python3.x/site-packages/`, which is visible in the venv thanks to `--system-site-packages`.

# How to run the benchmark?

Activate the virtual environment, then run the comparison test on the Sachs dataset:
```bash
source venv/bin/activate
python tests/test_comparison_sachs.py
```

This runs all 8 algorithms (CPC, CPC2, CMIIC, CMIIC2, MIIC, GHC+BDeu, NOTEARS, DirectLiNGAM) on the Sachs observational dataset, computes SHD against the ground truth, and displays pairwise comparisons.

# Project Structure

```
cbnsl_benchmark/
├── pipeline/              # Core pipeline components
│   ├── Pipeline.py        # Main orchestration
│   ├── Dataset.py         # Dataset wrapper (with feature_names, to_dataframe)
│   ├── Result.py          # Result storage
│   └── Structure.py       # Structure representation (CPDAG)
├── algorithms/            # Algorithm adapters
│   ├── AlgorithmAdapter.py    # Base adapter interface
│   ├── CPCAdapter.py          # CPC/CPC2 continu (otagrum)
│   ├── CMIICAdapter.py        # CMIIC/CMIIC2 continu (otagrum)
│   ├── MIICAdapter.py         # MIIC discret + discrétisation (pyAgrum)
│   ├── GHCBDeuAdapter.py      # GHC+BDeu discret + discrétisation (pyAgrum)
│   ├── NOTEARSAdapter.py      # NOTEARS (optimisation continue)
│   └── LiNGAMAdapter.py       # DirectLiNGAM (non-gaussianité)
├── metrics/              # Evaluation metrics
│   ├── MetricAdapter.py      # Base metric interface
│   └── SHDMetric.py         # Structural Hamming Distance
├── analysis/             # Benchmark analysis and visualization
│   └── BenchmarkAnalyzer.py  # Metrics vs golden, pairwise comparisons
├── data/                # Datasets
│   ├── generators.py        # Synthetic data generation from CBN
│   ├── sachs/               # Sachs Protein Signaling dataset + ground truth
│   └── synthetic/           # Synthetic datasets
├── tests/               # Tests
│   ├── test_comparison_sachs.py  # Full benchmark comparison (8 algorithms)
│   ├── test_cpc_sachs.py         # CPC on Sachs dataset
│   ├── test_cpc_shd.py           # CPC with SHD metric
│   └── test_cpc_with_golden.py   # CPC with golden structure
├── results/             # Benchmark outputs (gitignored)
├── install.sh           # Automated installation script
└── pyproject.toml       # Package configuration
```

# Developing Algorithm Adapters

To add a new structure learning algorithm to the benchmark, create an adapter that:
1. Implements the `AlgorithmAdapter` interface
2. Returns a `Structure` object containing a CPDAG (Completed Partially Directed Acyclic Graph)

**Important:** The CPDAG must be a `gum.MixedGraph` (or subclass like `gum.PDAG`).

See **[algorithms/ADAPTER_GUIDE.md](algorithms/ADAPTER_GUIDE.md)** for detailed instructions on:
- How to extract a CPDAG from different DAG types (otagrum.NamedDAG, gum.DAG, etc.)
- Common pitfalls to avoid
- Complete examples

Example adapters:
- [CPCAdapter.py](algorithms/CPCAdapter.py) - CPC/CPC2 continuous algorithm (otagrum)
- [MIICAdapter.py](algorithms/MIICAdapter.py) - Discrete MIIC with internal discretization (pyAgrum)
- [GHCBDeuAdapter.py](algorithms/GHCBDeuAdapter.py) - Greedy Hill Climbing + BDeu score with internal discretization (pyAgrum)
- [NOTEARSAdapter.py](algorithms/NOTEARSAdapter.py) - NOTEARS continuous optimization (notears)
- [LiNGAMAdapter.py](algorithms/LiNGAMAdapter.py) - DirectLiNGAM non-Gaussian model (lingam)
