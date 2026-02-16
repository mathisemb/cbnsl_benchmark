# Continuous Bayesian Network Structure Learning benchmark
This repository provides wrappers for different structure learning algorithms for bayesian networks in the continuous case. The pipeline allows them to be executed on the same dataset and to compare their results.

# Quick Installation

## Automated Installation (Recommended)
Run the installation script that handles all dependencies:
```bash
./install.sh
```

The script will guide you through:
- Creating a virtual environment
- Installing base dependencies
- Choosing your otagrum installation method (conda or build from source)
- Installing NO TEARS
- Installing development dependencies (optional)

## Manual Installation
If you prefer manual installation:

```bash
# 1. Create and activate virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 2. Install base dependencies
pip install -e .

# 3. Install otagrum and notears (see detailed instructions below)
```

# Detailed Installation Instructions
Before running the algorithms you should install all the requirements.

Install **pyAgrum** with:
```bash
pip install pyAgrum
```

Install **OpenTURNS** with:
```bash
pip install openturns
```

If you don't want to run CPC2 and CMIIC2 (only CPC and CMIIC) you can install **OTAgrum** with:
```bash
conda install otagrum
```

If you want to run CPC2 and CMIIC2 you can clone https://github.com/mathisemb/otagrum (branch cpc2_cmiic2):
```bash
git clone https://github.com/mathisemb/otagrum.git
```
and install otagrum with the following commands (make sure you have a compiler and cmake installed):
- ```bash
    mkdir build
    cd build
    ```
- ```bash
    cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local
    ```
- ```bash
    cmake --build .
    cmake --build . --target install
    ```
    (and if you want to run the otagrum unit tests)
    ```bash
    cmake --build . --target cppcheck
    ctest
    ```

Install **NO TEARS** by cloning https://github.com/xunzheng/notears:
```bash
git clone https://github.com/xunzheng/notears.git
```

Install **LiNGAM** with:
```bash
pip install lingam
```

# How to run the benchmark?

...

# Project Structure

```
cbnsl_benchmark/
├── pipeline/              # Core pipeline components
│   ├── Pipeline.py        # Main orchestration
│   ├── Dataset.py         # Dataset wrapper
│   ├── Result.py          # Result storage
│   └── Structure.py       # Structure representation
├── algorithms/            # Algorithm adapters
│   ├── AlgorithmAdapter.py    # Base adapter interface
│   └── adapters/          # Specific algorithm implementations
│       └── CPCAdapter.py
├── metrics/              # Evaluation metrics
│   ├── MetricAdapter.py      # Base metric interface
│   └── SHDMetric.py         # Structural Hamming Distance
├── discretization/       # Discretization strategies (TODO)
├── visualization/        # Visualization tools (TODO)
├── tests/               # Tests
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── fixtures/        # Test data
├── examples/            # Usage examples
│   └── basic_usage.py
├── data/                # Datasets
│   └── synthetic/       # Synthetic datasets
├── results/             # Benchmark outputs (gitignored)
├── install.sh           # Automated installation script
├── pyproject.toml       # Package configuration
├── requirements.txt     # Base dependencies
└── requirements-git.txt # Git-based dependencies
```
