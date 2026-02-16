# Continuous Bayesian Network Structure Learning benchmark
This repository provides wrappers for different structure learning algorithms for bayesian networks in the continuous case. The pipeline allows them to be executed on the same dataset and to compare their results.

# Installation
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
