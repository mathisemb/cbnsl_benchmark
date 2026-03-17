"""
Run all benchmark configurations and save results to results/.

Each configuration runs a full grid search and saves:
- manifest.json (scores, params, grid search results)
- golden.json, structures/, grid_structures/

Results can then be visualized in views/visualize_gridsearch.ipynb
via Benchmark.load("results/<folder>").

Usage:
    python -m views.run_all_benchmarks
"""

import pyagrum as gum
from tqdm.auto import tqdm 
from pipeline.Benchmark import Benchmark

# ---------------------------------------------------------------------------
# DAGs
# ---------------------------------------------------------------------------

def dag_5vars():
    dag = gum.DAG()
    dag.addNodes(5)
    for t, h in [(1, 0), (1, 2), (3, 2), (3, 4), (4, 2)]:
        dag.addArc(t, h)
    return dag


def dag_20vars():
    dag = gum.DAG()
    dag.addNodes(20)
    for t, h in [
        (1, 0), (0, 2), (2, 3), (1, 3), (3, 4), (4, 5), (4, 6), (5, 7),
        (6, 7), (6, 8), (9, 8), (2, 9), (11, 7), (0, 15), (9, 10), (12, 10),
        (12, 11), (16, 15), (14, 15), (16, 14), (13, 12), (14, 13), (13, 17),
        (16, 17), (17, 18), (18, 19),
    ]:
        dag.addArc(t, h)
    return dag


# ---------------------------------------------------------------------------
# Configurations: (factory_method, kwargs)
# ---------------------------------------------------------------------------

CONFIGS = [
    # --- Sachs (real data) ---
    ("sachs", {"variant": "raw"}),
    # ("sachs", {"variant": "log"}),
    ("sachs", {"variant": "preprocessed"}),
    ("sachs", {"variant": "raw", "repetition_nb": 3}),
    ("sachs", {"variant": "preprocessed", "repetition_nb": 3}),

    # --- CBN Gaussian (Uniform + NormalCopula) ---
    ("synthetic_cbn", {"dag": dag_5vars(), "n_samples": 200, "marginal_type": "Uniform", "lcc_types": "NormalCopula"}),
    ("synthetic_cbn", {"dag": dag_5vars(), "n_samples": 2000, "marginal_type": "Uniform", "lcc_types": "NormalCopula"}),
    ("synthetic_cbn", {"dag": dag_20vars(), "n_samples": 1000, "marginal_type": "Uniform", "lcc_types": "NormalCopula"}),
    # ("synthetic_cbn", {"dag": dag_20vars(), "n_samples": 5000, "marginal_type": "Uniform", "lcc_types": "NormalCopula"}),

    # --- CBN Non-Gaussian (Exponential + ClaytonCopula) ---
    ("synthetic_cbn", {"dag": dag_5vars(), "n_samples": 200, "marginal_type": "Exponential", "lcc_types": "ClaytonCopula"}),
    ("synthetic_cbn", {"dag": dag_5vars(), "n_samples": 2000, "marginal_type": "Exponential", "lcc_types": "ClaytonCopula"}),
    ("synthetic_cbn", {"dag": dag_20vars(), "n_samples": 1000, "marginal_type": "Exponential", "lcc_types": "ClaytonCopula"}),
    # ("synthetic_cbn", {"dag": dag_20vars(), "n_samples": 5000, "marginal_type": "Exponential", "lcc_types": "ClaytonCopula"}),

    # --- CBN Non-Gaussian (Uniform + MixtureCopula) ---
    ("synthetic_cbn", {"dag": dag_5vars(), "n_samples": 200, "marginal_type": "Uniform", "lcc_types": "MixtureCopula"}),
    ("synthetic_cbn", {"dag": dag_5vars(), "n_samples": 2000, "marginal_type": "Uniform", "lcc_types": "MixtureCopula"}),
    ("synthetic_cbn", {"dag": dag_20vars(), "n_samples": 1000, "marginal_type": "Uniform", "lcc_types": "MixtureCopula"}),

    # --- SEM Gaussian ---
    ("synthetic_sem", {"dag": dag_5vars(), "n_samples": 200, "noise_type": "gaussian"}),
    ("synthetic_sem", {"dag": dag_5vars(), "n_samples": 2000, "noise_type": "gaussian"}),
    # ("synthetic_sem", {"dag": dag_20vars(), "n_samples": 1000, "noise_type": "gaussian"}),
    # ("synthetic_sem", {"dag": dag_20vars(), "n_samples": 5000, "noise_type": "gaussian"}),

    # --- SEM Non-Gaussian (laplace) ---
    ("synthetic_sem", {"dag": dag_5vars(), "n_samples": 200, "noise_type": "laplace"}),
    ("synthetic_sem", {"dag": dag_5vars(), "n_samples": 2000, "noise_type": "laplace"}),
    # ("synthetic_sem", {"dag": dag_20vars(), "n_samples": 1000, "noise_type": "laplace"}),
    # ("synthetic_sem", {"dag": dag_20vars(), "n_samples": 5000, "noise_type": "laplace"}),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for method, kwargs in tqdm(CONFIGS, desc="Benchmarks"):
        factory = getattr(Benchmark, method)
        bench = factory(**kwargs)
        bench.run()


if __name__ == "__main__":
    main()
