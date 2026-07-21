"""
Run grid search for newly added algorithms only:
- NOTEARS NL (nonlinear)
- DAGMA (linear)
- DAGMA NL (nonlinear)

Uses the same datasets/DAGs as run_all_benchmarks.py (10 vars, 2000 samples).
Results are saved in views/results/.

Usage:
    python -m views.run_new_algorithms
"""

import pyagrum as gum
from tqdm.auto import tqdm
from pipeline.Benchmark import Benchmark
from pipeline.GridSearch import GridSearch
from algorithms.NOTEARSNonlinearAdapter import NOTEARSNonlinearAdapter
from algorithms.NOTEARSSobolevAdapter import NOTEARSSobolevAdapter
from algorithms.DAGMAAdapter import DAGMAAdapter
from algorithms.DAGMANonlinearAdapter import DAGMANonlinearAdapter
from metrics import ALL_METRICS, OBJECTIVES


def dag_10vars():
    dag = gum.DAG()
    dag.addNodes(10)
    for t, h in [
        (1, 0), (0, 2), (2, 3), (3, 4), (3, 5), (4, 5), (4, 6), (5, 7),
        (6, 7), (6, 8), (7, 8), (9, 8), (2, 9), (9, 7)
    ]:
        dag.addArc(t, h)
    return dag

"""
NEW_ALGORITHMS = {
    "NOTEARS NL": (NOTEARSNonlinearAdapter, {}, None),
    "DAGMA":      (DAGMAAdapter, {}, None),
    "DAGMA NL":   (DAGMANonlinearAdapter, {}, None),
}
"""
NEW_ALGORITHMS = {
    "DAGMA":      (DAGMAAdapter, {}, None),
}

CONFIGS = [
    ("sachs", {"variant": "preprocessed"})
]
"""
# --- CBN Gaussian (Uniform + NormalCopula) ---
("synthetic_cbn", {"dag": dag_10vars(), "n_samples": 2000, "marginal_type": "Uniform", "lcc_types": "NormalCopula"}),

# --- CBN Non-Gaussian (Exponential + ClaytonCopula) ---
("synthetic_cbn", {"dag": dag_10vars(), "n_samples": 2000, "marginal_type": "Exponential", "lcc_types": "ClaytonCopula"}),

# --- CBN Non-Gaussian (Uniform + MixtureCopula) ---
("synthetic_cbn", {"dag": dag_10vars(), "n_samples": 2000, "marginal_type": "Uniform", "lcc_types": "MixtureCopula"}),

# --- SEM Gaussian ---
("synthetic_sem", {"dag": dag_10vars(), "n_samples": 2000, "noise_type": "gaussian"}),

# --- SEM Non-Gaussian (laplace) ---
("synthetic_sem", {"dag": dag_10vars(), "n_samples": 2000, "noise_type": "laplace"}),
"""

def main():
    for method, kwargs in tqdm(CONFIGS, desc="Benchmarks"):
        factory = getattr(Benchmark, method)
        bench = factory(**kwargs)

        # Replace the auto-registered GridSearch with one containing only new algos
        bench._gs = GridSearch(
            dataset=bench.dataset,
            golden_structure=bench.golden_structure,
            metrics=list(ALL_METRICS),
            objectives=dict(OBJECTIVES),
        )
        for name, (algo_cls, fixed_params, random_seeds) in NEW_ALGORITHMS.items():
            bench._gs.add(name, algo_cls, fixed_params=fixed_params,
                          random_seeds=random_seeds)

        bench.run()


if __name__ == "__main__":
    main()
