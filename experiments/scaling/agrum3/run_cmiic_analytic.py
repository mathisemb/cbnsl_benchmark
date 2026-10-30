"""
CMIIC with the analytic threshold (otagrum mode Analytic), alone, on the same graphs and
datasets as the July 2026 CPC / CMIIC scaling run: same generators, same grid, same seed.

Results go to agrum3/<timestamp>_<generator>/ like any ScalingRunner run, and are
compared offline to the July rows, whose CMIIC is the Naive mode (constant alpha).

Usage, from agrum3/ with the cbnsl environment active:
    python run_cmiic_analytic.py sem_gaussian cbn_exp_clayton
    python run_cmiic_analytic.py all
In the background:
    nohup python run_cmiic_analytic.py all > run_cmiic_analytic.log 2>&1 &
"""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from algorithms.CMIICAdapter import CMIICAdapter
from data.generators import generate_from_sem, generate_from_cbn_dag
from experiments.scaling.runner import ScalingRunner

# Same generators as the five scaling_study_*.ipynb notebooks of July 2026.
GENERATORS = {
    "sem_gaussian": partial(generate_from_sem, noise_type="gaussian"),
    "sem_laplace": partial(generate_from_sem, noise_type="laplace"),
    "cbn_unif_gauss": partial(generate_from_cbn_dag, marginal_type="Uniform", lcc_types="NormalCopula"),
    "cbn_exp_clayton": partial(generate_from_cbn_dag, marginal_type="Exponential", lcc_types="ClaytonCopula"),
    "cbn_unif_mixture": partial(generate_from_cbn_dag, marginal_type="Uniform", lcc_types="MixtureCopula"),
}

ALGO_CONFIGS = [
    ("CMIIC_analytic", CMIICAdapter, {"alpha": 0.05, "kmode": "Analytic"}),
]


def run(generator_name):
    runner = ScalingRunner(
        algo_configs=ALGO_CONFIGS,
        generator=GENERATORS[generator_name],
        generator_name=generator_name,
        n_vars_list=[10, 20],
        n_samples_list=[200, 500, 1000, 3000, 5000],
        density=1.5,
        n_graphs=30,
        seed=42,
    )
    print(f"=== {generator_name} -> {runner.results_dir}", flush=True)
    runner.run()
    print(f"=== done {generator_name}", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:]
    if not names or names == ["all"]:
        names = list(GENERATORS)
    unknown = [n for n in names if n not in GENERATORS]
    if unknown:
        sys.exit(f"unknown generator(s) {unknown}, choose among {list(GENERATORS)}")
    for name in names:
        run(name)
