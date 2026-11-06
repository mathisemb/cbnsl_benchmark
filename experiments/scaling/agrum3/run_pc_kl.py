"""
PC with the Kullback-Leibler test (conditional information, sample split, analytic
threshold mu_n + z_{1-alpha} sigma_tot), alone, on the same graphs and datasets as the
July 2026 CPC / CMIIC scaling run: same generators, same grid, same seed.

Same PC as CPC (otagrum ContinuousPC), only the independence test changes, so the
comparison with the July CPC rows (Hellinger test of Bouezmarni, Rombouts and Taamouti)
isolates the test. Requires otagrum branch pc_entropie.

Results go to agrum3/<timestamp>_<generator>_pc_kl/ and are compared offline.

Usage, from agrum3/ with the cbnsl environment active:
    python run_pc_kl.py sem_gaussian cbn_exp_clayton
    python run_pc_kl.py all
In the background:
    nohup python run_pc_kl.py all > run_pc_kl.log 2>&1 &
"""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from algorithms.CPCAdapter import CPCAdapter
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

# Same alpha and same (default) max conditioning set size as the July CPC.
SUFFIX = "pc_kl"
ALGO_CONFIGS = [
    ("PC_kl", CPCAdapter, {"alpha": 0.05, "test": "kullback_leibler"}),
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
    runner.results_dir = runner.results_dir.with_name(f"{runner.results_dir.name}_{SUFFIX}")
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
