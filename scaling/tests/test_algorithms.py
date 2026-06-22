"""
Tests for all algorithm adapters: learn_structure and learn_dag on a synthetic CBN.

To run: python -m pytest tests/test_algorithms.py
"""

import numpy as np
import pyagrum as gum
import pytest

from pipeline.Dataset import Dataset
from pipeline.Structure import Structure
from data.generators import create_simple_cbn, generate_from_cbn
from algorithms.CPCAdapter import CPCAdapter
from algorithms.CMIICAdapter import CMIICAdapter
from algorithms.MIICAdapter import MIICAdapter
from algorithms.GHCBDeuAdapter import GHCBDeuAdapter
from algorithms.NOTEARSAdapter import NOTEARSAdapter
from algorithms.NOTEARSDiscreteAdapter import NOTEARSDiscreteAdapter
from algorithms.LiNGAMAdapter import LiNGAMAdapter


def create_linear_sem_data(dag, n_samples=2000, seed=42,
                           noise_type="uniform", weight_range=(0.5, 1.5)):
    """Generate data from a linear SEM.

    X_i = sum_j(w_ji * X_j) + e_i

    noise_type controls the noise distribution:
    - "uniform": e_i ~ Uniform(-1, 1) — non-Gaussian, suits LiNGAM
    - "gaussian": e_i ~ Normal(0, 1) — smoother, suits discretization

    Returns (Dataset, Structure) where Structure is the golden CPDAG.
    """
    rng = np.random.default_rng(seed)
    n_vars = dag.size()

    # Edge weights: strong enough to be detectable, mixed signs
    W = np.zeros((n_vars, n_vars))
    for tail, head in dag.arcs():
        W[tail, head] = rng.choice([-1, 1]) * rng.uniform(*weight_range)

    # Topological order generation
    X = np.zeros((n_samples, n_vars))
    for node in dag.topologicalOrder():
        if noise_type == "uniform":
            noise = rng.uniform(-1, 1, size=n_samples)
        else:
            noise = rng.standard_normal(size=n_samples)
        X[:, node] = X @ W[:, node] + noise

    dataset = Dataset(X)

    # Golden CPDAG via EssentialGraph
    bn = gum.BayesNet()
    bn.addVariables([str(n) for n in dag.nodes()], 2)
    for tail, head in dag.arcs():
        bn.addArc(tail, head)
    golden = Structure(gum.EssentialGraph(bn).pdag())

    return dataset, golden


def _make_dag():
    """V-structure 0 -> 2 <- 1, and 0 -> 3.
    Expected CPDAG: {0->2, 1->2} directed, {0--3} undirected."""
    dag = gum.DAG()
    dag.addNodes(4)
    dag.addArc(0, 2)
    dag.addArc(1, 2)
    dag.addArc(0, 3)
    return dag


# Continuous: uniform noise (non-Gaussian → LiNGAM identifiable, linear → NOTEARS)
@pytest.fixture(scope="module")
def continuous_data():
    return create_linear_sem_data(_make_dag(), noise_type="uniform")


# Discrete (MIIC, GHCBDeu): copula-based CBN (same generation as benchmarks)
@pytest.fixture(scope="module")
def discrete_data():
    dag = _make_dag()
    cbn = create_simple_cbn(dag, copula_correlation=0.8)
    return generate_from_cbn(cbn, n_samples=2000)


# --- Continuous algorithms ---

CONTINUOUS_ALGOS = [
    CPCAdapter(alpha=0.05),
    CMIICAdapter(alpha=0.05),
    NOTEARSAdapter(lambda1=0.05, w_threshold_notears=0.2),
    LiNGAMAdapter(threshold_lingam=0.1),
]

@pytest.mark.parametrize("algo", CONTINUOUS_ALGOS, ids=lambda a: a.name())
def test_learn_structure(algo, continuous_data):
    dataset, golden = continuous_data
    structure = algo.learn_structure(dataset)
    assert structure.cpdag == golden.cpdag


@pytest.mark.parametrize("algo", CONTINUOUS_ALGOS, ids=lambda a: a.name())
def test_learn_dag(algo, continuous_data):
    dataset, golden = continuous_data
    dag = algo.learn_dag(dataset)
    assert isinstance(dag, gum.DAG)
    bn = gum.BayesNet()
    bn.addVariables([str(n) for n in dag.nodes()], 2)
    for tail, head in dag.arcs():
        bn.addArc(tail, head)
    assert gum.EssentialGraph(bn).pdag() == golden.cpdag


# --- Discrete algorithms (MIIC, GHCBDeu — copula CBN) ---

DISCRETE_ALGOS = [
    MIICAdapter(n_bins=3, discretization_method="quantile"),
    GHCBDeuAdapter(n_bins=3, discretization_method="quantile"),
]

@pytest.mark.parametrize("algo", DISCRETE_ALGOS, ids=lambda a: a.name())
def test_learn_structure_discrete(algo, discrete_data):
    dataset, golden = discrete_data
    structure = algo.learn_structure(dataset)
    assert structure.cpdag == golden.cpdag


@pytest.mark.parametrize("algo", DISCRETE_ALGOS, ids=lambda a: a.name())
def test_learn_dag_discrete(algo, discrete_data):
    dataset, golden = discrete_data
    dag = algo.learn_dag(dataset)
    assert isinstance(dag, gum.DAG)
    bn = gum.BayesNet()
    bn.addVariables([str(n) for n in dag.nodes()], 2)
    for tail, head in dag.arcs():
        bn.addArc(tail, head)
    assert gum.EssentialGraph(bn).pdag() == golden.cpdag


# --- NOTEARS discrete (copula CBN) ---
# WARNING: NOTEARS discrete cannot reliably orient v-structures after
# discretization. We only check skeleton equality (undirected adjacencies),
# not full CPDAG equality.

def _skeleton(pdag):
    """Return the skeleton (unoriented adjacencies) of a PDAG."""
    pairs = set()
    for a, b in pdag.arcs():
        pairs.add((min(a, b), max(a, b)))
    for a, b in pdag.edges():
        pairs.add((min(a, b), max(a, b)))
    return pairs


def test_learn_structure_notears_discrete(discrete_data):
    dataset, golden = discrete_data
    algo = NOTEARSDiscreteAdapter(
        lambda1=0.05, w_threshold_notears=0.2, n_bins=3,
        discretization_method="quantile"
    )
    structure = algo.learn_structure(dataset)
    assert _skeleton(structure.cpdag) == _skeleton(golden.cpdag)


def test_learn_dag_notears_discrete(discrete_data):
    dataset, golden = discrete_data
    algo = NOTEARSDiscreteAdapter(
        lambda1=0.05, w_threshold_notears=0.2, n_bins=3,
        discretization_method="quantile"
    )
    dag = algo.learn_dag(dataset)
    assert isinstance(dag, gum.DAG)
    bn = gum.BayesNet()
    bn.addVariables([str(n) for n in dag.nodes()], 2)
    for tail, head in dag.arcs():
        bn.addArc(tail, head)
    assert _skeleton(gum.EssentialGraph(bn).pdag()) == _skeleton(golden.cpdag)
