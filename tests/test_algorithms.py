"""
Tests for all algorithm adapters: learn_structure and learn_dag on a synthetic CBN.
"""

import pyagrum as gum
import pytest

from data.generators import create_simple_cbn, generate_from_cbn
from algorithms.CPCAdapter import CPCAdapter
from algorithms.CMIICAdapter import CMIICAdapter
from algorithms.MIICAdapter import MIICAdapter
from algorithms.GHCBDeuAdapter import GHCBDeuAdapter
from algorithms.NOTEARSAdapter import NOTEARSAdapter
from algorithms.NOTEARSDiscreteAdapter import NOTEARSDiscreteAdapter
from algorithms.LiNGAMAdapter import LiNGAMAdapter


# Shared fixture: synthetic CBN with 0 -> 1 -> 2, 0 -> 2
@pytest.fixture(scope="module")
def synthetic_data():
    dag = gum.DAG()
    dag.addNodes(3)
    dag.addArc(0, 1)
    dag.addArc(1, 2)
    dag.addArc(0, 2)
    cbn = create_simple_cbn(dag, copula_correlation=0.8)
    dataset, golden = generate_from_cbn(cbn, n_samples=2000)
    return dataset, golden


# --- Continuous algorithms ---

CONTINUOUS_ALGOS = [
    CPCAdapter(alpha=0.05),
    CMIICAdapter(alpha=0.05),
    NOTEARSAdapter(lambda1=0.1, w_threshold=0.3),
    LiNGAMAdapter(threshold=0.1),
]

@pytest.mark.parametrize("algo", CONTINUOUS_ALGOS, ids=lambda a: a.name())
def test_learn_structure(algo, synthetic_data):
    dataset, golden = synthetic_data
    structure = algo.learn_structure(dataset)
    assert structure.cpdag.size() == 3


@pytest.mark.parametrize("algo", CONTINUOUS_ALGOS, ids=lambda a: a.name())
def test_learn_dag(algo, synthetic_data):
    dataset, golden = synthetic_data
    dag = algo.learn_dag(dataset)
    assert isinstance(dag, gum.DAG)
    assert dag.size() == 3


# --- Discrete algorithms (need discretization) ---

DISCRETE_ALGOS = [
    MIICAdapter(n_bins=3, discretization_method="quantile"),
    GHCBDeuAdapter(n_bins=3, discretization_method="quantile"),
    NOTEARSDiscreteAdapter(n_bins=3, discretization_method="quantile"),
]

@pytest.mark.parametrize("algo", DISCRETE_ALGOS, ids=lambda a: a.name())
def test_learn_structure_discrete(algo, synthetic_data):
    dataset, golden = synthetic_data
    structure = algo.learn_structure(dataset)
    assert structure.cpdag.size() == 3


@pytest.mark.parametrize("algo", DISCRETE_ALGOS, ids=lambda a: a.name())
def test_learn_dag_discrete(algo, synthetic_data):
    dataset, golden = synthetic_data
    dag = algo.learn_dag(dataset)
    assert isinstance(dag, gum.DAG)
    assert dag.size() == 3
