"""
Synthetic data generation for continuous Bayesian networks.

This module provides utilities to generate synthetic datasets from
continuous Bayesian networks using otagrum.
"""

import numpy as np
import pyagrum as gum
import openturns as ot
import otagrum
from typing import Tuple, Optional, List
from pipeline.Dataset import Dataset
from pipeline.Structure import Structure


def generate_from_cbn(
    cbn: otagrum.ContinuousBayesianNetwork,
    n_samples: int = 1000,
    seed: int = 42
) -> Tuple[Dataset, Structure]:
    """
    Generate a synthetic dataset from a given continuous Bayesian network.

    Args:
        cbn: The continuous Bayesian network to sample from
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (Dataset, Structure) where Structure contains the golden CPDAG

    Example:
        >>> # Create your own CBN
        >>> dag = gum.DAG()
        >>> # ... define structure and distributions ...
        >>> cbn = otagrum.ContinuousBayesianNetwork(structure, marginals, copulas)
        >>> dataset, golden = generate_from_cbn(cbn, n_samples=1000)
    """
    ot.RandomGenerator.SetSeed(seed)

    # Generate samples from the CBN
    sample = cbn.getSample(n_samples)
    data = np.array(sample)

    # Extract the DAG structure and convert to CPDAG via BayesNet
    # (EssentialGraph requires a DAGmodel, not a simple DAG)
    dag = cbn.getDAG()

    # Create temporary BayesNet to build EssentialGraph (same as CPCAdapter)
    bn = gum.BayesNet()
    for node_id in dag.nodes():
        bn.add(gum.LabelizedVariable(f"X{node_id}", f"X{node_id}", 2))
    for node_id in dag.nodes():
        for child_id in dag.children(node_id):
            bn.addArc(node_id, child_id)

    essential_graph = gum.EssentialGraph(bn)
    golden_structure = Structure(essential_graph.pdag())

    # Create Dataset with golden structure
    dataset = Dataset(data)

    return dataset, golden_structure


def create_simple_cbn(
    dag: gum.DAG,
    var_names: Optional[List[str]] = None,
    marginal_type: str = "uniform",
    copula_correlation: float = 0.8
) -> otagrum.ContinuousBayesianNetwork:
    """
    Create a simple continuous Bayesian network with default distributions.

    Args:
        dag: The DAG structure
        var_names: Names for variables (default: X0, X1, X2, ...)
        marginal_type: Type of marginal distribution ("uniform" or "normal")
        copula_correlation: Correlation coefficient for Gaussian copulas

    Returns:
        The created continuous Bayesian network

    Example:
        >>> # Create a chain: 0 -> 1 -> 2
        >>> dag = gum.DAG()
        >>> dag.addNode()
        >>> dag.addNode()
        >>> dag.addNode()
        >>> dag.addArc(0, 1)
        >>> dag.addArc(1, 2)
        >>> cbn = create_simple_cbn(dag)
        >>> dataset, golden = generate_from_cbn(cbn, n_samples=1000)
    """
    n_vars = dag.size()

    # Default variable names
    if var_names is None:
        var_names = [f"X{i}" for i in range(n_vars)]

    structure = otagrum.NamedDAG(dag, var_names)

    # Define marginal distributions
    if marginal_type == "uniform":
        marginals = [ot.Uniform(0.0, 1.0) for _ in range(n_vars)]
    elif marginal_type == "normal":
        marginals = [ot.Normal(0.0, 1.0) for _ in range(n_vars)]
    else:
        raise ValueError(f"Unknown marginal type: {marginal_type}")

    # Define local conditional copulas
    local_conditional_copulas = []
    for i in range(n_vars):
        parents = structure.getParents(i)
        n_parents = parents.getSize()
        dim_lcc = n_parents + 1  # parents + current variable

        # Create correlation matrix with correlation between all pairs
        R = ot.CorrelationMatrix(dim_lcc)
        if dim_lcc > 1:
            # Set correlation between all pairs of variables
            for j in range(dim_lcc):
                for k in range(j):
                    R[j, k] = copula_correlation

        # Create Gaussian copula
        copula = ot.NormalCopula(R)
        local_conditional_copulas.append(copula)

    # Create the continuous Bayesian network
    cbn = otagrum.ContinuousBayesianNetwork(structure, marginals, local_conditional_copulas)

    return cbn


def create_default_cbn(
    dag: gum.DAG,
    var_names: Optional[List[str]] = None,
    marginal_type: str = "Uniform",
    lcc_types: str = "NormalCopula",
) -> otagrum.ContinuousBayesianNetwork:
    """
    Create a simple continuous Bayesian network with default distributions and copulas
    based on the given DAG structure.

    Args:
        dag: The DAG structure
        var_names: Names for variables (default: X0, X1, X2, ...)
        marginal_type: Type of marginal distribution, same for all variables
            choices:
                Uniform
                Normal
                Exponential
        lcc_types: Types of local conditional copulas, same for all variables
            choices:
                NormalCopula
                ClaytonCopula
                GumbelCopula
                FrankCopula
                MinCopula

    Returns:
        The created continuous Bayesian network

    Example:
        >>> # Create a chain: 0 -> 1 -> 2
        >>> dag = gum.DAG()
        >>> dag.addNode()
        >>> dag.addNode()
        >>> dag.addNode()
        >>> dag.addArc(0, 1)
        >>> dag.addArc(1, 2)
        >>> cbn = create_default_cbn(dag)
        >>> dataset, golden = generate_from_cbn(cbn, n_samples=1000)
    """
    n_vars = dag.size()

    # Default variable names
    if var_names is None:
        var_names = [f"X{i}" for i in range(n_vars)]

    structure = otagrum.NamedDAG(dag, var_names)

    # Define marginal distributions
    if marginal_type == "Uniform":
        marginals = [ot.Uniform(0.0, 1.0) for _ in range(n_vars)]
    elif marginal_type == "Normal":
        marginals = [ot.Normal(0.0, 1.0) for _ in range(n_vars)]
    elif marginal_type == "Exponential":
        marginals = [ot.Exponential(1.0) for _ in range(n_vars)]
    else:
        raise ValueError(f"Unknown marginal type: {marginal_type}")

    # Define local conditional copulas
    def _make_normal_copula(dim, correlation=0.8):
        R = ot.CorrelationMatrix(dim)
        if dim > 1:
            for j in range(dim):
                for k in range(j):
                    R[j, k] = correlation
        return ot.NormalCopula(R)

    def _make_archimedean_copula(copula_class, theta, dim):
        """Create an Archimedean copula, falling back to NormalCopula for dim > 2."""
        if dim == 1:
            return ot.IndependentCopula(1)
        if dim == 2:
            return copula_class(theta)
        # Archimedean copulas are bivariate only in OpenTURNS
        return _make_normal_copula(dim)

    copula_factories = {
        "NormalCopula": lambda dim: _make_normal_copula(dim) if dim > 1 else ot.IndependentCopula(1),
        "ClaytonCopula": lambda dim: _make_archimedean_copula(ot.ClaytonCopula, 2.0, dim),
        "GumbelCopula": lambda dim: _make_archimedean_copula(ot.GumbelCopula, 2.0, dim),
        "FrankCopula": lambda dim: _make_archimedean_copula(ot.FrankCopula, 2.0, dim),
        "MinCopula": lambda dim: ot.MinCopula(dim),
    }

    if lcc_types not in copula_factories:
        raise ValueError(
            f"Unknown copula type: {lcc_types}. "
            f"Choices: {list(copula_factories.keys())}"
        )

    local_conditional_copulas = []
    for i in range(n_vars):
        parents = structure.getParents(i)
        n_parents = parents.getSize()
        dim_lcc = n_parents + 1  # parents + current variable

        copula = copula_factories[lcc_types](dim_lcc)
        local_conditional_copulas.append(copula)

    # Create the continuous Bayesian network
    cbn = otagrum.ContinuousBayesianNetwork(structure, marginals, local_conditional_copulas)

    return cbn
