"""
Synthetic data generation for continuous Bayesian networks and SEMs.

This module provides utilities to generate synthetic datasets from:
- Continuous Bayesian networks (CBN) using otagrum
- Linear Structural Equation Models (SEM) with configurable noise
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

    def _make_mixture_copula(dim, corr_high=0.8, corr_low=-0.3):
        """Extract a non-Gaussian copula from a Mixture of two Normals via getCopula().

        The two components have different correlation structures, producing
        a bimodal dependence pattern that is genuinely non-Gaussian.
        """
        if dim == 1:
            return ot.IndependentCopula(1)
        # For a dim×dim equicorrelation matrix, PD requires ρ > -1/(dim-1)
        if dim > 1:
            corr_low = max(corr_low, -1.0 / (dim - 1) + 1e-6)
        R_high = ot.CorrelationMatrix(dim)
        R_low = ot.CorrelationMatrix(dim)
        for j in range(dim):
            for k in range(j):
                R_high[j, k] = corr_high
                R_low[j, k] = corr_low
        d1 = ot.Normal([0.0] * dim, [1.0] * dim, R_high)
        d2 = ot.Normal([0.0] * dim, [1.0] * dim, R_low)
        mixture = ot.Mixture([d1, d2], [0.5, 0.5])
        return mixture.getCopula()

    copula_factories = {
        "NormalCopula": lambda dim: _make_normal_copula(dim) if dim > 1 else ot.IndependentCopula(1),
        "ClaytonCopula": lambda dim: _make_archimedean_copula(ot.ClaytonCopula, 2.0, dim),
        "GumbelCopula": lambda dim: _make_archimedean_copula(ot.GumbelCopula, 2.0, dim),
        "FrankCopula": lambda dim: _make_archimedean_copula(ot.FrankCopula, 2.0, dim),
        "MixtureCopula": lambda dim: _make_mixture_copula(dim),
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


def generate_from_cbn_dag(
    dag: gum.DAG,
    n_samples: int = 1000,
    seed: int = 42,
    marginal_type: str = "Uniform",
    lcc_types: str = "NormalCopula",
) -> Tuple[Dataset, Structure]:
    """
    Generate synthetic data from a DAG via a CBN with default distributions.

    Builds a CBN from the DAG using ``create_default_cbn``, then samples from it.
    Same signature as ``generate_from_sem`` so both can be used interchangeably.
    """
    var_names = [f"X{i}" for i in range(dag.size())]
    cbn = create_default_cbn(dag, var_names=var_names,
                             marginal_type=marginal_type, lcc_types=lcc_types)
    return generate_from_cbn(cbn, n_samples=n_samples, seed=seed)


def generate_from_sem(
    dag: gum.DAG,
    n_samples: int = 1000,
    seed: int = 42,
    noise_type: str = "gaussian",
    weight_range: tuple = (0.3, 0.8),
) -> Tuple[Dataset, Structure]:
    """
    Generate a synthetic dataset from a linear SEM with configurable noise.

    Each variable is generated as:
        X_i = sum(w_ji * X_j for j in parents(i)) + e_i
    where e_i is drawn from the chosen noise distribution.

    Args:
        dag: The ground-truth DAG structure.
        n_samples: Number of samples to generate.
        seed: Random seed for reproducibility.
        noise_type: Noise distribution — ``"gaussian"``, ``"laplace"``,
            ``"uniform"``, or ``"exp"`` (centred exponential).
        weight_range: ``(low, high)`` for uniform sampling of
            absolute edge weights (sign is random ±1).

    Returns:
        Tuple of (Dataset, Structure) where Structure contains the golden CPDAG.

    Example:
        >>> dag = gum.DAG()
        >>> dag.addNodes(5)
        >>> dag.addArc(0, 1); dag.addArc(1, 2)
        >>> dataset, golden = generate_from_sem(dag, n_samples=1000, noise_type="laplace")
    """
    rng = np.random.default_rng(seed)
    n_vars = dag.size()
    var_names = [f"X{i}" for i in range(n_vars)]
    topo = dag.topologicalOrder()

    # Sample edge weights
    weights = {}
    for node in topo:
        for parent in dag.parents(node):
            w = rng.uniform(*weight_range)
            sign = rng.choice([-1, 1])
            weights[(parent, node)] = sign * w

    # Generate noise
    if noise_type == "gaussian":
        noise = rng.normal(loc=0.0, scale=0.1, size=(n_samples, n_vars))
    elif noise_type == "laplace":
        noise = rng.laplace(loc=0.0, scale=0.1, size=(n_samples, n_vars))
    elif noise_type == "uniform":
        noise = rng.uniform(-1.0, 1.0, size=(n_samples, n_vars))
    elif noise_type == "exp":
        noise = rng.exponential(scale=1.0, size=(n_samples, n_vars))
        noise -= noise.mean(axis=0)
    else:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. "
            "Choose from 'gaussian', 'laplace', 'uniform', 'exp'."
        )

    # Generate data following topological order
    data = np.zeros((n_samples, n_vars))
    for node in topo:
        data[:, node] = noise[:, node]
        for parent in dag.parents(node):
            data[:, node] += weights[(parent, node)] * data[:, parent]

    # Build ground-truth CPDAG
    bn = gum.BayesNet()
    for node_id in dag.nodes():
        bn.add(gum.LabelizedVariable(f"X{node_id}", f"X{node_id}", 2))
    for node_id in dag.nodes():
        for child_id in dag.children(node_id):
            bn.addArc(node_id, child_id)
    golden = Structure(gum.EssentialGraph(bn).pdag())

    dataset = Dataset(data, feature_names=var_names)
    return dataset, golden
