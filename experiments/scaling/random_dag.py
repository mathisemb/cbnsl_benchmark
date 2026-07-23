"""
Random DAG generation for scaling studies.

Uses pyAgrum's BNGenerator which implements the MCMC method of Ide & Cozman (2002)
to sample DAGs (approximately) uniformly from the space of DAGs with a fixed number
of nodes and arcs.

Reference:
    Ide, J.S., Cozman, F.G. (2002). Random Generation of Bayesian Networks. SBIA 2002.
    http://sites.poli.usp.br/pmr/ltd/People/jside/IdeCozman_sbia02.pdf
"""

import pyagrum as gum


def random_dag(
    n_vars: int,
    n_arcs: int,
    seed: int = 0,
) -> gum.DAG:
    """
    Sample a random DAG with a fixed number of nodes and arcs.

    Uses ``pyagrum.BNGenerator`` which implements the MCMC chain of Ide & Cozman
    (2002). The chain converges to a uniform distribution over all DAGs with the
    given number of nodes and arcs.

    Parameters
    ----------
    n_vars : int
        Number of variables (nodes).
    n_arcs : int
        Number of arcs. Must satisfy ``n_arcs <= n_vars * (n_vars - 1) / 2``.
    seed : int
        Random seed passed to ``pyagrum.initRandom``.

    Returns
    -------
    gum.DAG
        A random DAG with node ids 0, 1, ..., n_vars - 1.
    """
    gum.initRandom(seed)
    generator = gum.BNGenerator()
    bn = generator.generate(n_nodes=n_vars, n_arcs=n_arcs)

    # Need to remove the node names since agrum3 (otherwise, conflict with NamedDAG) ===============
    result = gum.DAG()
    result.addNodes(n_vars)
    for arc in bn.dag().arcs():
        result.addArc(*arc)
    # ==============================================================================================

    return result
