"""
Adapter for the NOTEARS algorithm.

Uses notears.linear.notears_linear to learn a DAG from continuous data,
then converts the weighted adjacency matrix to a CPDAG via EssentialGraph.
"""

import numpy as np
import pyagrum as gum
from notears.linear import notears_linear
from algorithms.AlgorithmAdapter import AlgorithmAdapter
from pipeline.Structure import Structure
from pipeline.Dataset import Dataset


class NOTEARSAdapter(AlgorithmAdapter):
    """
    Adapter for NOTEARS (Non-combinatorial Optimization via Trace Exponential
    and Augmented lagRangian for Structure learning).

    Learns a DAG via continuous optimization with acyclicity constraint,
    then converts to CPDAG via EssentialGraph.
    """

    DEFAULT_PARAM_GRID = {
        "lambda1": [0.0, 0.05, 0.1, 0.3, 0.5],
        "w_threshold_notears": [0.0, 0.1, 0.3, 0.5, 0.7],
    }

    def __init__(self, lambda1: float = 0.1, loss_type: str = "l2",
                 w_threshold_notears: float = 0.3,
                 W_est: np.ndarray | None = None):
        """
        Initialize the NOTEARS adapter.

        Parameters
        ----------
        lambda1 : float, optional
            L1 penalty parameter for sparsity (default: 0.1)
        loss_type : str, optional
            Loss type: 'l2' for continuous, 'logistic' for binary (default: 'l2')
        w_threshold_notears : float, optional
            Threshold for pruning weak edges (default: 0.3)
        W_est : np.ndarray, optional
            Pre-computed weight matrix from notears_linear. If provided,
            skips the expensive L-BFGS optimization and only applies
            w_threshold_notears. Used by GridSearch to avoid redundant
            optimizations when only w_threshold_notears varies.
        """
        self.lambda1 = lambda1
        self.loss_type = loss_type
        self.w_threshold_notears = w_threshold_notears
        self._W_est_precomputed = W_est

    def learn_dag(self, dataset: Dataset) -> gum.DAG:
        """
        Learn DAG using NOTEARS.

        Runs notears_linear and converts the weighted adjacency matrix
        to a gum.DAG. Convention: W[i, j] != 0 means i -> j.

        Parameters
        ----------
        dataset : Dataset
            The dataset to learn from

        Returns
        -------
        gum.DAG
            The learned DAG
        """
        if self._W_est_precomputed is not None:
            # Reuse pre-computed W matrix (optimization already done)
            W_est = self._W_est_precomputed.copy()
            W_est[np.abs(W_est) < self.w_threshold_notears] = 0
        else:
            # Run full L-BFGS optimization with w_threshold=0 so we can
            # cache the raw weight matrix for other threshold values
            X = dataset.data
            W_est = notears_linear(X,
                                   lambda1=self.lambda1,
                                   loss_type=self.loss_type,
                                   w_threshold=0)
            self._W_est_raw = W_est.copy()
            W_est[np.abs(W_est) < self.w_threshold_notears] = 0

        dag = gum.DAG()
        d = W_est.shape[0]
        for i in range(d):
            dag.addNodeWithId(i)

        for i in range(d):
            for j in range(d):
                if W_est[i, j] != 0:
                    dag.addArc(i, j)

        return dag

    def learn_structure(self, dataset: Dataset) -> Structure:
        """
        Learn structure using NOTEARS.

        Calls learn_dag() then converts the DAG to CPDAG via EssentialGraph.

        Parameters
        ----------
        dataset : Dataset
            The dataset to learn from

        Returns
        -------
        Structure
            The learned structure (CPDAG)
        """
        dag = self.learn_dag(dataset)

        # Convert DAG to BayesNet (needed by EssentialGraph)
        bn = gum.BayesNet()
        for node_id in dag.nodes():
            name = dataset.feature_names[node_id]
            bn.add(gum.LabelizedVariable(name, name, 2))
        for tail, head in dag.arcs():
            bn.addArc(tail, head)

        pdag = gum.EssentialGraph(bn).pdag()
        return Structure(pdag)

    def name(self) -> str:
        return f"NOTEARS_l1={self.lambda1}"
