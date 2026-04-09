"""
Adapter for the DAGMA nonlinear (MLP) algorithm.

Uses dagma.nonlinear.DagmaNonlinear to learn a DAG from continuous data,
then converts the weighted adjacency matrix to a CPDAG via EssentialGraph.
"""

import numpy as np
import pyagrum as gum
import torch
from dagma.nonlinear import DagmaNonlinear, DagmaMLP
from algorithms.AlgorithmAdapter import AlgorithmAdapter
from pipeline.Structure import Structure
from pipeline.Dataset import Dataset


class DAGMANonlinearAdapter(AlgorithmAdapter):
    """
    Adapter for DAGMA nonlinear (MLP variant).

    Uses a Multi-Layer Perceptron to model nonlinear structural equations.
    Learns a DAG via continuous optimization with log-det acyclicity constraint,
    then converts to CPDAG via EssentialGraph.
    """

    DEFAULT_PARAM_GRID = {
        "lambda1": [0.01, 0.02, 0.05, 0.1],
        "lambda2": [0.001, 0.005, 0.01, 0.05],
        "w_threshold_dagma": [0.0, 0.1, 0.3, 0.5, 0.7],
    }

    def __init__(self, lambda1: float = 0.02, lambda2: float = 0.005,
                 w_threshold_dagma: float = 0.3,
                 hidden_units: int = 10,
                 T: int = 4,
                 mu_init: float = 0.1,
                 mu_factor: float = 0.1,
                 s: float = 1.0,
                 warm_iter: int = 50000,
                 max_iter: int = 80000,
                 lr: float = 0.0002,
                 W_est: np.ndarray | None = None):
        """
        Initialize the DAGMA nonlinear adapter.

        Parameters
        ----------
        lambda1 : float, optional
            L1 penalty parameter for sparsity (default: 0.02)
        lambda2 : float, optional
            L2 penalty parameter for regularization (default: 0.005)
        w_threshold_dagma : float, optional
            Threshold for pruning weak edges (default: 0.3)
        hidden_units : int, optional
            Number of hidden units in the MLP (default: 10)
        T : int, optional
            Number of DAGMA iterations (default: 4)
        mu_init : float, optional
            Initial value of mu (default: 0.1)
        mu_factor : float, optional
            Decay factor for mu (default: 0.1)
        s : float, optional
            Controls the domain of M-matrices (default: 1.0)
        warm_iter : int, optional
            Iterations for t < T (default: 50000)
        max_iter : int, optional
            Iterations for t = T (default: 80000)
        lr : float, optional
            Learning rate (default: 0.0002)
        W_est : np.ndarray, optional
            Pre-computed weight matrix. If provided, skips the expensive
            optimization and only applies w_threshold_dagma.
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.w_threshold_dagma = w_threshold_dagma
        self.hidden_units = hidden_units
        self.T = T
        self.mu_init = mu_init
        self.mu_factor = mu_factor
        self.s = s
        self.warm_iter = warm_iter
        self.max_iter = max_iter
        self.lr = lr
        self._W_est_precomputed = W_est

    def learn_dag(self, dataset: Dataset) -> gum.DAG:
        """
        Learn DAG using DAGMA nonlinear (MLP).

        Converts the weighted adjacency matrix to a gum.DAG.
        Convention: W[i, j] != 0 means i -> j.

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
            W_est = self._W_est_precomputed.copy()
            W_est[np.abs(W_est) < self.w_threshold_dagma] = 0
        else:
            torch.manual_seed(42)
            X = dataset.data
            d = X.shape[1]
            eq_model = DagmaMLP(dims=[d, self.hidden_units, 1], bias=True,
                                dtype=torch.double)
            model = DagmaNonlinear(eq_model, dtype=torch.double)
            W_est = model.fit(X,
                              lambda1=self.lambda1,
                              lambda2=self.lambda2,
                              T=self.T,
                              mu_init=self.mu_init,
                              mu_factor=self.mu_factor,
                              s=self.s,
                              warm_iter=self.warm_iter,
                              max_iter=self.max_iter,
                              lr=self.lr,
                              w_threshold=0)  # no pruning here: raw W is cached for GridSearch
            self._W_est_raw = W_est.copy()
            W_est[np.abs(W_est) < self.w_threshold_dagma] = 0

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
        Learn structure using DAGMA nonlinear.

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
        return f"DAGMA_NL_l1={self.lambda1}"
