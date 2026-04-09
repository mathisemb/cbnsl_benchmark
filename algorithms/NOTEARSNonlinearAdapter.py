"""
Adapter for the NOTEARS nonlinear (MLP) algorithm.

Uses notears.nonlinear.notears_nonlinear to learn a DAG from continuous data,
then converts the weighted adjacency matrix to a CPDAG via EssentialGraph.
"""

import numpy as np
import pyagrum as gum
import torch
from notears.nonlinear import NotearsMLP, notears_nonlinear
from algorithms.AlgorithmAdapter import AlgorithmAdapter
from pipeline.Structure import Structure
from pipeline.Dataset import Dataset


class NOTEARSNonlinearAdapter(AlgorithmAdapter):
    """
    Adapter for NOTEARS nonlinear (MLP variant).

    Uses a Multi-Layer Perceptron to model nonlinear structural equations.
    Learns a DAG via continuous optimization with acyclicity constraint,
    then converts to CPDAG via EssentialGraph.
    """

    DEFAULT_PARAM_GRID = {
        "lambda1": [0.0, 0.01, 0.05, 0.1, 0.3],
        "lambda2": [0.0, 0.005, 0.01, 0.05],
        "w_threshold_notears": [0.0, 0.1, 0.3, 0.5, 0.7],
    }

    def __init__(self, lambda1: float = 0.0, lambda2: float = 0.0,
                 w_threshold_notears: float = 0.3,
                 hidden_units: int = 10,
                 max_iter: int = 100,
                 h_tol: float = 1e-8,
                 rho_max: float = 1e+16,
                 W_est: np.ndarray | None = None):
        """
        Initialize the NOTEARS nonlinear adapter.

        Parameters
        ----------
        lambda1 : float, optional
            L1 penalty parameter for sparsity (default: 0.0)
        lambda2 : float, optional
            L2 penalty parameter for regularization (default: 0.0)
        w_threshold_notears : float, optional
            Threshold for pruning weak edges (default: 0.3)
        hidden_units : int, optional
            Number of hidden units in the MLP (default: 10)
        max_iter : int, optional
            Max number of dual ascent steps (default: 100)
        h_tol : float, optional
            Exit if |h(W)| <= h_tol (default: 1e-8)
        rho_max : float, optional
            Exit if rho >= rho_max (default: 1e+16)
        W_est : np.ndarray, optional
            Pre-computed weight matrix. If provided, skips the expensive
            optimization and only applies w_threshold_notears.
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.w_threshold_notears = w_threshold_notears
        self.hidden_units = hidden_units
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self._W_est_precomputed = W_est

    def learn_dag(self, dataset: Dataset) -> gum.DAG:
        """
        Learn DAG using NOTEARS nonlinear (MLP).

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
            W_est[np.abs(W_est) < self.w_threshold_notears] = 0
        else:
            torch.set_default_dtype(torch.double)
            torch.manual_seed(42)
            X = dataset.data
            d = X.shape[1]
            model = NotearsMLP(dims=[d, self.hidden_units, 1], bias=True)
            W_est = notears_nonlinear(model, X,
                                      lambda1=self.lambda1,
                                      lambda2=self.lambda2,
                                      max_iter=self.max_iter,
                                      h_tol=self.h_tol,
                                      rho_max=self.rho_max,
                                      w_threshold=0)  # no pruning here: raw W is cached for GridSearch
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
        Learn structure using NOTEARS nonlinear.

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
        return f"NOTEARS_NL_l1={self.lambda1}"
