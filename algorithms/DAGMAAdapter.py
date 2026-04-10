"""
Adapter for the DAGMA linear algorithm.

Uses dagma.linear.DagmaLinear to learn a DAG from continuous data,
then converts the weighted adjacency matrix to a CPDAG via EssentialGraph.
"""

import numpy as np
import pyagrum as gum
from dagma.linear import DagmaLinear
from algorithms.AlgorithmAdapter import AlgorithmAdapter
from pipeline.Structure import Structure
from pipeline.Dataset import Dataset


class DAGMAAdapter(AlgorithmAdapter):
    """
    Adapter for DAGMA linear (DAGs via M-matrices and a Log-Determinant
    Acyclicity characterization).

    Learns a DAG via continuous optimization with log-det acyclicity constraint,
    then converts to CPDAG via EssentialGraph.
    """

    DEFAULT_PARAM_GRID = {
        "lambda1": [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2],
        "w_threshold_dagma": [0.0, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
    }

    def __init__(self, lambda1: float = 0.03,
                 w_threshold_dagma: float = 0.3,
                 loss_type: str = "l2",
                 T: int = 5,
                 mu_init: float = 1.0,
                 mu_factor: float = 0.1,
                 s: list[float] | float = [1.0, 0.9, 0.8, 0.7, 0.6],
                 warm_iter: int = 30000,
                 max_iter: int = 60000,
                 lr: float = 0.0003,
                 beta_1: float = 0.99,
                 beta_2: float = 0.999,
                 W_est: np.ndarray | None = None):
        """
        Initialize the DAGMA linear adapter.

        Parameters
        ----------
        lambda1 : float, optional
            L1 penalty parameter for sparsity (default: 0.03)
        w_threshold_dagma : float, optional
            Threshold for pruning weak edges (default: 0.3)
        loss_type : str, optional
            Loss type: 'l2' or 'logistic' (default: 'l2')
        T : int, optional
            Number of DAGMA iterations (default: 5)
        mu_init : float, optional
            Initial value of mu (default: 1.0)
        mu_factor : float, optional
            Decay factor for mu (default: 0.1)
        s : list[float] or float, optional
            Controls the domain of M-matrices
            (default: [1.0, 0.9, 0.8, 0.7, 0.6])
        warm_iter : int, optional
            Iterations for t < T (default: 30000)
        max_iter : int, optional
            Iterations for t = T (default: 60000)
        lr : float, optional
            Learning rate (default: 0.0003)
        beta_1 : float, optional
            Adam beta_1 (default: 0.99)
        beta_2 : float, optional
            Adam beta_2 (default: 0.999)
        W_est : np.ndarray, optional
            Pre-computed weight matrix. If provided, skips the expensive
            optimization and only applies w_threshold_dagma.
        """
        self.lambda1 = lambda1
        self.w_threshold_dagma = w_threshold_dagma
        self.loss_type = loss_type
        self.T = T
        self.mu_init = mu_init
        self.mu_factor = mu_factor
        self.s = s
        self.warm_iter = warm_iter
        self.max_iter = max_iter
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self._W_est_precomputed = W_est

    def learn_dag(self, dataset: Dataset) -> gum.DAG:
        """
        Learn DAG using DAGMA linear.

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
            X = dataset.data
            model = DagmaLinear(loss_type=self.loss_type, verbose=False)
            W_est = model.fit(X,
                              lambda1=self.lambda1,
                              T=self.T,
                              mu_init=self.mu_init,
                              mu_factor=self.mu_factor,
                              s=self.s,
                              warm_iter=self.warm_iter,
                              max_iter=self.max_iter,
                              lr=self.lr,
                              beta_1=self.beta_1,
                              beta_2=self.beta_2,
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
        Learn structure using DAGMA linear.

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
        return f"DAGMA_l1={self.lambda1}"
