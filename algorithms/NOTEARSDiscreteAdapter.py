"""
Adapter for NOTEARS on discretized data.

Discretizes continuous data, then runs notears_linear on the ordinal bin indices.
"""

import numpy as np
import pandas as pd
import pyagrum as gum
from notears.linear import notears_linear
from algorithms.AlgorithmAdapter import AlgorithmAdapter
from pipeline.Structure import Structure
from pipeline.Dataset import Dataset
from algorithms.hartemink import hartemink_discretize


class NOTEARSDiscreteAdapter(AlgorithmAdapter):
    """
    NOTEARS on discretized data.

    Discretizes continuous data into ordinal bins, then applies
    notears_linear with l2 loss on the integer-valued matrix.
    """

    DEFAULT_PARAM_GRID = [
        {
            "lambda1": [0.0, 0.05, 0.1, 0.3, 0.5],
            "w_threshold_notears": [0.0, 0.1, 0.3, 0.5, 0.7],
            "n_bins": [2, 4, 6, 8, 10],
            "discretization_method": ["quantile"],
        },
        {
            "lambda1": [0.0, 0.05, 0.1, 0.3, 0.5],
            "w_threshold_notears": [0.0, 0.1, 0.3, 0.5, 0.7],
            "n_bins": [2, 4, 6, 8, 10],
            "discretization_method": ["hartemink"],
            "initial_bins": [20],
        },
    ]

    def __init__(self, lambda1: float = 0.1, w_threshold_notears: float = 0.3,
                 n_bins: int = 3, discretization_method: str = "quantile",
                 initial_bins: int | None = None,
                 discretized_df: pd.DataFrame | None = None,
                 W_est: np.ndarray | None = None):
        """
        Parameters
        ----------
        lambda1 : float, optional
            L1 penalty parameter for sparsity (default: 0.1).
        w_threshold_notears : float, optional
            Threshold for pruning weak edges (default: 0.3).
        n_bins : int, optional
            Number of bins for discretization (default: 3).
        discretization_method : str, optional
            Method: 'quantile' or 'hartemink' (default: 'quantile').
        initial_bins : int | None, optional
            Initial bins before merging (Hartemink only, default: n_bins * 3).
        discretized_df : pd.DataFrame, optional
            Pre-discretized data. If provided, skips internal discretization.
        W_est : np.ndarray, optional
            Pre-computed weight matrix from notears_linear. If provided,
            skips both discretization and L-BFGS optimization, only applies
            w_threshold_notears. Used by GridSearch to avoid redundant
            optimizations when only w_threshold_notears varies.
        """
        self.lambda1 = lambda1
        self.w_threshold_notears = w_threshold_notears
        self.n_bins = n_bins
        self.discretization_method = discretization_method
        self.initial_bins = initial_bins
        self._discretized_df = discretized_df
        self._W_est_precomputed = W_est

    def learn_dag(self, dataset: Dataset) -> gum.DAG:
        """
        Discretize data then learn DAG using NOTEARS.

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
            # Discretize data
            if self._discretized_df is not None:
                X = self._discretized_df.values.astype(float)
            elif self.discretization_method == "hartemink":
                df = dataset.to_dataframe()
                discretized_df = hartemink_discretize(
                    df, n_bins=self.n_bins, initial_bins=self.initial_bins
                )
                X = discretized_df.values.astype(float)
            elif self.discretization_method == "quantile":
                df = dataset.to_dataframe()
                X = df.apply(
                    lambda col: pd.qcut(col, self.n_bins, labels=False,
                                        duplicates="drop")
                ).values.astype(float)
            else:
                raise ValueError(
                    f"Unknown discretization method: "
                    f"{self.discretization_method}. "
                    "Supported: 'quantile', 'hartemink'."
                )

            # Run full L-BFGS optimization with w_threshold=0 so we can
            # cache the raw weight matrix for other threshold values
            W_est = notears_linear(
                X, lambda1=self.lambda1, loss_type="l2", w_threshold=0,
            )
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
        Discretize data then learn structure using NOTEARS.

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
        return f"NOTEARS_discrete_{self.discretization_method}_{self.n_bins}bins"
