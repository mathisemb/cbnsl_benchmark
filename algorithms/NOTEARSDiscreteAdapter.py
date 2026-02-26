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
from preprocessing.hartemink import hartemink_discretize


class NOTEARSDiscreteAdapter(AlgorithmAdapter):
    """
    NOTEARS on discretized data.

    Discretizes continuous data into ordinal bins, then applies
    notears_linear with l2 loss on the integer-valued matrix.
    """

    def __init__(self, lambda1: float = 0.1, w_threshold: float = 0.3,
                 n_bins: int = 3, discretization_method: str = "quantile",
                 initial_bins: int | None = None):
        """
        Parameters
        ----------
        lambda1 : float, optional
            L1 penalty parameter for sparsity (default: 0.1).
        w_threshold : float, optional
            Threshold for pruning weak edges (default: 0.3).
        n_bins : int, optional
            Number of bins for discretization (default: 3).
        discretization_method : str, optional
            Method: 'quantile' or 'hartemink' (default: 'quantile').
        initial_bins : int | None, optional
            Initial bins before merging (Hartemink only, default: n_bins * 3).
        """
        self.lambda1 = lambda1
        self.w_threshold = w_threshold
        self.n_bins = n_bins
        self.discretization_method = discretization_method
        self.initial_bins = initial_bins

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
        df = dataset.to_dataframe()

        if self.discretization_method == "hartemink":
            discretized_df = hartemink_discretize(
                df, n_bins=self.n_bins, initial_bins=self.initial_bins
            )
            X = discretized_df.values.astype(float)
        elif self.discretization_method == "quantile":
            X = df.apply(
                lambda col: pd.qcut(col, self.n_bins, labels=False, duplicates="drop")
            ).values.astype(float)
        else:
            raise ValueError(
                f"Unknown discretization method: {self.discretization_method}. "
                "Supported: 'quantile', 'hartemink'."
            )

        W_est = notears_linear(
            X, lambda1=self.lambda1, loss_type="l2",
            w_threshold=self.w_threshold,
        )

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
