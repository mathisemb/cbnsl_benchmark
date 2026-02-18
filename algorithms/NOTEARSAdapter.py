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

    def __init__(self, lambda1: float = 0.1, loss_type: str = "l2",
                 w_threshold: float = 0.3):
        """
        Initialize the NOTEARS adapter.

        Parameters
        ----------
        lambda1 : float, optional
            L1 penalty parameter for sparsity (default: 0.1)
        loss_type : str, optional
            Loss type: 'l2' for continuous, 'logistic' for binary (default: 'l2')
        w_threshold : float, optional
            Threshold for pruning weak edges (default: 0.3)
        """
        self.lambda1 = lambda1
        self.loss_type = loss_type
        self.w_threshold = w_threshold

    def learn_structure(self, dataset: Dataset) -> Structure:
        """
        Learn structure using NOTEARS.

        Parameters
        ----------
        dataset : Dataset
            The dataset to learn from

        Returns
        -------
        Structure
            The learned structure (CPDAG)
        """
        X = dataset.data

        # Run NOTEARS: returns weighted adjacency matrix (d, d)
        # Convention: W[i, j] != 0 means i -> j
        W_est = notears_linear(X, lambda1=self.lambda1, loss_type=self.loss_type,
                               w_threshold=self.w_threshold)

        # Build BayesNet from binary adjacency matrix
        bn = gum.BayesNet()
        d = W_est.shape[0]
        for i in range(d):
            name = dataset.feature_names[i]
            bn.add(gum.LabelizedVariable(name, name, 2))

        for i in range(d):
            for j in range(d):
                if W_est[i, j] != 0:  # i -> j
                    bn.addArc(bn.idFromName(dataset.feature_names[i]),
                              bn.idFromName(dataset.feature_names[j]))

        # Convert DAG to CPDAG via EssentialGraph
        pdag = gum.EssentialGraph(bn).pdag()
        return Structure(pdag)

    def name(self) -> str:
        return f"NOTEARS_l1={self.lambda1}"
