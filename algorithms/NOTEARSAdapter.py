"""
Adapter for the NOTEARS algorithm.

Uses notears.linear.notears_linear to learn a DAG from continuous data,
then converts the weighted adjacency matrix to a CPDAG via EssentialGraph.
"""

import numpy as np
import pandas as pd
import pyagrum as gum
from notears.linear import notears_linear
from algorithms.AlgorithmAdapter import AlgorithmAdapter
from pipeline.Structure import Structure
from pipeline.Dataset import Dataset
from preprocessing.hartemink import hartemink_discretize


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
        X = dataset.data

        W_est = notears_linear(X, lambda1=self.lambda1, loss_type=self.loss_type,
                               w_threshold=self.w_threshold)

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
