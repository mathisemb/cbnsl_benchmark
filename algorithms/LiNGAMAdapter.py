"""
Adapter for the DirectLiNGAM algorithm.

Uses lingam.DirectLiNGAM to learn a DAG from continuous data
(assumes non-Gaussian noise), then converts the weighted adjacency
matrix to a CPDAG via EssentialGraph.
"""

import numpy as np
import pyagrum as gum
import lingam
from algorithms.AlgorithmAdapter import AlgorithmAdapter
from pipeline.Structure import Structure
from pipeline.Dataset import Dataset


class LiNGAMAdapter(AlgorithmAdapter):
    """
    Adapter for DirectLiNGAM (Linear Non-Gaussian Acyclic Model).

    Identifies the full causal DAG using non-Gaussianity of error terms,
    then converts to CPDAG via EssentialGraph.
    """

    def __init__(self, random_state: int = 42, measure: str = "pwling",
                 threshold: float = 0.01):
        """
        Initialize the DirectLiNGAM adapter.

        Parameters
        ----------
        random_state : int, optional
            Random seed for reproducibility (default: 42)
        measure : str, optional
            Independence measure: 'pwling' or 'kernel' (default: 'pwling')
        threshold : float, optional
            Threshold for pruning weak edges in the adjacency matrix (default: 0.01)
        """
        self.random_state = random_state
        self.measure = measure
        self.threshold = threshold

    def learn_structure(self, dataset: Dataset) -> Structure:
        """
        Learn structure using DirectLiNGAM.

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

        # Run DirectLiNGAM
        model = lingam.DirectLiNGAM(random_state=self.random_state,
                                    measure=self.measure)
        model.fit(X)

        # Adjacency matrix: B[i, j] != 0 means j -> i
        B = model.adjacency_matrix_

        # Build BayesNet from thresholded adjacency matrix
        bn = gum.BayesNet()
        d = B.shape[0]
        for i in range(d):
            name = dataset.feature_names[i]
            bn.add(gum.LabelizedVariable(name, name, 2))

        for i in range(d):
            for j in range(d):
                if abs(B[i, j]) > self.threshold:  # j -> i
                    bn.addArc(bn.idFromName(dataset.feature_names[j]),
                              bn.idFromName(dataset.feature_names[i]))

        # Convert DAG to CPDAG via EssentialGraph
        pdag = gum.EssentialGraph(bn).pdag()
        return Structure(pdag)

    def name(self) -> str:
        return "DirectLiNGAM"
