"""
Adapter for the DirectLiNGAM algorithm.

Uses lingam.DirectLiNGAM to learn a DAG from continuous data
(assumes non-Gaussian noise), then converts the weighted adjacency
matrix to a CPDAG via EssentialGraph.
"""

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

    DEFAULT_PARAM_GRID = {
        "threshold": [0.1, 0.5, 0.75, 1.0, 1.25, 1.5],
    }

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

    def learn_dag(self, dataset: Dataset) -> gum.DAG:
        """
        Learn DAG using DirectLiNGAM.

        Adjacency matrix convention: B[i, j] != 0 means j -> i.

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

        model = lingam.DirectLiNGAM(random_state=self.random_state,
                                    measure=self.measure)
        model.fit(X)
        B = model.adjacency_matrix_

        dag = gum.DAG()
        d = B.shape[0]
        for i in range(d):
            dag.addNodeWithId(i)

        for i in range(d):
            for j in range(d):
                if abs(B[i, j]) > self.threshold:  # j -> i
                    dag.addArc(j, i)

        return dag

    def learn_structure(self, dataset: Dataset) -> Structure:
        """
        Learn structure using DirectLiNGAM.

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
        bn.addVariables([str(node) for node in dag.nodes()], 2)
        for tail, head in dag.arcs():
            bn.addArc(tail, head)

        pdag = gum.EssentialGraph(bn).pdag()
        return Structure(pdag)

    def name(self) -> str:
        return "DirectLiNGAM"
