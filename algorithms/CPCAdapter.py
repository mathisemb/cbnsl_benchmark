import otagrum
from algorithms.AlgorithmAdapter import AlgorithmAdapter
from pipeline.Structure import Structure
from pipeline.Dataset import Dataset
import pyagrum as gum

class CPCAdapter(AlgorithmAdapter):
    """
    Adapter for the Continuous PC (CPC) algorithm from otagrum.

    CPC is a constraint-based algorithm for learning Bayesian Network structures
    from continuous data using conditional independence tests.

    Supports both CPC (version 1) and CPC2 (version 2).
    """

    DEFAULT_PARAM_GRID = {
        "alpha": [0.01, 0.05, 0.10],
        "max_conditioning_set_size": list(range(2, 9)),
    }

    def __init__(self, alpha: float = 0.05, max_conditioning_set_size: int = None, version: int = 1):
        """
        Initialize the CPC adapter

        Parameters
        ----------
        alpha : float, optional
            Significance level for independence tests (default: 0.05)
        max_conditioning_set_size : int, optional
            Maximum size of conditioning sets. If None, uses data dimension - 1
        version : int, optional
            Algorithm version: 1 for CPC, 2 for CPC2 (default: 1)
        """
        self.alpha = alpha
        self.max_conditioning_set_size = max_conditioning_set_size
        self.version = version

        if self.version not in [1, 2]:
            raise ValueError(f"Unsupported CPC version: {self.version}. Must be 1 or 2.")

    def learn_dag(self, dataset: Dataset) -> gum.DAG:
        """
        Learn DAG using CPC algorithm.

        Parameters
        ----------
        dataset : Dataset
            The dataset to learn from

        Returns
        -------
        gum.DAG
            The learned DAG
        """
        max_cond_set = self.max_conditioning_set_size
        if max_cond_set is None:
            max_cond_set = dataset.data.shape[1] - 2

        # Create learner based on version
        if self.version == 1:
            learner = otagrum.ContinuousPC(dataset.data, max_cond_set, self.alpha)
        elif self.version == 2:
            learner = otagrum.ContinuousPC2(dataset.data, max_cond_set, self.alpha)
        else:
            raise ValueError(f"Unsupported CPC version: {self.version}")

        learner.setVerbosity(False)
        named_dag = learner.learnDAG()
        return named_dag.getDAG()

    def learn_structure(self, dataset: Dataset) -> Structure:
        """
        Learn Bayesian Network structure using CPC algorithm.

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

        # Convert DAG to CPDAG via EssentialGraph
        pdag = gum.EssentialGraph(bn).pdag()
        return Structure(pdag)
    
    def name(self) -> str:
        """
        Returns the name of the algorithm

        Returns
        -------
        str
            The algorithm name
        """
        if self.version == 1:
            return "ContinuousPC"
        else:
            return f"ContinuousPC{self.version}"
