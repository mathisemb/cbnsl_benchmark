import otagrum
from algorithms.AlgorithmAdapter import AlgorithmAdapter
from pipeline.Structure import Structure
from pipeline.Dataset import Dataset
import pyagrum as gum

class CMIICAdapter(AlgorithmAdapter):
    """
    Adapter for the Continuous MIIC (CMIIC) algorithm from otagrum.

    CMIIC is a constraint-based algorithm for learning Bayesian Network structures
    from continuous data using conditional independence tests with MIIC approach.

    Supports both CMIIC (version 1) and CMIIC2 (version 2).
    """

    DEFAULT_PARAM_GRID = {
        "alpha": [0.01, 0.05, 0.10],
    }

    def __init__(self, alpha: float = 0.05, version: int = 1):
        """
        Initialize the CMIIC adapter

        Parameters
        ----------
        alpha : float, optional
            Significance level for independence tests (default: 0.05).
            Note: CMIIC's internal default is 0.01 (see CorrectedMutualInformation.hxx),
            but we use 0.05 for consistency with CPC and standard statistical practice.
            Source: https://github.com/openturns/otagrum/blob/master/lib/src/otagrum/CorrectedMutualInformation.hxx
        version : int, optional
            Algorithm version: 1 for CMIIC, 2 for CMIIC2 (default: 1)

        Note
        ----
        CMIIC does not use max_conditioning_set_size (unlike CPC).
        Alpha is configured via setAlpha() after construction, not in the constructor.
        """
        self.alpha = alpha
        self.version = version

        if self.version not in [1, 2]:
            raise ValueError(f"Unsupported CMIIC version: {self.version}. Must be 1 or 2.")

    def learn_dag(self, dataset: Dataset) -> gum.DAG:
        """
        Learn DAG using CMIIC algorithm.

        Parameters
        ----------
        dataset : Dataset
            The dataset to learn from

        Returns
        -------
        gum.DAG
            The learned DAG
        """
        # Create learner based on version (only takes data in constructor)
        if self.version == 1:
            learner = otagrum.ContinuousMIIC(dataset.data)
        elif self.version == 2:
            learner = otagrum.ContinuousMIIC2(dataset.data)
        else:
            raise ValueError(f"Unsupported CMIIC version: {self.version}")

        # Configure alpha via setter (CMIIC uses setter, not constructor parameter)
        learner.setAlpha(self.alpha)
        learner.setVerbosity(False)
        named_dag = learner.learnDAG()
        return named_dag.getDAG()

    def learn_structure(self, dataset: Dataset) -> Structure:
        """
        Learn Bayesian Network structure using CMIIC algorithm.

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
            return "ContinuousMIIC"
        else:
            return f"ContinuousMIIC{self.version}"
