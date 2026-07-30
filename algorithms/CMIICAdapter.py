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

    """

    DEFAULT_PARAM_GRID = {
        "alpha": [0.01, 0.05, 0.10],
    }

    def __init__(self, alpha: float = 0.05):
        """
        Initialize the CMIIC adapter

        Parameters
        ----------
        alpha : float, optional
            Significance level for independence tests (default: 0.05).
            Note: CMIIC's internal default is 0.01 (see CorrectedMutualInformation.hxx),
            but we use 0.05 for consistency with CPC and standard statistical practice.

        Note
        ----
        CMIIC does not use max_conditioning_set_size (unlike CPC).
        Alpha is configured via setAlpha() after construction, not in the constructor.
        """
        self.alpha = alpha

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
        learner = otagrum.ContinuousMIIC(dataset.data)

        # Configure alpha via setter (CMIIC uses setter, not constructor parameter)
        learner.setAlpha(self.alpha)
        learner.setVerbosity(False)
        named_dag = learner.learnDAG()
        return named_dag.getDAG()

    def learn_structure(self, dataset: Dataset) -> Structure:
        """
        Learn Bayesian Network structure using CMIIC algorithm.

        The CPDAG is the Meek closure (propagateToCPDAG) of the raw pattern
        returned by learnPDAG (skeleton + v-structures). The closure only
        orients compelled arcs: unlike the former learnDAG + EssentialGraph
        chain, it never invents arbitrary orientations. The raw pattern is
        kept in the Structure so post-processing can be replayed offline.

        Parameters
        ----------
        dataset : Dataset
            The dataset to learn from

        Returns
        -------
        Structure
            The learned structure (CPDAG + raw pattern)
        """
        learner = otagrum.ContinuousMIIC(dataset.data)

        # Configure alpha via setter (CMIIC uses setter, not constructor parameter)
        learner.setAlpha(self.alpha)
        learner.setVerbosity(False)
        mixed = learner.learnPDAG()

        cpdag = gum.MeekRules().propagateToCPDAG(mixed)
        return Structure(cpdag, pdag=mixed)

    def name(self) -> str:
        """
        Returns the name of the algorithm

        Returns
        -------
        str
            The algorithm name
        """
        return "ContinuousMIIC"
