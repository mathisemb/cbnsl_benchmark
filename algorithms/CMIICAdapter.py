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

    KMODES = {
        "NoCorr": otagrum.CorrectedMutualInformation.KModeTypes_NoCorr,
        "Naive": otagrum.CorrectedMutualInformation.KModeTypes_Naive,
        "Analytic": otagrum.CorrectedMutualInformation.KModeTypes_Analytic,
    }

    def __init__(self, alpha: float = 0.05, kmode: str = "Naive", orientation_offset: float = None):
        """
        Initialize the CMIIC adapter

        Parameters
        ----------
        alpha : float, optional
            Significance level for independence tests (default: 0.05).
            Note: CMIIC's internal default is 0.01 (see CorrectedMutualInformation.hxx),
            but we use 0.05 for consistency with CPC and standard statistical practice.
        kmode : str, optional
            Threshold mode of the conditional independence test (default: "Naive").
            "Naive" : constant penalty alpha, the historical CMIIC.
            "Analytic" : sample split and the analytic threshold mu_n + z_{1-alpha} sigma_tot,
            alpha being then the level of the test (otagrum branch seuil_analytique).
            "NoCorr" : no penalty.
        orientation_offset : float, optional
            Offset added to the three-point information before MIIC reads its sign
            to declare a v-structure. None (default) keeps otagrum's behaviour,
            alpha in "Naive" mode, 0 otherwise. Set it to decouple the orientation
            from the two-point test, e.g. 0.05 with kmode="Analytic".

        Note
        ----
        CMIIC does not use max_conditioning_set_size (unlike CPC).
        Alpha and kmode are configured via setters after construction.
        """
        if kmode not in self.KMODES:
            raise ValueError(f"kmode must be one of {sorted(self.KMODES)}, got {kmode!r}")
        self.alpha = alpha
        self.kmode = kmode
        self.orientation_offset = orientation_offset

    def _configure(self, learner):
        """Settings shared by learn_dag and learn_structure (CMIIC uses setters, not constructor parameters)."""
        learner.setKMode(self.KMODES[self.kmode])
        learner.setAlpha(self.alpha)
        if self.orientation_offset is not None:
            learner.setOrientationOffset(self.orientation_offset)
        learner.setVerbosity(False)

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
        self._configure(learner)
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
        self._configure(learner)
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
