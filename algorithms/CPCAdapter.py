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

    """

    DEFAULT_PARAM_GRID = {
        "alpha": [0.01, 0.05, 0.10],
        "max_conditioning_set_size": list(range(2, 9)),
    }

    TESTS = {
        "hellinger": otagrum.ContinuousPC.TestTypes_Hellinger,
        "kullback_leibler": otagrum.ContinuousPC.TestTypes_KullbackLeibler,
    }

    def __init__(self, alpha: float = 0.05, max_conditioning_set_size: int = None,
                 test: str = "hellinger"):
        """
        Initialize the CPC adapter

        Parameters
        ----------
        alpha : float, optional
            Significance level for independence tests (default: 0.05)
        max_conditioning_set_size : int, optional
            Maximum size of conditioning sets. If None, uses data dimension - 1
        test : str, optional
            Conditional independence test (default: "hellinger").
            "hellinger" : the Hellinger test of Bouezmarni, Rombouts and Taamouti,
            the historical CPC.
            "kullback_leibler" : the conditional information with sample split and
            the analytic threshold mu_n + z_{1-alpha} sigma_tot (otagrum branch
            pc_entropie). Only the test changes, PC is the same.
        """
        if test not in self.TESTS:
            raise ValueError(f"test must be one of {sorted(self.TESTS)}, got {test!r}")
        self.alpha = alpha
        self.max_conditioning_set_size = max_conditioning_set_size
        self.test = test

    def _configure(self, learner):
        """Settings shared by learn_dag and learn_structure."""
        learner.setTestType(self.TESTS[self.test])
        learner.setVerbosity(False)

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

        learner = otagrum.ContinuousPC(dataset.data, max_cond_set, self.alpha)
        self._configure(learner)
        named_dag = learner.learnDAG()
        return named_dag.getDAG()

    def learn_structure(self, dataset: Dataset) -> Structure:
        """
        Learn Bayesian Network structure using CPC algorithm.

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
        max_cond_set = self.max_conditioning_set_size
        if max_cond_set is None:
            max_cond_set = dataset.data.shape[1] - 2

        learner = otagrum.ContinuousPC(dataset.data, max_cond_set, self.alpha)
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
        return "ContinuousPC"
