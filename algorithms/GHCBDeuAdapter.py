"""
Adapter for Greedy Hill Climbing with BDeu score from pyAgrum.

Uses BNLearner with useGreedyHillClimbing() + useScoreBDeu()
after discretizing continuous data via DiscreteTypeProcessor.
"""

import pyagrum as gum
from pyagrum.lib.discreteTypeProcessor import DiscreteTypeProcessor
from algorithms.AlgorithmAdapter import AlgorithmAdapter
from pipeline.Structure import Structure
from pipeline.Dataset import Dataset
from preprocessing.hartemink import hartemink_discretize


class GHCBDeuAdapter(AlgorithmAdapter):
    """
    Adapter for Greedy Hill Climbing + BDeu score structure learning (pyAgrum BNLearner).

    Discretizes continuous data internally using DiscreteTypeProcessor,
    then learns the structure with BNLearner.useGreedyHillClimbing() + useScoreBDeu().
    """

    def __init__(self, n_bins: int = 3, discretization_method: str = "quantile", initial_bins: int | None = None):
        """
        Initialize the GHC+BDeu adapter

        Parameters
        ----------
        n_bins : int, optional
            Number of bins for discretization (default: 3)
        discretization_method : str, optional
            Discretization method: 'quantile', 'uniform', or 'kmeans' (default: 'quantile')
        initial_bins : int, optional
            Number of initial bins before merging (Hartemink only, default: n_bins * 3)
        """
        self.n_bins = n_bins
        self.discretization_method = discretization_method
        self.initial_bins = initial_bins

    def _make_learner(self, dataset: Dataset) -> gum.BNLearner:
        """Discretize data and create a BNLearner configured with GHC + BDeu."""
        df = dataset.to_dataframe()

        if self.discretization_method == "hartemink":
            discretized_df = hartemink_discretize(df, n_bins=self.n_bins, initial_bins=self.initial_bins)
            learner = gum.BNLearner(discretized_df)
        else:
            dtp = DiscreteTypeProcessor()
            dtp.setDiscretizationParameters(None, self.discretization_method, self.n_bins)
            template = dtp.discretizedTemplate(df)
            learner = gum.BNLearner(df, template)
        learner.useGreedyHillClimbing()
        learner.useScoreBDeu()
        learner.setVerbosity(False)
        return learner

    def learn_dag(self, dataset: Dataset) -> gum.DAG:
        """
        Learn DAG using Greedy Hill Climbing + BDeu.

        GHC is score-based and directly produces a BN (DAG).

        Parameters
        ----------
        dataset : Dataset
            The dataset to learn from

        Returns
        -------
        gum.DAG
            The learned DAG
        """
        learner = self._make_learner(dataset)
        bn = learner.learnBN()
        return bn.dag()

    def learn_structure(self, dataset: Dataset) -> Structure:
        """
        Learn Bayesian Network structure using Greedy Hill Climbing + BDeu.

        GHC is score-based: learnPDAG() is not supported,
        so we use learnBN() then extract the CPDAG via EssentialGraph.

        Parameters
        ----------
        dataset : Dataset
            The dataset to learn from

        Returns
        -------
        Structure
            The learned structure (CPDAG)
        """
        learner = self._make_learner(dataset)
        bn = learner.learnBN()
        pdag = gum.EssentialGraph(bn).pdag()
        return Structure(pdag)

    def name(self) -> str:
        return f"GHC_BDeu_{self.discretization_method}_{self.n_bins}bins"
