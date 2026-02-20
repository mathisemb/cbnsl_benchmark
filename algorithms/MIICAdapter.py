"""
Adapter for discrete MIIC algorithm from pyAgrum.

Uses BNLearner with useMIIC() after discretizing continuous data
via DiscreteTypeProcessor.
"""

import pyagrum as gum
from pyagrum.lib.discreteTypeProcessor import DiscreteTypeProcessor
from algorithms.AlgorithmAdapter import AlgorithmAdapter
from pipeline.Structure import Structure
from pipeline.Dataset import Dataset
from preprocessing.hartemink import hartemink_discretize


class MIICAdapter(AlgorithmAdapter):
    """
    Adapter for discrete MIIC structure learning (pyAgrum BNLearner).

    Discretizes continuous data internally using DiscreteTypeProcessor,
    then learns the structure with BNLearner.useMIIC().
    """

    def __init__(self, n_bins: int = 3, discretization_method: str = "quantile"):
        """
        Initialize the MIIC adapter

        Parameters
        ----------
        n_bins : int, optional
            Number of bins for discretization (default: 3)
        discretization_method : str, optional
            Discretization method: 'quantile', 'uniform', or 'kmeans' (default: 'quantile')
        """
        self.n_bins = n_bins
        self.discretization_method = discretization_method

    def learn_structure(self, dataset: Dataset) -> Structure:
        """
        Learn Bayesian Network structure using discrete MIIC

        Parameters
        ----------
        dataset : Dataset
            The dataset to learn from

        Returns
        -------
        Structure
            The learned structure (CPDAG)
        """
        df = dataset.to_dataframe()

        # Discretize
        if self.discretization_method == "hartemink":
            discretized_df = hartemink_discretize(df, n_bins=self.n_bins)
            learner = gum.BNLearner(discretized_df)
        else:
            dtp = DiscreteTypeProcessor()
            dtp.setDiscretizationParameters(None, self.discretization_method, self.n_bins)
            template = dtp.discretizedTemplate(df)
            learner = gum.BNLearner(df, template)
        learner.useMIIC()
        learner.setVerbosity(False)

        # MIIC is constraint-based: learnPDAG() returns the CPDAG directly
        pdag = learner.learnPDAG()
        return Structure(pdag)

    def name(self) -> str:
        return f"MIIC_{self.discretization_method}_{self.n_bins}bins"
