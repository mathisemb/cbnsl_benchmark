"""
Adapter for discrete MIIC algorithm from pyAgrum.

Uses BNLearner with useMIIC() after discretizing continuous data
via DiscreteTypeProcessor.
"""

import pandas as pd
import pyagrum as gum
from pyagrum.lib.discreteTypeProcessor import DiscreteTypeProcessor
from algorithms.AlgorithmAdapter import AlgorithmAdapter
from pipeline.Structure import Structure
from pipeline.Dataset import Dataset
from algorithms.hartemink import hartemink_discretize


class MIICAdapter(AlgorithmAdapter):
    """
    Adapter for discrete MIIC structure learning (pyAgrum BNLearner).

    Discretizes continuous data internally using DiscreteTypeProcessor,
    then learns the structure with BNLearner.useMIIC().
    """

    DEFAULT_PARAM_GRID = [
        {"n_bins": [2, 4, 6, 8, 10], "discretization_method": ["quantile"]},
        {"n_bins": [2, 4, 6, 8, 10], "discretization_method": ["hartemink"], "initial_bins": [20]},
    ]

    def __init__(self, n_bins: int = 3, discretization_method: str = "quantile",
                 initial_bins: int | None = None,
                 discretized_df: pd.DataFrame | None = None):
        """
        Initialize the MIIC adapter

        Parameters
        ----------
        n_bins : int, optional
            Number of bins for discretization (default: 3)
        discretization_method : str, optional
            Discretization method: 'quantile' or 'hartemink' (default: 'quantile')
        initial_bins : int, optional
            Number of initial bins before merging (Hartemink only, default: n_bins * 3)
        discretized_df : pd.DataFrame, optional
            Pre-discretized data. If provided, skips internal discretization.
        """
        self.n_bins = n_bins
        self.discretization_method = discretization_method
        self.initial_bins = initial_bins
        self._discretized_df = discretized_df

    def _make_learner(self, dataset: Dataset) -> gum.BNLearner:
        """Discretize data and create a BNLearner configured with MIIC."""
        if self._discretized_df is not None:
            learner = gum.BNLearner(self._discretized_df)
        elif self.discretization_method == "hartemink":
            df = dataset.to_dataframe()
            discretized_df = hartemink_discretize(df, n_bins=self.n_bins, initial_bins=self.initial_bins)
            learner = gum.BNLearner(discretized_df)
        else:
            df = dataset.to_dataframe()
            dtp = DiscreteTypeProcessor()
            dtp.setDiscretizationParameters(None, self.discretization_method, self.n_bins)
            template = dtp.discretizedTemplate(df)
            learner = gum.BNLearner(df, template)
        learner.useMIIC()
        learner.setVerbosity(False)
        return learner

    def learn_dag(self, dataset: Dataset) -> gum.DAG:
        """
        Learn DAG using discrete MIIC.

        Uses learnBN() which completes all edge orientations
        (via meekRules_.propagateToDAG) to produce a full DAG.

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
        Learn Bayesian Network structure using discrete MIIC.

        Uses learnPDAG() which returns the CPDAG directly
        (via meekRules_.propagateToCPDAG after v-structure detection).

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
        cpdag = learner.learnPDAG()
        # Note: learnPDAG() returns the CPDAG...
        # See: https://gitlab.com/agrumery/aGrUM/-/blob/master/src/agrum/BN/learning/Miic.cpp#L178
        # https://gitlab.com/agrumery/aGrUM/-/blob/master/src/agrum/base/graphs/algorithms/MeekRules.cpp#L66
        # https://gitlab.com/agrumery/aGrUM/-/blob/master/src/agrum/base/graphs/algorithms/MeekRules.cpp#L177
        return Structure(cpdag)

    def name(self) -> str:
        return f"MIIC_{self.discretization_method}_{self.n_bins}bins"
