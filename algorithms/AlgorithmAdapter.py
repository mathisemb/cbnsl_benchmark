from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pipeline.Structure import Structure

if TYPE_CHECKING:
    from pipeline.Dataset import Dataset


class AlgorithmAdapter(ABC):
    """
    Abstract base class for all algorithm adapters.

    Any concrete algorithm implementation should inherit from this class
    and implement the required methods.

    All algorithms return a Structure object containing a CPDAG (Completed
    Partially Directed Acyclic Graph), regardless of the underlying library
    used internally. This provides a unified interface for structure comparison.
    """

    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the algorithm

        Returns
        -------
        str
            The algorithm name
        """
        pass

    @abstractmethod
    def learn_structure(self, dataset: Dataset) -> Structure:
        """
        Learn the Bayesian Network structure from the dataset

        Parameters
        ----------
        dataset : Dataset
            The dataset to learn from

        Returns
        -------
        Structure
            The learned structure (CPDAG)
        """
        pass
