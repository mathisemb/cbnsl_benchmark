from abc import ABC, abstractmethod
from enum import Enum

from pipeline.Structure import Structure


class DataType(Enum):
    """Enum for data types supported by algorithms"""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


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
    def required_data_type(self) -> DataType:
        """
        Returns the data type required by this algorithm
        
        Returns
        -------
        DataType
            CONTINUOUS or DISCRETE
        """
        pass

    @abstractmethod
    def learn_structure(self, dataset) -> Structure:
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
