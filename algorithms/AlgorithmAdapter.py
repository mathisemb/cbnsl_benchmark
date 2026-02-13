from abc import ABC, abstractmethod
from enum import Enum
from typing import Any
import pyagrum as gum


class DataType(Enum):
    """Enum for data types supported by algorithms"""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


class AlgorithmAdapter(ABC):
    """
    Abstract base class for all algorithm adapters.
    
    Any concrete algorithm implementation should inherit from this class
    and implement the required methods.
    
    All algorithms return a gum.DAG (pyAgrum DAG) with only the structure
    (nodes and arcs), regardless of the underlying library used internally.
    This provides a unified interface for structure comparison.
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
    def learn_structure(self, dataset) -> gum.DAG:
        """
        Learn the Bayesian Network structure from the dataset
        
        Parameters
        ----------
        dataset : Dataset
            The dataset to learn from
            
        Returns
        -------
        gum.DAG
            The learned DAG (structure only, no parameters)
        """
        pass
