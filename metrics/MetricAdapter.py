"""
MetricAdapter abstract class for computing comparison metrics between DAGs.
"""

from abc import ABC, abstractmethod
from typing import Optional
import pyagrum as gum


class MetricAdapter(ABC):
    """
    Abstract base class for all metric adapters.
    
    Metrics compute values based on one or two DAGs.
    """

    @abstractmethod
    def name(self) -> str:
        """
        Returns the metric name
        
        Returns
        -------
        str
            The metric name
        """
        pass

    @abstractmethod
    def compute(self, learned_dag: gum.DAG, golden_dag: Optional[gum.DAG] = None) -> float:
        """
        Compute the metric value
        
        Parameters
        ----------
        learned_dag : gum.DAG
            First DAG to compute metric on
        golden_dag : gum.DAG, optional
            Second DAG for comparison (if needed by the metric)
            
        Returns
        -------
        float
            The computed metric value
            
        Raises
        ------
        ValueError
            If the metric cannot be computed with the provided DAGs
        """
        pass
