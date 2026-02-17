"""
MetricAdapter abstract class for computing comparison metrics between structures.
"""

from abc import ABC, abstractmethod

# Avoid circular import by using TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pipeline.Structure import Structure


class MetricAdapter(ABC):
    """
    Abstract base class for all metric adapters.

    Metrics compute comparison values between two Structure objects containing CPDAGs.
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
    def compute(self, ref: "Structure", test: "Structure") -> float:
        """
        Compute the metric value by comparing two structures

        Parameters
        ----------
        ref : Structure
            Reference structure (typically ground truth)
        test : Structure
            Test structure to compare (typically learned structure)

        Returns
        -------
        float
            The computed metric value

        Raises
        ------
        ValueError
            If the metric cannot be computed with the provided structures
        """
        pass
