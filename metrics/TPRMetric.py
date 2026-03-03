"""
True Positive Rate (recall) metric for comparing CPDAG structures.

Uses pyAgrum's StructuralComparator to compute recall on PDAGs.
"""

import pyagrum as gum
from metrics.MetricAdapter import MetricAdapter
from pipeline.Structure import Structure


class TPRMetric(MetricAdapter):
    """True Positive Rate (recall) metric for CPDAG comparison.

    Uses gum.StructuralComparator to compare two PDAGs and returns recall.
    """

    def name(self) -> str:
        return "TPR"

    def compute(self, ref: Structure, test: Structure) -> float:
        """Compute True Positive Rate between two CPDAG structures.

        Args:
            ref: Reference structure (ground truth).
            test: Test structure (learned).

        Returns:
            TPR in [0, 1]. Returns 0.0 if there are no positives in ref.
        """
        sc = gum.StructuralComparator()
        sc.compare(ref.cpdag, test.cpdag)
        recall = sc.recall()
        return 0.0 if recall != recall else recall  # NaN guard
