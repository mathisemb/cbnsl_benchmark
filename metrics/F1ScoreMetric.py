"""
F1-Score metric for comparing CPDAG structures.

Uses pyAgrum's StructuralComparator to compute F1-Score on PDAGs.
"""

import pyagrum as gum
from metrics.MetricAdapter import MetricAdapter
from pipeline.Structure import Structure


class F1ScoreMetric(MetricAdapter):
    """F1-Score metric for CPDAG comparison.

    Uses gum.StructuralComparator to compare two PDAGs and returns f_score.
    """

    def name(self) -> str:
        return "F1-Score"

    def compute(self, ref: Structure, test: Structure) -> float:
        """Compute F1-Score between two CPDAG structures.

        Args:
            ref: Reference structure (ground truth).
            test: Test structure (learned).

        Returns:
            F1-Score in [0, 1]. Returns 0.0 if both precision and recall are 0.
        """
        sc = gum.StructuralComparator()
        sc.compare(ref.cpdag, test.cpdag)
        f1 = sc.f_score()
        return 0.0 if f1 != f1 else f1  # NaN guard (StructuralComparator returns NaN when TP+FP=0)
