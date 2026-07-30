"""
Structural Hamming Distance (SHD) metric for comparing CPDAG structures.

Uses pyAgrum's StructuralMetrics to compute SHD on PDAGs.
"""

import pyagrum as gum
from metrics.MetricAdapter import MetricAdapter
from pipeline.Structure import Structure


class SHDMetric(MetricAdapter):
    """Structural Hamming Distance (SHD) metric for CPDAG comparison.

    Uses gum.StructuralMetrics to compare two PDAGs and returns shd.
    """

    def name(self) -> str:
        return "SHD"

    def compute(self, ref: Structure, test: Structure) -> float:
        """Compute Structural Hamming Distance between two CPDAG structures.

        Args:
            ref: Reference structure (ground truth).
            test: Test structure (learned).

        Returns:
            SHD value (number of arc/edge differences, >= 0).
        """
        sc = gum.StructuralMetrics()
        sc.compare(ref.cpdag, test.cpdag)
        return float(sc.shd())

    def compute_skeleton(self, ref: Structure, test: Structure) -> float:
        """Compute Structural Hamming Distance between the skeletons of two CPDAG structures.

        Args:
            ref: Reference structure (ground truth).
            test: Test structure (learned).

        Returns:
            Skeleton SHD value (number of edge differences, >= 0).
        """
        sc = gum.StructuralMetrics()
        sc.compare(ref.cpdag, test.cpdag)
        return float(sc.shd_skeleton())
