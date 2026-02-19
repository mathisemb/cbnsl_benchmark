"""
True Positive Rate (recall) metric for comparing CPDAG structures.

Computes TPR = TP / (TP + FN) directly on MixedGraph CPDAGs,
distinguishing arcs (directed) from edges (undirected).
"""

from metrics.MetricAdapter import MetricAdapter
from metrics.F1ScoreMetric import count_tp_fp_fn
from pipeline.Structure import Structure


class TPRMetric(MetricAdapter):
    """True Positive Rate (recall) metric for CPDAG comparison.

    Computes TPR = TP / (TP + FN) where TP and FN are based on
    structural match of arcs and edges between two CPDAGs (MixedGraphs).
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
        tp, fp, fn = count_tp_fp_fn(ref.cpdag, test.cpdag)

        if (tp + fn) == 0:
            return 0.0
        return tp / (tp + fn)
