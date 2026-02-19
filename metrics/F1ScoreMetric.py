"""
F1-Score metric for comparing CPDAG structures.

Computes precision, recall, and F1-Score directly on MixedGraph CPDAGs,
distinguishing arcs (directed) from edges (undirected).

Uses the same counting strategy as aGrUM's StructuralComparator for PDAGs:
- TP: exact match (same arc direction or same edge)
- FP: any link in test that is wrong (misoriented, wrong type, or extra)
- FN: link in ref completely absent from test
A misoriented arc or a type mismatch (arc vs edge) counts as FP only, not FN.
"""

from itertools import combinations

import pyagrum as gum
from metrics.MetricAdapter import MetricAdapter
from pipeline.Structure import Structure


def count_tp_fp_fn(ref_cpdag: gum.MixedGraph, test_cpdag: gum.MixedGraph) -> tuple:
    """Count TP, FP, FN for structural comparison of two CPDAGs.

    Follows StructuralComparator's PDAG strategy: for each unordered pair
    of nodes, classifies the relationship into one of 10 categories, then:
    - TP = true_arc + true_edge
    - FP = misoriented_arc + wrong_edge_arc + wrong_arc_edge
           + wrong_arc_none + wrong_edge_none
    - FN = wrong_none_arc + wrong_none_edge

    Args:
        ref_cpdag: Reference CPDAG (MixedGraph).
        test_cpdag: Test CPDAG (MixedGraph).

    Returns:
        Tuple (tp, fp, fn).
    """
    true_arc = 0
    true_edge = 0
    misoriented_arc = 0
    wrong_edge_arc = 0   # ref=arc, test=edge
    wrong_arc_edge = 0   # ref=edge, test=arc
    wrong_arc_none = 0   # ref=none, test=arc
    wrong_edge_none = 0  # ref=none, test=edge
    wrong_none_arc = 0   # ref=arc, test=none
    wrong_none_edge = 0  # ref=edge, test=none

    # Check arcs in ref
    for arc in ref_cpdag.arcs():
        tail, head = arc[0], arc[1]
        if test_cpdag.existsArc(tail, head):
            true_arc += 1
        elif test_cpdag.existsArc(head, tail):
            misoriented_arc += 1
        elif test_cpdag.existsEdge(tail, head):
            wrong_edge_arc += 1
        else:
            wrong_none_arc += 1

    # Check edges in ref
    for edge in ref_cpdag.edges():
        first, second = edge[0], edge[1]
        if test_cpdag.existsEdge(first, second):
            true_edge += 1
        elif test_cpdag.existsArc(first, second) or test_cpdag.existsArc(second, first):
            wrong_arc_edge += 1
        else:
            wrong_none_edge += 1

    # Check arcs in test that are completely new (not in ref as arc or edge)
    for arc in test_cpdag.arcs():
        tail, head = arc[0], arc[1]
        if (not ref_cpdag.existsArc(tail, head)
                and not ref_cpdag.existsArc(head, tail)
                and not ref_cpdag.existsEdge(tail, head)):
            wrong_arc_none += 1

    # Check edges in test that are completely new (not in ref as arc or edge)
    for edge in test_cpdag.edges():
        first, second = edge[0], edge[1]
        if (not ref_cpdag.existsEdge(first, second)
                and not ref_cpdag.existsArc(first, second)
                and not ref_cpdag.existsArc(second, first)):
            wrong_edge_none += 1

    tp = true_arc + true_edge
    fp = (misoriented_arc + wrong_edge_arc + wrong_arc_edge
          + wrong_arc_none + wrong_edge_none)
    fn = wrong_none_arc + wrong_none_edge

    return tp, fp, fn


class F1ScoreMetric(MetricAdapter):
    """F1-Score metric for CPDAG comparison.

    Computes F1 = 2 * precision * recall / (precision + recall)
    where precision and recall are based on structural match of arcs
    and edges between two CPDAGs (MixedGraphs).
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
        tp, fp, fn = count_tp_fp_fn(ref.cpdag, test.cpdag)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)
