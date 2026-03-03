"""
Tests for metrics: SHD(G,G)=0, F1(G,G)=1, TPR(G,G)=1.
"""

import pyagrum as gum

from metrics.SHDMetric import SHDMetric
from metrics.F1ScoreMetric import F1ScoreMetric
from metrics.TPRMetric import TPRMetric
from pipeline.Structure import Structure


def _make_structure():
    """Simple PDAG: 0 -> 1, 0 -> 2, 1 - 2."""
    g = gum.PDAG()
    g.addNodeWithId(0)
    g.addNodeWithId(1)
    g.addNodeWithId(2)
    g.addArc(0, 1)
    g.addArc(0, 2)
    g.addEdge(1, 2)
    return Structure(g)


def test_shd_identity():
    s = _make_structure()
    assert SHDMetric().compute(s, s) == 0


def test_f1_identity():
    s = _make_structure()
    assert F1ScoreMetric().compute(s, s) == 1.0


def test_tpr_identity():
    s = _make_structure()
    assert TPRMetric().compute(s, s) == 1.0
