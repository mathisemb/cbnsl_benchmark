"""
Tests for metrics

To run: python -m tests.test_metrics
or
python -m pytest tests/test_metrics.py
"""
import pyagrum as gum

from metrics.SHDMetric import SHDMetric
from metrics.F1ScoreMetric import F1ScoreMetric
from metrics.TPRMetric import TPRMetric
from pipeline.Structure import Structure


# ===== SHD(G,G)=0, F1(G,G)=1, TPR(G,G)=1. =====

def _make_structure_G():
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
    s = _make_structure_G()
    assert SHDMetric().compute(s, s) == 0

def test_f1_identity():
    s = _make_structure_G()
    assert F1ScoreMetric().compute(s, s) == 1.0

def test_tpr_identity():
    s = _make_structure_G()
    assert TPRMetric().compute(s, s) == 1.0


# ====== SHD(G, G_1)=2, F1(G, G_1)<1, TPR(G, G_1)<1 =====

def _make_structure_G1_G2():
    """Simple PDAGs:
    0 -> 1, 2
    0 -> 1 -> 2 """
    g1 = gum.PDAG()
    g1.addNode()
    g1.addNode()
    g1.addNode()
    g1.addArc(0, 1)

    g2 = gum.PDAG()
    g2.addNode()
    g2.addNode()
    g2.addNode()
    g2.addArc(0, 1)
    g2.addArc(1, 2)
    return Structure(g1), Structure(g2)

def test_shd_G1_G2():
    g1, g2 = _make_structure_G1_G2()
    shd = SHDMetric().compute(g1, g2)
    print("SHD(G1, G2) =", shd)
    assert shd == 1

def test_f1_G1_G2():
    g1, g2 = _make_structure_G1_G2()
    f1 = F1ScoreMetric().compute(g1, g2)
    print("F1(G1, G2) =", f1)
    assert f1 < 1.0

def test_tpr_G1_G2():
    g1, g2 = _make_structure_G1_G2()
    tpr = TPRMetric().compute(g2, g1)
    print("TPR(G2, G1) =", tpr)
    assert tpr < 1.0


if __name__ == "__main__":
    tests = [
        test_shd_identity,
        test_f1_identity,
        test_tpr_identity,
        test_shd_G1_G2,
        test_f1_G1_G2,
        test_tpr_G1_G2
    ]
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {test.__name__}: {e}")
