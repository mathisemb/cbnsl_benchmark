"""
Tests for F1 score between CPDAGs.

Running the tests:
python -m pytest tests/test_F1.py -v -s
-v: verbose, shows the name of each test and if it passed or failed.
-s: deactivate output capture, allows print statements to show in the console for debugging.
"""

import pyagrum as gum

from metrics.F1ScoreMetric import F1ScoreMetric, count_tp_fp_fn
from pipeline.Structure import Structure

def test_f1_identity_arcs():
    """F1(G, G) == 1 pour un graphe avec uniquement des arcs."""
    # 0 -> 1 -> 2
    g = gum.MixedGraph()
    g.addNodeWithId(0)
    g.addNodeWithId(1)
    g.addNodeWithId(2)
    g.addArc(0, 1)
    g.addArc(1, 2)
    result = F1ScoreMetric().compute(Structure(g), Structure(g))
    print(f"g = {g}, F1(g, g) = {result}")
    assert result == 1


def test_f1_identity_edges():
    """F1(G, G) == 1 pour un graphe avec uniquement des edges."""
    # 0 - 1 - 2
    g = gum.MixedGraph()
    g.addNodeWithId(0)
    g.addNodeWithId(1)
    g.addNodeWithId(2)
    g.addEdge(0, 1)
    g.addEdge(1, 2)
    result = F1ScoreMetric().compute(Structure(g), Structure(g))
    print(f"g = {g}, F1(g, g) = {result}")
    assert result == 1


def test_f1_identity_mixed():
    """F1(G, G) == 1 pour un graphe avec arcs et edges."""
    # 0 -> 1 - 2
    g = gum.MixedGraph()
    g.addNodeWithId(0)
    g.addNodeWithId(1)
    g.addNodeWithId(2)
    g.addArc(0, 1)
    g.addEdge(1, 2)
    result = F1ScoreMetric().compute(Structure(g), Structure(g))
    print(f"g = {g}, F1(g, g) = {result}")
    assert result == 1


def test_f1_empty_graphs():
    """F1(vide, vide) == 0."""
    # (0)  — un seul noeud, aucun lien
    g = gum.MixedGraph()
    g.addNodeWithId(0)
    result = F1ScoreMetric().compute(Structure(g), Structure(g))
    print(f"g = {g}, F1(g, g) = {result}")
    assert result == 0


def test_f1_no_overlap():
    """Aucun lien en commun => F1 == 0."""
    # ref:  0 -> 1    2    3
    # test: 0    1    2 -> 3
    ref = gum.MixedGraph()
    test = gum.MixedGraph()
    for i in range(4):
        ref.addNodeWithId(i)
        test.addNodeWithId(i)
    ref.addArc(0, 1)
    test.addArc(2, 3)
    result = F1ScoreMetric().compute(Structure(ref), Structure(test))
    print(f"ref = {ref}, test = {test}, F1(ref, test) = {result}")
    assert result == 0


def test_f1_all_misoriented():
    """Tous les arcs inversés => F1 == 0."""
    # ref:  0 -> 1 -> 2
    # test: 0 <- 1 <- 2
    ref = gum.MixedGraph()
    test = gum.MixedGraph()
    for i in range(3):
        ref.addNodeWithId(i)
        test.addNodeWithId(i)
    ref.addArc(0, 1)
    ref.addArc(1, 2)
    test.addArc(1, 0)
    test.addArc(2, 1)
    result = F1ScoreMetric().compute(Structure(ref), Structure(test))
    print(f"ref = {ref}, test = {test}, F1(ref, test) = {result}")
    assert result == 0


def test_f1_edges_vs_dag_same_skeleton():
    """CPDAG avec edges vs DAG même squelette => F1 == 0."""
    # ref:  0 - 1 - 2
    # test: 0 -> 1 -> 2
    ref = gum.MixedGraph()
    test = gum.MixedGraph()
    for i in range(3):
        ref.addNodeWithId(i)
        test.addNodeWithId(i)
    ref.addEdge(0, 1)
    ref.addEdge(1, 2)
    test.addArc(0, 1)
    test.addArc(1, 2)
    result = F1ScoreMetric().compute(Structure(ref), Structure(test))
    print(f"ref = {ref}, test = {test}, F1(ref, test) = {result}")
    assert result == 0


def test_tp_fp_fn_identity():
    """count_tp_fp_fn(G, G) => tp=nb_liens, fp=0, fn=0."""
    # 0 -> 1 - 2
    g = gum.MixedGraph()
    g.addNodeWithId(0)
    g.addNodeWithId(1)
    g.addNodeWithId(2)
    g.addArc(0, 1)
    g.addEdge(1, 2)
    tp, fp, fn = count_tp_fp_fn(g, g)
    print(f"g = {g}, count_tp_fp_fn(g, g) = ({tp}, {fp}, {fn})")
    assert tp == 2
    assert fp == 0
    assert fn == 0
