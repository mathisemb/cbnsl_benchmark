"""
Tests for the generic Pareto front function.

To run: python -m pytest tests/test_pareto.py
"""

from pipeline.GridSearch import pareto_front


def test_pareto_simple():
    """Two objectives, three points: one dominated."""
    points = [
        {"SHD": 5, "F1": 0.8},   # 0
        {"SHD": 3, "F1": 0.9},   # 1 — dominates 0
        {"SHD": 2, "F1": 0.7},   # 2 — not dominated (best SHD)
    ]
    objectives = {"SHD": True, "F1": False}  # SHD lower better, F1 higher better
    front = pareto_front(points, objectives)
    assert set(front) == {1, 2}


def test_pareto_all_equal():
    """All points identical: none dominates another, all on front."""
    points = [{"A": 1, "B": 2}] * 4
    objectives = {"A": True, "B": True}
    front = pareto_front(points, objectives)
    assert len(front) == 4


def test_pareto_single_point():
    points = [{"X": 10, "Y": 20}]
    front = pareto_front(points, {"X": True, "Y": False})
    assert front == [0]


def test_pareto_missing_key():
    """Points missing an objective key are excluded."""
    points = [
        {"SHD": 5},
        {"SHD": 3, "F1": 0.9},
    ]
    front = pareto_front(points, {"SHD": True, "F1": False})
    assert front == [1]
