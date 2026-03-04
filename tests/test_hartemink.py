"""
Tests for Hartemink information-preserving discretization.

To run: python tests/test_hartemink.py
so it also runs the demos
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algorithms.hartemink import hartemink_discretize, hartemink_discretize_multi, _mutual_information_naive

def test_mutual_information_independent():
    """MI of independent variables should be close to 0."""
    rng = np.random.default_rng(42)
    x = rng.integers(0, 3, size=10000)
    y = rng.integers(0, 3, size=10000)
    mi = _mutual_information_naive(x, y)
    print(f"MI of independent vars: {mi:.4f}")
    assert mi < 0.01, f"MI of independent vars should be ~0, got {mi}"

def test_mutual_information_dependent():
    """MI of dependent variables should be significantly different from 0."""
    rng = np.random.default_rng(42)
    x = rng.integers(0, 3, size=10000)
    y = x + rng.integers(0, 3, size=10000)  # Make y dependent on x
    mi = _mutual_information_naive(x, y)
    print(f"MI of dependent vars: {mi:.4f}")
    assert mi > 0.1, f"MI of dependent vars should be >0, got {mi}"

def test_multi_matches_individual():
    """hartemink_discretize_multi must produce the same results as individual calls."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "A": rng.standard_normal(200),
        "B": rng.standard_normal(200),
        "C": rng.standard_normal(200),
    })
    initial_bins = 15

    # Individual calls
    r3 = hartemink_discretize(df, n_bins=3, initial_bins=initial_bins)
    r5 = hartemink_discretize(df, n_bins=5, initial_bins=initial_bins)

    # Multi call
    multi = hartemink_discretize_multi(df, target_bins=[3, 5], initial_bins=initial_bins)

    print("r3:", r3)
    print("r5:", r5)
    print("multi[3, 5]:", multi)

    assert set(multi.keys()) == {3, 5}, f"Expected keys {{3, 5}}, got {set(multi.keys())}"
    assert (r3 == multi[3]).all().all(), "n_bins=3: multi result differs from individual call"
    assert (r5 == multi[5]).all().all(), "n_bins=5: multi result differs from individual call"


def test_multi_single_target():
    """hartemink_discretize_multi with a single target must match hartemink_discretize."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "X": rng.standard_normal(100),
        "Y": rng.standard_normal(100),
    })
    single = hartemink_discretize(df, n_bins=4)
    multi = hartemink_discretize_multi(df, target_bins=[4])

    print("single:\n", single)
    print("multi[4]:\n", multi[4])

    assert (single == multi[4]).all().all(), "Single-target multi differs from hartemink_discretize"


def test_hartemink_preserves_more_mi_than_quantile():
    """Hartemink should preserve more MI between correlated variables than quantile."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    y = x + rng.standard_normal(500) * 0.3
    z = rng.standard_normal(500)
    df = pd.DataFrame({"X": x, "Y": y, "Z": z})

    n_bins = 3
    hart = hartemink_discretize(df, n_bins=n_bins)
    quant = df.apply(lambda col: pd.qcut(col, n_bins, labels=["0", "1", "2"], duplicates="drop"))

    mi_hart = _mutual_information_naive(hart["X"], hart["Y"])
    mi_quant = _mutual_information_naive(quant["X"], quant["Y"])
    print(f"MI(X,Y) Hartemink: {mi_hart:.4f}, Quantile: {mi_quant:.4f}")
    assert mi_hart > mi_quant, "Hartemink should preserve more MI between X and Y than quantile"


def test_mi_symmetry():
    """Mutual information must be symmetric: MI(X,Y) == MI(Y,X)."""
    rng = np.random.default_rng(42)
    x = rng.integers(0, 5, size=5000)
    y = (x + rng.integers(0, 3, size=5000)) % 5
    mi_xy = _mutual_information_naive(x, y)
    mi_yx = _mutual_information_naive(y, x)
    print(f"MI(X,Y): {mi_xy:.4f}, MI(Y,X): {mi_yx:.4f}")
    assert abs(mi_xy - mi_yx) < 1e-12, (
        f"MI should be symmetric: MI(X,Y)={mi_xy}, MI(Y,X)={mi_yx}"
    )


def test_mi_self_equals_entropy():
    """MI(X,X) should equal the entropy of X."""
    rng = np.random.default_rng(42)
    x = rng.integers(0, 4, size=10000)
    mi_self = _mutual_information_naive(x, x)
    _, counts = np.unique(x, return_counts=True)
    p = counts / counts.sum()
    entropy = -np.sum(p * np.log(p))
    print(f"MI(X,X): {mi_self:.4f}, H(X): {entropy:.4f}")
    assert abs(mi_self - entropy) < 1e-10, (
        f"MI(X,X)={mi_self:.6f} should equal H(X)={entropy:.6f}"
    )


def test_monotonic_mi_loss():
    """More bins should preserve at least as much MI as fewer bins.

    If we discretize correlated variables into 5, 4, and 3 bins,
    the total pairwise MI should be non-increasing as bins decrease.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    y = x + rng.standard_normal(500) * 0.5
    df = pd.DataFrame({"X": x, "Y": y})

    results = hartemink_discretize_multi(df, target_bins=[3, 4, 5])

    def total_mi(disc_df):
        return _mutual_information_naive(disc_df["X"].values, disc_df["Y"].values)

    mi_5 = total_mi(results[5])
    mi_4 = total_mi(results[4])
    mi_3 = total_mi(results[3])
    print(f"MI: 5bins={mi_5:.4f}, 4bins={mi_4:.4f}, 3bins={mi_3:.4f}")
    assert mi_5 >= mi_4 - 1e-10, f"MI should not increase when reducing from 5 to 4 bins"
    assert mi_4 >= mi_3 - 1e-10, f"MI should not increase when reducing from 4 to 3 bins"


if __name__ == "__main__":
    tests = [
        test_mutual_information_independent,
        test_mutual_information_dependent,
        test_mi_symmetry,
        test_mi_self_equals_entropy,
        test_multi_matches_individual,
        test_multi_single_target,
        test_hartemink_preserves_more_mi_than_quantile,
        test_monotonic_mi_loss,
    ]
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {test.__name__}: {e}")
