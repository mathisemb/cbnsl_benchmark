"""
Tests for Hartemink information-preserving discretization.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from preprocessing.hartemink import hartemink_discretize, _mutual_information


def test_mutual_information_independent():
    """MI of independent variables should be close to 0."""
    rng = np.random.default_rng(42)
    x = rng.integers(0, 3, size=10000)
    y = rng.integers(0, 3, size=10000)
    mi = _mutual_information(x, y)
    assert mi < 0.01, f"MI of independent vars should be ~0, got {mi}"


def demo_correlated():
    """Show Hartemink vs quantile on correlated data."""
    print("=" * 60)
    print("DEMO: Correlated variables (X, Y=X+noise, Z=independent)")
    print("=" * 60)

    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    y = x + rng.standard_normal(500) * 0.3
    z = rng.standard_normal(500)
    df = pd.DataFrame({"X": x, "Y": y, "Z": z})

    n_bins = 3
    hart = hartemink_discretize(df, n_bins=n_bins)
    quant = df.apply(
        lambda col: pd.qcut(col, n_bins, labels=["0", "1", "2"], duplicates="drop")
    )

    for col in df.columns:
        same = (hart[col] == quant[col]).mean() * 100
        print(f"\n  {col}: {same:.1f}% identical bins")
        print(f"    Hartemink  bin counts: {dict(hart[col].value_counts().sort_index())}")
        print(f"    Quantile   bin counts: {dict(quant[col].value_counts().sort_index())}")
    print()


def demo_sachs():
    """Show Hartemink discretization on the Sachs dataset."""
    print("=" * 60)
    print("DEMO: Sachs dataset")
    print("=" * 60)

    data_path = project_root / "data" / "sachs" / "sachs_observational.csv"
    sachs = pd.read_csv(data_path, sep="\t")
    print(f"  Shape: {sachs.shape}")

    for n_bins in [3, 5]:
        hart = hartemink_discretize(sachs, n_bins=n_bins)
        quant = sachs.apply(
            lambda col: pd.qcut(col, n_bins, labels=False, duplicates="drop").astype(str)
        )

        print(f"\n  n_bins={n_bins}:")
        for col in sachs.columns:
            same = (hart[col] == quant[col]).mean() * 100
            h_nunique = hart[col].nunique()
            q_nunique = quant[col].nunique()
            print(f"    {col:<6} hartemink={h_nunique} bins, quantile={q_nunique} bins, {same:.0f}% identical")
    print()


if __name__ == "__main__":
    try:
        test_mutual_information_independent()
        print(f"  PASS  {test_mutual_information_independent.__name__}")
    except AssertionError as e:
        print(f"  FAIL  {test_mutual_information_independent.__name__}: {e}")
    except Exception as e:
        print(f"  ERROR {test_mutual_information_independent.__name__}: {e}")

    demo_correlated()
    demo_sachs()
