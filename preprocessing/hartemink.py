"""
Hartemink information-preserving discretization.

Implements the discretization method from:
    Hartemink, A. (2001). Principled Computational Methods for the Validation
    and Discovery of Genetic Regulatory Networks. PhD thesis, MIT.

Unlike univariate methods (quantile, uniform, kmeans), this method considers
pairwise mutual information between all variables when deciding which bins
to merge, preserving inter-variable dependencies.
"""

import numpy as np
import pandas as pd


def _mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Compute mutual information between two discrete integer arrays.

    Uses the empirical joint distribution from a contingency table.
    """
    # Build contingency table using integer labels directly
    x_vals = np.unique(x)
    y_vals = np.unique(y)
    x_map = {v: i for i, v in enumerate(x_vals)}
    y_map = {v: i for i, v in enumerate(y_vals)}

    cont = np.zeros((len(x_vals), len(y_vals)), dtype=np.float64)
    for xi, yi in zip(x, y):
        cont[x_map[xi], y_map[yi]] += 1.0

    # Normalize to joint probability
    n = cont.sum()
    if n == 0:
        return 0.0
    pxy = cont / n
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    mi = 0.0
    for i in range(len(px)):
        for j in range(len(py)):
            if pxy[i, j] > 0:
                mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))
    return mi


def hartemink_discretize(
    df: pd.DataFrame,
    n_bins: int,
    initial_bins: int | None = None,
    initial_method: str = "quantile",
) -> pd.DataFrame:
    """Hartemink information-preserving discretization.

    Parameters
    ----------
    df : pd.DataFrame
        Continuous data (n_samples x n_features).
    n_bins : int
        Target number of bins for each variable.
    initial_bins : int, optional
        Number of bins for the initial discretization step.
        Arbitrary default: ``n_bins * 3`` (enough headroom for
        meaningful merges).
    initial_method : str
        Method for the initial discretization: ``"quantile"`` or ``"uniform"``. 

    Returns
    -------
    pd.DataFrame
        Discretized data with string labels (``"0"``, ``"1"``, ...).
    """
    if initial_bins is None:
        initial_bins = n_bins * 3
    if initial_bins <= n_bins:
        raise ValueError(
            f"initial_bins ({initial_bins}) must be strictly greater than n_bins ({n_bins})"
        )

    columns = list(df.columns)
    n_vars = len(columns)

    # ------------------------------------------------------------------
    # Step 1: initial discretization (integer labels)
    # ------------------------------------------------------------------
    bin_labels = np.empty((len(df), n_vars), dtype=np.int32)  # (n_samples, n_vars)

    for j, col in enumerate(columns):
        values = df[col].values
        if initial_method == "quantile":
            _, edges = pd.qcut(values, initial_bins, retbins=True, duplicates="drop")
        elif initial_method == "uniform":
            _, edges = pd.cut(values, initial_bins, retbins=True)
        else:
            raise ValueError(
                f"initial_method must be 'quantile' or 'uniform', got '{initial_method}'"
            )
        bin_labels[:, j] = np.digitize(values, edges[1:-1], right=False)

    # ------------------------------------------------------------------
    # Step 2: iterative merging
    # ------------------------------------------------------------------
    n_bins_per_var = np.array([len(np.unique(bin_labels[:, j])) for j in range(n_vars)])

    while np.any(n_bins_per_var > n_bins):
        for j in range(n_vars):
            if n_bins_per_var[j] <= n_bins:
                continue

            col_bins = bin_labels[:, j]
            adjacent_bins = np.sort(np.unique(col_bins))

            best_mi = -np.inf
            best_pair = None

            for k in range(len(adjacent_bins) - 1):
                bin_lo, bin_hi = adjacent_bins[k], adjacent_bins[k + 1]

                merged = col_bins.copy()  # try merging bin_hi into bin_lo
                merged[merged == bin_hi] = bin_lo

                total_mi = 0.0  # sum of MI(merged_var_j, var_other) over all other vars
                for other_j in range(n_vars):
                    if other_j != j:
                        total_mi += _mutual_information(merged, bin_labels[:, other_j])

                if total_mi > best_mi:  # keep the merge that preserves the most MI
                    best_mi = total_mi
                    best_pair = (bin_lo, bin_hi)

            if best_pair is not None:
                bin_lo, bin_hi = best_pair
                bin_labels[:, j][bin_labels[:, j] == bin_hi] = bin_lo
                n_bins_per_var[j] -= 1

    # ------------------------------------------------------------------
    # Step 3: relabel to consecutive string labels for pyAgrum
    # ------------------------------------------------------------------
    result = pd.DataFrame(index=df.index, columns=columns)
    for j, col in enumerate(columns):
        unique_sorted = np.sort(np.unique(bin_labels[:, j]))
        label_map = {v: str(i) for i, v in enumerate(unique_sorted)}
        result[col] = [label_map[v] for v in bin_labels[:, j]]

    return result
