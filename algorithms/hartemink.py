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


def _mutual_information_naive(x: np.ndarray, y: np.ndarray) -> float:
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


def _relabel_to_dataframe(
    bin_labels: np.ndarray, columns: list, index
) -> pd.DataFrame:
    """Relabel integer bin_labels to consecutive string labels for pyAgrum.

    For example if the final bin labels for a variable are [0, 2, 5]
    we want to relabel them to ["0", "1", "2"] for pyAgrum.

    Parameters
    ----------
    bin_labels : np.ndarray
        Integer bin labels (n_samples x n_vars).
    columns : list
        Column names.
    index : pd.Index
        DataFrame index.

    Returns
    -------
    pd.DataFrame
        Discretized data with string labels (``"0"``, ``"1"``, ...).
    """
    result = pd.DataFrame(index=index, columns=columns)
    for j, col in enumerate(columns):
        unique_sorted = np.sort(np.unique(bin_labels[:, j]))
        label_map = {v: str(i) for i, v in enumerate(unique_sorted)}
        result[col] = [label_map[v] for v in bin_labels[:, j]]
    return result


def hartemink_discretize_multi(
    df: pd.DataFrame,
    target_bins: list[int],
    initial_bins: int | None = None,
    initial_method: str = "quantile",
    progress: bool = False,
) -> dict[int, pd.DataFrame]:
    """Hartemink information-preserving discretization for multiple target bin counts.

    Runs the merge loop once from ``initial_bins`` down to ``min(target_bins)``,
    saving a snapshot at each requested bin count. This avoids redundant
    computation when the same initial discretization is shared across
    several target bin counts.

    Parameters
    ----------
    df : pd.DataFrame
        Continuous data (n_samples x n_features).
    target_bins : list[int]
        Target bin counts to snapshot (e.g. ``[3, 4, 5]``).
    initial_bins : int, optional
        Number of bins for the initial discretization step.
        Default: ``max(target_bins) * 3``.
    initial_method : str
        Method for the initial discretization: ``"quantile"`` or ``"uniform"``.

    Returns
    -------
    dict[int, pd.DataFrame]
        Mapping from ``n_bins`` to the corresponding discretized DataFrame.
    """
    if not target_bins:
        raise ValueError("target_bins must not be empty")

    max_target = max(target_bins)
    min_target = min(target_bins)

    if initial_bins is None:
        initial_bins = max_target * 3
    if initial_bins <= max_target:
        raise ValueError(
            f"initial_bins ({initial_bins}) must be strictly greater "
            f"than max(target_bins) ({max_target})"
        )

    columns = list(df.columns)
    n_vars = len(columns)

    # ------------------------------------------------------------------
    # Step 1: initial discretization (integer labels)
    # ------------------------------------------------------------------
    bin_labels = np.empty((len(df), n_vars), dtype=np.int32)

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
    # Step 2: iterative merging with snapshots
    # ------------------------------------------------------------------
    n_bins_per_var = np.array(
        [len(np.unique(bin_labels[:, j])) for j in range(n_vars)]
    )

    sorted_targets = sorted(set(target_bins), reverse=True)
    snapshots = {}

    total_merges = int(n_bins_per_var.sum() - n_vars * min_target)
    pbar = None
    if progress:
        from tqdm.auto import tqdm

        pbar = tqdm(total=total_merges, desc="Hartemink merging", leave=True)

    for target in sorted_targets:
        while np.any(n_bins_per_var > target):
            for j in range(n_vars):
                if n_bins_per_var[j] <= target:
                    continue

                col_bins = bin_labels[:, j]
                sorted_bins = np.sort(np.unique(col_bins))
                nb_adjacent_pairs = len(sorted_bins) - 1

                best_mi = -np.inf
                best_pair = None

                for k in range(nb_adjacent_pairs):
                    bin_lo, bin_hi = sorted_bins[k], sorted_bins[k + 1]

                    merged = col_bins.copy()
                    merged[merged == bin_hi] = bin_lo

                    total_mi = 0.0
                    for other_j in range(n_vars):
                        if other_j != j:
                            total_mi += _mutual_information_naive(
                                merged, bin_labels[:, other_j]
                            )

                    if total_mi > best_mi:
                        best_mi = total_mi
                        best_pair = (bin_lo, bin_hi)

                if best_pair is not None:
                    bin_lo, bin_hi = best_pair
                    bin_labels[:, j][bin_labels[:, j] == bin_hi] = bin_lo
                    n_bins_per_var[j] -= 1
                    if pbar is not None:
                        pbar.update(1)

        snapshots[target] = _relabel_to_dataframe(
            bin_labels.copy(), columns, df.index
        )

    if pbar is not None:
        pbar.close()

    return snapshots


def hartemink_discretize(
    df: pd.DataFrame,
    n_bins: int,
    initial_bins: int | None = None,
    initial_method: str = "quantile",
    progress: bool = True,
) -> pd.DataFrame:
    """Hartemink information-preserving discretization.

    Convenience wrapper around :func:`hartemink_discretize_multi` for a
    single target bin count.

    Parameters
    ----------
    df : pd.DataFrame
        Continuous data (n_samples x n_features).
    n_bins : int
        Target number of bins for each variable.
    initial_bins : int, optional
        Number of bins for the initial discretization step.
        Default: ``n_bins * 3``.
    initial_method : str
        Method for the initial discretization: ``"quantile"`` or ``"uniform"``.

    Returns
    -------
    pd.DataFrame
        Discretized data with string labels (``"0"``, ``"1"``, ...).
    """
    return hartemink_discretize_multi(
        df,
        target_bins=[n_bins],
        initial_bins=initial_bins,
        initial_method=initial_method,
        progress=progress,
    )[n_bins]
