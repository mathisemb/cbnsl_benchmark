"""
Plotting utilities for scaling study results.

All aggregation (mean ± std over graph_idx / repeat_idx) is handled
automatically by seaborn. Filter the DataFrame before calling these
functions to control what is shown.

Examples
--------
::

    from scaling.plot import plot_metric, plot_facet
    from scaling.io import load_results

    df = load_results("scaling/results/sem_laplace/results.parquet")

    # Single n_vars — vary n_samples
    plot_metric(df[df.n_vars == 10], y="F1-Score")

    # Multiple n_vars — one subplot per n_vars value
    plot_facet(df, y="F1-Score", col="n_vars")
"""

import seaborn as sns
import pandas as pd

from metrics import ALL_METRICS

METRIC_NAMES = [m.name() for m in ALL_METRICS]
METRIC_COLS = METRIC_NAMES + [f"{m}_skeleton" for m in METRIC_NAMES]


def plot_metric(
    df: pd.DataFrame,
    y: str = "F1-Score",
    x: str = "n_samples",
    **kwargs,
) -> sns.FacetGrid:
    """
    Plot mean ± std of one metric (or time_s) vs. x, one curve per algo.

    Parameters
    ----------
    df : pd.DataFrame
    y : str
        Column to plot on y-axis. Any metric column or ``"time_s"``.
    x : str
        Column to plot on x-axis. Usually ``"n_samples"`` or ``"n_vars"``.
    **kwargs
        Forwarded to ``sns.relplot``.
    """
    n_graphs = df["graph_idx"].nunique()
    g = sns.relplot(
        data=df, x=x, y=y, hue="algo",
        kind="line", errorbar="se", marker="o",
        **kwargs,
    )
    g.figure.subplots_adjust(top=0.9)
    g.figure.suptitle(f"{y} vs. {x} (mean +/- se over {n_graphs} graphs)")
    g._legend.set_title(None)
    return g


def plot_facet(
    df: pd.DataFrame,
    y: str = "F1-Score",
    x: str = "n_samples",
    col: str = "n_vars",
    **kwargs,
) -> sns.FacetGrid:
    """
    Same as :func:`plot_metric` with faceting on a third dimension.

    Parameters
    ----------
    df : pd.DataFrame
    y : str
    x : str
    col : str
        Column used to split into subplots (e.g. ``"n_vars"``).
    **kwargs
        Forwarded to ``sns.relplot``.
    """
    n_graphs = df["graph_idx"].nunique()
    g = sns.relplot(
        data=df, x=x, y=y, hue="algo", col=col,
        kind="line", errorbar="se", marker="o",
        **kwargs,
    )
    g.figure.subplots_adjust(top=0.85)
    g.figure.suptitle(f"{y} vs. {x} (mean +/- se over {n_graphs} graphs)")
    g._legend.set_title(None)
    return g
