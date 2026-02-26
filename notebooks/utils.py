"""Utility functions for benchmark notebooks (plotting, grid search, Pareto selection)."""

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyagrum.lib.notebook as gnb
from tqdm.auto import tqdm

from analysis.GridSearch import GridSearch
from analysis.ParetoSelector import pareto_front, best_pareto
from pipeline.Structure import dag_to_structure
from preprocessing.hartemink import hartemink_discretize_multi


# ---------------------------------------------------------------------------
# Structure helpers
# ---------------------------------------------------------------------------

def cpdag_to_dot(structure, feature_names=None):
    """Convert a Structure's CPDAG to a labeled dot string."""
    mg = structure.cpdag
    lines = ["digraph {"]
    for node_id in sorted(mg.nodes()):
        label = feature_names[node_id] if feature_names else str(node_id)
        lines.append(f'  {node_id} [label="{label}"];')
    for tail, head in mg.arcs():
        lines.append(f"  {tail} -> {head};")
    for n1, n2 in mg.edges():
        lines.append(f"  {n1} -> {n2} [dir=none];")
    lines.append("}")
    return "\n".join(lines)


def discretization_label(row):
    """Compact label: Q(n_bins) or H(n_bins, initial_bins)."""
    method = row["discretization_method"]
    n = int(row["n_bins"])
    if method == "hartemink":
        init = int(row["initial_bins"])
        return f"H({n}, {init})"
    return f"Q({n})"


# ---------------------------------------------------------------------------
# Grid search execution
# ---------------------------------------------------------------------------

def _count_combinations(param_grid):
    """Count total parameter combinations for a param_grid (dict or list of dicts)."""
    if isinstance(param_grid, list):
        return sum(
            len(list(itertools.product(*sg.values()))) for sg in param_grid
        )
    return len(list(itertools.product(*param_grid.values())))


def _precompute_hartemink(algo_configs, dataset):
    """Scan algo_configs for hartemink sub-grids and precompute discretizations.

    Returns a dict mapping (n_bins, initial_bins) -> discretized DataFrame.
    """
    df = dataset.to_dataframe()
    # Collect all (initial_bins, [n_bins_list]) groups
    groups = {}  # initial_bins -> set of n_bins
    for _name, _cls, param_grid, fixed_params in algo_configs:
        grids = param_grid if isinstance(param_grid, list) else [param_grid]
        for grid in grids:
            methods = grid.get("discretization_method", [fixed_params.get("discretization_method")])
            if "hartemink" not in methods:
                continue
            n_bins_list = grid.get("n_bins", [fixed_params.get("n_bins", 3)])
            initial_bins_list = grid.get("initial_bins", [fixed_params.get("initial_bins")])
            for ib in initial_bins_list:
                groups.setdefault(ib, set()).update(n_bins_list)

    if not groups:
        return {}

    precomputed = {}
    for initial_bins, n_bins_set in groups.items():
        target_bins = sorted(n_bins_set)
        results = hartemink_discretize_multi(df, target_bins, initial_bins=initial_bins)
        for nb, disc_df in results.items():
            precomputed[(nb, initial_bins)] = disc_df

    return precomputed


def run_grid_searches(algo_configs, dataset, ground_truth, metrics, objectives,
                      verbose=True, learn_method="learn_structure"):
    """Run grid search for all algorithm configurations.

    Automatically precomputes Hartemink discretizations shared across
    algorithms and parameter combinations to avoid redundant computation.

    Args:
        algo_configs: List of (name, algo_class, param_grid, fixed_params) tuples.
        dataset: Dataset instance.
        ground_truth: Golden Structure.
        metrics: List of MetricAdapter instances.
        objectives: Dict mapping metric name -> lower_is_better bool.
        verbose: If True, print detailed text progress. If False, show tqdm bars.
        learn_method: Method to call on the adapter ("learn_structure" or "learn_dag").

    Returns:
        Dict mapping algo name -> GridSearch instance (fitted).
    """
    # Precompute all Hartemink discretizations once
    hartemink_precomputed = _precompute_hartemink(algo_configs, dataset)

    grid_searches = {}
    use_bars = not verbose

    # Single progress bar: total = sum of all parameter combinations across algos
    total_combos = sum(_count_combinations(pg) for _, _, pg, _ in algo_configs)
    pbar = tqdm(total=total_combos, desc="Grid search", leave=True) if use_bars else None

    for name, algo_class, param_grid, fixed_params in algo_configs:
        if pbar is not None:
            pbar.set_postfix_str(name)

        if isinstance(param_grid, list):
            all_results = []
            for sub_grid in param_grid:
                sub_gs = GridSearch(
                    algorithm_class=algo_class,
                    param_grid=sub_grid,
                    dataset=dataset,
                    golden_structure=ground_truth,
                    metrics=metrics,
                    fixed_params=fixed_params if fixed_params else None,
                    objectives=objectives,
                    verbose=verbose,
                    learn_method=learn_method,
                    hartemink_precomputed=hartemink_precomputed,
                )
                sub_gs.run(_pbar=pbar)
                all_results.extend(sub_gs.results)
            gs = GridSearch(
                algorithm_class=algo_class,
                param_grid=param_grid[0],
                dataset=dataset,
                golden_structure=ground_truth,
                metrics=metrics,
                fixed_params=fixed_params if fixed_params else None,
                objectives=objectives,
                verbose=verbose,
                learn_method=learn_method,
                hartemink_precomputed=hartemink_precomputed,
            )
            gs.results = all_results
            gs._is_fitted = True
        else:
            gs = GridSearch(
                algorithm_class=algo_class,
                param_grid=param_grid,
                dataset=dataset,
                golden_structure=ground_truth,
                metrics=metrics,
                fixed_params=fixed_params if fixed_params else None,
                objectives=objectives,
                verbose=verbose,
                learn_method=learn_method,
                hartemink_precomputed=hartemink_precomputed,
            )
            gs.run(_pbar=pbar)

        grid_searches[name] = gs

    return grid_searches


# ---------------------------------------------------------------------------
# Per-algorithm visualisation
# ---------------------------------------------------------------------------

def plot_grid_search_results(name, gs, param_grid, metric_names, pareto_objectives):
    """Plot bar charts / heatmaps per metric + SHD vs F1 scatter with Pareto front."""
    df = gs.get_results_dataframe()
    param_cols = [c for c in df.columns if c not in metric_names + ["error"]]

    if isinstance(param_grid, list):
        df = df.dropna(subset=metric_names, how="all")
        df["discretization"] = df.apply(discretization_label, axis=1)
        order = sorted(df["discretization"].unique(),
                       key=lambda l: (0 if l.startswith("Q") else 1, l))
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"{name} : score par discrétisation  (Q=quantile, H=hartemink)",
                     fontsize=13, fontweight="bold")
        for ax, mn in zip(axes, metric_names):
            sns.barplot(data=df, x="discretization", y=mn, ax=ax, color="steelblue",
                        order=order)
            ax.set_title(mn)
            ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.show()

    elif len(param_cols) == 1:
        p = param_cols[0]
        df[p] = df[p].astype(str)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle(f"{name} : score par {p}", fontsize=13, fontweight="bold")
        for ax, mn in zip(axes, metric_names):
            sns.barplot(data=df, x=p, y=mn, ax=ax, color="steelblue")
            ax.set_title(mn)
        plt.tight_layout()
        plt.show()

    elif len(param_cols) == 2:
        p1, p2 = param_cols
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"{name} : {p1} x {p2}", fontsize=13, fontweight="bold")
        for ax, mn, cmap in zip(axes, metric_names, ["rocket_r", "viridis", "viridis"]):
            pivot = df.pivot_table(index=p1, columns=p2, values=mn, aggfunc="mean")
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                        cbar_kws={"shrink": 0.8})
            ax.set_title(mn)
        plt.tight_layout()
        plt.show()

    # Scatter SHD vs F1 + Pareto front
    valid = [r for r in gs.results if "SHD" in r.scores and "F1-Score" in r.scores]
    if valid:
        front = pareto_front(gs.results, pareto_objectives)
        front_ids = {id(r) for r in front}

        scatter_df = pd.DataFrame([
            {"SHD": r.scores["SHD"], "F1-Score": r.scores["F1-Score"],
             "Pareto": "Pareto front" if id(r) in front_ids else "Profile"}
            for r in valid
        ])

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=scatter_df[scatter_df["Pareto"] == "Profile"],
            x="SHD", y="F1-Score", color="#cccccc", s=50, ax=ax, label="Profiles"
        )
        sns.scatterplot(
            data=scatter_df[scatter_df["Pareto"] == "Pareto front"],
            x="SHD", y="F1-Score", color="#e74c3c", s=120, ax=ax,
            edgecolor="black", linewidth=1.5, label="Pareto front", zorder=5
        )
        if len(front) > 1:
            front_sorted = sorted(front, key=lambda r: r.scores["SHD"])
            ax.plot(
                [r.scores["SHD"] for r in front_sorted],
                [r.scores["F1-Score"] for r in front_sorted],
                color="#e74c3c", linestyle="--", alpha=0.5
            )
        ax.set_xlabel("SHD (lower is better)")
        ax.set_ylabel("F1-Score (higher is better)")
        ax.set_title(f"{name} : SHD vs F1-Score")
        ax.legend()
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def plot_summary_table(grid_searches, metric_names):
    """Build and return a summary DataFrame with best scores per algo."""
    rows = []
    for name, gs in grid_searches.items():
        row = {"Algorithm": name}
        for mn in metric_names:
            row[f"Best {mn}"] = gs.best_score(mn)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Algorithm")


# ---------------------------------------------------------------------------
# Pareto selection (single metric)
# ---------------------------------------------------------------------------

def select_best_profiles(grid_searches, pareto_objectives, rank_by):
    """Select one Pareto-optimal profile per algorithm.

    Args:
        grid_searches: Dict mapping algo name -> GridSearch.
        pareto_objectives: Dict mapping metric name -> lower_is_better.
        rank_by: Metric to rank by ("SHD" or "F1-Score").

    Returns:
        Dict mapping algo name -> best GridSearchResult (or None).
    """
    rank_lower_is_better = pareto_objectives.get(rank_by, True)
    selection = {}
    for name, gs in grid_searches.items():
        selection[name] = best_pareto(
            gs.results, pareto_objectives,
            rank_by=rank_by, rank_lower_is_better=rank_lower_is_better,
        )
    return selection


def print_selection(selection, rank_by):
    """Print the selected profiles."""
    lower = "min" if rank_by == "SHD" else "max"
    print(f"Selection (Pareto {lower} {rank_by}):")
    for name, r in selection.items():
        if r:
            print(f"  {name:<15} SHD={r.scores['SHD']:<6.1f} F1={r.scores['F1-Score']:<6.3f} params={r.params}")
        else:
            print(f"  {name:<15} No valid result")


# ---------------------------------------------------------------------------
# Comparison scatter
# ---------------------------------------------------------------------------

def plot_comparison_scatter(selection, algo_names, rank_by):
    """Scatter plot comparing all algos (one point per algo)."""
    palette = sns.color_palette("tab10", len(algo_names))
    lower = "min" if rank_by == "SHD" else "max"
    title = f"Pareto-optimal, {lower} {rank_by}"

    fig, ax = plt.subplots(figsize=(10, 7))
    for idx, name in enumerate(algo_names):
        r = selection.get(name)
        if r is None:
            continue
        params_str = ", ".join(f"{k}={v}" for k, v in r.params.items())
        ax.scatter(
            r.scores["SHD"], r.scores["F1-Score"],
            color=palette[idx], s=150, edgecolors="black", linewidths=1.5,
            label=f"{name} ({params_str})", zorder=5,
        )
    ax.set_xlabel("SHD (lower is better)")
    ax.set_ylabel("F1-Score (higher is better)")
    ax.set_title(title)
    ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Re-run best profiles
# ---------------------------------------------------------------------------

def run_best_profiles(selection, algo_configs_map, dataset, learn_method="learn_structure",
                      hartemink_precomputed=None):
    """Re-run the best profile for each algo and return learned structures.

    Args:
        hartemink_precomputed: Optional dict mapping (n_bins, initial_bins) -> discretized DataFrame.
            If provided, injects discretized_df into hartemink adapters to avoid recomputation.
    """
    structures = {}
    for name, r in selection.items():
        if r is None:
            continue
        algo_class, fixed_params = algo_configs_map[name]
        all_params = {**fixed_params, **r.params}
        # Inject pre-discretized data if available
        if hartemink_precomputed and all_params.get("discretization_method") == "hartemink":
            key = (all_params.get("n_bins"), all_params.get("initial_bins"))
            if key in hartemink_precomputed:
                all_params["discretized_df"] = hartemink_precomputed[key]
        algo = algo_class(**all_params)
        result_obj = getattr(algo, learn_method)(dataset)
        if learn_method == "learn_dag":
            structures[name] = dag_to_structure(result_obj)
        else:
            structures[name] = result_obj
    return structures


# ---------------------------------------------------------------------------
# CPDAG display
# ---------------------------------------------------------------------------

def plot_cpdags(structures, selection, feature_names):
    """Display the learned CPDAG for each algorithm."""
    for name, structure in structures.items():
        r = selection[name]
        params_str = ", ".join(f"{k}={v}" for k, v in r.params.items())
        print(f"\n{name} ({params_str}) — {structure.cpdag.sizeArcs()} arcs, {structure.cpdag.sizeEdges()} edges")
        gnb.showDot(cpdag_to_dot(structure, feature_names))


# ---------------------------------------------------------------------------
# Pairwise heatmaps
# ---------------------------------------------------------------------------

def plot_pairwise_heatmaps(structures, title_prefix, metrics, objectives, golden_structure=None):
    """Plot pairwise heatmaps for each metric.

    Rows = ref, columns = test.
    If golden_structure is provided, it is added as the first row.
    """
    names = list(structures.keys())
    n = len(names)

    for metric in metrics:
        row_names = names
        if golden_structure is not None:
            row_names = ["Golden BN"] + names

        matrix = np.zeros((len(row_names), n))
        for i, ref_name in enumerate(row_names):
            ref = golden_structure if ref_name == "Golden BN" else structures[ref_name]
            for j, test_name in enumerate(names):
                matrix[i, j] = metric.compute(ref=ref, test=structures[test_name])

        lower_better = objectives.get(metric.name(), True)
        cmap = "rocket_r" if lower_better else "viridis"

        matrix_df = pd.DataFrame(matrix, index=row_names, columns=names)
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(matrix_df, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                    cbar_kws={"shrink": 0.8}, linewidths=0.5, square=True)
        ax.set_title(f"{title_prefix} - Pairwise {metric.name()}")
        plt.tight_layout()
        plt.show()
