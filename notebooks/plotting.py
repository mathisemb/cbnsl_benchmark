"""Visualization functions for benchmark results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline.GridSearch import pareto_front


# ---------------------------------------------------------------------------
# Structure helpers
# ---------------------------------------------------------------------------

def cpdag_to_dot(structure, feature_names=None):
    """Convert a Structure's CPDAG to a labeled dot string."""
    cpdag = structure.cpdag
    lines = ["digraph {"]
    for node_id in sorted(cpdag.nodes()):
        label = feature_names[node_id] if feature_names else str(node_id)
        lines.append(f'  {node_id} [label="{label}"];')
    for tail, head in cpdag.arcs():
        lines.append(f"  {tail} -> {head};")
    for n1, n2 in cpdag.edges():
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
# Per-algorithm grid search plots
# ---------------------------------------------------------------------------

def _compact_disc_label(row):
    """Ultra-compact discretization label for heatmap cells: Q3, H4-20, etc."""
    method = row["discretization_method"]
    n = int(row["n_bins"])
    if method == "hartemink":
        init = int(row["initial_bins"])
        return f"H{n}-{init}"
    return f"Q{n}"


def _plot_best_over_discretizations(df, name, metric_names):
    """Heatmap of lambda1 x w_threshold_notears showing the best score across discretizations.

    Each cell is annotated with the score and the compact discretization label
    (e.g. Q3, H4) that achieved it.
    """
    df = df.copy()
    df["_disc_label"] = df.apply(_compact_disc_label, axis=1)

    cmaps = {"SHD": "rocket_r", "F1-Score": "viridis", "TPR": "viridis"}
    lower_better = {"SHD": True, "F1-Score": False, "TPR": False}

    fig, axes = plt.subplots(1, len(metric_names), figsize=(18, 5))
    fig.suptitle(f"{name} : best over discretizations",
                 fontsize=13, fontweight="bold")

    for ax, mn in zip(axes, metric_names):
        ascending = lower_better.get(mn, True)
        # For each (lambda1, w_threshold_notears), find the row with the best score
        best = (df.sort_values(mn, ascending=ascending)
                .drop_duplicates(subset=["lambda1", "w_threshold_notears"], keep="first"))

        score_pivot = best.pivot_table(
            index="lambda1", columns="w_threshold_notears", values=mn, aggfunc="first")
        label_pivot = best.pivot_table(
            index="lambda1", columns="w_threshold_notears", values="_disc_label",
            aggfunc="first")

        # Build annotation matrix: "score\nlabel"
        annot = score_pivot.copy().astype(object)
        for r in score_pivot.index:
            for c in score_pivot.columns:
                val = score_pivot.loc[r, c]
                lab = label_pivot.loc[r, c]
                if pd.notna(val):
                    annot.loc[r, c] = f"{val:.2f}\n{lab}"
                else:
                    annot.loc[r, c] = ""

        cmap = cmaps.get(mn, "viridis")
        sns.heatmap(score_pivot, annot=annot, fmt="", cmap=cmap, ax=ax,
                    cbar_kws={"shrink": 0.8})
        ax.set_title(mn)

    plt.tight_layout()
    plt.show()


def plot_grid_search_results(name, gs, param_grid, metric_names, pareto_objectives,
                             random_seeds=None):
    """Plot bar charts / heatmaps per metric + SHD vs F1 scatter with Pareto front.

    Args:
        name: Algorithm display name.
        gs: GridSearch instance (unified).
        param_grid: The param_grid used for this algo (dict or list of dicts).
        metric_names: List of metric names.
        pareto_objectives: Dict of objectives for Pareto front.
        random_seeds: List of seeds if stochastic algorithm, None otherwise.
    """
    df = gs.get_results_dataframe(name)
    results = gs.results.get(name, [])
    param_cols = [c for c in df.columns if c not in metric_names + ["error"]]

    if isinstance(param_grid, list):
        df = df.dropna(subset=metric_names, how="all")
        df["discretization"] = df.apply(discretization_label, axis=1)
        order = sorted(df["discretization"].unique(),
                       key=lambda l: (0 if l.startswith("Q") else 1, l))

        # Bar chart: scores by discretization (averaged over lambda1/w_threshold_notears)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"{name} : score par discrétisation",
                     fontsize=13, fontweight="bold")
        for ax, mn in zip(axes, metric_names):
            sns.barplot(data=df, x="discretization", y=mn, ax=ax, color="steelblue",
                        order=order)
            ax.set_title(mn)
            ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.show()

        # Heatmap: best score over all discretizations for each (lambda1, w_threshold_notears)
        # Each cell is annotated with the score and the discretization that achieved it
        if "lambda1" in df.columns and "w_threshold_notears" in df.columns:
            _plot_best_over_discretizations(df, name, metric_names)

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
            if pivot.empty or pivot.isna().all().all():
                ax.set_title(f"{mn} (N/A)")
            else:
                sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                            cbar_kws={"shrink": 0.8})
                ax.set_title(mn)
        plt.tight_layout()
        plt.show()

    # Scatter SHD vs F1 + Pareto front
    valid = [r for r in results if "SHD" in r.scores and "F1-Score" in r.scores]
    if valid:
        front_idx = set(pareto_front([r.scores for r in results], pareto_objectives))

        scatter_df = pd.DataFrame([
            {"SHD": r.scores["SHD"], "F1-Score": r.scores["F1-Score"],
             "Pareto": "Pareto front" if i in front_idx else "Profile"}
            for i, r in enumerate(results) if "SHD" in r.scores and "F1-Score" in r.scores
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
        front_results = [results[i] for i in front_idx]
        if len(front_results) > 1:
            front_sorted = sorted(front_results, key=lambda r: r.scores["SHD"])
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
# Best scores comparison scatter
# ---------------------------------------------------------------------------

def plot_best_scores(scores_by_algo, params_by_algo, seed_counts=None):
    """Scatter plot comparing all algos (one point per algo) on SHD vs F1-Score.

    Args:
        scores_by_algo: {algo_name: {metric_name: value}}.
        params_by_algo: {algo_name: {param_name: value}}.
        seed_counts: {algo_name: n_seeds} for stochastic algorithms.
    """
    seed_counts = seed_counts or {}
    algo_names = list(scores_by_algo.keys())
    palette = sns.color_palette("tab10", len(algo_names))

    fig, ax = plt.subplots(figsize=(10, 7))
    for idx, name in enumerate(algo_names):
        scores = scores_by_algo[name]
        params = params_by_algo.get(name, {})
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        label = f"{name} ({params_str})"
        if name in seed_counts:
            label += f" [moy. {seed_counts[name]} seeds]"
        ax.scatter(
            scores["SHD"], scores["F1-Score"],
            color=palette[idx], s=150, edgecolors="black", linewidths=1.5,
            label=label, zorder=5,
        )
    ax.set_xlabel("SHD (lower is better)")
    ax.set_ylabel("F1-Score (higher is better)")
    ax.set_title("Best profiles : SHD vs F1-Score")
    ax.legend(
        fontsize=8, bbox_to_anchor=(0.5, -0.15), loc="upper center",
        ncol=1,
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CPDAG display
# ---------------------------------------------------------------------------

def plot_cpdags(structures, params_by_algo, feature_names):
    """Display the learned CPDAG for each algorithm.

    For stochastic algorithms, ``structures[name]`` is a list of Structures
    (one per seed) and all are displayed.

    Args:
        structures: {algo_name: Structure | List[Structure]}.
        params_by_algo: {algo_name: {param_name: value}}.
        feature_names: List of variable names.
    """
    import pyagrum.lib.notebook as gnb
    for name, struct_or_list in structures.items():
        params = params_by_algo.get(name, {})
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())

        if isinstance(struct_or_list, list):
            # Stochastic algorithm: display every seed structure
            for i, structure in enumerate(struct_or_list):
                print(
                    f"\n{name} seed {i} ({params_str}) — "
                    f"{structure.cpdag.sizeArcs()} arcs, "
                    f"{structure.cpdag.sizeEdges()} edges"
                )
                gnb.showDot(cpdag_to_dot(structure, feature_names))
        else:
            structure = struct_or_list
            print(
                f"\n{name} ({params_str}) — "
                f"{structure.cpdag.sizeArcs()} arcs, "
                f"{structure.cpdag.sizeEdges()} edges"
            )
            gnb.showDot(cpdag_to_dot(structure, feature_names))


# ---------------------------------------------------------------------------
# Pairwise heatmaps
# ---------------------------------------------------------------------------

def _compute_metric_value(metric, ref, test):
    """Compute a metric between ref and test, averaging over seeds if needed.

    If ref or test is a list of structures (stochastic algo), the metric is
    computed for every combination and averaged.
    """
    ref_list = ref if isinstance(ref, list) else [ref]
    test_list = test if isinstance(test, list) else [test]

    values = [
        metric.compute(ref=r, test=t)
        for r in ref_list
        for t in test_list
    ]
    return float(np.mean(values))


def plot_pairwise_heatmaps(structures, title_prefix, metrics, objectives, golden_structure=None):
    """Plot pairwise heatmaps for each metric.

    For stochastic algorithms whose ``structures[name]`` is a list, the
    metric is averaged over all seed structures.
    """
    names = list(structures.keys())
    n = len(names)

    # Build display names: append "[moy. N seeds]" for seeded algos (only if >1 seed)
    col_labels = []
    for name in names:
        s = structures[name]
        if isinstance(s, list) and len(s) > 1:
            col_labels.append(f"{name}\n[moy. {len(s)} seeds]")
        else:
            col_labels.append(name)

    for metric in metrics:
        row_labels = list(col_labels)
        if golden_structure is not None:
            row_labels = ["Golden BN"] + row_labels

        row_names = list(names)
        if golden_structure is not None:
            row_names = ["Golden BN"] + row_names

        matrix = np.zeros((len(row_names), n))
        for i, ref_name in enumerate(row_names):
            ref = golden_structure if ref_name == "Golden BN" else structures[ref_name]
            for j, test_name in enumerate(names):
                matrix[i, j] = _compute_metric_value(metric, ref, structures[test_name])

        lower_better = objectives.get(metric.name(), True)
        cmap = "rocket_r" if lower_better else "viridis"

        matrix_df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(matrix_df, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                    cbar_kws={"shrink": 0.8}, linewidths=0.5, square=True)
        ax.set_title(f"{title_prefix} - Pairwise {metric.name()}")
        plt.tight_layout()
        plt.show()
