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


def cpdag_diff_dot(ref_structure, test_structure, feature_names=None):
    """Build a dot string showing the diff between a reference and a learned CPDAG.

    Colour code:
      - **green**: arc/edge present in both (correct)
      - **red**: arc/edge in *test* only (added / false positive)
      - **orange**: arc present in both but inverted in *test*
      - **grey dashed**: arc/edge in *ref* only (missing / false negative)
    """
    ref = ref_structure.cpdag
    test = test_structure.cpdag

    ref_arcs = set(ref.arcs())
    test_arcs = set(test.arcs())
    ref_edges = {frozenset(e) for e in ref.edges()}
    test_edges = {frozenset(e) for e in test.edges()}

    all_nodes = set(ref.nodes()) | set(test.nodes())

    lines = ["digraph {"]
    for node_id in sorted(all_nodes):
        label = feature_names[node_id] if feature_names else str(node_id)
        lines.append(f'  {node_id} [label="{label}"];')

    # --- arcs ---
    for t, h in test_arcs:
        if (t, h) in ref_arcs:
            # correct arc
            lines.append(f'  {t} -> {h} [color="forestgreen", penwidth=2];')
        elif (h, t) in ref_arcs:
            # inverted arc
            lines.append(f'  {t} -> {h} [color="orange", penwidth=2];')
        else:
            # false positive arc
            lines.append(f'  {t} -> {h} [color="red", penwidth=2];')

    for t, h in ref_arcs:
        if (t, h) not in test_arcs and (h, t) not in test_arcs:
            # check if it appears as an edge in test
            if frozenset((t, h)) not in test_edges:
                # missing arc
                lines.append(
                    f'  {t} -> {h} [color="grey", style=dashed, penwidth=1];'
                )

    # --- edges ---
    for e in test_edges:
        n1, n2 = sorted(e)
        if e in ref_edges:
            lines.append(
                f'  {n1} -> {n2} [dir=none, color="forestgreen", penwidth=2];'
            )
        else:
            lines.append(
                f'  {n1} -> {n2} [dir=none, color="red", penwidth=2];'
            )

    for e in ref_edges:
        if e not in test_edges:
            n1, n2 = sorted(e)
            # check if it appears as an arc in test
            if (n1, n2) not in test_arcs and (n2, n1) not in test_arcs:
                lines.append(
                    f'  {n1} -> {n2} [dir=none, color="grey", style=dashed, penwidth=1];'
                )

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
                             random_seeds=None, compare_mode="cpdag"):
    """Plot bar charts / heatmaps per metric + SHD vs F1 scatter with Pareto front.

    Args:
        name: Algorithm display name.
        gs: GridSearch instance (unified).
        param_grid: The param_grid used for this algo (dict or list of dicts).
        metric_names: List of metric names.
        pareto_objectives: Dict of objectives for Pareto front.
        random_seeds: List of seeds if stochastic algorithm, None otherwise.
        compare_mode: ``"cpdag"`` or ``"skeleton"`` — which score set to use.
    """
    df = gs.get_results_dataframe(name, compare_mode=compare_mode)
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
            if mn not in df.columns:
                ax.set_title(f"{mn} (N/A)")
                continue
            sns.barplot(data=df, x=p, y=mn, ax=ax, color="steelblue")
            ax.set_title(mn)
        plt.tight_layout()
        plt.show()

    elif len(param_cols) == 2:
        p1, p2 = param_cols
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"{name} : {p1} x {p2}", fontsize=13, fontweight="bold")
        for ax, mn, cmap in zip(axes, metric_names, ["rocket_r", "viridis", "viridis"]):
            if mn not in df.columns:
                ax.set_title(f"{mn} (N/A)")
                continue
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
    _get = lambda r: r.scores_skeleton if compare_mode == "skeleton" else r.scores
    valid = [r for r in results if "SHD" in _get(r) and "F1-Score" in _get(r)]
    if valid:
        scores_list = [_get(r) for r in results]
        front_idx = set(pareto_front(scores_list, pareto_objectives))

        scatter_df = pd.DataFrame([
            {"SHD": _get(r)["SHD"], "F1-Score": _get(r)["F1-Score"],
             "Pareto": "Pareto front" if i in front_idx else "Profile"}
            for i, r in enumerate(results) if "SHD" in _get(r) and "F1-Score" in _get(r)
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
            front_sorted = sorted(front_results, key=lambda r: _get(r)["SHD"])
            ax.plot(
                [_get(r)["SHD"] for r in front_sorted],
                [_get(r)["F1-Score"] for r in front_sorted],
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

def plot_best_scores(scores_by_algo, params_by_algo, seed_counts=None,
                     compare_mode="cpdag"):
    """Scatter plot comparing all algos (one point per algo) on SHD vs F1-Score.

    Args:
        scores_by_algo: {algo_name: {metric_name: value}}.
        params_by_algo: {algo_name: {param_name: value}}.
        seed_counts: {algo_name: n_seeds} for stochastic algorithms.
    """
    from collections import defaultdict

    seed_counts = seed_counts or {}
    algo_names = list(scores_by_algo.keys())
    palette = sns.color_palette("tab10", len(algo_names))

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
    sns.set_theme(style="whitegrid", font_scale=1.6,
                  rc={"text.usetex": True, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(7, 5))

    # Group algos by (x, y) position to detect overlaps.
    # Round to avoid floating-point mismatches.
    groups = defaultdict(list)
    for idx, name in enumerate(algo_names):
        scores = scores_by_algo[name]
        x, y = scores["SHD"], scores["F1-Score"]
        key = (round(float(x), 6), round(float(y), 6))
        groups[key].append((idx, name))

    # Compute marker sizes: overlapping points are drawn as concentric
    # circles (largest first) so every colour is visible.
    marker_sizes = {}
    for (x, y), members in groups.items():
        n = len(members)
        if n == 1:
            marker_sizes[members[0][0]] = 150
        else:
            for rank, (idx, _name) in enumerate(members):
                marker_sizes[idx] = 150 + (n - 1 - rank) * 80

    # Draw in decreasing size order so smaller circles appear on top.
    draw_order = sorted(range(len(algo_names)),
                        key=lambda i: -marker_sizes[i])
    for idx in draw_order:
        name = algo_names[idx]
        scores = scores_by_algo[name]
        params = params_by_algo.get(name, {})
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        x, y = scores["SHD"], scores["F1-Score"]
        label = f"{name} ({params_str})"
        ax.scatter(
            x, y,
            color=palette[idx], s=marker_sizes[idx],
            edgecolors="none",
            label=label, zorder=5, alpha=0.85,
        )

    # Annotate: single points get a simple label, overlapping points get a
    # grouped text block listing all algo names line by line.
    for (x, y), members in groups.items():
        if len(members) == 1:
            ax.annotate(
                members[0][1], (x, y), fontsize=11,
                textcoords="offset points", xytext=(8, 8), zorder=6,
            )
        else:
            block = "\n".join(name for _, name in members)
            ax.annotate(
                block, (x, y), fontsize=11,
                textcoords="offset points", xytext=(12, 12), zorder=6,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="gray", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
            )

    mode_label = "CPDAGs" if compare_mode == "cpdag" else "skeletons"
    ax.set_xlabel(f"SHD (on {mode_label})")
    ax.set_ylabel(f"F1 (on {mode_label})")
    plt.tight_layout()
    plt.show()
    return fig  # pour pouvoir la sauvegarder (fig.savefig(...))


# ---------------------------------------------------------------------------
# CPDAG display
# ---------------------------------------------------------------------------

def _save_dot(dot_string, filepath):
    """Render a DOT string to a file (PDF/PNG/SVG based on extension)."""
    import graphviz
    from pathlib import Path

    p = Path(filepath)
    fmt = p.suffix.lstrip(".")
    src = graphviz.Source(dot_string)
    src.render(outfile=str(p), format=fmt, cleanup=True)


def plot_cpdags(structures, params_by_algo, feature_names,
                golden_structure=None, save_dir=None):
    """Display the learned CPDAG for each algorithm, optionally with golden BN and diff.

    For stochastic algorithms, ``structures[name]`` is a list of Structures
    (one per seed) and all are displayed.

    When *golden_structure* is provided, each learned structure is shown
    side-by-side with the golden CPDAG and a colour-coded diff
    (green = correct, red = extra, orange = inverted, grey dashed = missing).

    Args:
        structures: {algo_name: Structure | List[Structure]}.
        params_by_algo: {algo_name: {param_name: value}}.
        feature_names: List of variable names.
        golden_structure: Optional golden Structure for comparison.
        save_dir: If given, save each graph as PDF in this directory
            instead of displaying inline.
    """
    import pyagrum.lib.notebook as gnb
    from pathlib import Path

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

    golden_dot = (cpdag_to_dot(golden_structure, feature_names)
                  if golden_structure is not None else None)

    def _show_structure(structure, title, file_prefix=None):
        if save_dir is not None:
            learned_dot = cpdag_to_dot(structure, feature_names)
            _save_dot(learned_dot, save_path / f"{file_prefix}_learned.pdf")
            if golden_structure is not None:
                diff_dot = cpdag_diff_dot(golden_structure, structure, feature_names)
                _save_dot(diff_dot, save_path / f"{file_prefix}_diff.pdf")
            print(f"  Saved {file_prefix}")
        elif golden_structure is not None:
            diff_dot = cpdag_diff_dot(golden_structure, structure, feature_names)
            gnb.sideBySide(
                gnb.getDot(cpdag_to_dot(structure, feature_names)),
                gnb.getDot(golden_dot),
                gnb.getDot(diff_dot),
                captions=[title, "Golden BN", "Diff (ref=golden)"],
            )
        else:
            print(title)
            gnb.showDot(cpdag_to_dot(structure, feature_names))

    if save_dir is not None and golden_structure is not None:
        _save_dot(golden_dot, Path(save_dir) / "golden.pdf")
        print("  Saved golden")

    for name, struct_or_list in structures.items():
        params = params_by_algo.get(name, {})
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())

        if isinstance(struct_or_list, list):
            for i, structure in enumerate(struct_or_list):
                title = (
                    f"{name} seed {i} ({params_str}) — "
                    f"{structure.cpdag.sizeArcs()} arcs, "
                    f"{structure.cpdag.sizeEdges()} edges"
                )
                safe_name = name.replace(" ", "_").replace("+", "_")
                _show_structure(structure, title,
                                file_prefix=f"{safe_name}_seed{i}")
        else:
            structure = struct_or_list
            title = (
                f"{name} ({params_str}) — "
                f"{structure.cpdag.sizeArcs()} arcs, "
                f"{structure.cpdag.sizeEdges()} edges"
            )
            safe_name = name.replace(" ", "_").replace("+", "_")
            _show_structure(structure, title, file_prefix=safe_name)


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


def _to_skeleton(struct_or_list):
    """Convert a Structure (or list of Structures) to skeleton form."""
    if isinstance(struct_or_list, list):
        return [s.skeleton() for s in struct_or_list]
    return struct_or_list.skeleton()


def plot_pairwise_heatmaps(structures, title_prefix, metrics, objectives,
                           golden_structure=None, compare_mode="cpdag"):
    """Plot pairwise heatmaps for each metric.

    For stochastic algorithms whose ``structures[name]`` is a list, the
    metric is averaged over all seed structures.

    Args:
        compare_mode: ``"cpdag"`` or ``"skeleton"`` — if skeleton, structures
            are converted to skeletons before computing metrics.
    """
    # Convert to skeletons if needed
    if compare_mode == "skeleton":
        structures = {name: _to_skeleton(s) for name, s in structures.items()}
        if golden_structure is not None:
            golden_structure = golden_structure.skeleton()

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
