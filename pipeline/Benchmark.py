"""
High-level benchmark facade for structure learning algorithm comparison.

Usage in notebooks::

    from pipeline.Benchmark import Benchmark
    bench = Benchmark.sachs()
    bench.run()
    bench.summary()
    bench.plot_grid_search()
    bench.plot_best_scores()
    bench.plot_structures()
    bench.plot_pairwise_heatmaps()
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import seaborn as sns

from algorithms import ALL_ALGORITHMS
from metrics import ALL_METRICS, OBJECTIVES
from pipeline.Dataset import Dataset
from pipeline.Structure import Structure
from pipeline.GridSearch import GridSearch


class Benchmark:
    """Facade for the complete benchmarking workflow.

    Supports two modes:

    - **Grid search** (``run()``): finds the best hyperparameters for every
      registered algorithm, selects Pareto-optimal profiles, and re-runs
      them to obtain the learned structures.
    - **Fixed params** (``run_fixed(configs)``): runs algorithms with
      user-specified hyperparameters (no search).

    After either mode, call ``summary()``, ``plot_*()`` to analyse results.
    """

    PARETO_OBJECTIVES = {"SHD": True, "F1-Score": False}

    def __init__(
        self,
        dataset: Dataset,
        golden_structure: Structure,
        rank_by: str = "F1-Score",
    ):
        sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.1)

        self.dataset = dataset
        self.golden_structure = golden_structure
        self.rank_by = rank_by
        self.metrics = list(ALL_METRICS)
        self.metric_names = [m.name() for m in self.metrics]
        self.objectives = dict(OBJECTIVES)

        # Internal state (populated by run / run_fixed)
        self._gs: Optional[GridSearch] = None
        self._selection: Optional[dict] = None
        # For stochastic algorithms, _structures[name] is a List[Structure]
        self._structures: Optional[dict] = None
        self._scores: Optional[Dict[str, Dict[str, float]]] = None
        self._params: Optional[Dict[str, Dict[str, Any]]] = None
        # Number of seeds per algorithm (only for stochastic algos)
        self._seed_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def sachs(cls, rank_by: str = "F1-Score") -> "Benchmark":
        """Load Sachs protein signaling dataset with BN18 ground truth."""
        from data.sachs.load_ground_truth import load_sachs_ground_truth

        sachs_path = Path(__file__).parent.parent / "data" / "sachs"
        sachs_data = pd.read_csv(
            sachs_path / "sachs_observational.csv", sep="\t"
        )
        golden = load_sachs_ground_truth(version="bn18", as_structure=True)

        dataset = Dataset(
            sachs_data.to_numpy(),
            name="sachs_observational",
            feature_names=list(sachs_data.columns),
        )

        bench = cls(dataset, golden, rank_by=rank_by)
        bench._register_all_algorithms()
        return bench

    @classmethod
    def synthetic_cbn_unif_gauss(
        cls,
        dag,
        n_samples: int = 2000,
        seed: int = 42,
        rank_by: str = "F1-Score",
        var_names: Optional[List[str]] = None,
        **cbn_kwargs,
    ) -> "Benchmark":
        """Generate synthetic data from a known DAG structure.

        Args:
            dag: ``gum.DAG`` defining the ground-truth structure.
            n_samples: Number of samples to generate.
            seed: Random seed.
            rank_by: Metric used to rank Pareto-optimal profiles.
            var_names: Variable names (default ``X0, X1, ...``).
            **cbn_kwargs: Forwarded to ``create_simple_cbn``
                (e.g. ``copula_correlation``).
        """
        from data.generators import create_simple_cbn, generate_from_cbn

        if var_names is None:
            var_names = [f"X{i}" for i in range(dag.size())]

        cbn = create_simple_cbn(dag, var_names=var_names, **cbn_kwargs)
        dataset, golden = generate_from_cbn(cbn, n_samples=n_samples, seed=seed)

        dataset = Dataset(
            dataset.data,
            name=f"synthetic_{dag.size()}nodes",
            feature_names=var_names,
        )

        bench = cls(dataset, golden, rank_by=rank_by)
        bench._register_all_algorithms()
        return bench

    @classmethod
    def synthetic_nongausslinSEM(
        cls,
        dag,
        n_samples: int = 2000,
        seed: int = 42,
        rank_by: str = "F1-Score",
        var_names: Optional[List[str]] = None,
        noise_type: str = "laplace",
        weight_range: tuple = (0.5, 2.0),
    ) -> "Benchmark":
        """Generate synthetic data from a known DAG using a linear SEM with non-Gaussian noise.

        Each variable is generated as:
        ``X_i = sum(w_ji * X_j for j in parents(i)) + e_i``
        where ``e_i`` is drawn from a non-Gaussian distribution.

        Args:
            dag: ``gum.DAG`` defining the ground-truth structure.
            n_samples: Number of samples to generate.
            seed: Random seed.
            rank_by: Metric used to rank Pareto-optimal profiles.
            var_names: Variable names (default ``X0, X1, ...``).
            noise_type: Noise distribution — ``"laplace"``, ``"uniform"``,
                or ``"exp"`` (centred exponential).
            weight_range: ``(low, high)`` for uniform sampling of
                absolute edge weights (sign is random).
        """
        import numpy as np
        import pyagrum as gum
        from pipeline.Structure import Structure, dag_as_a_structure

        rng = np.random.default_rng(seed)
        n_vars = dag.size()

        if var_names is None:
            var_names = [f"X{i}" for i in range(dag.size())]

        # Topological order
        topo = dag.topologicalOrder()

        # Sample edge weights
        weights = {}  # (parent, child) -> weight
        for node in topo:
            for parent in dag.parents(node):
                w = rng.uniform(*weight_range)
                sign = rng.choice([-1, 1])
                weights[(parent, node)] = sign * w

        # Generate noise
        if noise_type == "laplace":
            noise = rng.laplace(loc=0.0, scale=1.0, size=(n_samples, n_vars))
        elif noise_type == "uniform":
            noise = rng.uniform(-1.0, 1.0, size=(n_samples, n_vars))
        elif noise_type == "exp":
            noise = rng.exponential(scale=1.0, size=(n_samples, n_vars))
            noise -= noise.mean(axis=0)  # centre
        else:
            raise ValueError(
                f"Unknown noise_type '{noise_type}'. "
                "Choose from 'laplace', 'uniform', 'exp'."
            )

        # Generate data following topological order
        data = np.zeros((n_samples, n_vars))
        for node in topo:
            data[:, node] = noise[:, node]
            for parent in dag.parents(node):
                data[:, node] += weights[(parent, node)] * data[:, parent]

        # Golden structure (CPDAG)
        bn = gum.BayesNet()
        for node_id in dag.nodes():
            bn.add(gum.LabelizedVariable(f"X{node_id}", f"X{node_id}", 2))
        for node_id in dag.nodes():
            for child_id in dag.children(node_id):
                bn.addArc(node_id, child_id)
        essential_graph = gum.EssentialGraph(bn)
        golden = Structure(essential_graph.pdag())

        dataset = Dataset(
            data,
            name=f"synthetic_nongausslinSEM_{n_vars}nodes",
            feature_names=var_names,
        )

        bench = cls(dataset, golden, rank_by=rank_by)
        bench._register_all_algorithms()
        return bench

    # ------------------------------------------------------------------
    # Algorithm registration
    # ------------------------------------------------------------------

    def _register_all_algorithms(self) -> None:
        """Register all algorithms from the global registry."""
        self._gs = GridSearch(
            dataset=self.dataset,
            golden_structure=self.golden_structure,
            metrics=self.metrics,
            objectives=self.objectives,
        )
        for name, (algo_cls, fixed_params, random_seeds) in ALL_ALGORITHMS.items():
            self._gs.add(name, algo_cls, fixed_params=fixed_params,
                         random_seeds=random_seeds)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self) -> "Benchmark":
        """Run grid search for all registered algorithms.

        After completion, the best Pareto-optimal profiles are selected
        (according to ``rank_by``) and re-run to obtain learned structures.
        """
        if self._gs is None:
            self._register_all_algorithms()

        self._gs.run()
        self._selection = self._gs.select_best(self.rank_by)

        # The grid search stores params and scores but not the learned Structure
        # objects, to save memory across all combinations. Re-run the selected
        # Pareto-optimal profiles to obtain structures needed for visualization.
        # For stochastic algorithms, all seed structures are returned as a list.
        print(f"\nRe-running best profiles (rank_by={self.rank_by})...")
        self._structures = self._gs.rerun_best(self.rank_by)

        # Build scores and params dicts from selection
        # (scores are already seed-averaged for stochastic algorithms)
        self._scores = {}
        self._params = {}
        self._seed_counts = {}
        for name, r in self._selection.items():
            if r is not None:
                self._scores[name] = r.scores
                self._params[name] = r.params
        for n, _, _, _, seeds in self._gs._configs:
            if seeds is not None:
                self._seed_counts[n] = len(seeds)

        print("Done.\n")
        return self

    def run_fixed(self, configs: Dict[str, Dict[str, Any]]) -> "Benchmark":
        """Run algorithms with fixed hyperparameters (no grid search).

        Args:
            configs: ``{algo_display_name: {param: value, ...}}``.
                Display names must match keys in ``ALL_ALGORITHMS``
                (e.g. ``"CPC v1"``, ``"MIIC"``, ``"LiNGAM"``).

        Example::

            bench.run_fixed({
                "LiNGAM": {"threshold_lingam": 0.2},
                "NOTEARS": {"lambda1": 0.15, "w_threshold_notears": 0.3},
            })
        """
        self._gs = None
        self._selection = None
        self._structures = {}
        self._scores = {}
        self._params = configs

        for name, params in configs.items():
            if name not in ALL_ALGORITHMS:
                available = ", ".join(ALL_ALGORITHMS.keys())
                raise ValueError(
                    f"Unknown algorithm '{name}'. Available: {available}"
                )

            algo_cls, default_fixed, _ = ALL_ALGORITHMS[name]
            all_params = {**default_fixed, **params}

            try:
                algo = algo_cls(**all_params)
                structure = algo.learn_structure(self.dataset)
                self._structures[name] = structure

                scores = {}
                for metric in self.metrics:
                    scores[metric.name()] = metric.compute(
                        ref=self.golden_structure, test=structure
                    )
                self._scores[name] = scores
                print(f"  {name}: OK")
            except Exception as e:
                print(f"  {name}: FAILED ({e})")

        print("Done.\n")
        return self

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def show_golden(self) -> None:
        """Display the golden CPDAG in the notebook."""
        import pyagrum.lib.notebook as gnb
        from notebooks.plotting import cpdag_to_dot

        s = self.golden_structure
        print(
            f"Golden CPDAG: {s.cpdag.sizeArcs()} arcs, "
            f"{s.cpdag.sizeEdges()} edges"
        )
        gnb.showDot(cpdag_to_dot(s, self.dataset.feature_names))

    def plot_grid_search(self) -> None:
        """Plot per-algorithm grid search results (bars/heatmaps + Pareto)."""
        if self._gs is None:
            print("No grid search results (run_fixed was used).")
            return
        self._gs.plot()

    def plot_best_scores(self) -> None:
        """Scatter plot comparing best scores of each algorithm (SHD vs F1)."""
        if self._scores is None:
            raise RuntimeError("Call run() or run_fixed() first.")
        from notebooks.plotting import plot_best_scores

        plot_best_scores(self._scores, self._params, seed_counts=self._seed_counts)

    def plot_structures(self) -> None:
        """Display learned CPDAGs for each algorithm.

        For stochastic algorithms (e.g. LiNGAM), all seed structures are shown.
        """
        if self._structures is None:
            raise RuntimeError("Call run() or run_fixed() first.")
        from notebooks.plotting import plot_cpdags

        plot_cpdags(self._structures, self._params, self.dataset.feature_names)

    def plot_pairwise_heatmaps(self) -> None:
        """Plot pairwise metric heatmaps between all learned structures."""
        if self._structures is None:
            raise RuntimeError("Call run() or run_fixed() first.")
        from notebooks.plotting import plot_pairwise_heatmaps

        title = (
            f"Best profiles (rank_by={self.rank_by})"
            if self._selection
            else "Fixed hyperparameters"
        )
        plot_pairwise_heatmaps(
            self._structures,
            title,
            self.metrics,
            self.objectives,
            golden_structure=self.golden_structure,
        )

    def __repr__(self) -> str:
        status = (
            "ready"
            if self._structures is None
            else f"{len(self._structures)} algorithms"
        )
        return f"Benchmark(dataset={self.dataset.name!r}, {status})"
