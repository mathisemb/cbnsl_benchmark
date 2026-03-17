"""
High-level benchmark facade for structure learning algorithm comparison.

Usage::

    bench = Benchmark.sachs()
    bench.run()                         # runs grid search, saves to results/
    bench.plot_grid_search()
    bench.plot_best_scores()
    bench.plot_structures()
    bench.plot_pairwise_heatmaps()

    # Or reload from disk:
    bench = Benchmark.load("results/sachs_raw__11vars_853s__2026-03-17_15h42")
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns

from algorithms import ALL_ALGORITHMS
from metrics import ALL_METRICS, OBJECTIVES
from pipeline.Dataset import Dataset
from pipeline.Structure import Structure
from pipeline.GridSearch import GridSearch, GridSearchResult
from scaling.io import save_structure, load_structure


class Benchmark:
    """Facade for the complete benchmarking workflow.

    Supports two modes:

    - **Grid search** (``run()``): finds the best hyperparameters for every
      registered algorithm and selects Pareto-optimal profiles.
    - **Fixed params** (``run_fixed(configs)``): runs algorithms with
      user-specified hyperparameters (no search).

    After either mode, call ``plot_*()`` to analyse results.
    Results are saved automatically and can be reloaded with ``Benchmark.load()``.
    """

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
        # Selections and scores for both compare modes
        self._selection_cpdag: Optional[dict] = None
        self._selection_skeleton: Optional[dict] = None
        # For stochastic algorithms, _structures[name] is a List[Structure]
        self._structures: Optional[dict] = None
        self._scores_cpdag: Optional[Dict[str, Dict[str, float]]] = None
        self._scores_skeleton: Optional[Dict[str, Dict[str, float]]] = None
        self._params_cpdag: Optional[Dict[str, Dict[str, Any]]] = None
        self._params_skeleton: Optional[Dict[str, Dict[str, Any]]] = None
        # Number of seeds per algorithm (only for stochastic algos)
        self._seed_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    _SACHS_VARIANTS = {
        "raw": "sachs_observational.csv",
        "log": "log_sachs_observational.csv",
        "preprocessed": "sachs_observational_preprocessed.csv",
    }

    @classmethod
    def sachs(
        cls,
        variant: str = "raw",
        rank_by: str = "F1-Score",
        repetition_nb: int = 1,
    ) -> "Benchmark":
        """Load Sachs protein signaling dataset with BN18 ground truth.

        Args:
            variant: ``"raw"``, ``"log"``, or ``"preprocessed"``.
            rank_by: Metric used to rank Pareto-optimal profiles.
            repetition_nb: Tile the dataset this many times (for sample-size experiments).
        """
        from data.sachs.load_ground_truth import load_sachs_ground_truth

        if variant not in cls._SACHS_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Choose from {list(cls._SACHS_VARIANTS.keys())}"
            )

        sachs_path = Path(__file__).parent.parent / "data" / "sachs"
        sachs_data = pd.read_csv(sachs_path / cls._SACHS_VARIANTS[variant], sep="\t")
        golden = load_sachs_ground_truth(version="bn18", as_structure=True)

        data = sachs_data.to_numpy()
        if repetition_nb > 1:
            data = np.tile(data, (repetition_nb, 1))

        n_samples = data.shape[0]
        n_vars = data.shape[1]
        rep_tag = f"_x{repetition_nb}" if repetition_nb > 1 else ""
        dataset = Dataset(
            data,
            name=f"sachs_{variant}{rep_tag}__{n_vars}vars_{n_samples}s",
            feature_names=list(sachs_data.columns),
        )

        bench = cls(dataset, golden, rank_by=rank_by)
        bench._register_all_algorithms()
        return bench

    # Alias (used in report_comparison.ipynb)
    @classmethod
    def preprocessed_sachs(cls, **kw) -> "Benchmark":
        return cls.sachs(variant="preprocessed", **kw)

    @classmethod
    def synthetic_cbn(
        cls,
        dag,
        n_samples: int = 2000,
        seed: int = 42,
        rank_by: str = "F1-Score",
        var_names: Optional[List[str]] = None,
        marginal_type: str = "Uniform",
        lcc_types: str = "NormalCopula",
    ) -> "Benchmark":
        """Generate synthetic data from a CBN built on the given DAG.

        Args:
            dag: ``gum.DAG`` defining the ground-truth structure.
            n_samples: Number of samples to generate.
            seed: Random seed.
            rank_by: Metric used to rank Pareto-optimal profiles.
            var_names: Variable names (default ``X0, X1, ...``).
            marginal_type: ``"Uniform"``, ``"Normal"``, ``"Exponential"``.
            lcc_types: ``"NormalCopula"``, ``"ClaytonCopula"``,
                ``"MixtureCopula"``, etc.
        """
        from data.generators import create_default_cbn, generate_from_cbn

        if var_names is None:
            var_names = [f"X{i}" for i in range(dag.size())]

        cbn = create_default_cbn(dag, var_names=var_names,
                                 marginal_type=marginal_type, lcc_types=lcc_types)
        dataset, golden = generate_from_cbn(cbn, n_samples=n_samples, seed=seed)
        dataset = Dataset(dataset.data,
                          name=f"cbn_{marginal_type}_{lcc_types}__{dag.size()}vars_{n_samples}s",
                          feature_names=var_names)

        bench = cls(dataset, golden, rank_by=rank_by)
        bench._register_all_algorithms()
        return bench

    @classmethod
    def synthetic_sem(
        cls,
        dag,
        n_samples: int = 2000,
        seed: int = 42,
        rank_by: str = "F1-Score",
        noise_type: str = "gaussian",
        weight_range: tuple = (0.3, 0.8),
    ) -> "Benchmark":
        """Generate synthetic data from a linear SEM.

        Args:
            dag: ``gum.DAG`` defining the ground-truth structure.
            n_samples: Number of samples to generate.
            seed: Random seed.
            rank_by: Metric used to rank Pareto-optimal profiles.
            noise_type: ``"gaussian"``, ``"laplace"``, ``"uniform"``,
                or ``"exp"``.
            weight_range: ``(low, high)`` for absolute edge weights.
        """
        from data.generators import generate_from_sem

        dataset, golden = generate_from_sem(
            dag, n_samples=n_samples, seed=seed,
            noise_type=noise_type, weight_range=weight_range,
        )
        dataset = Dataset(dataset.data,
                          name=f"sem_{noise_type}__{dag.size()}vars_{n_samples}s",
                          feature_names=[f"X{i}" for i in range(dag.size())])

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
        (according to ``rank_by``). Structures are now kept in the grid
        search results, so no re-run is needed. Results are saved to disk
        automatically.
        """
        if self._gs is None:
            self._register_all_algorithms()

        self._gs.run()
        self._selection_cpdag = self._gs.select_best(self.rank_by, compare_mode="cpdag")
        self._selection_skeleton = self._gs.select_best(self.rank_by, compare_mode="skeleton")

        # Extract best structures directly from grid search results
        self._structures = self._extract_best_structures()

        # Build scores and params dicts from both selections
        self._scores_cpdag = {}
        self._scores_skeleton = {}
        self._params_cpdag = {}
        self._params_skeleton = {}
        self._seed_counts = {}
        for name, r in self._selection_cpdag.items():
            if r is not None:
                self._scores_cpdag[name] = r.scores
                self._params_cpdag[name] = r.params
        for name, r in self._selection_skeleton.items():
            if r is not None:
                self._scores_skeleton[name] = r.scores_skeleton
                self._params_skeleton[name] = r.params
        for n, _, _, _, seeds in self._gs._configs:
            if seeds is not None:
                self._seed_counts[n] = len(seeds)

        save_dir = self.save()
        print(f"Results saved to {save_dir}\n")
        return self

    def _extract_best_structures(self) -> Dict[str, Any]:
        """Extract structures for the best cpdag profiles from grid search results.

        For stochastic algorithms (with random_seeds), returns a list of
        structures (one per seed matching the best params).
        """
        configs_map = {name: seeds for name, _, _, _, seeds in self._gs._configs}
        structures = {}

        for name, best_r in self._selection_cpdag.items():
            if best_r is None:
                continue
            seeds = configs_map.get(name)
            if seeds is not None:
                # Stochastic: collect all structures with matching params
                seed_structures = [
                    r.structure for r in self._gs.results[name]
                    if r.structure is not None and r.params == best_r.params
                ]
                if seed_structures:
                    structures[name] = seed_structures
            else:
                # Deterministic: find the structure matching best params
                for r in self._gs.results[name]:
                    if r.params == best_r.params and r.structure is not None:
                        structures[name] = r.structure
                        break

        return structures

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
        self._selection_cpdag = None
        self._selection_skeleton = None
        self._structures = {}
        self._scores_cpdag = {}
        self._scores_skeleton = {}
        self._params_cpdag = configs
        self._params_skeleton = configs

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

                scores_cpdag = {}
                scores_skel = {}
                for metric in self.metrics:
                    scores_cpdag[metric.name()] = metric.compute(
                        ref=self.golden_structure, test=structure)
                    scores_skel[metric.name()] = metric.compute(
                        ref=self.golden_structure.skeleton(), test=structure.skeleton())
                self._scores_cpdag[name] = scores_cpdag
                self._scores_skeleton[name] = scores_skel
                print(f"  {name}: OK")
            except Exception as e:
                print(f"  {name}: FAILED ({e})")

        print("Done.\n")
        return self

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: Optional[str | Path] = None) -> Path:
        """Save all results (scores, structures, grid search) to disk.

        Args:
            path: Directory to save to. Auto-generated if None.

        Returns:
            The directory where results were saved.
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%M")
            path = Path(__file__).parent.parent / "views" / "results" / f"{self.dataset.name}__{timestamp}"
        path = Path(path)

        # Save golden structure
        save_structure(self.golden_structure, path / "golden.json")

        # Save best structures
        if self._structures:
            for name, s in self._structures.items():
                if isinstance(s, list):
                    for i, si in enumerate(s):
                        save_structure(si, path / "structures" / f"{name}_seed{i}.json")
                else:
                    save_structure(s, path / "structures" / f"{name}.json")

        # Save ALL grid search structures
        if self._gs is not None:
            for algo_name, results_list in self._gs.results.items():
                for idx, r in enumerate(results_list):
                    if r.structure is not None:
                        save_structure(
                            r.structure,
                            path / "grid_structures" / f"{algo_name}_{idx}.json",
                        )

        # Build manifest
        manifest = {
            "dataset_name": self.dataset.name,
            "feature_names": self.dataset.feature_names,
            "rank_by": self.rank_by,
            "scores_cpdag": self._scores_cpdag,
            "scores_skeleton": self._scores_skeleton,
            "params_cpdag": self._params_cpdag,
            "params_skeleton": self._params_skeleton,
            "seed_counts": self._seed_counts,
        }

        # Save grid search results (scores + params, no structures)
        if self._gs is not None:
            gs_results = {}
            for algo_name, results_list in self._gs.results.items():
                gs_results[algo_name] = [
                    {"params": r.params, "scores": r.scores,
                     "scores_skeleton": r.scores_skeleton, "error": r.error}
                    for r in results_list
                ]
            manifest["grid_search_results"] = gs_results
            manifest["grid_search_configs"] = [
                {"name": n, "param_grid": pg, "random_seeds": rs}
                for n, _, pg, _, rs in self._gs._configs
            ]

        path.mkdir(parents=True, exist_ok=True)
        with open(path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        return path

    @classmethod
    def load(cls, path: str | Path) -> "Benchmark":
        """Load a Benchmark from a saved results directory.

        The loaded Benchmark supports all ``plot_*`` methods but cannot
        ``run()`` again (no dataset).

        Args:
            path: Directory containing ``manifest.json``.
        """
        path = Path(path)
        with open(path / "manifest.json") as f:
            manifest = json.load(f)

        # Reconstruct golden structure
        golden = load_structure(path / "golden.json")

        # Reconstruct a minimal Dataset (no data, just metadata for plotting)
        dataset = Dataset(
            np.empty((0, len(manifest["feature_names"]))),
            name=manifest["dataset_name"],
            feature_names=manifest["feature_names"],
        )

        bench = cls(dataset, golden, rank_by=manifest["rank_by"])
        bench._scores_cpdag = manifest["scores_cpdag"]
        bench._scores_skeleton = manifest["scores_skeleton"]
        bench._params_cpdag = manifest["params_cpdag"]
        bench._params_skeleton = manifest["params_skeleton"]
        bench._seed_counts = {k: int(v) for k, v in manifest.get("seed_counts", {}).items()}

        # Load best structures
        structures_dir = path / "structures"
        if structures_dir.exists():
            bench._structures = {}
            for algo_name in manifest["scores_cpdag"]:
                single = structures_dir / f"{algo_name}.json"
                if single.exists():
                    bench._structures[algo_name] = load_structure(single)
                else:
                    # Stochastic: load all seed files
                    seed_files = sorted(structures_dir.glob(f"{algo_name}_seed*.json"))
                    if seed_files:
                        bench._structures[algo_name] = [load_structure(f) for f in seed_files]

        # Reconstruct GridSearch stub for plot_grid_search()
        if "grid_search_results" in manifest:
            bench._gs = GridSearch(
                dataset=dataset,
                golden_structure=golden,
                metrics=list(ALL_METRICS),
                objectives=dict(OBJECTIVES),
            )
            # Populate results
            for algo_name, results_data in manifest["grid_search_results"].items():
                bench._gs.results[algo_name] = [
                    GridSearchResult(
                        params=r["params"],
                        scores=r["scores"],
                        scores_skeleton=r["scores_skeleton"],
                        error=r.get("error"),
                    )
                    for r in results_data
                ]
            # Populate _configs (algo_class=None since we never re-run)
            bench._gs._configs = [
                (c["name"], None, c["param_grid"], {}, c["random_seeds"])
                for c in manifest["grid_search_configs"]
            ]

        return bench

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def show_golden(self) -> None:
        """Display the golden CPDAG in the notebook."""
        import pyagrum.lib.notebook as gnb
        from views.plotting import cpdag_to_dot

        s = self.golden_structure
        print(
            f"Golden CPDAG: {s.cpdag.sizeArcs()} arcs, "
            f"{s.cpdag.sizeEdges()} edges"
        )
        gnb.showDot(cpdag_to_dot(s, self.dataset.feature_names))

    def plot_grid_search(self, compare_mode: str = "cpdag") -> None:
        """Plot per-algorithm grid search results (bars/heatmaps + Pareto)."""
        if self._gs is None:
            print("No grid search results (run_fixed was used).")
            return
        self._gs.plot(compare_mode=compare_mode)

    def plot_best_scores(self, compare_mode: str = "cpdag") -> None:
        """Scatter plot comparing best scores of each algorithm (SHD vs F1)."""
        scores = self._scores_cpdag if compare_mode == "cpdag" else self._scores_skeleton
        params = self._params_cpdag if compare_mode == "cpdag" else self._params_skeleton
        if scores is None:
            raise RuntimeError("Call run() or run_fixed() first.")
        from views.plotting import plot_best_scores

        plot_best_scores(scores, params, seed_counts=self._seed_counts)

    def plot_structures(self) -> None:
        """Display learned CPDAGs for each algorithm alongside the golden BN and a diff.

        For stochastic algorithms (e.g. LiNGAM), all seed structures are shown.
        """
        if self._structures is None:
            raise RuntimeError("Call run() or run_fixed() first.")
        from views.plotting import plot_cpdags

        plot_cpdags(
            self._structures, self._params_cpdag, self.dataset.feature_names,
            golden_structure=self.golden_structure,
        )

    def plot_pairwise_heatmaps(self, compare_mode: str = "cpdag") -> None:
        """Plot pairwise metric heatmaps between all learned structures."""
        if self._structures is None:
            raise RuntimeError("Call run() or run_fixed() first.")
        from views.plotting import plot_pairwise_heatmaps

        title = (
            f"Best profiles (rank_by={self.rank_by})"
            if self._gs is not None
            else "Fixed hyperparameters"
        )
        plot_pairwise_heatmaps(
            self._structures,
            title,
            self.metrics,
            self.objectives,
            golden_structure=self.golden_structure,
            compare_mode=compare_mode,
        )

    def __repr__(self) -> str:
        status = (
            "ready"
            if self._structures is None
            else f"{len(self._structures)} algorithms"
        )
        return f"Benchmark(dataset={self.dataset.name!r}, {status})"
