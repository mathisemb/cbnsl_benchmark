"""
Grid search for hyperparameter optimization of structure learning algorithms.

Usage:
    gs = GridSearch(dataset, golden_structure, metrics, objectives)
    gs.add("CPC", CPCAdapter)                      # uses DEFAULT_PARAM_GRID
    gs.add("MIIC", MIICAdapter, param_grid={...})   # override grid
    gs.run()
    gs.summary()
    gs.select_best("F1-Score")
"""

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import pandas as pd
from tqdm.auto import tqdm

from algorithms.AlgorithmAdapter import AlgorithmAdapter
from metrics.MetricAdapter import MetricAdapter
from pipeline.Dataset import Dataset
from pipeline.Structure import Structure, dag_as_a_structure


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GridSearchResult:
    """Stores the result of a single grid search trial."""

    params: Dict[str, Any]
    scores: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Pareto helpers (generic: work on any list of score dicts)
# ---------------------------------------------------------------------------

def _dominates(a: Dict[str, float], b: Dict[str, float], objectives: Dict[str, bool]) -> bool:
    """True if point a dominates point b on all objectives."""
    strictly_better = False
    for name, lower_is_better in objectives.items():
        av, bv = a[name], b[name]
        if lower_is_better:
            if av > bv:
                return False
            if av < bv:
                strictly_better = True
        else:
            if av < bv:
                return False
            if av > bv:
                strictly_better = True
    return strictly_better


def pareto_front(
    points: List[Dict[str, float]],
    objectives: Dict[str, bool],
) -> List[int]:
    """Return indices of Pareto-optimal (non-dominated) points.

    Args:
        points: List of score dicts, e.g. [{"SHD": 5, "F1": 0.8}, ...].
        objectives: {metric_name: lower_is_better}.

    Returns:
        List of indices into *points* that are on the Pareto front.
    """
    valid = [(i, p) for i, p in enumerate(points) if all(m in p for m in objectives)]
    front = []
    for i, pi in valid:
        if not any(_dominates(pj, pi, objectives) for j, pj in valid if j != i):
            front.append(i)
    return front


# ---------------------------------------------------------------------------
# GridSearch
# ---------------------------------------------------------------------------

def _count_combinations(param_grid):
    if isinstance(param_grid, list):
        return sum(len(list(itertools.product(*sg.values()))) for sg in param_grid)
    return len(list(itertools.product(*param_grid.values())))


class GridSearch:
    """Multi-algorithm grid search with discretization precomputation.

    Register one or more algorithms with add(), then call run().
    """

    def __init__(
        self,
        dataset: Dataset,
        golden_structure: Structure,
        metrics: List[MetricAdapter],
        objectives: Dict[str, bool],
        verbose: bool = False,
        learn_method: str = "learn_structure",
    ):
        self.dataset = dataset
        self.golden_structure = golden_structure
        self.metrics = metrics
        self.objectives = objectives
        self.verbose = verbose
        self.learn_method = learn_method

        self._configs: List[tuple] = []  # (name, algo_class, param_grid, fixed_params)
        self.results: Dict[str, List[GridSearchResult]] = {}

    # ----- registration -----------------------------------------------------

    def add(
        self,
        name: str,
        algo_class: Type[AlgorithmAdapter],
        param_grid: Optional[Dict[str, List[Any]] | List[Dict]] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
    ) -> "GridSearch":
        if param_grid is None:
            param_grid = algo_class.DEFAULT_PARAM_GRID
        if fixed_params is None:
            fixed_params = algo_class.DEFAULT_FIXED_PARAMS
        self._configs.append((name, algo_class, param_grid, fixed_params))
        return self

    # ----- execution --------------------------------------------------------

    def run(self) -> "GridSearch":
        hartemink_cache = self._precompute_hartemink()

        use_bars = not self.verbose
        total = sum(_count_combinations(pg) for _, _, pg, _ in self._configs)
        pbar = tqdm(total=total, desc="Grid search", leave=True) if use_bars else None

        for name, algo_class, param_grid, fixed_params in self._configs:
            if pbar is not None:
                pbar.set_postfix_str(name)
            grids = param_grid if isinstance(param_grid, list) else [param_grid]
            algo_results: List[GridSearchResult] = []
            for grid in grids:
                algo_results.extend(
                    self._run_grid(algo_class, grid, fixed_params, hartemink_cache, pbar)
                )
            self.results[name] = algo_results

        if pbar is not None:
            pbar.close()
        return self

    def _run_grid(self, algo_class, param_grid, fixed_params, hartemink_cache, pbar):
        """Run a single param_grid for a single algorithm.

        Includes two caching mechanisms to avoid redundant computation:
        - Hartemink cache: pre-computed discretizations injected via discretized_df
        - W matrix cache: when w_threshold is in the grid, the expensive
          notears_linear optimization only depends on lambda1 (and discretization
          params). The raw W matrix is cached and reused across w_threshold values,
          giving a ~5x speedup for NOTEARS-based algorithms.
        """
        param_names = list(param_grid.keys())
        combos = list(itertools.product(*param_grid.values()))
        results = []

        # Cache for NOTEARS weight matrices: avoids re-running L-BFGS
        # optimization when only w_threshold changes
        use_w_cache = "w_threshold" in param_names
        w_cache = {}
        # Params that are injected objects, not part of the cache key
        _injected_params = ("w_threshold", "W_est", "discretized_df")

        for combo in combos:
            params = {k: round(v, 3) if isinstance(v, float) else v
                      for k, v in zip(param_names, combo)}
            try:
                all_params = {**fixed_params, **params}
                # Inject pre-computed Hartemink discretizations
                if hartemink_cache and all_params.get("discretization_method") == "hartemink":
                    key = (all_params.get("n_bins"), all_params.get("initial_bins"))
                    if key in hartemink_cache:
                        all_params["discretized_df"] = hartemink_cache[key]

                # Inject cached W matrix if available
                if use_w_cache:
                    cache_key = tuple(sorted(
                        (k, v) for k, v in all_params.items()
                        if k not in _injected_params
                    ))
                    if cache_key in w_cache:
                        all_params["W_est"] = w_cache[cache_key]

                algo = algo_class(**all_params)
                result_obj = getattr(algo, self.learn_method)(self.dataset)
                learned = dag_as_a_structure(result_obj) if self.learn_method == "learn_dag" else result_obj

                # Cache raw W matrix after first computation
                if use_w_cache and hasattr(algo, '_W_est_raw') \
                        and cache_key not in w_cache:
                    w_cache[cache_key] = algo._W_est_raw

                scores = {m.name(): m.compute(ref=self.golden_structure, test=learned)
                          for m in self.metrics}
                result = GridSearchResult(params=params, scores=scores)
                if self.verbose:
                    params_str = ", ".join(f"{k}={v}" for k, v in params.items())
                    scores_str = ", ".join(f"{k}={v:.4f}" for k, v in scores.items())
                    print(f"  {params_str} -> {scores_str}")
            except Exception as e:
                result = GridSearchResult(params=params, error=str(e))
                if self.verbose:
                    print(f"  {params} -> FAILED: {e}")

            results.append(result)
            if pbar is not None:
                pbar.update(1)

        return results

    # ----- queries ----------------------------------------------------------

    def best_result(self, algo_name: str, metric_name: str) -> Optional[GridSearchResult]:
        """Best result for a given algorithm and metric."""
        valid = [r for r in self.results.get(algo_name, []) if metric_name in r.scores]
        if not valid:
            return None
        lower = self.objectives.get(metric_name, True)
        return min(valid, key=lambda r: r.scores[metric_name]) if lower \
            else max(valid, key=lambda r: r.scores[metric_name])

    def best_score(self, algo_name: str, metric_name: str) -> Optional[float]:
        best = self.best_result(algo_name, metric_name)
        return best.scores[metric_name] if best else None

    def get_results_dataframe(self, algo_name: str) -> pd.DataFrame:
        rows = []
        for r in self.results.get(algo_name, []):
            row = dict(r.params)
            row.update(r.scores)
            row["error"] = r.error
            rows.append(row)
        return pd.DataFrame(rows)

    def select_best(self, rank_by: str = "SHD") -> Dict[str, Optional[GridSearchResult]]:
        """Select one Pareto-optimal profile per algorithm."""
        pareto_obj = {k: v for k, v in self.objectives.items() if k in ("SHD", "F1-Score")}
        rank_lower = pareto_obj.get(rank_by, True)
        selection = {}
        for name, res in self.results.items():
            front_idx = pareto_front([r.scores for r in res], pareto_obj)
            if not front_idx:
                selection[name] = None
                continue
            front = [res[i] for i in front_idx]
            key = lambda r: r.scores[rank_by]
            selection[name] = min(front, key=key) if rank_lower else max(front, key=key)
        return selection

    # ----- re-run best profiles ---------------------------------------------

    def rerun_best(self, rank_by: str = "SHD") -> Dict[str, Structure]:
        selection = self.select_best(rank_by)
        hartemink_cache = self._precompute_hartemink()
        configs_map = {name: (cls, fp) for name, cls, _, fp in self._configs}
        structures = {}
        for name, r in selection.items():
            if r is None:
                continue
            algo_class, fixed_params = configs_map[name]
            all_params = {**fixed_params, **r.params}
            if hartemink_cache and all_params.get("discretization_method") == "hartemink":
                key = (all_params.get("n_bins"), all_params.get("initial_bins"))
                if key in hartemink_cache:
                    all_params["discretized_df"] = hartemink_cache[key]
            algo = algo_class(**all_params)
            result_obj = getattr(algo, self.learn_method)(self.dataset)
            structures[name] = dag_as_a_structure(result_obj) if self.learn_method == "learn_dag" else result_obj
        return structures

    # ----- plotting (delegates to notebooks.plotting) -----------------------

    def plot(self) -> None:
        from notebooks.plotting import plot_grid_search_results
        metric_names = [m.name() for m in self.metrics]
        pareto_obj = {k: v for k, v in self.objectives.items() if k in ("SHD", "F1-Score")}
        for name, _, param_grid, _ in self._configs:
            plot_grid_search_results(name, self, param_grid, metric_names, pareto_obj)

    def plot_comparison(self, rank_by: str = "SHD") -> None:
        from notebooks.plotting import plot_best_scores
        selection = self.select_best(rank_by)
        scores_by_algo = {name: r.scores for name, r in selection.items() if r is not None}
        params_by_algo = {name: r.params for name, r in selection.items() if r is not None}
        plot_best_scores(scores_by_algo, params_by_algo)

    # ----- Hartemink precomputation -----------------------------------------

    def _precompute_hartemink(self) -> Dict:
        from algorithms.hartemink import hartemink_discretize_multi

        df = self.dataset.to_dataframe()
        groups = {}
        for _name, _cls, param_grid, fixed_params in self._configs:
            grids = param_grid if isinstance(param_grid, list) else [param_grid]
            for grid in grids:
                methods = grid.get("discretization_method",
                                   [fixed_params.get("discretization_method")])
                if "hartemink" not in methods:
                    continue
                n_bins_list = grid.get("n_bins", [fixed_params.get("n_bins", 3)])
                initial_bins_list = grid.get("initial_bins",
                                             [fixed_params.get("initial_bins")])
                for ib in initial_bins_list:
                    groups.setdefault(ib, set()).update(n_bins_list)

        if not groups:
            return {}

        precomputed = {}
        for initial_bins, n_bins_set in groups.items():
            target_bins = sorted(n_bins_set)
            multi_results = hartemink_discretize_multi(df, target_bins, initial_bins=initial_bins, progress=True)
            for nb, disc_df in multi_results.items():
                precomputed[(nb, initial_bins)] = disc_df
        return precomputed
