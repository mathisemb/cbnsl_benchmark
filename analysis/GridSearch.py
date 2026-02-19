"""
Grid search for hyperparameter optimization of structure learning algorithms.
"""

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import pandas as pd

from algorithms.AlgorithmAdapter import AlgorithmAdapter
from metrics.MetricAdapter import MetricAdapter
from pipeline.Dataset import Dataset
from pipeline.Structure import Structure


@dataclass
class GridSearchResult:
    """Stores the result of a single grid search trial.

    Attributes:
        params: The parameter combination used.
        scores: Metric values keyed by metric name (empty if the run failed).
        error: Error message if the run failed, None otherwise.
    """

    params: Dict[str, Any]
    scores: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


class GridSearch:
    """Hyperparameter grid search for structure learning algorithms.

    Generates all combinations of parameters from a parameter grid,
    runs the algorithm for each combination, computes metrics against
    a golden (ground truth) structure, and identifies the best parameters.

    Args:
        algorithm_class: The algorithm adapter class (not an instance).
        param_grid: Dictionary mapping parameter names to lists of values.
            All combinations are explored (Cartesian product).
        dataset: The dataset to run the algorithm on.
        golden_structure: The ground truth structure for metric computation.
        metrics: List of metrics to compute for each profile.
        fixed_params: Parameters passed to the algorithm constructor but not
            varied in the grid (e.g. version=1 for CPC).
        objectives: Dictionary mapping metric names to their optimization
            direction (True = lower is better, False = higher is better).
            If not provided, all metrics default to lower is better.
    """

    def __init__(
        self,
        algorithm_class: Type[AlgorithmAdapter],
        param_grid: Dict[str, List[Any]],
        dataset: Dataset,
        golden_structure: Structure,
        metrics: List[MetricAdapter],
        fixed_params: Optional[Dict[str, Any]] = None,
        objectives: Optional[Dict[str, bool]] = None,
    ):
        self.algorithm_class = algorithm_class
        self.param_grid = param_grid
        self.dataset = dataset
        self.golden_structure = golden_structure
        self.metrics = metrics
        self.fixed_params = fixed_params or {}
        self.objectives = objectives or {m.name(): True for m in metrics}

        self.results: List[GridSearchResult] = []
        self._is_fitted: bool = False

    def run(self) -> "GridSearch":
        """Execute the grid search over all parameter combinations.

        For each combination:
        1. Instantiate the algorithm with those parameters.
        2. Run learn_structure() on the dataset.
        3. Compute all metrics vs the golden structure.
        4. Record the result (or the error if it failed).

        Returns:
            Self, for method chaining.
        """
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        combinations = list(itertools.product(*param_values))
        n_combinations = len(combinations)

        metric_names = [m.name() for m in self.metrics]

        print(f"\n{'='*60}")
        print(f"Grid Search: {self.algorithm_class.__name__}")
        if self.fixed_params:
            fixed_str = ", ".join(f"{k}={v}" for k, v in self.fixed_params.items())
            print(f"  Fixed params: {fixed_str}")
        print(f"  Metrics: {', '.join(metric_names)}")
        print(f"  Parameter combinations: {n_combinations}")
        print(f"{'='*60}\n")

        self.results = []

        for i, combo in enumerate(combinations, 1):
            params = dict(zip(param_names, combo))
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"  [{i}/{n_combinations}] {params_str} ... ", end="", flush=True)

            try:
                all_params = {**self.fixed_params, **params}
                algorithm = self.algorithm_class(**all_params)
                learned_structure = algorithm.learn_structure(self.dataset)

                scores = {}
                for metric in self.metrics:
                    scores[metric.name()] = metric.compute(
                        ref=self.golden_structure,
                        test=learned_structure,
                    )

                result = GridSearchResult(params=params, scores=scores)
                scores_str = ", ".join(f"{k}={v:.4f}" for k, v in scores.items())
                print(scores_str)
            except Exception as e:
                result = GridSearchResult(params=params, error=str(e))
                print(f"FAILED - {e}")

            self.results.append(result)

        self._is_fitted = True

        for metric_name in metric_names:
            best = self.best_result(metric_name)
            if best is not None:
                print(f"\n  Best {metric_name}: {best.scores[metric_name]:.4f}")
                print(f"    params: {best.params}")

        if all(self.best_result(m) is None for m in metric_names):
            print("\n  WARNING: All parameter combinations failed.")

        print(f"\n{'='*60}\n")
        return self

    def best_result(self, metric_name: str) -> Optional[GridSearchResult]:
        """Return the best GridSearchResult for a given metric.

        Args:
            metric_name: Name of the metric to optimize.

        Returns:
            The result with the best metric value, or None if all runs failed.

        Raises:
            RuntimeError: If run() has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError("GridSearch has not been run yet. Call run() first.")

        valid_results = [r for r in self.results if metric_name in r.scores]
        if not valid_results:
            return None

        lower_is_better = self.objectives.get(metric_name, True)
        if lower_is_better:
            return min(valid_results, key=lambda r: r.scores[metric_name])
        else:
            return max(valid_results, key=lambda r: r.scores[metric_name])

    def best_params(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Return the best parameter combination for a given metric."""
        best = self.best_result(metric_name)
        return best.params if best is not None else None

    def best_score(self, metric_name: str) -> Optional[float]:
        """Return the best score for a given metric."""
        best = self.best_result(metric_name)
        return best.scores[metric_name] if best is not None else None

    def get_results_dataframe(self) -> pd.DataFrame:
        """Get all grid search results as a pandas DataFrame.

        Returns:
            DataFrame with one row per parameter combination, containing
            one column per parameter, one column per metric, and an error column.

        Raises:
            RuntimeError: If run() has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError("GridSearch has not been run yet. Call run() first.")

        rows = []
        for result in self.results:
            row = dict(result.params)
            for metric_name, score in result.scores.items():
                row[metric_name] = score
            row["error"] = result.error
            rows.append(row)

        return pd.DataFrame(rows)
