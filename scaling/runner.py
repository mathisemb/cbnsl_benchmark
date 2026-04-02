"""
ScalingRunner: measures algorithm performance as a function of dataset parameters.

Results schema (one row per algo × graph × n_samples):
    generator_name    : str   - human-readable generator description
    algo              : str   - algorithm display name
    algo_params       : str   - JSON-encoded hyperparameters
    n_vars            : int   - number of variables
    n_arcs            : int   - number of arcs in the ground-truth DAG
    n_samples         : int   - number of samples
    graph_idx         : int   - random graph index
    time_s            : float - elapsed wall-clock time (NaN if failed)
    SHD               : float - (CPDAG)
    F1-Score          : float - (CPDAG)
    TPR               : float - (CPDAG)
    SHD_skeleton      : float
    F1-Score_skeleton : float
    TPR_skeleton      : float
    error_msg         : str   - empty if success, error description otherwise
    golden_path       : str   - path to saved golden Structure JSON
    learned_path      : str   - path to saved learned Structure JSON (empty if failed)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import pandas as pd

from algorithms.AlgorithmAdapter import AlgorithmAdapter
from metrics import ALL_METRICS
from pipeline.Structure import dag_as_a_structure
from .io import save_results, save_structure
from .random_dag import random_dag


class ScalingRunner:
    """
    Measures computation time and metrics as a function of dataset parameters.

    Iterates over a grid of ``(n_vars, n_arcs, n_samples)`` combinations.
    For each combination, generates ``n_graphs`` random DAGs.

    Parameters
    ----------
    algo_configs : list of (name, adapter_class, params)
        Each entry is ``(display_name, AdapterClass, fixed_params_dict)``.
    generator : callable
        Function ``(dag, n_samples, seed) -> (Dataset, Structure)``.
        Use ``generate_from_sem`` or ``generate_from_cbn`` from ``data.generators``.
    generator_name : str
        Human-readable name for the generator (used in results and folder name).
    n_vars_list : list of int
        Number of variables to test (e.g. ``[5, 10, 20]``).
    n_samples_list : list of int
        Sample sizes to test (e.g. ``[200, 500, 1000, 2000, 5000]``).
    density : float
        Average out-degree. ``n_arcs = round(n_vars * density)``.
    n_graphs : int
        Number of independent random graphs per (n_vars, n_arcs) pair.
    seed : int
        Base random seed. All derived seeds are deterministic from this value.
    """

    def __init__(
        self,
        algo_configs: List[AlgoConfig],
        generator,
        generator_name: str,
        n_vars_list: List[int],
        n_samples_list: List[int],
        density: float = 1.5,
        n_graphs: int = 10,
        seed: int = 42,
    ):
        self.algo_configs = algo_configs
        self.generator = generator
        self.generator_name = generator_name
        self.n_graphs = n_graphs
        self.seed = seed
        self.grid = _make_grid(n_vars_list, n_samples_list, density)
        timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%M")
        self.results_dir = (
            Path(__file__).parent / "results" / f"{timestamp}_{generator_name}"
        )
        self._results: Optional[pd.DataFrame] = None

    def run(self) -> pd.DataFrame:
        """
        Run the scaling experiment.

        Results are saved incrementally to ``{results_dir}/results.csv``
        (one row appended after each algorithm run).

        Returns
        -------
        pd.DataFrame
            Raw results (one row per algo × graph × n_samples).
        """
        metrics = list(ALL_METRICS)
        metric_names = [m.name() for m in metrics]
        records: List[Dict[str, Any]] = []
        csv_path = self.results_dir / "results.csv"

        var_arc_pairs = sorted({(nv, na) for nv, na, _ in self.grid})
        n_samples_for = {
            (nv, na): sorted({ns for nv2, na2, ns in self.grid if nv2 == nv and na2 == na})
            for nv, na in var_arc_pairs
        }
        total = sum(
            self.n_graphs * len(n_samples_for[(nv, na)]) * len(self.algo_configs)
            for nv, na in var_arc_pairs
        )
        done = 0

        for n_vars, n_arcs in var_arc_pairs:
            for graph_idx in range(self.n_graphs):
                graph_seed = self.seed + graph_idx # so we sample different graphs
                dag = random_dag(n_vars, n_arcs, seed=graph_seed)
                golden = dag_as_a_structure(dag)

                golden_path = (
                    self.results_dir / "graphs"
                    / f"golden__v{n_vars}_a{n_arcs}_g{graph_idx}.json"
                )
                save_structure(golden, golden_path)

                for n_samples in n_samples_for[(n_vars, n_arcs)]:
                    dataset, _ = self.generator(dag, n_samples, seed=self.seed)

                    for algo_name, algo_cls, algo_params in self.algo_configs:
                        done += 1
                        print(
                            f"[{done}/{total}] {algo_name} | "
                            f"v={n_vars} a={n_arcs} s={n_samples} g={graph_idx}",
                            end="\r",
                        )

                        row: Dict[str, Any] = {
                            "generator_name": self.generator_name,
                            "algo": algo_name,
                            "algo_params": json.dumps(algo_params),
                            "n_vars": n_vars,
                            "n_arcs": n_arcs,
                            "n_samples": n_samples,
                            "graph_idx": graph_idx,
                            "golden_path": str(golden_path),
                            "learned_path": "",
                            "error_msg": "",
                        }

                        try:
                            algo = algo_cls(**algo_params)
                            t0 = time.perf_counter()
                            learned = algo.learn_structure(dataset)
                            row["time_s"] = time.perf_counter() - t0

                            learned_path = (
                                self.results_dir / "graphs"
                                / f"{algo_name}__v{n_vars}_a{n_arcs}"
                                  f"_g{graph_idx}_s{n_samples}.json"
                            )
                            save_structure(learned, learned_path)
                            row["learned_path"] = str(learned_path)

                            for metric in metrics:
                                row[metric.name()] = metric.compute(golden, learned)
                                row[f"{metric.name()}_skeleton"] = metric.compute(
                                    golden.skeleton(), learned.skeleton()
                                )

                        except Exception as e:
                            print(
                                f"\n  [{algo_name}] v={n_vars} a={n_arcs} "
                                f"g={graph_idx} s={n_samples}: FAILED ({e})"
                            )
                            row["time_s"] = float("nan")
                            row["error_msg"] = str(e)
                            for name in metric_names:
                                row[name] = float("nan")
                                row[f"{name}_skeleton"] = float("nan")

                        records.append(row)

                        # Incremental save: append this row to CSV immediately
                        row_df = pd.DataFrame([row])
                        header = not csv_path.exists()
                        csv_path.parent.mkdir(parents=True, exist_ok=True)
                        row_df.to_csv(
                            csv_path, mode="a", index=False, header=header,
                        )

        print()
        self._results = pd.DataFrame(records)
        return self._results

    @property
    def results(self) -> pd.DataFrame:
        """Raw results DataFrame (one row per algo × graph × n_samples)."""
        if self._results is None:
            raise RuntimeError("Call run() first.")
        return self._results


def _make_grid(
    n_vars_list: List[int],
    n_samples_list: List[int],
    density: float = 1.5,
) -> List[Tuple[int, int, int]]:
    """
    Build a grid of ``(n_vars, n_arcs, n_samples)`` tuples.

    ``n_arcs`` is derived as ``round(n_vars * density)``, capped at
    ``n_vars * (n_vars - 1) / 2``.

    Parameters
    ----------
    n_vars_list : list of int
    n_samples_list : list of int
    density : float
        Average out-degree (``n_arcs = round(n_vars * density)``).
    """
    grid = []
    for n_vars in n_vars_list:
        n_arcs = min(round(n_vars * density), n_vars * (n_vars - 1) // 2)
        for n_samples in n_samples_list:
            grid.append((n_vars, n_arcs, n_samples))
    return grid
