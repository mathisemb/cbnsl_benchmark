"""
Grid search for best hyperparameters on the Sachs dataset.

Runs a grid search for each algorithm separately and displays
the best parameters found for each, plus Pareto front (SHD vs F1).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algorithms.CPCAdapter import CPCAdapter
from algorithms.CMIICAdapter import CMIICAdapter
from algorithms.MIICAdapter import MIICAdapter
from algorithms.GHCBDeuAdapter import GHCBDeuAdapter
from algorithms.NOTEARSAdapter import NOTEARSAdapter
from algorithms.LiNGAMAdapter import LiNGAMAdapter
from pipeline.Dataset import Dataset
from metrics.SHDMetric import SHDMetric
from metrics.F1ScoreMetric import F1ScoreMetric
from metrics.TPRMetric import TPRMetric
from data.sachs.load_ground_truth import load_sachs_ground_truth
from analysis.GridSearch import GridSearch
from analysis.ParetoSelector import pareto_front


def test_grid_search_sachs():
    """Grid search for best hyperparameters on Sachs dataset."""

    print("=" * 70)
    print("GRID SEARCH: All algorithms on Sachs dataset")
    print("=" * 70)
    print()

    # 1. Load Sachs observational dataset
    print("1. Loading Sachs observational dataset...")
    data_path = project_root / "data" / "sachs" / "sachs_observational.csv"
    sachs_data = pd.read_csv(data_path, sep="\t")
    print(f"   Dataset shape: {sachs_data.shape}")
    print()

    # 2. Load ground truth
    print("2. Loading ground truth structure (BN18)...")
    ground_truth = load_sachs_ground_truth(version="bn18", as_structure=True)
    print()

    # 3. Create Dataset
    np_data = sachs_data.to_numpy()
    dataset = Dataset(
        np_data,
        golden_structure=ground_truth,
        name="sachs_observational",
        feature_names=list(sachs_data.columns),
    )

    metrics = [SHDMetric(), F1ScoreMetric(), TPRMetric()]
    objectives = {"SHD": True, "F1-Score": False, "TPR": False}

    # 4. Define algorithm configurations
    algo_configs = [
        (CPCAdapter, {"alpha": [0.001, 0.01, 0.05, 0.10], "max_conditioning_set_size": [2, 3, 4, 5]}, {"version": 1}),
        (CPCAdapter, {"alpha": [0.001, 0.01, 0.05, 0.10], "max_conditioning_set_size": [2, 3, 4, 5]}, {"version": 2}),
        (CMIICAdapter, {"alpha": [0.001, 0.01, 0.05, 0.10]}, {"version": 1}),
        (CMIICAdapter, {"alpha": [0.001, 0.01, 0.05, 0.10]}, {"version": 2}),
        (MIICAdapter, {"n_bins": [2, 3, 4, 5, 6], "discretization_method": ["quantile", "uniform", "kmeans"]}, {}),
        (GHCBDeuAdapter, {"n_bins": [2, 3, 4, 5, 6], "discretization_method": ["quantile", "uniform", "kmeans"]}, {}),
        (NOTEARSAdapter, {"lambda1": [0.01, 0.05, 0.1, 0.2, 0.5], "w_threshold": [0.1, 0.2, 0.3, 0.5]}, {}),
        (LiNGAMAdapter, {"measure": ["pwling"], "threshold": [0.001, 0.005, 0.01, 0.05]}, {}),
    ]

    # 5. Run grid searches
    print("3. Running grid searches...")
    print()

    grid_searches = []
    for algo_class, param_grid, fixed_params in algo_configs:
        gs = GridSearch(
            algorithm_class=algo_class,
            param_grid=param_grid,
            dataset=dataset,
            golden_structure=ground_truth,
            metrics=metrics,
            fixed_params=fixed_params if fixed_params else None,
            objectives=objectives,
        )
        gs.run()
        grid_searches.append(gs)

    # 6. Summary: best per metric
    metric_names = [m.name() for m in metrics]

    print("=" * 70)
    print("SUMMARY: Best score per algorithm and metric")
    print("=" * 70)
    print()

    header = f"{'Algorithm':<30} " + " ".join(f"{name:<12}" for name in metric_names)
    print(header)
    print("-" * 70)

    for i, (algo_class, _, fixed_params) in enumerate(algo_configs):
        algo_name = algo_class.__name__
        if fixed_params:
            fixed_str = ", ".join(f"{k}={v}" for k, v in fixed_params.items())
            algo_name = f"{algo_name} ({fixed_str})"

        scores = []
        for metric_name in metric_names:
            best = grid_searches[i].best_score(metric_name)
            if best is not None:
                scores.append(f"{best:<12.3f}")
            else:
                scores.append(f"{'FAILED':<12}")

        print(f"{algo_name:<30} {' '.join(scores)}")

    print("-" * 70)
    print()

    # 7. Pareto front per algorithm (SHD vs F1)
    pareto_objectives = {"SHD": True, "F1-Score": False}

    print("=" * 70)
    print("PARETO FRONT: SHD vs F1-Score (per algorithm)")
    print("=" * 70)
    print()

    for i, (algo_class, _, fixed_params) in enumerate(algo_configs):
        algo_name = algo_class.__name__
        if fixed_params:
            fixed_str = ", ".join(f"{k}={v}" for k, v in fixed_params.items())
            algo_name = f"{algo_name} ({fixed_str})"

        front = pareto_front(grid_searches[i].results, pareto_objectives)

        print(f"{algo_name}:")
        if not front:
            print("  No valid results")
        else:
            for r in sorted(front, key=lambda r: r.scores["SHD"]):
                shd = r.scores["SHD"]
                f1 = r.scores["F1-Score"]
                tpr = r.scores.get("TPR", float("nan"))
                print(f"  SHD={shd:<6.1f} F1={f1:<6.3f} TPR={tpr:<6.3f}  params={r.params}")
        print()

    print("=" * 70)
    print("GRID SEARCH COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    test_grid_search_sachs()
