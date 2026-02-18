"""
Comprehensive comparison test of all algorithms.

This test runs all algorithms on the Sachs dataset and compares
their learned structures against the known ground truth using SHD.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algorithms.CPCAdapter import CPCAdapter
from algorithms.CMIICAdapter import CMIICAdapter
from algorithms.MIICAdapter import MIICAdapter
from algorithms.GHCBDeuAdapter import GHCBDeuAdapter
from algorithms.NOTEARSAdapter import NOTEARSAdapter
from algorithms.LiNGAMAdapter import LiNGAMAdapter
from pipeline.Dataset import Dataset
from pipeline.Pipeline import StructureLearningPipeline
from metrics.SHDMetric import SHDMetric
from data.sachs.load_ground_truth import load_sachs_ground_truth
from analysis.BenchmarkAnalyzer import BenchmarkAnalyzer


def test_comparison_sachs():
    """Compare all algorithms on Sachs observational dataset."""

    print("=" * 70)
    print("COMPARISON: All algorithms")
    print("Dataset: Sachs Protein Signaling")
    print("=" * 70)
    print()

    # 1. Load Sachs observational dataset
    print("1. Loading Sachs observational dataset...")
    data_path = project_root / "data" / "sachs" / "sachs_observational.csv"
    sachs_data = pd.read_csv(data_path, sep="\t")
    print(f"   Dataset shape: {sachs_data.shape}")
    print(f"   Variables: {', '.join(sachs_data.columns)}")
    print()

    # 2. Load ground truth structure
    print("2. Loading ground truth structure (BN18)...")
    ground_truth = load_sachs_ground_truth(version="bn18", as_structure=True)
    print(f"   Ground truth: {ground_truth.cpdag.size()} nodes, {ground_truth.cpdag.sizeArcs()} edges")
    print()

    # Convert to numpy and create Dataset with ground truth
    np_data = sachs_data.to_numpy()
    dataset = Dataset(
        np_data,
        golden_structure=ground_truth,
        name="sachs_observational",
        feature_names=list(sachs_data.columns)
    )

    # 3. Configure all algorithms
    print("3. Configuring algorithms...")
    alpha = 0.05
    max_cond_set = 3

    n_bins = 3

    algorithms = [
        CPCAdapter(alpha=alpha, max_conditioning_set_size=max_cond_set, version=1),
        CPCAdapter(alpha=alpha, max_conditioning_set_size=max_cond_set, version=2),
        CMIICAdapter(alpha=alpha, version=1),
        CMIICAdapter(alpha=alpha, version=2),
        MIICAdapter(n_bins=n_bins),
        GHCBDeuAdapter(n_bins=n_bins),
        NOTEARSAdapter(lambda1=0.1),
        LiNGAMAdapter(),
    ]

    for algo in algorithms:
        print(f"   - {algo.name()}")
    print()

    # 4. Create pipeline
    print("4. Creating pipeline...")
    pipeline = StructureLearningPipeline(dataset)
    for algo in algorithms:
        pipeline.add_algorithm(algo)
    print("   Pipeline ready")
    print()

    # 5. Run benchmark
    print("5. Running benchmark...")
    results = pipeline.run()
    print()

    # 6. Compute metrics using BenchmarkAnalyzer
    print("6. Computing metrics...")
    analyzer = BenchmarkAnalyzer(results, golden_structure=ground_truth)
    analyzer.compute_vs_golden([SHDMetric()])
    print()

    # 7. Display results summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    # Create comparison table
    print("Comparison vs Ground Truth (BN18):")
    print("-" * 70)
    print(f"{'Algorithm':<20} {'Nodes':<10} {'Edges':<10} {'SHD':<10}")
    print("-" * 70)

    for result in results:
        algo_name = result.algorithm_name
        n_nodes = result.learned_structure.cpdag.size()
        n_edges = result.learned_structure.cpdag.sizeArcs()
        shd = result.metrics.get('SHD', 'N/A')
        shd_str = f"{shd:.1f}" if isinstance(shd, (int, float)) else shd
        print(f"{algo_name:<20} {n_nodes:<10} {n_edges:<10} {shd_str:<10}")

    print("-" * 70)
    print()

    # Display ground truth for reference
    print("=" * 70)
    print("Ground Truth (BN18)")
    print("=" * 70)
    print()
    ground_truth.display(show_structure=True)
    print()

    # 8. Compute pairwise comparison
    print("=" * 70)
    print("PAIRWISE COMPARISON")
    print("=" * 70)
    print()
    pairwise_shd = analyzer.compute_pairwise(SHDMetric())
    print()

    # 9. Export results to DataFrame
    print("=" * 70)
    print("EXPORTING RESULTS")
    print("=" * 70)
    print()
    df = analyzer.get_results_dataframe()
    print(df.to_string(index=False))
    print()

    print("=" * 70)
    print("BENCHMARK COMPLETED")
    print("=" * 70)
    print()


if __name__ == "__main__":
    test_comparison_sachs()
