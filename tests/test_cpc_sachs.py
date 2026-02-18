"""
Test CPC algorithm on the Sachs protein signaling dataset.

This test runs CPC on the real-world Sachs dataset and compares
the learned structure against the known ground truth.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algorithms.CPCAdapter import CPCAdapter
from pipeline.Dataset import Dataset
from pipeline.Pipeline import StructureLearningPipeline
from metrics.SHDMetric import SHDMetric
from data.sachs.load_ground_truth import load_sachs_ground_truth
from analysis.BenchmarkAnalyzer import BenchmarkAnalyzer


def test_cpc_sachs():
    """Test CPC on Sachs observational dataset with ground truth comparison (BN18)."""

    print("=" * 70)
    print("CPC Algorithm on Sachs Protein Signaling Dataset")
    print("=" * 70)
    print()

    # 1. Load Sachs observational dataset
    print("1. Loading Sachs observational dataset...")
    data_path = project_root / "data" / "sachs" / "sachs_observational.csv"
    sachs_data = pd.read_csv(data_path, sep="\t") # panda to read the columns names
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

    # 3. Configure CPC algorithm
    print("3. Configuring CPC algorithm...")
    cpc = CPCAdapter(alpha=0.05, max_conditioning_set_size=3)
    print(f"   Algorithm: {cpc.name()}")
    print(f"   Alpha: {cpc.alpha}")
    print(f"   Max conditioning set size: 3")
    print()

    # 4. Create pipeline
    print("4. Creating pipeline...")
    pipeline = StructureLearningPipeline(dataset)
    pipeline.add_algorithm(cpc)
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

    # 7. Display results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    for result in results:
        print(f"Algorithm: {result.algorithm_name}")
        print(f"  Learned structure: {result.learned_structure.cpdag.size()} nodes, {result.learned_structure.cpdag.sizeArcs()} edges")

        if result.metrics:
            print("  Metrics:")
            for metric_name, value in result.metrics.items():
                print(f"    - {metric_name}: {value:.4f}")
        else:
            print("  No metrics computed")

    print()
    print("=" * 70)
    print("Ground Truth (BN18)")
    print("=" * 70)
    print()
    ground_truth.display(show_structure=True)
    print()
    print("=" * 70)
    print()


if __name__ == "__main__":
    test_cpc_sachs()
