"""
Basic usage example of the cbnsl_benchmark pipeline.

This example shows how to:
1. Create a synthetic dataset
2. Configure the pipeline with an algorithm
3. Add metrics
4. Run the benchmark and view results
"""

import numpy as np
import openturns as ot
from algorithms.AlgorithmAdapter import DataType
from algorithms.adapters.CPCAdapter import CPCAdapter
from pipeline.Dataset import Dataset
from pipeline.Pipeline import StructureLearningPipeline
from metrics.SHDMetric import SHDMetric


def generate_sample_data(n_samples=1000):
    """
    Generate synthetic data from a known copula structure.

    This creates a 7-variable dataset with block-independent structure:
    - Block 1: 2 variables with Frank copula
    - Block 2: 3 variables with Normal copula
    - Block 3: 2 variables with Clayton copula
    """
    R = ot.CorrelationMatrix(3)
    R[0, 1] = 0.5
    R[0, 2] = 0.45

    collection = [
        ot.FrankCopula(3.0),
        ot.NormalCopula(R),
        ot.ClaytonCopula(2.0)
    ]

    copula = ot.BlockIndependentCopula(collection)
    copula.setDescription(["A", "B", "C", "D", "E", "F", "G"])

    return copula.getSample(n_samples)


def main():
    """Run the basic benchmark example."""

    print("=" * 60)
    print("CBNSL Benchmark - Basic Usage Example")
    print("=" * 60)
    print()

    # 1. Generate synthetic dataset
    print("Step 1: Generating synthetic dataset...")
    ot_sample = generate_sample_data(n_samples=1000)
    np_data = np.array(ot_sample)
    dataset = Dataset(np_data, DataType.CONTINUOUS)
    print(f"  Created dataset: {dataset}")
    print()

    # 2. Create pipeline
    print("Step 2: Creating pipeline...")
    pipeline = StructureLearningPipeline(dataset)
    print("  Pipeline created")
    print()

    # 3. Add algorithm(s)
    print("Step 3: Adding CPC algorithm...")
    cpc = CPCAdapter(alpha=0.1, max_conditioning_set_size=3)
    pipeline.add_algorithm(cpc)
    print(f"  Added algorithm: {cpc.name()}")
    print()

    # 4. Add metric(s)
    print("Step 4: Adding metrics...")
    pipeline.add_metric(SHDMetric())
    print("  Added metric: SHD")
    print()

    # 5. Run benchmark
    print("Step 5: Running benchmark...")
    results = pipeline.run()
    print()

    # 6. Display results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    for algo_name, result in results.items():
        print(f"\nAlgorithm: {algo_name}")
        print(f"  Learned structure: {result.learned_structure}")
        if result.metrics:
            print("  Metrics:")
            for metric_name, value in result.metrics.items():
                print(f"    - {metric_name}: {value:.4f}")
        else:
            print("  No metrics computed")
    print()


if __name__ == "__main__":
    main()
