"""
Integration test for CPC algorithm with golden structure.

Tests that CPC can learn a simple chain structure and computes
the Structural Hamming Distance against the golden structure.
"""

import pyagrum as gum
from algorithms.CPCAdapter import CPCAdapter
from pipeline.Pipeline import StructureLearningPipeline
from metrics.SHDMetric import SHDMetric
from data.generators import create_simple_cbn, generate_from_cbn


def test_cpc_with_golden_chain():
    """Test CPC algorithm with SHD metric on a simple chain structure."""

    # Create a simple chain DAG: X0 -> X1 -> X2
    print("\n" + "="*60)
    print("Creating golden chain structure: X0 -> X1 -> X2")
    print("="*60)

    dag = gum.DAG()
    dag.addNode()  # X0
    dag.addNode()  # X1
    dag.addNode()  # X2
    dag.addArc(0, 1)  # X0 -> X1
    dag.addArc(1, 2)  # X1 -> X2

    # Create CBN and generate samples
    cbn = create_simple_cbn(dag, var_names=["X0", "X1", "X2"])
    dataset, golden_structure = generate_from_cbn(cbn, n_samples=1000, seed=42)

    print(f"Dataset: {dataset}")
    print(f"Golden structure: {golden_structure.cpdag}")

    # Create CPC adapter
    alpha = 0.1
    max_cond_set_size = 3
    cpc_adapter = CPCAdapter(alpha=alpha, max_conditioning_set_size=max_cond_set_size)

    # Run pipeline
    print("\n" + "="*60)
    print("Running pipeline with CPC algorithm...")
    print("="*60)
    pipeline = StructureLearningPipeline(dataset)
    pipeline.add_algorithm(cpc_adapter)
    pipeline.add_metric(SHDMetric())
    results = pipeline.run()

    # Display results
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    for algo_name, result in results.items():
        print(f"\nAlgorithm: {algo_name}")
        print(f"Learned structure: {result.learned_structure.cpdag}")
        print(f"Metrics: {result.metrics}")

        # Verify that SHD was calculated
        assert "SHD" in result.metrics, "SHD metric should be calculated"
        shd_value = result.metrics["SHD"]
        print(f"\nStructural Hamming Distance: {shd_value}")
        print(f"  (0 = perfect match, higher = more differences)")


if __name__ == "__main__":
    # Run with: python -m tests.integration.test_cpc_with_golden
    test_cpc_with_golden_chain()
