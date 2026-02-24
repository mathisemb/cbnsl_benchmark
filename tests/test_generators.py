"""Test synthetic data generation with create_simple_cbn + generate_from_cbn."""

import pyagrum as gum
from data.generators import create_simple_cbn, generate_from_cbn
from algorithms.CPCAdapter import CPCAdapter
from metrics.SHDMetric import SHDMetric
from metrics.F1ScoreMetric import F1ScoreMetric
from metrics.TPRMetric import TPRMetric


def test_generators():
    # Create a simple DAG: 0 -> 1 -> 2, 0 -> 2
    dag = gum.DAG()
    dag.addNodes(3)
    dag.addArc(0, 1)
    dag.addArc(1, 2)
    dag.addArc(0, 2)

    # Generate synthetic data from it
    cbn = create_simple_cbn(dag, copula_correlation=0.8)
    dataset, golden = generate_from_cbn(cbn, n_samples=2000)

    print(f"Dataset shape: {dataset.data.shape}")
    print(f"Golden CPDAG arcs: {golden.cpdag.arcs()}")
    print(f"Golden CPDAG edges: {golden.cpdag.edges()}")

    # Learn structure with CPC
    cpc = CPCAdapter(alpha=0.1)
    learned = cpc.learn_structure(dataset)

    print(f"\nLearned arcs: {learned.cpdag.arcs()}")
    print(f"Learned edges: {learned.cpdag.edges()}")

    # Compute metrics against golden
    metrics = [SHDMetric(), F1ScoreMetric(), TPRMetric()]
    for metric in metrics:
        score = metric.compute(golden, learned)
        print(f"{metric.name()}: {score}")


if __name__ == "__main__":
    test_generators()
