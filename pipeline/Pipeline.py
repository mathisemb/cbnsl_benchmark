"""
Main pipeline for structure learning and evaluation.
"""

from typing import List
from pipeline.Dataset import Dataset
from pipeline.Result import Result
from algorithms.AlgorithmAdapter import AlgorithmAdapter


class StructureLearningPipeline:
    """
    Main pipeline for learning and evaluating Bayesian Network structures.
    
    This class orchestrates the execution of multiple algorithms on a dataset,
    computes metrics, and stores results. It handles automatic adaptation of
    data types (e.g., discretization) when needed.
    
    The pipeline works with gum.DAG representations for structure comparison.
    """

    def __init__(self, dataset: Dataset):
        """
        Initialize the pipeline
        
        Parameters
        ----------
        dataset : Dataset
            The dataset to learn from
        """
        self.dataset = dataset
        self.algorithms: List[AlgorithmAdapter] = []

    def add_algorithm(self, algorithm: AlgorithmAdapter) -> None:
        """
        Add an algorithm to the pipeline

        Parameters
        ----------
        algorithm : AlgorithmAdapter
            The algorithm adapter to add
        """
        self.algorithms.append(algorithm)

    def run(self) -> List[Result]:
        """
        Execute the pipeline

        For each algorithm:
        1. Learn the Bayesian Network structure
        2. Store the result

        Returns
        -------
        List[Result]
            List of results for all algorithms
        """
        self.results = []

        for algorithm in self.algorithms:
            print(f"\n{'='*60}")
            print(f"Running {algorithm.name()}...")
            print(f"{'='*60}")

            # Learn structure
            learned_structure = algorithm.learn_structure(self.dataset)

            # Display learned structure
            print()
            learned_structure.display(show_structure=True)

            # Create result
            result = Result(
                algorithm_name=algorithm.name(),
                learned_structure=learned_structure,
                dataset=self.dataset
            )

            self.results.append(result)

        print(f"\n{'='*60}")
        print(f"Pipeline completed. {len(self.results)} algorithms executed.")
        print(f"{'='*60}\n")

        return self.results

    def get_results(self) -> List[Result]:
        """
        Get the results from the last pipeline run

        Returns
        -------
        List[Result]
            The results
        """
        return self.results
