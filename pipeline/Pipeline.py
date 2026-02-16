"""
Main pipeline for structure learning and evaluation.
"""

from typing import Dict, List, Optional
from pipeline.Dataset import Dataset
from pipeline.Result import Result
from algorithms.AlgorithmAdapter import AlgorithmAdapter, DataType
from metrics.MetricAdapter import MetricAdapter
from pipeline.Structure import Structure


class DiscretizationStrategy:
    """Interface for discretization strategies"""
    
    def discretize(self, dataset: Dataset) -> Dataset:
        """
        Convert continuous dataset to discrete
        
        Parameters
        ----------
        dataset : Dataset
            The continuous dataset to discretize
            
        Returns
        -------
        Dataset
            The discretized dataset
        """
        raise NotImplementedError


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
        self.metrics: List[MetricAdapter] = []
        self.discretization_strategy: Optional[DiscretizationStrategy] = None
        # self.results = {} will be set in run()

    def add_algorithm(self, algorithm: AlgorithmAdapter) -> None:
        """
        Add an algorithm to the pipeline
        
        Parameters
        ----------
        algorithm : AlgorithmAdapter
            The algorithm adapter to add
        """
        self.algorithms.append(algorithm)

    def add_metric(self, metric: MetricAdapter) -> None:
        """
        Add a metric to the pipeline
        
        Parameters
        ----------
        metric : MetricAdapter
            The metric adapter to add
        """
        self.metrics.append(metric)

    def set_discretization_strategy(self, strategy: DiscretizationStrategy) -> None:
        """
        Set the discretization strategy for converting continuous to discrete data
        
        Parameters
        ----------
        strategy : DiscretizationStrategy
            The discretization strategy to use
        """
        self.discretization_strategy = strategy

    def adapt_dataset(self, algorithm: AlgorithmAdapter) -> Dataset:
        """
        Adapt the dataset to match the algorithm's requirements
        
        If the algorithm requires a different data type than the dataset provides,
        this method performs the necessary conversion (e.g., discretization).
        
        Parameters
        ----------
        algorithm : AlgorithmAdapter
            The algorithm requiring data adaptation
            
        Returns
        -------
        Dataset
            The adapted dataset
            
        Raises
        ------
        ValueError
            If adaptation is required but no strategy is available
        """
        required_type = algorithm.required_data_type()
        current_type = self.dataset.data_type

        if required_type == current_type:
            return self.dataset

        if required_type == DataType.DISCRETE and current_type == DataType.CONTINUOUS:
            if self.discretization_strategy is None:
                raise ValueError(
                    f"Algorithm {algorithm.name()} requires discrete data, "
                    "but no discretization strategy is set."
                )
            return self.discretization_strategy.discretize(self.dataset)

        raise ValueError(
            f"Cannot adapt dataset from {current_type} to {required_type}"
        )

    def run(self) -> Dict[str, Result]:
        """
        Execute the pipeline
        
        For each algorithm:
        1. Adapt the dataset if necessary
        2. Learn the Bayesian Network structure
        3. Compute all metrics
        4. Store the result
        
        Returns
        -------
        Dict[str, Result]
            Dictionary of results, keyed by algorithm name
        """
        self.results = {}

        for algorithm in self.algorithms:
            print(f"\n{'='*60}")
            print(f"Running {algorithm.name()}...")
            print(f"{'='*60}")

            # Adapt dataset if necessary
            try:
                adapted_dataset = self.adapt_dataset(algorithm)
            except Exception as e:
                print(f"ERROR adapting dataset for {algorithm.name()}: {str(e)}")
                continue

            # Learn structure
            learned_structure = algorithm.learn_structure(adapted_dataset)

            # Create result
            result = Result(
                algorithm_name=algorithm.name(),
                learned_structure=learned_structure,
                dataset=adapted_dataset
            )

            # Compute metrics if golden structure is available
            if adapted_dataset.golden_structure is not None:
                for metric in self.metrics:
                    try:
                        # Extract CPDAGs from Structure objects
                        metric_value = metric.compute(
                            ref=adapted_dataset.golden_structure.cpdag,
                            test=learned_structure.cpdag
                        )
                        result.add_metric(metric.name(), metric_value)
                    except Exception as e:
                        print(f"ERROR computing {metric.name()} for {algorithm.name()}: {str(e)}")

            self.results[algorithm.name()] = result

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

    def get_results_dataframe(self):
        """
        Get results in pandas DataFrame format for easy analysis
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with algorithms as rows and metrics as columns
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for this method")

        data = []
        for algorithm_name, result in self.results.items():
            row = {"Algorithm": algorithm_name}
            row.update(result.metrics)
            data.append(row)

        return pd.DataFrame(data)
