"""
Result class for storing pipeline output.
"""

from typing import Dict, Optional
import pyagrum as gum
from pipeline.Dataset import Dataset
from pipeline.Structure import Structure

class Result:
    """
    Stores the result of running an algorithm on a dataset.

    Contains the learned structure, dataset metadata, and all computed metrics.
    """

    def __init__(
        self,
        algorithm_name: str,
        learned_structure: Structure,
        dataset: Dataset
    ):
        """
        Initialize a result

        Parameters
        ----------
        algorithm_name : str
            Name of the algorithm that produced this result
        learned_structure : Structure
            The learned structure
        dataset : Dataset
            The dataset used for learning (only metadata is stored)
        """
        self.algorithm_name = algorithm_name
        self.learned_structure = learned_structure

        # Store only dataset metadata to save memory
        self.dataset_name = dataset.name
        self.dataset_n_samples = dataset.n_samples
        self.dataset_n_features = dataset.n_features
        self.metrics: Dict[str, float] = {}

    def add_metric(self, metric_name: str, value: float) -> None:
        """
        Add a metric value to this result
        
        Parameters
        ----------
        metric_name : str
            Name of the metric
        value : float
            The computed value
        """
        self.metrics[metric_name] = value

    def __str__(self):
        output  = f"Result for {self.algorithm_name}:\n"
        output += f"  Learned Structure: {self.learned_structure}\n"
        output += f"  Dataset: {self.dataset_name} ({self.dataset_n_samples} samples, {self.dataset_n_features} features)\n"
        output += "  Metrics:\n"
        for metric, value in self.metrics.items():
            output += f"    {metric}: {value:.4f}\n"
        return output

    def __repr__(self) -> str:
        metrics_str = ", ".join(
            [f"{k}={v:.4f}" for k, v in self.metrics.items()]
        )
        return f"Result(algorithm={self.algorithm_name}, structure={self.learned_structure}, metrics=[{metrics_str}])"
