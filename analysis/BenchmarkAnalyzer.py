"""
BenchmarkAnalyzer for computing metrics and generating visualizations.
"""

from typing import List, Optional
import numpy as np
from pipeline.Result import Result
from pipeline.Structure import Structure
from metrics.MetricAdapter import MetricAdapter


class BenchmarkAnalyzer:
    """
    Analyzes benchmark results by computing metrics and generating visualizations.

    This class handles:
    - Computing metrics between learned structures and ground truth
    - Computing pairwise metrics between different algorithms
    - Generating comparison tables and visualizations
    """

    def __init__(
        self,
        results: List[Result],
        golden_structure: Optional[Structure] = None
    ):
        """
        Initialize the analyzer

        Parameters
        ----------
        results : List[Result]
            List of results from pipeline execution
        golden_structure : Structure, optional
            Ground truth structure for comparison
        """
        self.results = results
        self.golden_structure = golden_structure

    def compute_vs_golden(self, metrics: List[MetricAdapter]) -> None:
        """
        Compute metrics comparing each result against the golden structure

        This method adds metrics to each Result object in-place.

        Parameters
        ----------
        metrics : List[MetricAdapter]
            List of metrics to compute

        Raises
        ------
        ValueError
            If no golden structure is available
        """
        if self.golden_structure is None:
            raise ValueError("No golden structure available for comparison")

        print(f"\n{'='*60}")
        print(f"Computing metrics vs golden structure...")
        print(f"{'='*60}\n")

        for result in self.results:
            for metric in metrics:
                try:
                    metric_value = metric.compute(
                        ref=self.golden_structure,
                        test=result.learned_structure
                    )
                    result.add_metric(metric.name(), metric_value)
                    print(f"  {result.algorithm_name} - {metric.name()}: {metric_value:.4f}")
                except Exception as e:
                    print(f"  ERROR computing {metric.name()} for {result.algorithm_name}: {str(e)}")

        print()

    def compute_pairwise(self, metric: MetricAdapter):
        """
        Compute pairwise metrics between all algorithms

        Parameters
        ----------
        metric : MetricAdapter
            The metric to compute for pairwise comparisons

        Returns
        -------
        pandas.DataFrame
            Matrix of pairwise metric values
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for this method")

        n_algos = len(self.results)
        algo_names = [r.algorithm_name for r in self.results]

        # Initialize matrix
        matrix = np.zeros((n_algos, n_algos))

        print(f"\n{'='*60}")
        print(f"Computing pairwise {metric.name()} between algorithms...")
        print(f"{'='*60}\n")

        # Compute pairwise metrics
        for i, result_i in enumerate(self.results):
            for j, result_j in enumerate(self.results):
                if i == j:
                    matrix[i, j] = 0.0  # Distance to self is 0
                else:
                    try:
                        metric_value = metric.compute(
                            ref=result_i.learned_structure,
                            test=result_j.learned_structure
                        )
                        matrix[i, j] = metric_value
                    except Exception as e:
                        print(f"  ERROR computing {metric.name()} between {algo_names[i]} and {algo_names[j]}: {str(e)}")
                        matrix[i, j] = np.nan

        # Create DataFrame
        df = pd.DataFrame(matrix, index=algo_names, columns=algo_names)
        print(f"\nPairwise {metric.name()} matrix:")
        print(df)
        print()

        return df

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
        for result in self.results:
            row = {
                "Algorithm": result.algorithm_name,
                "Dataset": result.dataset_name,
                "N_Samples": result.dataset_n_samples,
                "N_Features": result.dataset_n_features
            }
            row.update(result.metrics)
            data.append(row)

        return pd.DataFrame(data)

    def plot_heatmap(self, pairwise_matrix, metric_name: str, save_path: Optional[str] = None):
        """
        Plot a heatmap of pairwise metrics

        Parameters
        ----------
        pairwise_matrix : pandas.DataFrame
            The pairwise metric matrix from compute_pairwise()
        metric_name : str
            Name of the metric for the plot title
        save_path : str, optional
            Path to save the plot. If None, displays the plot.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("matplotlib and seaborn are required for this method")

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pairwise_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn_r',
            cbar_kws={'label': metric_name}
        )
        plt.title(f'Pairwise {metric_name} Between Algorithms')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        else:
            plt.show()

        plt.close()
