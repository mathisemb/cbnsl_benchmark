"""
Dataset class for encapsulating data and metadata.
"""

from typing import List, Optional
import numpy as np
from pipeline.Structure import Structure

class Dataset:
    """
    Encapsulates a dataset along with its metadata.

    Attributes
    ----------
    data : np.ndarray
        The data matrix (samples × features)
    feature_names : list of str
        Names of the features
    """

    def __init__(
        self,
        data: np.ndarray,
        golden_structure: Optional[Structure] = None,
        name: Optional[str] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize the dataset

        Parameters
        ----------
        data : np.ndarray
            The data matrix (samples × features)
        golden_structure : Structure, optional
            The true/reference structure for comparison
        name : str, optional
            Name of the dataset (e.g., "sachs_observational")
        feature_names : list of str, optional
            Names of the features. If None, generates ["X0", "X1", ...]
        """
        self.data = data
        self.golden_structure = golden_structure
        self.name = name or "unnamed"
        self.feature_names = feature_names or [f"X{i}" for i in range(data.shape[1])]

    @property
    def n_samples(self) -> int:
        """Number of samples"""
        return self.data.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features"""
        return self.data.shape[1]

    def set_golden_structure(self, golden_structure: Structure) -> None:
        """
        Set the golden/reference structure for this dataset.

        Parameters
        ----------
        golden_structure : Structure
            The true structure to compare against
        """
        self.golden_structure = golden_structure

    def to_dataframe(self):
        """
        Convert the dataset to a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with feature_names as column names
        """
        import pandas as pd
        return pd.DataFrame(self.data, columns=self.feature_names)

    def __repr__(self) -> str:
        return (
            f"Dataset(name='{self.name}', n_samples={self.n_samples}, n_features={self.n_features}, "
            f"has_golden_structure={self.golden_structure is not None})"
        )
