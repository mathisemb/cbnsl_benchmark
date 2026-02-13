"""
Dataset class for encapsulating data and metadata.
"""

from typing import List, Optional
import numpy as np
from algorithms.AlgorithmAdapter import DataType
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
    data_type : DataType
        Whether data is continuous or discrete
    """

    def __init__(
        self,
        data: np.ndarray,
        data_type: DataType = DataType.CONTINUOUS,
        golden_structure: Optional[Structure] = None
    ):
        """
        Initialize the dataset
        
        Parameters
        ----------
        data : np.ndarray
            The data matrix (samples × features)
        data_type : DataType, optional
            CONTINUOUS (default) or DISCRETE
        golden_structure : Structure, optional
        """
        self.data = data
        self.data_type = data_type
        self.golden_structure = golden_structure

    @property
    def n_samples(self) -> int:
        """Number of samples"""
        return self.data.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features"""
        return self.data.shape[1]

    def __repr__(self) -> str:
        return (
            f"Dataset(n_samples={self.n_samples}, n_features={self.n_features}, "
            f"data_type={self.data_type.value}, has_golden_structure={self.golden_structure is not None})"
        )
