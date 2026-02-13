"""
Structure class for storing the learned structure (CPDAG).
"""

import pyagrum as gum

class Structure:
    """
    Stores the structure of the learned CPDAG.
    
    Contains the CPDAG learned by the algorithm.
    """

    def __init__(
        self,
        cpdag: gum.MixedGraph
    ):
        """
        Initialize a structure
        
        Parameters
        ----------
        cpdag : gum.MixedGraph
            The learned CPDAG.
        """
        self.cpdag = cpdag
    
    def __str__(self):
        return self.cpdag.pdag().__str__() # ATTENTION MOCHE : CAR SELF.CPDAG EST UN ESSENTIALGRAPH ICI
    
    def __repr__(self) -> str:
        return f"Structure(cpdag={self.cpdag})"
