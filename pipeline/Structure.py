"""
Structure class for storing the learned structure (CPDAG).
"""

import pyagrum as gum

class Structure:
    """
    Stores the structure of the learned CPDAG.

    Contains the CPDAG learned by the algorithm. The CPDAG should be a
    gum.MixedGraph (or subclass like gum.PDAG) representing the canonical
    form of the learned DAG.

    For algorithm adapters: see algorithms/ADAPTER_GUIDE.md for how to
    extract a CPDAG from different DAG types.
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
        # Note: For PDAG, calling pdag() returns itself
        # For EssentialGraph (legacy), it converts to PDAG
        if hasattr(self.cpdag, 'pdag'):
            return self.cpdag.pdag().__str__()
        return str(self.cpdag)

    def __repr__(self) -> str:
        return f"Structure(cpdag={self.cpdag})"

    def display(self, show_structure: bool = True) -> None:
        """
        Display the structure information.

        Parameters
        ----------
        show_structure : bool, optional
            If True, display the arcs and edges. If False, only show counts.
        """
        print(f"Structure summary:")
        print(f"  Nodes: {self.cpdag.size()}")
        print(f"  Directed arcs: {self.cpdag.sizeArcs()}")
        print(f"  Undirected edges: {self.cpdag.sizeEdges()}")

        if not show_structure:
            return

        # Display arcs if any
        if self.cpdag.sizeArcs() > 0:
            arcs_list = list(self.cpdag.arcs())
            print(f"\n  Arcs ({len(arcs_list)}):")
            for tail, head in arcs_list:
                print(f"    {tail} → {head}")

        # Display undirected edges if any
        if self.cpdag.sizeEdges() > 0:
            edges_list = list(self.cpdag.edges())
            print(f"\n  Undirected edges ({len(edges_list)}):")
            for node1, node2 in edges_list:
                print(f"    {node1} - {node2}")
