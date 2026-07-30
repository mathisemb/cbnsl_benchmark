"""
Structure class for storing the learned structure (CPDAG or DAG).
"""

import pyagrum as gum


def dag_as_a_structure(dag: gum.DAG) -> "Structure":
    """Represent a gum.DAG as a Structure (possible because a DAG is a special case of a PDAG).
    Because metrics expect a Structure object, not a DAG.
    """
    pdag = gum.PDAG()
    for node_id in dag.nodes():
        pdag.addNodeWithId(node_id)
    for tail, head in dag.arcs():
        pdag.addArc(tail, head)
    return Structure(pdag)


class Structure:
    """
    Stores the structure of the learned CPDAG.

    Contains the CPDAG learned by the algorithm. The CPDAG should be a
    gum.PDAG representing the canonical form of the learned DAG.
    """

    def __init__(
        self,
        cpdag: gum.PDAG,
        pdag: gum.MixedGraph = None
    ):
        """
        Initialize a structure

        Parameters
        ----------
        cpdag : gum.PDAG
            The learned CPDAG.
        pdag : gum.MixedGraph, optional
            The raw pattern learned by the algorithm (skeleton + v-structures,
            before Meek closure), when the algorithm exposes it. Stored so that
            any future post-processing can be recomputed offline without
            re-learning. Kept as a MixedGraph: a raw pattern may contain
            partially directed cycles, which gum.PDAG rejects.
        """
        self.cpdag = cpdag
        self.pdag = pdag

    def __str__(self):
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

    def skeleton(self) -> "Structure":
        """Return a new Structure containing only the skeleton (undirected edges).

        All directed arcs are converted to undirected edges.
        """
        skel = gum.PDAG()
        for node_id in self.cpdag.nodes():
            skel.addNodeWithId(node_id)
        for tail, head in self.cpdag.arcs():
            if not skel.existsEdge(tail, head):
                skel.addEdge(tail, head)
        for n1, n2 in self.cpdag.edges():
            if not skel.existsEdge(n1, n2):
                skel.addEdge(n1, n2)
        return Structure(skel)
