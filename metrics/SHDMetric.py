"""
Structural Hamming Distance (SHD) metric for comparing structures.

Uses pyAgrum's built-in GraphicalBNComparator.
"""

import pyagrum as gum
import pyagrum.lib.bn_vs_bn as bn_vs_bn
from metrics.MetricAdapter import MetricAdapter
from pipeline.Structure import Structure


class SHDMetric(MetricAdapter):
    """
    Structural Hamming Distance (SHD) metric.

    SHD compares the structure of two CPDAGs by counting differences in arcs and edges.
    Uses pyAgrum's GraphicalBNComparator.hamming() method.
    """

    def name(self) -> str:
        """Returns 'SHD'"""
        return "SHD"

    def compute(self, ref: Structure, test: Structure) -> float:
        """
        Compute Structural Hamming Distance between two structures

        Parameters
        ----------
        ref : Structure
            Reference structure (typically ground truth)
        test : Structure
            Test structure to compare

        Returns
        -------
        float
            The SHD value (number of differences)

        Raises
        ------
        ValueError
            If structures have different node sets
        """
        # Extract CPDAGs from Structure objects
        cpdag_ref = ref.cpdag
        cpdag_test = test.cpdag

        # Convert CPDAGs to BayesNets for pyAgrum's comparator
        bn_ref = self._cpdag_to_bn(cpdag_ref)
        bn_test = self._cpdag_to_bn(cpdag_test)

        # Use pyAgrum's built-in comparator
        comparator = bn_vs_bn.GraphicalBNComparator(bn_ref, bn_test)
        hamming_result = comparator.hamming()

        # Return structural hamming distance
        return float(hamming_result.get("structural hamming", 0.0))

    def _cpdag_to_bn(self, cpdag: gum.MixedGraph) -> gum.BayesNet:
        """
        Convert a CPDAG (MixedGraph) to a BayesNet.

        Uses MeekRules to orient undirected edges into a complete DAG.
        GraphicalBNComparator.hamming() will then recover the CPDAG
        via EssentialGraph before comparing.

        Parameters
        ----------
        cpdag : gum.MixedGraph
            The CPDAG to convert (can contain arcs and edges)

        Returns
        -------
        gum.BayesNet
            A BayesNet with the same equivalence class
        """

        # Add directed arcs
        # Note: Undirected edges in CPDAG are ignored for BayesNet conversion
        # as BayesNet only supports directed arcs
        
        """"
        Note:
        We do not directly convert the cpdag to a BayesNet.
        BayesNet only supports directed arcs (only addArc method).
        To avoid losing undirected edges, we first complete the CPDAG into a DAG using Meek rules.
        The resulting DAG will be in the same Markov equivalence class as the original CPDAG.
        Then, SHD will compare the CPDAGs by extracting the essential graph from the DAGs.
        Also, it's not mandatory to set CPTs to create a valid BayesNet.
        """

        # Complete the CPDAG into a DAG using Meek rules
        meek = gum.MeekRules()
        dag = meek.propagateToDAG(cpdag)

        # Build BayesNet from the DAG
        bn = gum.BayesNet()
        node_names = {}
        for node_id in dag.nodes():
            name = f"X{node_id}"
            node_names[node_id] = name
            bn.add(gum.LabelizedVariable(name, name, 2))

        for arc in dag.arcs():
            bn.addArc(node_names[arc[0]], node_names[arc[1]])

        return bn
