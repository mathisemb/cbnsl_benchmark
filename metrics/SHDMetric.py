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
        Convert a CPDAG (MixedGraph) to a minimal BayesNet with uniform CPDs

        Parameters
        ----------
        cpdag : gum.MixedGraph
            The CPDAG to convert (can contain arcs and edges)

        Returns
        -------
        gum.BayesNet
            A BayesNet with the same structure (CPDs are uniform)
        """
        bn = gum.BayesNet()

        # Create a mapping from node IDs to names
        node_names = {}
        for node_id in cpdag.nodes():
            name = f"X{node_id}"
            node_names[node_id] = name
            bn.add(gum.LabelizedVariable(name, name, 2))

        # Add directed arcs
        for node_id in cpdag.nodes():
            for child_id in cpdag.children(node_id):
                parent_name = node_names[node_id]
                child_name = node_names[child_id]
                bn.addArc(bn.idFromName(parent_name), bn.idFromName(child_name))

        # Note: Undirected edges in CPDAG are ignored for BayesNet conversion
        # as BayesNet only supports directed arcs

        # Set uniform CPDs
        for node_name in bn.names():
            cpt = bn.cpt(node_name)
            # Fill with uniform distribution
            uniform_value = 1.0 / cpt.variable(0).domainSize()
            bn.cpt(node_name).fillWith(uniform_value)

        return bn
