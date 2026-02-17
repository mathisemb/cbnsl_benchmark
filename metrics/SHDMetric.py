"""
Structural Hamming Distance (SHD) metric for comparing DAGs.

Uses pyAgrum's built-in GraphicalBNComparator.
"""

import pyagrum as gum
import pyagrum.lib.bn_vs_bn as bn_vs_bn
from metrics.MetricAdapter import MetricAdapter


class SHDMetric(MetricAdapter):
    """
    Structural Hamming Distance (SHD) metric.
    
    SHD compares the structure of two DAGs by counting differences in arcs.
    Uses pyAgrum's GraphicalBNComparator.hamming() method.
    """

    def name(self) -> str:
        """Returns 'SHD'"""
        return "SHD"

    def compute(self, ref: gum.DAG, test: gum.DAG = None) -> float:
        """
        Compute Structural Hamming Distance between two DAGs
        
        Parameters
        ----------
        ref : gum.DAG
            First DAG to compare
        test : gum.DAG, optional
            Second DAG to compare with. If None, returns 0.0
            
        Returns
        -------
        float
            The SHD value (number of differences)
            
        Raises
        ------
        ValueError
            If DAGs have different node sets
        """
        if test is None:
            return 0.0

        # Convert DAGs to BayesNets for pyAgrum's comparator
        bn_ref = self._dag_to_bn(ref)
        bn_test = self._dag_to_bn(test)

        # Use pyAgrum's built-in comparator
        comparator = bn_vs_bn.GraphicalBNComparator(bn_ref, bn_test)
        hamming_result = comparator.hamming()

        # Return structural hamming distance
        return float(hamming_result.get("structural hamming", 0.0))

    def _dag_to_bn(self, dag: gum.DAG) -> gum.BayesNet:
        """
        Convert a DAG to a minimal BayesNet with uniform CPDs
        
        Parameters
        ----------
        dag : gum.DAG
            The DAG to convert
            
        Returns
        -------
        gum.BayesNet
            A BayesNet with the same structure (CPDs are uniform)
        """
        bn = gum.BayesNet()

        # Create a mapping from node IDs to names
        node_names = {}
        for node_id in dag.nodes():
            name = f"X{node_id}"
            node_names[node_id] = name
            bn.add(gum.LabelizedVariable(name, name, 2))

        # Add arcs
        for node_id in dag.nodes():
            for child_id in dag.children(node_id):
                parent_name = node_names[node_id]
                child_name = node_names[child_id]
                bn.addArc(bn.idFromName(parent_name), bn.idFromName(child_name))

        # Set uniform CPDs
        for node_name in bn.names():
            cpt = bn.cpt(node_name)
            # Fill with uniform distribution
            uniform_value = 1.0 / cpt.variable(0).domainSize()
            bn.cpt(node_name).fillWith(uniform_value)

        return bn
