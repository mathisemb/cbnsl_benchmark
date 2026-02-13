import otagrum
from algorithms.AlgorithmAdapter import AlgorithmAdapter, DataType
from pipeline.Structure import Structure
import pyagrum as gum

class CPCAdapter(AlgorithmAdapter):
    """
    Adapter for the Continuous PC (CPC) algorithm from otagrum.
    
    CPC is a constraint-based algorithm for learning Bayesian Network structures
    from continuous data using conditional independence tests.
    """

    def __init__(self, alpha: float = 0.05, max_conditioning_set_size: int = None):
        """
        Initialize the CPC adapter
        
        Parameters
        ----------
        alpha : float, optional
            Significance level for independence tests (default: 0.05)
        max_conditioning_set_size : int, optional
            Maximum size of conditioning sets. If None, uses data dimension - 1
        """
        self.alpha = alpha
        self.max_conditioning_set_size = max_conditioning_set_size

    def name(self) -> str:
        """Returns the algorithm name"""
        return "CPC"

    def required_data_type(self) -> DataType:
        """CPC requires continuous data"""
        return DataType.CONTINUOUS

    def learn_structure(self, dataset) -> Structure:
        """
        Learn Bayesian Network structure using CPC algorithm
        
        Parameters
        ----------
        dataset : Dataset
            The dataset to learn from (must be continuous)
            
        Returns
        -------
        Structure
            The learned structure (structure only)
            
        Raises
        ------
        ValueError
            If dataset is not continuous
        """
        if dataset.data_type != DataType.CONTINUOUS:
            raise ValueError(f"CPC requires continuous data, got {dataset.data_type}")

        try:
            import otagrum
        except ImportError:
            raise ImportError(
                "otagrum is required for CPC."
            )

        # Set max conditioning set size
        max_cond_set = self.max_conditioning_set_size
        if max_cond_set is None:
            max_cond_set = dataset.data.shape[1] - 1

        # Create and run CPC learner
        learner = otagrum.ContinuousPC(dataset.data, max_cond_set, self.alpha)
        learner.setVerbosity(False)

        # Learn the skeleton, PDAG, and finally the DAG
        otagrum_dag = learner.learnDAG()

        return self._NamedDAG_to_Structure(otagrum_dag)

    def _NamedDAG_to_Structure(self, named_dag: otagrum.NamedDAG) -> Structure:
        """
        Convert otagrum DAG to Structure
        
        Parameters
        ----------
        named_dag : otagrum.NamedDAG
            The learned DAG from otagrum
            
        Returns
        -------
        Structure
            The converted DAG in Structure format
        """

        # from otagrum.NamedDAG to gum.BayesNet because pyAgrum's comparator works with BayesNets
        bn = gum.BayesNet()
        bn.addVariables([str(node) for node in named_dag.getDAG().nodes()], 2)
        for (tail_id, head_id) in named_dag.getDAG().arcs():
            bn.addArc(tail_id, head_id)

        return Structure(gum.EssentialGraph(bn))
    
    def name(self) -> str:
        """
        Returns the name of the algorithm
        
        Returns
        -------
        str
            The algorithm name
        """
        return "ContinuousPC"
