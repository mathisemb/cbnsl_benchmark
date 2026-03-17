"""
Algorithm adapter registry.

ALL_ALGORITHMS maps display names to (adapter_class, fixed_params, random_seeds) tuples.
Each adapter defines its own DEFAULT_PARAM_GRID used by GridSearch.
"""

from algorithms.CPCAdapter import CPCAdapter
from algorithms.CMIICAdapter import CMIICAdapter
from algorithms.MIICAdapter import MIICAdapter
from algorithms.GHCBDeuAdapter import GHCBDeuAdapter
from algorithms.NOTEARSAdapter import NOTEARSAdapter
from algorithms.NOTEARSDiscreteAdapter import NOTEARSDiscreteAdapter
from algorithms.LiNGAMAdapter import LiNGAMAdapter

# Each entry: (adapter_class, fixed_params, random_seeds)
# random_seeds is optional (None = deterministic algorithm, run once).
ALL_ALGORITHMS = {
    "CPC":           (CPCAdapter, {}, None),
    "CMIIC":         (CMIICAdapter, {}, None),
    "MIIC":          (MIICAdapter, {}, None),
    "GHC+BDeu":      (GHCBDeuAdapter, {}, None),
    "NOTEARS":       (NOTEARSAdapter, {}, None),
    "NOTEARS Disc.": (NOTEARSDiscreteAdapter, {}, None),
    #"LiNGAM":        (LiNGAMAdapter, {}, list(range(10))),
    "LiNGAM":        (LiNGAMAdapter, {}, [42]),
}
