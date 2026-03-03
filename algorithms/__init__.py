"""
Algorithm adapter registry.

ALL_ALGORITHMS maps display names to (adapter_class, fixed_params) tuples.
Each adapter defines its own DEFAULT_PARAM_GRID used by GridSearch.
"""

from algorithms.CPCAdapter import CPCAdapter
from algorithms.CMIICAdapter import CMIICAdapter
from algorithms.MIICAdapter import MIICAdapter
from algorithms.GHCBDeuAdapter import GHCBDeuAdapter
from algorithms.NOTEARSAdapter import NOTEARSAdapter
from algorithms.NOTEARSDiscreteAdapter import NOTEARSDiscreteAdapter
from algorithms.LiNGAMAdapter import LiNGAMAdapter

ALL_ALGORITHMS = {
    "CPC v1":        (CPCAdapter, {"version": 1}),
    "CPC v2":        (CPCAdapter, {"version": 2}),
    "CMIIC v1":      (CMIICAdapter, {"version": 1}),
    "CMIIC v2":      (CMIICAdapter, {"version": 2}),
    "MIIC":          (MIICAdapter, {}),
    "GHC+BDeu":      (GHCBDeuAdapter, {}),
    "NOTEARS":       (NOTEARSAdapter, {}),
    "NOTEARS Disc.": (NOTEARSDiscreteAdapter, {}),
    "LiNGAM":        (LiNGAMAdapter, {}),
}
