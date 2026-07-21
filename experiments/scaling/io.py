"""
I/O utilities for the scaling study.

- Structure serialization to/from JSON (pyAgrum has no native save for PDAG/MixedGraph)
- Results DataFrame save/load as CSV
"""

import json
from pathlib import Path
from typing import Union

import pandas as pd
import pyagrum as gum

from pipeline.Structure import Structure


# ---------------------------------------------------------------------------
# Structure serialization
# ---------------------------------------------------------------------------

def save_structure(structure: Structure, path: Union[str, Path]) -> None:
    """
    Save a Structure (CPDAG) to a JSON file.

    Stores node ids, directed arcs, and undirected edges so the Structure
    can be reconstructed exactly.

    Parameters
    ----------
    structure : Structure
        The structure to save.
    path : str or Path
        Destination file path (should end in ``.json``).
    """
    cpdag = structure.cpdag
    data = {
        "nodes": sorted(cpdag.nodes()),
        "arcs": sorted([tail, head] for tail, head in cpdag.arcs()),
        "edges": sorted([min(u, v), max(u, v)] for u, v in cpdag.edges()),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def load_structure(path: Union[str, Path]) -> Structure:
    """
    Load a Structure from a JSON file saved by :func:`save_structure`.

    Parameters
    ----------
    path : str or Path
        Source file path.

    Returns
    -------
    Structure
    """
    with open(path) as f:
        data = json.load(f)

    pdag = gum.PDAG()
    for node in data["nodes"]:
        pdag.addNodeWithId(node)
    for tail, head in data["arcs"]:
        pdag.addArc(tail, head)
    for u, v in data["edges"]:
        pdag.addEdge(u, v)
    return Structure(pdag)


# ---------------------------------------------------------------------------
# Results DataFrame persistence
# ---------------------------------------------------------------------------

def save_results(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Save (or append to) a results DataFrame as CSV.

    If the file already exists the new rows are appended without rewriting
    the header, allowing incremental accumulation across multiple runs.

    Parameters
    ----------
    df : pd.DataFrame
        New results to persist.
    path : str or Path
        Destination ``.csv`` file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_results(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a results DataFrame from a CSV file.

    Parameters
    ----------
    path : str or Path
        Source ``.csv`` file.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(path)
