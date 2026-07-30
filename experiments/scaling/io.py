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

def save_structure(structure: Structure, path: Union[str, Path],
                   dag: gum.DAG = None) -> None:
    """
    Save a Structure to a JSON file.

    Every key names the graph it describes, so a reader can never mistake
    one graph for another:

    - ``cpdag_arcs`` / ``cpdag_edges`` : the CPDAG (always present);
    - ``pdag_arcs`` / ``pdag_edges``   : the raw learned pattern, if any;
    - ``dag_arcs``                     : the true DAG (goldens), if given.

    (Legacy files written before this scheme used ambiguous ``arcs`` /
    ``edges`` keys; see :func:`load_structure` and :func:`load_golden_dag`.)

    Parameters
    ----------
    structure : Structure
        The structure to save.
    path : str or Path
        Destination file path (should end in ``.json``).
    dag : gum.DAG, optional
        The true DAG behind a golden structure, stored under ``dag_arcs``.
    """
    cpdag = structure.cpdag
    data = {
        "nodes": sorted(cpdag.nodes()),
        "cpdag_arcs": sorted([tail, head] for tail, head in cpdag.arcs()),
        "cpdag_edges": sorted([min(u, v), max(u, v)] for u, v in cpdag.edges()),
    }
    if structure.pdag is not None:
        pdag = structure.pdag
        data["pdag_arcs"] = sorted([tail, head] for tail, head in pdag.arcs())
        data["pdag_edges"] = sorted([min(u, v), max(u, v)] for u, v in pdag.edges())
    if dag is not None:
        data["dag_arcs"] = sorted([tail, head] for tail, head in dag.arcs())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def load_structure(path: Union[str, Path]) -> Structure:
    """
    Load a Structure from a JSON file saved by :func:`save_structure`.

    Legacy files (ambiguous ``arcs`` / ``edges`` keys) are read as CPDAGs,
    which is correct for all learned structures. Caution: in legacy
    ``golden__*.json`` files those keys hold the true DAG, not a CPDAG —
    use :func:`load_golden_dag` for those.

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

    def build(arcs, edges, graph_cls):
        g = graph_cls()
        for node in data["nodes"]:
            g.addNodeWithId(node)
        for tail, head in arcs:
            g.addArc(tail, head)
        for u, v in edges:
            g.addEdge(u, v)
        return g

    if "cpdag_arcs" in data:
        cpdag = build(data["cpdag_arcs"], data["cpdag_edges"], gum.PDAG)
    else:  # legacy keys
        cpdag = build(data["arcs"], data["edges"], gum.PDAG)
    pdag = None
    if "pdag_arcs" in data:
        # MixedGraph, not PDAG: a raw pattern may contain partially
        # directed cycles, which gum.PDAG rejects.
        pdag = build(data["pdag_arcs"], data["pdag_edges"], gum.MixedGraph)
    return Structure(cpdag, pdag=pdag)


def load_golden_dag(path: Union[str, Path]) -> gum.DAG:
    """
    Load the true DAG from a ``golden__*.json`` file.

    Single place that knows both eras: new files store the DAG under
    ``dag_arcs``; legacy golden files stored it directly under ``arcs``.

    Parameters
    ----------
    path : str or Path
        Source golden file path.

    Returns
    -------
    gum.DAG
    """
    with open(path) as f:
        data = json.load(f)

    arcs = data["dag_arcs"] if "dag_arcs" in data else data["arcs"]
    dag = gum.DAG()
    for node in data["nodes"]:
        dag.addNodeWithId(node)
    for tail, head in arcs:
        dag.addArc(tail, head)
    return dag


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
