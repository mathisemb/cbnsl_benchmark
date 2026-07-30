"""Generate once the frozen datasets used by compare_agrum2_agrum3.ipynb.

Run from this directory:  python generate_frozen_datasets.py

Writes one CSV per golden ({name}_dataset.csv). The CSVs are the frozen
reference: both kernels (agrum2 and agrum3) load the same files, so any
difference in the learned structures comes from the code, not from the data.
Do not regenerate them under another stack (openturns sampling may differ);
if you do, rerun the notebook in BOTH kernels afterwards.
"""

import sys
from pathlib import Path

sys.path.insert(0, "../..")

import pandas as pd
import pyagrum as gum

from data.generators import create_default_cbn, generate_from_cbn

N_SAMPLES = 2000
SEED = 42

def make_dag(n_nodes, arcs):
    dag = gum.DAG()
    dag.addNodes(n_nodes)
    for tail, head in arcs:
        dag.addArc(tail, head)
    return dag

GOLDEN_DAGS = {
    "collider": make_dag(3, [(0, 2), (1, 2)]),
    "fork":     make_dag(3, [(1, 0), (1, 2)]),
    "chain":    make_dag(3, [(0, 1), (1, 2)]),
    "report_10v": make_dag(10, [(1, 0), (0, 2), (2, 3), (3, 4), (3, 5), (4, 5), (4, 6),
                                (5, 7), (6, 7), (6, 8), (7, 8), (9, 8), (2, 9), (9, 7)]),
}

here = Path(__file__).parent
for name, dag in GOLDEN_DAGS.items():
    var_names = [f"X{i}" for i in range(dag.size())]
    cbn = create_default_cbn(dag, var_names=var_names)
    dataset, _ = generate_from_cbn(cbn, n_samples=N_SAMPLES, seed=SEED)
    df = pd.DataFrame(dataset.data, columns=var_names)
    df.to_csv(here / f"{name}_dataset.csv", index=False)
    print(f"{name}_dataset.csv : {df.shape[0]} lignes x {df.shape[1]} colonnes")

print("genere avec pyagrum", gum.__version__)
