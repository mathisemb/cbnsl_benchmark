"""Recalcul unifie des 6 metriques pour tous les runs du scaling.

Produit un CSV par generateur dans scaling/remetrics/, a partir des sources
listees en dur dans SOURCES (la table EST la specification) :
  - agrum2/first_run/  : les algos d'avril hors CPC/CMIIC (structures saines,
    metriques d'avril fausses), runs DAGMA du 11/04 compris ;
  - agrum2/clean_rerun/ : CPC/CMIIC reappris sous agrum2 (metriques NaN car
    StructuralMetrics absent d'agrum2) ;
  - agrum3/            : CPC/CMIIC du run agrum3.

Toutes les metriques (y compris agrum3, deja justes) sont recalculees depuis les
JSON par le meme code : un seul chemin de calcul pour tous les chiffres.
Les goldens herites (first_run) stockent le DAG : ils sont essentialises ici.
Les lignes sans structure apprise (echec d'apprentissage) sont recopiees
telles quelles (NaN + error_msg) ; pour les lignes recalculees, error_msg est
efface (fichier present => apprentissage reussi, l'erreur d'epoque etait
post-apprentissage et n'a plus d'objet).

Usage, dans remetrics/, env cbnsl actif :
    python remetrics.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pyagrum as gum

# --- Verification de la stack : les metriques exigent agrum3 (cbnsl). ---
print(f"python  : {sys.executable}")
print(f"pyagrum {gum.__version__}")
if not gum.__version__.startswith("3."):
    sys.exit("ERREUR : remetrics doit tourner sous agrum3 (cbnsl actif ?). Abandon.")

# --- Imports projet (racine du repo = 3 niveaux au-dessus de ce fichier) ---
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.scaling.io import load_golden_dag, load_structure
from metrics import ALL_METRICS
from pipeline.Structure import Structure

SCALING = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

ALGOS_AVRIL = ["GHC+BDeu", "LiNGAM", "MIIC", "NOTEARS", "NOTEARS Disc."]

# (dossier relatif a scaling/, version, algos a garder)
SOURCES = {
    "sem_gaussian": [
        ("agrum2/first_run/results/2026-04-02_10h34_sem_gaussian", "agrum2", ALGOS_AVRIL),
        ("agrum2/first_run/results/2026-04-11_00h40_sem_gaussian", "agrum2", ["DAGMA"]),
        ("agrum2/clean_rerun/2026-07-31_16h31_sem_gaussian",       "agrum2", ["CPC", "CMIIC"]),
        ("agrum3/2026-07-30_14h34_sem_gaussian",                   "agrum3", ["CPC", "CMIIC"]),
    ],
    "sem_laplace": [
        ("agrum2/first_run/results/2026-04-01_16h49_sem_laplace",  "agrum2", ALGOS_AVRIL),
        ("agrum2/first_run/results/2026-04-11_01h06_sem_laplace",  "agrum2", ["DAGMA"]),
        ("agrum2/clean_rerun/2026-07-31_19h53_sem_laplace",        "agrum2", ["CPC", "CMIIC"]),
        ("agrum3/2026-07-30_23h47_sem_laplace",                    "agrum3", ["CPC", "CMIIC"]),
    ],
    "cbn_unif_gauss": [
        ("agrum2/first_run/results/2026-04-02_17h41_cbn_unif_gauss", "agrum2", ALGOS_AVRIL),
        ("agrum2/first_run/results/2026-04-11_01h23_cbn_unif_gauss", "agrum2", ["DAGMA"]),
        ("agrum2/clean_rerun/2026-07-31_23h23_cbn_unif_gauss",       "agrum2", ["CPC", "CMIIC"]),
        ("agrum3/2026-07-31_02h24_cbn_unif_gauss",                   "agrum3", ["CPC", "CMIIC"]),
    ],
    "cbn_exp_clayton": [
        ("agrum2/first_run/results/2026-04-03_03h56_cbn_exp_clayton", "agrum2", ALGOS_AVRIL),
        ("agrum2/first_run/results/2026-04-11_01h37_cbn_exp_clayton", "agrum2", ["DAGMA"]),
        ("agrum2/clean_rerun/2026-08-01_04h40_cbn_exp_clayton",       "agrum2", ["CPC", "CMIIC"]),
        ("agrum3/2026-07-31_06h33_cbn_exp_clayton",                   "agrum3", ["CPC", "CMIIC"]),
    ],
    "cbn_unif_mixture": [
        ("agrum2/first_run/results/2026-04-03_12h05_cbn_unif_mixture", "agrum2", ALGOS_AVRIL),
        ("agrum2/first_run/results/2026-04-11_01h46_cbn_unif_mixture", "agrum2", ["DAGMA"]),
        ("agrum2/clean_rerun/2026-08-01_09h41_cbn_unif_mixture",       "agrum2", ["CPC", "CMIIC"]),
        ("agrum3/2026-07-31_10h35_cbn_unif_mixture",                   "agrum3", ["CPC", "CMIIC"]),
    ],
}


def golden_structure(folder: Path, n_vars: int, n_arcs: int, graph_idx: int,
                     cache: dict) -> Structure:
    """Golden CPDAG d'un graphe, quel que soit le format du fichier.

    Nouveaux fichiers : le CPDAG est stocke directement. Fichiers herites :
    ils stockent le DAG, essentialise ici (EssentialGraph, graphes creux).
    """
    path = folder / "graphs" / f"golden__v{n_vars}_a{n_arcs}_g{graph_idx}.json"
    if path not in cache:
        if "cpdag_arcs" in json.loads(path.read_text()):
            cache[path] = load_structure(path)
        else:
            dag = load_golden_dag(path)
            bn = gum.BayesNet()
            for i in dag.nodes():
                bn.add(gum.LabelizedVariable(f"X{i}", "", 2))
            for t, h in dag.arcs():
                bn.addArc(t, h)
            cache[path] = Structure(gum.EssentialGraph(bn).pdag())
    return cache[path]


def remetric_folder(folder: Path, version: str, algos: list) -> pd.DataFrame:
    """Recalcule les metriques d'un dossier de resultats, ligne par ligne.

    Les chemins des JSON sont reconstruits depuis les champs des lignes :
    les colonnes *_path des vieux CSV pointent vers l'ancien agencement.
    """
    df = pd.read_csv(folder / "results.csv")
    df = df[df["algo"].isin(algos)].copy()
    # une colonne error_msg entierement vide est lue comme float64 :
    # la passer en objet pour pouvoir y ecrire des chaines
    df["error_msg"] = df["error_msg"].astype(object)
    cache = {}
    n_done = n_kept_error = 0

    for idx, row in df.iterrows():
        learned_path = (folder / "graphs"
                        / f"{row['algo']}__v{row['n_vars']}_a{row['n_arcs']}"
                          f"_g{row['graph_idx']}_s{row['n_samples']}.json")
        if not learned_path.exists():
            n_kept_error += 1  # ligne en echec d'apprentissage : NaN conserves
            continue

        golden = golden_structure(folder, row["n_vars"], row["n_arcs"],
                                  row["graph_idx"], cache)
        learned = load_structure(learned_path)
        for m in ALL_METRICS:
            df.loc[idx, m.name()] = m.compute(golden, learned)
            df.loc[idx, f"{m.name()}_skeleton"] = m.compute_skeleton(golden, learned)
        # fichier present => apprentissage reussi ; toute erreur residuelle
        # etait post-apprentissage (metriques d'epoque) et n'a plus d'objet
        df.loc[idx, "error_msg"] = ""
        n_done += 1

    df["version"] = version
    df["source"] = str(folder.relative_to(SCALING))
    print(f"  {folder.relative_to(SCALING)} : {n_done} lignes recalculees, "
          f"{n_kept_error} erreurs conservees")
    return df


for generator, sources in SOURCES.items():
    print(f"=== {generator}")
    parts = [remetric_folder(SCALING / folder, version, algos)
             for folder, version, algos in sources]
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(HERE / f"{generator}.csv", index=False)
    print(f"  -> {generator}.csv : {len(out)} lignes")

print("Termine.")
