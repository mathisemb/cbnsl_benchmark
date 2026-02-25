# Résumé du développement du projet

Ce fichier contient les justifications de certains choix de développement

Le but de ce projet est de faire une pipeline python qui permet de comparer différents algorithmes d’apprentissage de structure dans des réseaux bayésiens continus. La pipeline prend au moins un dataset en entrée (et plus tard aussi un golden BN de référence), exécute tous les algorithmes et compare leurs résultats. Ce n’est qu’un travail d’orchestration. La difficulté est que les algorithmes sont implémentés dans des dépots différents et qu’il faut tout standardiser. 

Dans un premier temps les algorithmes à comparer sont les suivants :

- CMIIC
- CPC
- CMIIC2
- CPC2
- NOTEARS
- Lingam
- Discrétisation + MIIC
- Discrétisation + GHC avec score BDeu

Remarque : étant donné que CPC2 et CMIIC2 sont présentes uniquement dans la branche cpc2_cmiic2 de mon fork https://github.com/mathisemb/otagrum/tree/cpc2_cmiic2 d’otagrum, et que CPC et CMIIC sont également présents dedans, il peut être utile de seulement importer cette branche de mon fork et non la vraie librairie otagrum.

Ensuite les résultas sont résumés dans des heatmaps. Une heatmap par métrique. Les métriques utilisées seront dans un premier temps :

- F1-Score
- Structural Hamming Distance
- True Positive Rate

# Découpage de l'exécution

1. Dataset existant ou génération de données synthétique à partir d'un CBN connu
2. Exécution des algorithmes sur le dataset
3. Calcul des métriques
4. Visualisation des résultats

# Principes et choix de développement

Le code doit être maintenable et évolutif en respectant les conventions Python.

**Contraintes** :
- Les algorithmes et métriques proviennent de dépôts différents → couche *adapter* pour les encapsuler sans les modifier
- La pipeline ne connaît pas les algos/métriques concrètes → orchestration pure
- Ajout dynamique → créer un nouveau fichier suffit, sans modifier le reste
- Golden BN optionnel → fonctionne avec données simulées ou réelles
- Évaluation et visualisation séparées → modifications possibles sans toucher la pipeline

# Objectif initial

- lire tous les dépots, comprendre les inputs/ouputs de chaque algorithme.
- à partir de l’info des inputs/outputs des algorithmes, faire la meilleure classe adapter possible.
- idem pour les métriques.
- proposer des classes pour gérer les algos et les métriques en justifiant les choix.
- proposer une structure de fichiers en justifiant les choix.
- coder une première version de celles ci.
- télécharger et comprendre le dataset https://pubmed.ncbi.nlm.nih.gov/15845847/.
- à partir des informations de ce dataset, écrire une classe Dataset qui pourra adapter n’importe quel dataset pour qu’il soit utilisé par la pipeline.
- coder un premier exemple avec CPC, F1-Score et le dataset https://pubmed.ncbi.nlm.nih.gov/15845847/.
- continuer avec les autres algos.

# État actuel de l'implémentation

## Structure actuelle du projet

```
cbnsl_benchmark/
├── .context/                  # Documentation pour le dev
│   ├── decisions.md           # Décisions de développement (ce fichier)
│   └── useful_links.md        # Liens vers code/docs des dépendances
├── .gitignore
├── algorithms/                # Adapters d'algorithmes d'apprentissage de structure
│   ├── AlgorithmAdapter.py    # Classe abstraite de base
│   ├── CPCAdapter.py          # Un adapter par algorithme
│   └── ...
├── analysis/                  # Analyse et visualisation des résultats
│   ├── BenchmarkAnalyzer.py   # Heatmaps, comparaisons vs golden
│   ├── GridSearch.py          # Grid search sur paramètres (n_bins, etc.)
│   └── ParetoSelector.py      # Sélection Pareto-optimale
├── data/                      # Datasets
├── metrics/                   # Métriques de comparaison de structures
│   ├── MetricAdapter.py       # Classe abstraite de base
│   ├── F1ScoreMetric.py       # Une métrique par fichier
│   └── ...
├── notebooks/                 # Expérimentations
│   └── benchmark_sachs.ipynb  # Benchmark principal sur données Sachs
├── pipeline/                  # Composants core de la pipeline
│   ├── Dataset.py             # Wrapper de données
│   ├── Pipeline.py            # Orchestration principale
│   ├── Result.py              # Stockage des résultats
│   └── Structure.py           # Représentation du CPDAG (MixedGraph)
├── preprocessing/             # Prétraitement des données
│   └── hartemink.py           # Discrétisation Hartemink
├── tests/                     # Tests
│   ├── test_cpc_shd.py        # Un test par scénario
│   └── ...
├── install.sh                 # Script d'installation automatique
├── pyproject.toml             # Configuration du package
└── README.md
```

## Décisions architecturales (état actuel)

### Organisation du projet

**tests/** : tous les tests mélangés.

**Séparation notebooks/ et tests/** : Distinction claire entre exemples dans des notebooks et les tests. Les exemples montrent comment utiliser la lib, les tests vérifient la correction.

**data/ pour datasets** : Gitignoré pour éviter de versionner de gros fichiers.

**Dataset dans pipeline/ (pas dans data/)** : `Dataset` est une abstraction avec logique (wrapper + métadonnées), pas des données brutes. Appartient aux autres abstractions core comme `Result` et `Structure`.

**.context/** : Contient documentation architecture et liens externes. Pas partie du package, seulement pour dev/assistance IA.

### Système de types & Représentations

**Types de graphes pyAgrum** :
- **DAG** : graphe orienté acyclique (structure pure sans probabilités)
- **DAGmodel** : classe C++ interne pour modèles graphiques basés sur un DAG (non accessible directement en Python)
- **BayesNet** : réseau bayésien (chaîne d'héritage : `BayesNet → IBayesNet → DAGmodel → GraphicalModel`)
- **MixedGraph** : classe de base pour graphes avec arcs orientés ET arêtes non orientées
- **PDAG** : hérite de MixedGraph, représente un CPDAG
- **EssentialGraph** : classe utilitaire pour calculer le CPDAG à partir d'un **DAGmodel** (donc BayesNet, mais pas DAG)

**Structure.cpdag typé comme `gum.MixedGraph`** : `MixedGraph` est le type le plus général de pyAgrum supportant à la fois arcs dirigés et edges non-dirigés, ce qui correspond exactement à un CPDAG. `EssentialGraph` avait été envisagé mais il sert à *extraire* un CPDAG depuis un DAG, pas à le stocker. `PDAG` refuserait les cycles non-dirigés (ex : triangle sans v-structure). `MixedGraph` est donc le seul type adapté pour représenter un CPDAG arbitraire.

### Conversion CPDAG → BayesNet pour le calcul de SHD

**Problème** : `GraphicalBNComparator` de pyAgrum attend des `BayesNet` en entrée, mais un `BayesNet` hérite de `DAG` qui ne supporte que des arcs dirigés (`addArc`). Or nos structures apprises sont des CPDAGs (MixedGraph) qui contiennent des arêtes non-dirigées (edges). Une conversion naïve CPDAG → BayesNet perd ces edges, ce qui fausse complètement le calcul de SHD (toutes les distances deviennent 0).

**Solution** : Avant de construire le BayesNet, on complète le CPDAG en DAG complet via `gum.MeekRules().propagateToDAG(cpdag)`. Les Meek rules orientent les edges restantes de manière cohérente avec la classe d'équivalence de Markov. Ensuite, `GraphicalBNComparator.hamming()` retrouve le CPDAG original via `EssentialGraph` avant de comparer.

**Détail du calcul SHD par `hamming()`** : Pour chaque paire de variables, la méthode compare ce qui existe dans les deux CPDAGs et incrémente les compteurs :

| Ref | Test | pure hamming | structural hamming |
|---|---|---|---|
| `A→B` | `A→B` | 0 | 0 |
| `A→B` | `B→A` ou `A—B` | 0 | +1 (mauvaise orientation) |
| `A→B` | rien | +1 | +1 (arc manquant) |
| `A—B` | `A→B` ou `B→A` | 0 | +1 (orienté alors que non-dirigé) |
| `A—B` | `A—B` | 0 | 0 |
| `A—B` | rien | +1 | +1 (edge manquant) |
| rien | quelque chose | +1 | +1 (en trop) |

Notre code utilise `structural hamming` qui compte : arcs manquants + arcs en trop + arcs mal orientés.

**Note** : on pourrait aussi décider de compter 2 erreurs pour les orientations dans le mauvais sens ou les arcs orientés là où il ne devrait rien y avoir.

### Pourquoi F1-Score et TPR sont implémentés à la main

Pour le SHD, on utilise `GraphicalBNComparator.hamming()` de pyAgrum qui reconvertit internement les BayesNets en CPDAGs avant de comparer. Pour le F1-Score et le TPR (recall), deux options existaient dans pyAgrum :

1. **`GraphicalBNComparator.scores()`** (`bn_vs_bn.py`) : compare les **DAGs** directement (`existsArc` sur le BayesNet). Problème : la complétion du CPDAG en DAG via MeekRules est arbitraire au sein de la classe d'équivalence de Markov. Contrairement à `hamming()` qui reconvertit en CPDAG en interne, `scores()` ne le fait pas. Le F1/recall dépendrait donc du DAG choisi par MeekRules, pas du CPDAG réel.

2. **`gum.StructuralComparator`** (binding C++) : compare correctement des PDAG en prenant en compte arcs et edges. Mais le binding SWIG dispatche `MixedGraph` vers la surcharge `UndiGraph` (héritage multiple : `MixedGraph(UndiGraph, DiGraph)`), ce qui fait qu'il ne voit que les edges et ignore les arcs. Vérifié expérimentalement : passer deux `MixedGraph` avec uniquement des arcs donne `precision=nan, recall=nan`. Seuls les objets `gum.PDAG` fonctionnent correctement, mais on ne peut pas utiliser PDAG car un CPDAG peut contenir des cycles non-dirigés (ex : A — B — C — A, triangle sans v-structure) que PDAG refuserait.

**Solution retenue** : implémenter F1 et TPR directement sur `MixedGraph`, en reprenant la stratégie de comptage de `StructuralComparator`. Pour chaque paire de nœuds non ordonnée, on classifie la relation en 10 catégories :

| ref \ test | `→` (arc) | `—` (edge) | `X` (rien) |
|---|---|---|---|
| **`→`** | `true_arc` (TP) / `misoriented_arc` (FP) | `wrong_edge_arc` (FP) | `wrong_none_arc` (FN) |
| **`—`** | `wrong_arc_edge` (FP) | `true_edge` (TP) | `wrong_none_edge` (FN) |
| **`X`** | `wrong_arc_none` (FP) | `wrong_edge_none` (FP) | `true_none` (—) |

Voir https://gitlab.com/agrumery/aGrUM/-/blob/master/src/agrum/BN/algorithms/structuralComparator.h.

**Choix de la stratégie de comptage** : la distinction clé concerne les liens mal orientés ou de mauvais type (arc vs edge). `StructuralComparator` les compte comme FP uniquement (le lien existe dans test mais est incorrect), pas comme FN (le lien de ref n'est pas "absent" dans test, il est juste mal représenté). L'alternative (compter aussi comme FN) double-pénaliserait ces erreurs. On considère qu'un arc mal orienté ou un arc au lieu d'une edge est moins grave qu'un lien complètement absent, d'où le choix de ne compter qu'un FP.

Puis : recall (= TPR) = TP / (TP + FN), precision = TP / (TP + FP), F1 = 2·precision·recall/(precision+recall).

### Discrétisation
Pour les approches s’appliquant à des données discrètes, les performances dépendent fortement de la stratégie de discrétisation utilisée en amont deux méthodes sont explorées et une grille de bin de 1 à 10 sont testées grâce à ces méthode de discretisation :
- **Discrétisation par quantiles** : découpe l’échelle des variables continues en classes de même effectif.
- **Discrétisation Hartemink** : utilisée dans l’étude de Sachs, cette méthode commence par une discrétisation initiale (paramétrable dans notre pipeline), puis agrège les intervalles de manière à maximiser la conservation de l’information mutuelle conditionnelle entre les variables, préservant ainsi au mieux leurs dépendances.

Pour la discrétisation par quantiles on utilise la classe DiscreteTypeProcessor de agrum.

Pour la discrétisation d'Hartemink, on doit l'écrire car il n'existe pas de version disponible de celle ci en Python.

### Conversion vers Structure dans les adapters

**Pipeline de conversion standard (otagrum)** : `NamedDAG` → `BayesNet` → `EssentialGraph(bn)` → `.pdag()` (MixedGraph) → `Structure(cpdag)`.

**Algorithmes retournant une matrice d'adjacence (NOTEARS, LiNGAM)** : ces algorithmes retournent un `np.ndarray` de shape `(d, d)` pondéré. On seuille puis on construit `BayesNet` → `EssentialGraph` → `Structure`.
- NOTEARS : `W[i,j] != 0` ⇒ arc `i → j`. Seuil `w_threshold` (défaut `0.3`).
- LiNGAM : `B[i,j] != 0` ⇒ arc `j → i` (convention inversée par rapport à NOTEARS).

### Design Patterns

**Pattern Adapter pour Algorithmes** : Découple les implémentations externes de notre pipeline. Chaque algo (CPC, NOTEARS, LiNGAM) a son adapter implémentant l'interface `AlgorithmAdapter`.

**Pattern Adapter pour Métriques** : Similaire aux algos, permet d'ajouter nouvelles métriques sans modifier la pipeline.

**Injection de Dépendances** : Pipeline ne crée pas les algos ou métriques, ils sont injectés via `add_algorithm()`. Augmente flexibilité et testabilité.

### Discrétisation intégrée aux adapters

Les algorithmes discrets (MIIC, GHC+BDeu) gèrent la discrétisation en interne via `DiscreteTypeProcessor` de pyAgrum, ou Hartemink. Les paramètres (n_bins, méthode) font partie de la configuration de l'adapter. Il n'y a pas de `DataType` ni de mécanisme de conversion automatique dans la Pipeline : tous les adapters acceptent des données continues.

### Installation & Dépendances

**Environnement Python** : On utilise un venv avec `--system-site-packages` pour avoir accès aux packages C++ compilés (pyAgrum, openturns, otagrum) tout en isolant les packages pip (lingam, notears).

**Comment les packages sont installés (pour Manjaro) :**
- **pyAgrum, openturns** : installés via pacman (paquets Arch) → `/usr/lib/python3.x/site-packages`
- **otagrum** : compilé depuis le fork C++ et installé via `cmake --install` avec `CMAKE_INSTALL_PREFIX=$HOME/.local` → `~/.local/lib/python3.x/site-packages`
- **lingam, notears** : installés via `pip install` dans le venv → `venv/lib/python3.x/site-packages`

**Pourquoi `--system-site-packages` ?** Manjaro (Arch) applique PEP 668 qui interdit pip d'installer hors d'un venv. Un venv standard ne voit pas les packages système (pyAgrum, openturns) ni les packages utilisateur (otagrum dans `~/.local/`). Le flag `--system-site-packages` rend ces packages visibles dans le venv, tout en permettant `pip install` normal pour les packages Python purs (lingam, notears).

**pyproject.toml pour config package** : Standard moderne Python. Définit dépendances et métadonnées du package.

**Script install.sh automatisé** : Installation interactive gérant conda vs build cmake pour otagrum, NO TEARS optionnel, etc.

## Fonctionnalités à implémenter
- [ ] Faire des tests unitaires propres ?
- [ ] Export des résultats (CSV, JSON) ?
- [ ] Mesure du temps d'exécution ?

## Conventions de code
- Code et commentaires en anglais
- Type hints systématiques
- Docstrings au format Google
