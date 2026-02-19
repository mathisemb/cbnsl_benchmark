# Résumé du projet

Ce dossier .context/ contient la documentation pour l'assistance IA.

Le but de ce projet est de faire une pipeline python qui permet de comparer différents algorithmes d’apprentissage de structure dans des réseaux bayésiens continus. La pipeline prend au moins un dataset en entrée (et plus tard aussi un golden BN de référence), exécute tous les algorithmes et comparer leurs résultats. Ce n’est qu’un travail d’orchestration. La difficulté est que les algorithmes sont implémentés dans des dépots différents et qu’il faut tout standardiser. Aussi il faut que le code puisse accepter l’ajout d’un nouvel algorithme écrit dans un autre langage.

Dans un premier temps les algorithmes à comparer sont les suivants :

- CMIIC
- CPC
- CMIIC2
- CPC2
- NOTEARS
- Lingam
- Discrétisation + MIIC
- Discrétisation + GHC avec score BDeu

Remarque : étant donné que CPC2 et CMIIC2 sont présentes uniquement dans la branche cpc2_cmiic2 de mon fork https://github.com/mathisemb/otagrum/tree/cpc2_cmiic2 d’otagrum, et que CPC et CMIIC sont également présents dedans, il peut être utile de seulement importer cette branche de mon fork et non la vraie librairie otagrum. Peut être que c’est possible avec pip install git+url_depot.

Ensuite les résultas sont résumés dans des heatmaps. Une heatmap par métrique. Les métriques utilisées seront dans un premier temps :

- F1-Score
- Structural Hamming Distance
- True Positive Rate

# Découpage de l'exécution

1. Génération de données synthétique à partir d'un CBN connu (optionnel)
2. Exécution des algorithmes sur le dataset
3. Calcul des métriques
4. Présentation des résultats (heatmaps, exports)

# Principes et choix de développement

Le code doit être maintenable et évolutif en respectant les conventions Python.

**Contraintes** :
- Les algorithmes et métriques proviennent de dépôts différents → couche *adapter* pour les encapsuler sans les modifier
- La pipeline ne connaît pas les algos/métriques concrètes → orchestration pure, couplage minimal
- Ajout dynamique → créer un nouveau fichier suffit, sans modifier le reste
- Golden BN optionnel → fonctionne avec données simulées ou réelles
- Gestion centralisée des types de données → adaptations automatiques (discrétisation si nécessaire)
- Évaluation et visualisation séparées → modifications possibles sans toucher la pipeline
- Architecture extensible → rapide à implémenter, compatible avec futures extensions

---

# Les objectifs

- lire tous les dépots, comprendre et faire un résumé des inputs/ouputs de chaque algorithme.
- à partir de l’info des inputs/outputs des algorithmes, faire la meilleure classe adapter possible.
- idem pour les métriques.
- proposer des classes pour gérer les algos et les métriques en justifiant les choix.
- proposer une structure de fichiers en justifiant les choix.
- coder une première version de celles ci.
- télécharger, regarder et comprendre le dataset https://pubmed.ncbi.nlm.nih.gov/15845847/.
- à partir des informations de ce dataset, écrire une classe Dataset qui pourra adapter n’importe quel dataset pour qu’il soit utilisé par la pipeline.
- coder un premier exemple avec CPC, F1-Score et le dataset https://pubmed.ncbi.nlm.nih.gov/15845847/.
- continuer avec les autres algos.

---

# État actuel de l'implémentation

## Composants implémentés

### Pipeline principal (✅ Fait)
- **Pipeline.py** : Orchestration principale avec la classe `StructureLearningPipeline`
- **Dataset.py** : Wrapper de données avec gestion des types (continu/discret)
- **Result.py** : Stockage des résultats avec dictionnaire de métriques
- **Structure.py** : Représentation du CPDAG avec `gum.EssentialGraph`

### Algorithmes (🔄 En cours)
- **AlgorithmAdapter.py** : Classe abstraite de base pour tous les algorithmes
- **CPCAdapter.py** : Algorithme CPC continu depuis otagrum (✅ implémenté)
- Autres : TODO (CMIIC, CPC2, CMIIC2, NOTEARS, LiNGAM, discrétisation + MIIC/GHC)

### Métriques (🔄 En cours)
- **MetricAdapter.py** : Classe abstraite de base pour toutes les métriques
- **SHDMetric.py** : Structural Hamming Distance (✅ implémenté)
- Autres : TODO (F1-Score, TPR)

### Autres modules
- **discretization/** : TODO (stratégies de conversion continu→discret)
- **visualization/** : TODO (heatmaps et visualisation des résultats)

### Tests & Exemples
- **tests/integration/test_cpc_shd.py** : Test d'intégration basique avec CPC + SHD
- **examples/basic_usage.py** : Exemple d'utilisation complet avec données synthétiques

## Structure actuelle du projet

```
cbnsl_benchmark/
├── pipeline/              # Core pipeline components
│   ├── Pipeline.py        # Main orchestration
│   ├── Dataset.py         # Dataset wrapper
│   ├── Result.py          # Result storage
│   └── Structure.py       # Structure representation (CPDAG)
│
├── algorithms/            # Algorithm adapters
│   ├── AlgorithmAdapter.py    # Base adapter interface
│   ├── CPCAdapter.py          # CPC/CPC2 continu (otagrum) (✅)
│   ├── CMIICAdapter.py        # CMIIC/CMIIC2 continu (otagrum) (✅)
│   ├── MIICAdapter.py         # MIIC discret + discrétisation (pyAgrum) (✅)
│   └── GHCBDeuAdapter.py      # GHC+BDeu discret + discrétisation (pyAgrum) (✅)
│
├── metrics/              # Evaluation metrics
│   ├── MetricAdapter.py      # Base metric interface
│   └── SHDMetric.py         # Structural Hamming Distance (✅)
│
├── analysis/             # Benchmark analysis and visualization
│   └── BenchmarkAnalyzer.py  # Metrics vs golden, pairwise, heatmaps (✅)
│
├── tests/               # Tests
│   ├── unit/            # Unit tests (TODO)
│   ├── integration/     # Integration tests
│   │   └── test_cpc_shd.py
│   └── fixtures/        # Test data (TODO)
│
├── examples/            # Usage examples
│   └── basic_usage.py
│
├── data/                # Datasets
│   └── synthetic/       # Synthetic datasets
│
├── results/             # Benchmark outputs (gitignored)
│
├── .context/            # Contexte IA (pas dans le package)
│   ├── architecture.md  # Ce fichier
│   └── useful_links.md  # Liens vers code/docs externes
│
├── install.sh           # Script d'installation automatique
├── pyproject.toml       # Configuration du package
├── requirements.txt     # Dépendances de base
└── requirements-git.txt # Dépendances Git
```

## Décisions architecturales (état actuel)

### Organisation du projet

**tests/ (pluriel, pas test/)** : Suit la convention Python. Subdivisé en unit/, integration/ et fixtures/ pour une meilleure organisation.

**Séparation examples/ et tests/** : Distinction claire entre exemples d'utilisation et tests. Les exemples montrent comment utiliser la lib, les tests vérifient la correction.

**data/ pour datasets, results/ pour outputs** : Garde le code propre. Les deux sont gitignorés (sauf structure) pour éviter de versionner de gros fichiers.

**Dataset dans pipeline/ (pas dans data/)** : `Dataset` est une abstraction avec logique (wrapper + métadonnées), pas des données brutes. Appartient aux autres abstractions core comme `Result` et `Structure`.

**.context/ pour contexte IA** : Contient documentation architecture et liens externes. Pas partie du package, seulement pour dev/assistance IA.

### Système de types & Représentations

**Structure.cpdag typé comme `gum.EssentialGraph`** :
- PyAgrum utilise `EssentialGraph` comme représentation interne d'un CPDAG
- On garde le nom conceptuel "cpdag" (ça représente un CPDAG) mais avec des type hints honnêtes
- La méthode `.pdag()` convertit vers représentation PDAG pour l'affichage

**Pourquoi pas MixedGraph ?** : Initialement typé comme `MixedGraph`, mais c'était incorrect. `EssentialGraph` est plus spécifique et précis.

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

**Solution retenue** : implémenter F1 et TPR directement sur `MixedGraph`, en reprenant la stratégie de comptage de `StructuralComparator` (la logique C++ est correcte, c'est le binding SWIG qui pose problème). Pour chaque paire de nœuds non ordonnée, on classifie la relation en 10 catégories :

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
- **Discrétisation par quantiles** : découpe l’échelle des variables continues en classes de même effectif, garantissant une distribution uniforme des observations.
- **Discrétisation Hartemink** : utilisée dans l’étude de Sachs, cette méthode commence par une discrétisation initiale (paramétrable dans notre pipeline), puis agrège les intervalles de manière à maximiser la conservation de l’information mutuelle conditionnelle entre les variables, préservant ainsi au mieux leurs dépendances.

Pour la discrétisation par quantiles on utilise la classe DiscreteTypeProcessor de agrum.

### Design Patterns

**Pattern Adapter pour Algorithmes** : Découple les implémentations externes de notre pipeline. Chaque algo (CPC, NOTEARS, LiNGAM) a son adapter implémentant l'interface `AlgorithmAdapter`.

**Pattern Adapter pour Métriques** : Similaire aux algos, permet d'ajouter nouvelles métriques sans modifier la pipeline.

**Injection de Dépendances** : Pipeline ne crée pas les algos ou métriques, ils sont injectés via `add_algorithm()`. Augmente flexibilité et testabilité.

### Discrétisation intégrée aux adapters

Les algorithmes discrets (MIIC, GHC+BDeu) gèrent la discrétisation en interne via `DiscreteTypeProcessor` de pyAgrum. Les paramètres (n_bins, méthode) font partie de la configuration de l'adapter. Il n'y a pas de `DataType` ni de mécanisme de conversion automatique dans la Pipeline : tous les adapters acceptent des données continues.

### Installation & Dépendances

**Environnement Python** : On utilise un venv avec `--system-site-packages` pour avoir accès aux packages C++ compilés (pyAgrum, openturns, otagrum) tout en isolant les packages pip (lingam, notears).

**Comment les packages sont installés :**
- **pyAgrum, openturns** : installés via pacman (paquets Arch) → `/usr/lib/python3.x/site-packages`
- **otagrum** : compilé depuis le fork C++ et installé via `cmake --install` avec `CMAKE_INSTALL_PREFIX=$HOME/.local` → `~/.local/lib/python3.x/site-packages`
- **lingam, notears** : installés via `pip install` dans le venv → `venv/lib/python3.x/site-packages`

**Pourquoi `--system-site-packages` ?** Manjaro (Arch) applique PEP 668 qui interdit pip d'installer hors d'un venv. Un venv standard ne voit pas les packages système (pyAgrum, openturns) ni les packages utilisateur (otagrum dans `~/.local/`). Le flag `--system-site-packages` rend ces packages visibles dans le venv, tout en permettant `pip install` normal pour les packages Python purs (lingam, notears).

**Activation du venv :** `source venv/bin/activate` ou utiliser directement `venv/bin/python`.

**pyproject.toml pour config package** : Standard moderne Python (PEP 621). Définit dépendances et métadonnées du package.

**requirements-git.txt séparé** : Certaines dépendances (otagrum avec CPC2/CMIIC2, notears) viennent de dépôts Git, pas PyPI.

**Script install.sh automatisé** : Installation interactive gérant conda vs build cmake pour otagrum, NO TEARS optionnel, etc.

## Fonctionnalités à implémenter
- [ ] Algorithmes restants : NOTEARS, LiNGAM, Discrétisation Hartemink
- [ ] Métriques restantes : F1-Score, TPR
- [ ] Grille de bins (1-10) pour les algorithmes avec discrétisation
- [ ] Export des résultats (CSV, JSON)
- [ ] Mesure du temps d'exécution

## Conventions de code
- Code et commentaires en anglais
- Type hints systématiques
- Docstrings au format Google
- Pas d'emojis sauf demande explicite
