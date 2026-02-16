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
│   └── adapters/
│       └── CPCAdapter.py      # CPC algorithm (✅)
│
├── metrics/              # Evaluation metrics
│   ├── MetricAdapter.py      # Base metric interface
│   └── SHDMetric.py         # Structural Hamming Distance (✅)
│
├── discretization/       # Discretization strategies (TODO)
│   └── __init__.py
│
├── visualization/        # Visualization tools (TODO)
│   └── __init__.py
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

### Design Patterns

**Pattern Adapter pour Algorithmes** : Découple les implémentations externes de notre pipeline. Chaque algo (CPC, NOTEARS, LiNGAM) a son adapter implémentant l'interface `AlgorithmAdapter`.

**Pattern Adapter pour Métriques** : Similaire aux algos, permet d'ajouter nouvelles métriques sans modifier la pipeline.

**Pattern Strategy pour Discrétisation** : Prévu pour gérer différentes stratégies de discrétisation (uniforme, quantile, etc.) sans hard-coder dans les adapters d'algos.

**Injection de Dépendances** : Pipeline ne crée pas les algos ou métriques, ils sont injectés via `add_algorithm()` et `add_metric()`. Augmente flexibilité et testabilité.

### Gestion des types de données

**Adaptation automatique du Dataset** : Pipeline vérifie si le type de données requis par l'algo correspond au type du dataset. Sinon, applique stratégie de discrétisation automatiquement.

**Enum DataType** : Enum simple (CONTINUOUS/DISCRETE) centralise la gestion des types et évite les vérifications basées sur strings.

### Installation & Dépendances

**pyproject.toml pour config package** : Standard moderne Python (PEP 621). Définit dépendances et métadonnées du package.

**requirements-git.txt séparé** : Certaines dépendances (otagrum avec CPC2/CMIIC2, notears) viennent de dépôts Git, pas PyPI.

**Script install.sh automatisé** : Installation interactive gérant conda vs build cmake pour otagrum, NO TEARS optionnel, etc.

## Problèmes connus / TODO

### Bugs à corriger
- ⚠️ **Métriques non calculées dans Pipeline.run()** : Les métriques sont ajoutées au pipeline mais jamais calculées. Besoin d'appeler `metric.compute()` dans la boucle run.
- ⚠️ **Pas de logging** : Utilise actuellement des `print()`. Devrait utiliser le module `logging` Python pour niveaux de log et configuration appropriés.

### Fonctionnalités à implémenter
- [ ] Algorithmes restants : CMIIC, CPC2, CMIIC2, NOTEARS, LiNGAM
- [ ] Métriques restantes : F1-Score, TPR
- [ ] Stratégies de discrétisation et intégration
- [ ] Module de visualisation (heatmaps)
- [ ] Tests unitaires pour tous les composants
- [ ] Intégration dataset réel (Sachs protein dataset)
- [ ] Comparaison avec structure golden et benchmarking
- [ ] Export des résultats (CSV, JSON)
- [ ] Mesure du temps d'exécution
- [ ] Exécution parallèle des algorithmes

### Améliorations architecturales
- [ ] Ajouter logging structuré partout
- [ ] Amélioration gestion d'erreurs (actuellement try/except basique)
- [ ] Complétion des type hints (manquent à certains endroits)
- [ ] Génération documentation (Sphinx)
- [ ] Pipeline CI/CD (GitHub Actions)

## Conventions de code
- Code et commentaires en anglais
- Type hints systématiques
- Docstrings au format Google
- Pas d'emojis sauf demande explicite
