# Guide pour créer des adapters d'algorithmes

Ce guide explique comment créer un adapter pour intégrer un nouvel algorithme d'apprentissage de structure dans le benchmark.

## Principe de base

Un adapter implémente l'interface `AlgorithmAdapter` et doit :
1. Apprendre une structure à partir de données
2. Retourner un objet `Structure` contenant un **CPDAG** (Completed Partially Directed Acyclic Graph)

## Pourquoi un CPDAG ?

Le CPDAG est la **forme canonique** d'un DAG pour la comparaison de structures :
- Plusieurs DAGs peuvent représenter la même classe d'équivalence de Markov
- Le CPDAG représente cette classe de manière unique
- Il contient des arcs orientés (→) et des arêtes non orientées (-)
- Permet une comparaison équitable entre algorithmes

## Types de graphes pyAgrum

- **DAG** : graphe orienté acyclique (structure pure sans probabilités)
- **DAGmodel** : classe C++ interne pour modèles graphiques basés sur un DAG (non accessible directement en Python)
- **BayesNet** : réseau bayésien (chaîne d'héritage : `BayesNet → IBayesNet → DAGmodel → GraphicalModel`)
- **MixedGraph** : classe de base pour graphes avec arcs orientés ET arêtes non orientées
- **PDAG** : hérite de MixedGraph, représente un CPDAG
- **EssentialGraph** : classe utilitaire pour calculer le CPDAG à partir d'un **DAGmodel** (donc BayesNet ✓, mais pas DAG ✗)

## Exemple : CPCAdapter (otagrum)

### Étapes principales de l'apprentissage

1. **Apprendre la structure avec l'algorithme**
   ```python
   learner = otagrum.ContinuousPC(dataset.data, max_cond_set, self.alpha)
   otagrum_dag = learner.learnDAG()
   ```

2. **Convertir en Structure avec CPDAG**
   ```python
   return self._NamedDAG_to_Structure(otagrum_dag)
   ```

### Étapes de conversion : NamedDAG → Structure

1. **Convertir NamedDAG en BayesNet**
   ```python
   bn = gum.BayesNet()
   bn.addVariables([str(node) for node in named_dag.getDAG().nodes()], 2)
   for (tail_id, head_id) in named_dag.getDAG().arcs():
       bn.addArc(tail_id, head_id)
   ```

2. **Extraire le CPDAG via EssentialGraph**
   ```python
   essential_graph = gum.EssentialGraph(bn)
   cpdag = essential_graph.pdag()  # PDAG hérite de MixedGraph
   ```

3. **Créer la Structure**
   ```python
   return Structure(cpdag)
   ```

**Important** : `essential_graph.pdag()` retourne un `PDAG` qui **hérite de** `MixedGraph`.

## Workflow général

```
Algorithme apprend
       ↓
    DAG/NamedDAG/BayesNet
       ↓
Conversion en BayesNet (si nécessaire)
       ↓
EssentialGraph(bn) ← calcule le CPDAG
       ↓
.pdag() ← extrait le PDAG (MixedGraph)
       ↓
Structure(cpdag) ← wrapper final
```

## Vérification du type

Pour vérifier que votre adapter retourne le bon type :

```python
import pyagrum as gum

learned = my_adapter.learn_structure(dataset)

# Doit être True
assert isinstance(learned.cpdag, gum.MixedGraph)

# Souvent un PDAG
print(type(learned.cpdag))  # <class 'pyagrum.pyagrum.PDAG'>
```

## Questions fréquentes

**Q: Pourquoi ne pas utiliser directement EssentialGraph ?**

R: `EssentialGraph` est une classe utilitaire pour le calcul, mais n'hérite pas de `MixedGraph`. Utilisez `.pdag()` pour obtenir un `PDAG` qui hérite correctement de `MixedGraph`.

**Q: Pourquoi `gum.EssentialGraph(dag)` ne fonctionne pas avec un `gum.DAG` ?**

R: `EssentialGraph` attend un `DAGmodel` (classe C++ interne), pas un simple `DAG`. Convertissez d'abord votre DAG en `BayesNet`, puis passez-le à `EssentialGraph`. `BayesNet` fonctionne car il hérite de `DAGmodel` (chaîne : `BayesNet → IBayesNet → DAGmodel`).

**Q: Comment gérer les arêtes non orientées ?**

R: Le CPDAG peut contenir des arêtes non orientées (edges). Pour créer un MixedGraph directement, utilisez `mixed_graph.addEdge(node1, node2)`.
