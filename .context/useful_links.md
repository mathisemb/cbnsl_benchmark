# Useful Links for AI Context

This file contains links to external code, documentation, and papers that are relevant to the project.

## Les méthodes à comparer dans le benchmark

- **CPC** (otagrum)
  - [Documentation](https://openturns.github.io/otagrum/master/user_manual/_generated/otagrum.ContinuousPC.html)
  - [Code](https://github.com/openturns/otagrum/blob/master/lib/src/ContinuousPC.cxx)

- **CMIIC** (otagrum)
  - [Documentation](https://openturns.github.io/otagrum/master/user_manual/_generated/otagrum.ContinuousMIIC.html)
  - [Code](https://github.com/openturns/otagrum/blob/master/lib/src/ContinuousMIIC.cxx)

- **CPC2** (otagrum fork with CPC2/CMIIC2)
  - [Code](https://github.com/mathisemb/otagrum/blob/cpc2_cmiic2/lib/src/ContinuousPC2.cxx)

- **CMIIC2** (otagrum fork with CPC2/CMIIC2)
  - [Code](https://github.com/mathisemb/otagrum/blob/cpc2_cmiic2/lib/src/ContinuousMIIC2.cxx)

- **NOTEARS**
  - [Repository](https://github.com/xunzheng/notears)
  - [Paper](https://arxiv.org/pdf/1803.01422)

- **LiNGAM**
  - [Repository](https://github.com/cdt15/lingam)
  - [Paper](https://www.jmlr.org/papers/volume7/shimizu06a/shimizu06a.pdf)

- **Discrétisation + MIIC** (aGrUM, pyAgrum)
  - [Documentation discrétisation](https://pyagrum.gitlab.io/reference/6_pyagrumlib_modules/2_libdiscretetypeprocessor/)
  - [Code discrétisation](https://gitlab.com/agrumery/aGrUM/-/blob/master/wrappers/pyagrum/pyLibs/lib/discreteTypeProcessor.py)
  - [Documentation MIIC](https://pyagrum.gitlab.io/reference/2_bayesian_networks/4_bnlearning/#pyagrum.BNLearner.useMIIC)
  - [Code MIIC](https://gitlab.com/agrumery/aGrUM/-/blob/master/src/agrum/BN/learning/Miic.cpp)

- **Discrétisation + GHC avec score BDeu** (aGrUM, pyAgrum)
  - [Documentation discrétisation](https://pyagrum.gitlab.io/reference/6_pyagrumlib_modules/2_libdiscretetypeprocessor/)
  - [Code discrétisation](https://gitlab.com/agrumery/aGrUM/-/blob/master/wrappers/pyagrum/pyLibs/lib/discreteTypeProcessor.py)
  - [Documentation GHC](https://pyagrum.gitlab.io/reference/2_bayesian_networks/4_bnlearning/#pyagrum.BNLearner.useGreedyHillClimbing)
  - [Code GHC](https://gitlab.com/agrumery/aGrUM/-/blob/master/src/agrum/BN/learning/greedyHillClimbing_tpl.h)
  - [Documentation score BDeu](https://pyagrum.gitlab.io/reference/2_bayesian_networks/4_bnlearning/#pyagrum.BNLearner.useScoreBDeu)
  - [Code score BDeu](https://gitlab.com/agrumery/aGrUM/-/blob/master/src/agrum/BN/learning/scores_and_tests/scoreBDeu.cpp)

## Les métriques pour la comparaison

- **F1-Score** (aGrUM)
  - [Code](https://gitlab.com/agrumery/aGrUM/-/blob/master/src/agrum/BN/algorithms/structuralComparator.cpp)

- **Structural Hamming Distance** (pyAgrum)
  - [Code](https://gitlab.com/agrumery/aGrUM/-/blob/master/wrappers/pyagrum/pyLibs/lib/bn_vs_bn.py)

- **True Positive Rate** (aGrUM)
  - [Code](https://gitlab.com/agrumery/aGrUM/-/blob/master/src/agrum/BN/algorithms/structuralComparator.cpp)

## PyAgrum

- [Documentation principale](https://pyagrum.gitlab.io/reference/)
- [Dossier des graphes](https://gitlab.com/agrumery/aGrUM/-/tree/master/src/agrum/base/graphs)

### Les classes représentant des graphes

- **MixedGraph**
  - [Header](https://gitlab.com/agrumery/aGrUM/-/blob/master/src/agrum/base/graphs/mixedGraph.h)
  - [Implementation](https://gitlab.com/agrumery/aGrUM/-/blob/master/src/agrum/base/graphs/mixedGraph.cpp)

- **PDAG**
  - [Header](https://gitlab.com/agrumery/aGrUM/-/blob/master/src/agrum/base/graphs/PDAG.h)
  - [Implementation](https://gitlab.com/agrumery/aGrUM/-/blob/master/src/agrum/base/graphs/PDAG.cpp)

- **DAG**
  - [Header](https://gitlab.com/agrumery/aGrUM/-/blob/master/src/agrum/base/graphs/DAG.h)
  - [Implementation](https://gitlab.com/agrumery/aGrUM/-/blob/master/src/agrum/base/graphs/DAG.cpp)

- **EssentialGraph** (pour extraire le CPDAG à partir du DAG)
  - [Header](https://gitlab.com/agrumery/aGrUM/-/blob/master/src/agrum/BN/algorithms/essentialGraph.h)
  - [Implementation](https://gitlab.com/agrumery/aGrUM/-/blob/master/src/agrum/BN/algorithms/essentialGraph.cpp)

## Otagrum

- [Repository](https://github.com/openturns/otagrum)
- [Documentation](https://openturns.github.io/otagrum/master/index.html)

## OpenTURNS

- [Repository](https://github.com/openturns/openturns)
- [Documentation](https://openturns.github.io/openturns/latest/index.html)
