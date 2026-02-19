"""
Pareto front selection for multi-objective optimization.
"""

from typing import Dict, List, Optional

from analysis.GridSearch import GridSearchResult


def pareto_front(
    results: List[GridSearchResult],
    objectives: Dict[str, bool],
) -> List[GridSearchResult]:
    """Select Pareto-optimal results from a list of grid search results.

    A result A dominates B if A is at least as good as B on all objectives
    and strictly better on at least one. The Pareto front is the set of
    non-dominated results.

    Args:
        results: List of grid search results with scores.
        objectives: Dictionary mapping metric names to their optimization
            direction (True = lower is better, False = higher is better).

    Returns:
        List of Pareto-optimal results (non-dominated).
    """
    valid = [r for r in results if all(m in r.scores for m in objectives)]
    if not valid:
        return []

    front = []
    for candidate in valid:
        dominated = False
        for other in valid:
            if other is candidate:
                continue
            if _dominates(other, candidate, objectives):
                dominated = True
                break
        if not dominated:
            front.append(candidate)

    return front


def _dominates(
    a: GridSearchResult,
    b: GridSearchResult,
    objectives: Dict[str, bool],
) -> bool:
    """Check if result A dominates result B.

    A dominates B if A is at least as good on all objectives
    and strictly better on at least one.
    """
    at_least_as_good = True
    strictly_better = False

    for metric_name, lower_is_better in objectives.items():
        a_val = a.scores[metric_name]
        b_val = b.scores[metric_name]

        if lower_is_better:
            if a_val > b_val:
                at_least_as_good = False
                break
            if a_val < b_val:
                strictly_better = True
        else:
            if a_val < b_val:
                at_least_as_good = False
                break
            if a_val > b_val:
                strictly_better = True

    return at_least_as_good and strictly_better


def best_pareto(
    results: List[GridSearchResult],
    objectives: Dict[str, bool],
    rank_by: str,
    rank_lower_is_better: bool,
) -> Optional[GridSearchResult]:
    """Select the best Pareto-optimal result according to a ranking metric.

    First computes the Pareto front, then picks the result with the best
    value for `rank_by` among the Pareto-optimal results.

    Args:
        results: List of grid search results with scores.
        objectives: Dictionary mapping metric names to their optimization
            direction (True = lower is better, False = higher is better).
        rank_by: Metric name to rank Pareto-optimal results by.
        rank_lower_is_better: If True, pick the lowest value of rank_by.

    Returns:
        The best Pareto-optimal result, or None if no valid results.
    """
    front = pareto_front(results, objectives)
    if not front:
        return None

    if rank_lower_is_better:
        return min(front, key=lambda r: r.scores[rank_by])
    else:
        return max(front, key=lambda r: r.scores[rank_by])
