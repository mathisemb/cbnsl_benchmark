"""
Metric registry.

ALL_METRICS: default metric instances used by Benchmark.
OBJECTIVES: {metric_name: lower_is_better} for Pareto selection.
"""

from metrics.SHDMetric import SHDMetric
from metrics.F1ScoreMetric import F1ScoreMetric
from metrics.TPRMetric import TPRMetric

ALL_METRICS = [SHDMetric(), F1ScoreMetric(), TPRMetric()]

OBJECTIVES = {"SHD": True, "F1-Score": False, "TPR": False}
