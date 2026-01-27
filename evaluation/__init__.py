"""
Evaluation module for model assessment

This module provides comprehensive tools for:
- Computing evaluation metrics
- Generating visualizations
- Comparing against baselines
- Creating evaluation reports
"""

from .metrics import (
    RegressionMetrics,
    ClassificationMetrics,
    TrajectoryMetrics,
    QualityMetrics,
    UnifiedMetrics,
    compute_model_metrics,
)

from .visualizer import ResultVisualizer
from .benchmark import BenchmarkComparison, BaselineModel

__all__ = [
    'RegressionMetrics',
    'ClassificationMetrics',
    'TrajectoryMetrics',
    'QualityMetrics',
    'UnifiedMetrics',
    'compute_model_metrics',
    'ResultVisualizer',
    'BenchmarkComparison',
    'BaselineModel',
]
