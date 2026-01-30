"""
Benchmark comparison utilities

Compare model performance against baseline methods
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


class BaselineModel:
    """
    Simple baseline models for comparison
    """

    @staticmethod
    def mean_predictor(train_targets: np.ndarray, test_size: int) -> np.ndarray:
        """
        Mean predictor (predicts the mean of training data)

        Args:
            train_targets: Training targets
            test_size: Number of test samples

        Returns:
            Predictions (all equal to training mean)
        """
        mean_value = np.mean(train_targets)
        return np.full(test_size, mean_value)

    @staticmethod
    def last_value_predictor(train_targets: np.ndarray, test_size: int) -> np.ndarray:
        """
        Last value predictor (predicts the last value of training data)

        Args:
            train_targets: Training targets
            test_size: Number of test samples

        Returns:
            Predictions (all equal to last training value)
        """
        last_value = train_targets[-1]
        return np.full(test_size, last_value)

    @staticmethod
    def linear_regression_predictor(X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   X_test: np.ndarray) -> np.ndarray:
        """
        Linear regression predictor

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features

        Returns:
            Predictions
        """
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(X_train.reshape(-1, 1) if X_train.ndim == 1 else X_train, y_train)
        return model.predict(X_test.reshape(-1, 1) if X_test.ndim == 1 else X_test)


class BenchmarkComparison:
    """
    Compare model performance against baselines
    """

    def __init__(self, save_dir: str = 'results/benchmarks'):
        """
        Initialize benchmark comparison

        Args:
            save_dir: Directory to save benchmark results
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.results = {}

    def compare_regression(self,
                         model_predictions: np.ndarray,
                         targets: np.ndarray,
                         train_targets: Optional[np.ndarray] = None,
                         task_name: str = 'regression'):
        """
        Compare model against baseline regression methods

        Args:
            model_predictions: Model predictions
            targets: Ground truth targets
            train_targets: Training targets (for baselines that need it)
            task_name: Name of the task

        Returns:
            Dictionary with comparison results
        """
        from .metrics import RegressionMetrics

        test_size = len(targets)

        # Model metrics
        model_metrics = RegressionMetrics.compute(model_predictions, targets, prefix='model_')

        results = {
            'task_name': task_name,
            'model': model_metrics,
        }

        # Baseline 1: Mean predictor
        if train_targets is not None:
            mean_pred = BaselineModel.mean_predictor(train_targets, test_size)
            mean_metrics = RegressionMetrics.compute(mean_pred, targets, prefix='mean_baseline_')
            results['mean_baseline'] = mean_metrics

        # Baseline 2: Last value predictor
        if train_targets is not None:
            last_pred = BaselineModel.last_value_predictor(train_targets, test_size)
            last_metrics = RegressionMetrics.compute(last_pred, targets, prefix='last_baseline_')
            results['last_baseline'] = last_metrics

        # Baseline 3: Linear regression (if features available)
        # This would need features, skip for now

        # Compute improvements
        if 'mean_baseline' in results:
            results['improvements_vs_mean'] = self._compute_improvements(
                model_metrics, results['mean_baseline']
            )

        if 'last_baseline' in results:
            results['improvements_vs_last'] = self._compute_improvements(
                model_metrics, results['last_baseline']
            )

        self.results[task_name] = results

        return results

    def compare_classification(self,
                             model_predictions: np.ndarray,
                             targets: np.ndarray,
                             num_classes: int = 4,
                             task_name: str = 'classification'):
        """
        Compare model against baseline classification methods

        Args:
            model_predictions: Model predictions (class labels)
            targets: Ground truth labels
            num_classes: Number of classes
            task_name: Name of the task

        Returns:
            Dictionary with comparison results
        """
        from .metrics import ClassificationMetrics

        # Model metrics
        model_metrics = ClassificationMetrics.compute(
            model_predictions, targets, num_classes, prefix='model_'
        )

        results = {
            'task_name': task_name,
            'model': model_metrics,
        }

        # Baseline 1: Random predictor
        np.random.seed(42)
        random_pred = np.random.randint(0, num_classes, size=len(targets))
        random_metrics = ClassificationMetrics.compute(
            random_pred, targets, num_classes, prefix='random_baseline_'
        )
        results['random_baseline'] = random_metrics

        # Baseline 2: Majority class predictor
        from scipy.stats import mode
        majority_class = mode(targets, keepdims=True).mode[0]
        majority_pred = np.full(len(targets), majority_class)
        majority_metrics = ClassificationMetrics.compute(
            majority_pred, targets, num_classes, prefix='majority_baseline_'
        )
        results['majority_baseline'] = majority_metrics

        # Compute improvements
        results['improvements_vs_random'] = self._compute_improvements(
            model_metrics, results['random_baseline']
        )
        results['improvements_vs_majority'] = self._compute_improvements(
            model_metrics, results['majority_baseline']
        )

        self.results[task_name] = results

        return results

    def _compute_improvements(self,
                            model_metrics: Dict[str, float],
                            baseline_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Compute improvements over baseline

        Args:
            model_metrics: Model metrics
            baseline_metrics: Baseline metrics

        Returns:
            Dictionary with improvements
        """
        improvements = {}

        # For regression metrics (lower is better for errors, higher is better for R²)
        for key in model_metrics:
            if 'mse' in key or 'rmse' in key or 'mae' in key or 'mape' in key:
                # Error metrics (lower is better)
                if key in baseline_metrics:
                    baseline_val = baseline_metrics[key]
                    model_val = model_metrics[key]
                    improvement = (baseline_val - model_val) / (baseline_val + 1e-6) * 100
                    improvements[key] = float(improvement)
            elif 'r2' in key:
                # R² (higher is better)
                if key in baseline_metrics:
                    baseline_val = baseline_metrics[key]
                    model_val = model_metrics[key]
                    improvement = (model_val - baseline_val) / (abs(baseline_val) + 1e-6) * 100
                    improvements[key] = float(improvement)

        # For classification metrics (higher is better)
        for key in model_metrics:
            if 'accuracy' in key or 'precision' in key or 'recall' in key or 'f1' in key:
                if key in baseline_metrics:
                    baseline_val = baseline_metrics[key]
                    model_val = model_metrics[key]
                    improvement = (model_val - baseline_val) / (baseline_val + 1e-6) * 100
                    improvements[key] = float(improvement)

        return improvements

    def format_comparison(self, results: Dict) -> str:
        """
        Format comparison results for printing

        Args:
            results: Results dictionary

        Returns:
            Formatted string
        """
        lines = [
            "="*80,
            f"BENCHMARK COMPARISON: {results.get('task_name', 'Unknown').upper()}",
            "="*80,
        ]

        # Model performance
        if 'model' in results:
            lines.extend([
                "",
                "🎯 Model Performance:",
            ])
            for key, value in results['model'].items():
                if any(metric in key for metric in ['rmse', 'mae', 'r2', 'accuracy', 'f1']):
                    lines.append(f"  {key}: {value:.6f}")

        # Baseline comparisons
        for baseline_name in ['mean_baseline', 'last_baseline', 'random_baseline', 'majority_baseline']:
            if baseline_name in results:
                lines.extend([
                    "",
                    f"📊 {baseline_name.replace('_', ' ').title()}:",
                ])
                for key, value in results[baseline_name].items():
                    if any(metric in key for metric in ['rmse', 'mae', 'r2', 'accuracy', 'f1']):
                        lines.append(f"  {key}: {value:.6f}")

        # Improvements
        for improvement_key in ['improvements_vs_mean', 'improvements_vs_last',
                               'improvements_vs_random', 'improvements_vs_majority']:
            if improvement_key in results:
                lines.extend([
                    "",
                    f"🚀 {improvement_key.replace('_', ' ').title()}:",
                ])
                for key, value in results[improvement_key].items():
                    if any(metric in key for metric in ['rmse', 'mae', 'r2', 'accuracy', 'f1']):
                        lines.append(f"  {key}: {value:+.2f}%")

        lines.append("="*80)

        return "\n".join(lines)

    def save_comparison(self, results: Dict, filename: str = 'comparison.json'):
        """
        Save comparison results to JSON file

        Args:
            results: Results dictionary
            filename: Filename to save
        """
        save_path = self.save_dir / filename

        # Convert numpy types to Python types for JSON serialization
        def convert_to_python_types(obj):
            if isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            else:
                return obj

        results_serializable = convert_to_python_types(results)

        with open(save_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"Comparison results saved to {save_path}")

    def generate_comparison_report(self,
                                  model_predictions: Dict[str, np.ndarray],
                                  targets: Dict[str, np.ndarray],
                                  train_targets: Optional[Dict[str, np.ndarray]] = None):
        """
        Generate comprehensive comparison report for all tasks

        Args:
            model_predictions: Dictionary of model predictions
            targets: Dictionary of targets
            train_targets: Optional dictionary of training targets
        """
        print("\n" + "="*80)
        print("GENERATING BENCHMARK COMPARISON REPORT")
        print("="*80 + "\n")

        all_results = {}

        # RUL prediction comparison
        if 'rul' in model_predictions and 'rul' in targets:
            print("Comparing RUL prediction...")
            train_rul = train_targets['rul'] if train_targets and 'rul' in train_targets else None
            rul_results = self.compare_regression(
                model_predictions['rul'],
                targets['rul'],
                train_rul,
                task_name='RUL_prediction'
            )
            all_results['rul'] = rul_results
            print(self.format_comparison(rul_results))

        # Temperature prediction comparison
        if 'temperature' in model_predictions and 'temperature' in targets:
            print("\nComparing temperature prediction...")
            train_temp = train_targets['temperature'] if train_targets and 'temperature' in train_targets else None
            temp_results = self.compare_regression(
                model_predictions['temperature'],
                targets['temperature'],
                train_temp,
                task_name='Temperature_prediction'
            )
            all_results['temperature'] = temp_results
            print(self.format_comparison(temp_results))

        # Quality score comparison
        if 'quality_score' in model_predictions and 'quality_score' in targets:
            print("\nComparing quality score prediction...")
            train_quality = train_targets['quality_score'] if train_targets and 'quality_score' in train_targets else None
            quality_results = self.compare_regression(
                model_predictions['quality_score'],
                targets['quality_score'],
                train_quality,
                task_name='Quality_score_prediction'
            )
            all_results['quality_score'] = quality_results
            print(self.format_comparison(quality_results))

        # Fault classification comparison
        if 'fault_pred' in model_predictions and 'fault_label' in targets:
            print("\nComparing fault classification...")
            fault_results = self.compare_classification(
                model_predictions['fault_pred'].flatten(),
                targets['fault_label'],
                num_classes=4,
                task_name='Fault_classification'
            )
            all_results['fault'] = fault_results
            print(self.format_comparison(fault_results))

        # Trajectory correction comparison
        if 'displacement_x' in model_predictions and 'displacement_x' in targets:
            print("\nComparing trajectory correction...")
            train_traj = train_targets['displacement_x'] if train_targets and 'displacement_x' in train_targets else None

            # Combine x, y, z for overall trajectory error
            pred_traj = np.stack([
                model_predictions['displacement_x'].flatten(),
                model_predictions['displacement_y'].flatten(),
                model_predictions['displacement_z'].flatten()
            ], axis=1)

            target_traj = np.stack([
                targets['displacement_x'].flatten(),
                targets['displacement_y'].flatten(),
                targets['displacement_z'].flatten()
            ], axis=1)

            traj_results = self._compare_trajectory(
                pred_traj, target_traj, train_traj, task_name='Trajectory_correction'
            )
            all_results['trajectory'] = traj_results
            print(self.format_comparison(traj_results))

        # Save all results
        self.save_comparison(all_results, 'full_comparison.json')

        print("\n✅ Benchmark comparison completed!")
        print(f"📊 Results saved to {self.save_dir}/full_comparison.json")

        return all_results

    def _compare_trajectory(self,
                           model_predictions: np.ndarray,
                           targets: np.ndarray,
                           train_targets: Optional[np.ndarray] = None,
                           task_name: str = 'trajectory') -> Dict:
        """
        Compare trajectory correction performance

        Args:
            model_predictions: Model predictions [n, 3]
            targets: Ground truth [n, 3]
            train_targets: Training targets
            task_name: Task name

        Returns:
            Comparison results
        """
        from .metrics import TrajectoryMetrics

        # Model metrics
        model_metrics = TrajectoryMetrics.compute_displacement_error(
            model_predictions, targets, prefix='model_'
        )

        results = {
            'task_name': task_name,
            'model': model_metrics,
        }

        # Baseline: zero correction (no correction)
        zero_pred = np.zeros_like(targets)
        zero_metrics = TrajectoryMetrics.compute_displacement_error(
            zero_pred, targets, prefix='zero_baseline_'
        )
        results['zero_baseline'] = zero_metrics

        # Baseline: mean correction
        if train_targets is not None:
            mean_correction = np.mean(train_targets, axis=0)
            mean_pred = np.tile(mean_correction, (len(targets), 1))
            mean_metrics = TrajectoryMetrics.compute_displacement_error(
                mean_pred, targets, prefix='mean_baseline_'
            )
            results['mean_baseline'] = mean_metrics

        # Compute improvements (lower error is better)
        results['improvements_vs_zero'] = {}
        for key in model_metrics:
            if 'error' in key and key in zero_metrics:
                baseline_val = zero_metrics[key]
                model_val = model_metrics[key]
                improvement = (baseline_val - model_val) / (baseline_val + 1e-6) * 100
                results['improvements_vs_zero'][key] = float(improvement)

        if 'mean_baseline' in results:
            results['improvements_vs_mean'] = {}
            for key in model_metrics:
                if 'error' in key and key in mean_metrics:
                    baseline_val = mean_metrics[key]
                    model_val = model_metrics[key]
                    improvement = (baseline_val - model_val) / (baseline_val + 1e-6) * 100
                    results['improvements_vs_mean'][key] = float(improvement)

        return results
