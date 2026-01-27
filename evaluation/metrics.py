"""
Evaluation metrics for model performance assessment
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, roc_curve
)


class RegressionMetrics:
    """
    Regression metrics for quality prediction and trajectory correction
    """

    @staticmethod
    def compute(predictions: np.ndarray,
                targets: np.ndarray,
                prefix: str = '') -> Dict[str, float]:
        """
        Compute regression metrics

        Args:
            predictions: Predicted values [n_samples] or [n_samples, 1]
            targets: Ground truth values [n_samples] or [n_samples, 1]
            prefix: Prefix for metric names

        Returns:
            Dictionary with metric names and values
        """
        # Flatten if needed
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        if targets.ndim > 1:
            targets = targets.flatten()

        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        # Mean Absolute Percentage Error (MAPE)
        mask = np.abs(targets) > 1e-6
        if mask.sum() > 0:
            mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
        else:
            mape = 0.0

        metrics = {
            f'{prefix}mse': float(mse),
            f'{prefix}rmse': float(rmse),
            f'{prefix}mae': float(mae),
            f'{prefix}r2': float(r2),
            f'{prefix}mape': float(mape),
        }

        return metrics

    @staticmethod
    def compute_for_sequence(predictions: np.ndarray,
                            targets: np.ndarray,
                            prefix: str = '') -> Dict[str, float]:
        """
        Compute metrics for sequence predictions

        Args:
            predictions: Predicted sequences [n_samples, seq_len, n_dims]
            targets: Target sequences [n_samples, seq_len, n_dims]
            prefix: Prefix for metric names

        Returns:
            Dictionary with metric names and values
        """
        # Compute metrics for each dimension
        n_dims = predictions.shape[-1]
        all_metrics = {}

        for dim in range(n_dims):
            dim_metrics = RegressionMetrics.compute(
                predictions[:, :, dim],
                targets[:, :, dim],
                prefix=f'{prefix}dim{dim}_'
            )
            all_metrics.update(dim_metrics)

        # Compute average metrics across all dimensions
        avg_metrics = RegressionMetrics.compute(
                predictions.flatten(),
                targets.flatten(),
                prefix=f'{prefix}avg_'
            )
        all_metrics.update(avg_metrics)

        return all_metrics


class ClassificationMetrics:
    """
    Classification metrics for fault detection
    """

    @staticmethod
    def compute(predictions: np.ndarray,
                targets: np.ndarray,
                num_classes: int = 4,
                prefix: str = '') -> Dict[str, float]:
        """
        Compute classification metrics

        Args:
            predictions: Predicted class labels [n_samples]
            targets: Ground truth class labels [n_samples]
            num_classes: Number of classes
            prefix: Prefix for metric names

        Returns:
            Dictionary with metric names and values
        """
        # Accuracy
        accuracy = accuracy_score(targets, predictions)

        # Precision, Recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )

        metrics = {
            f'{prefix}accuracy': float(accuracy),
            f'{prefix}precision': float(precision),
            f'{prefix}recall': float(recall),
            f'{prefix}f1': float(f1),
        }

        # Add per-class metrics
        for i in range(min(num_classes, len(precision_per_class))):
            metrics[f'{prefix}class{i}_precision'] = float(precision_per_class[i])
            metrics[f'{prefix}class{i}_recall'] = float(recall_per_class[i])
            metrics[f'{prefix}class{i}_f1'] = float(f1_per_class[i])

        return metrics

    @staticmethod
    def compute_confusion_matrix(predictions: np.ndarray,
                                targets: np.ndarray,
                                num_classes: int = 4) -> np.ndarray:
        """
        Compute confusion matrix

        Args:
            predictions: Predicted class labels [n_samples]
            targets: Ground truth class labels [n_samples]
            num_classes: Number of classes

        Returns:
            Confusion matrix [num_classes, num_classes]
        """
        cm = confusion_matrix(targets, predictions, labels=list(range(num_classes)))
        return cm

    @staticmethod
    def compute_with_probabilities(predictions_prob: np.ndarray,
                                  targets: np.ndarray,
                                  prefix: str = '') -> Dict[str, float]:
        """
        Compute classification metrics with probability predictions

        Args:
            predictions_prob: Predicted class probabilities [n_samples, num_classes]
            targets: Ground truth class labels [n_samples]
            prefix: Prefix for metric names

        Returns:
            Dictionary with metric names and values
        """
        # Get predicted classes
        predictions = np.argmax(predictions_prob, axis=1)

        # Compute standard metrics
        metrics = ClassificationMetrics.compute(predictions, targets, prefix=prefix)

        # Compute AUC (one-vs-rest)
        try:
            num_classes = predictions_prob.shape[1]
            if num_classes == 2:
                auc = roc_auc_score(targets, predictions_prob[:, 1])
                metrics[f'{prefix}auc'] = float(auc)
            else:
                # Multi-class AUC (one-vs-rest)
                auc = roc_auc_score(targets, predictions_prob, multi_class='ovr', average='weighted')
                metrics[f'{prefix}auc'] = float(auc)
        except Exception as e:
            # AUC may fail if only one class is present
            pass

        return metrics


class TrajectoryMetrics:
    """
    Specialized metrics for trajectory correction
    """

    @staticmethod
    def compute_displacement_error(predicted_displacement: np.ndarray,
                                  target_displacement: np.ndarray,
                                  prefix: str = '') -> Dict[str, float]:
        """
        Compute displacement error metrics

        Args:
            predicted_displacement: Predicted displacement [n_samples, 3] (dx, dy, dz)
            target_displacement: Target displacement [n_samples, 3]
            prefix: Prefix for metric names

        Returns:
            Dictionary with metric names and values
        """
        # Compute error for each axis
        errors = predicted_displacement - target_displacement

        # Magnitude of error
        error_magnitude = np.linalg.norm(errors, axis=1)

        metrics = {
            f'{prefix}error_magnitude_mean': float(np.mean(error_magnitude)),
            f'{prefix}error_magnitude_std': float(np.std(error_magnitude)),
            f'{prefix}error_magnitude_max': float(np.max(error_magnitude)),
            f'{prefix}error_magnitude_median': float(np.median(error_magnitude)),
        }

        # Per-axis metrics
        for i, axis in enumerate(['x', 'y', 'z']):
            axis_error = errors[:, i]
            metrics[f'{prefix}error_{axis}_mean'] = float(np.mean(np.abs(axis_error)))
            metrics[f'{prefix}error_{axis}_rmse'] = float(np.sqrt(np.mean(axis_error ** 2)))

        return metrics

    @staticmethod
    def compute_improvement_ratio(original_error: np.ndarray,
                                 corrected_error: np.ndarray) -> Dict[str, float]:
        """
        Compute improvement ratio after trajectory correction

        Args:
            original_error: Error before correction [n_samples]
            corrected_error: Error after correction [n_samples]

        Returns:
            Dictionary with improvement metrics
        """
        original_magnitude = np.linalg.norm(original_error, axis=1) if original_error.ndim > 1 else np.abs(original_error)
        corrected_magnitude = np.linalg.norm(corrected_error, axis=1) if corrected_error.ndim > 1 else np.abs(corrected_error)

        # Improvement ratio
        improvement = (original_magnitude - corrected_magnitude) / (original_magnitude + 1e-6)

        # Percentage improvement
        improvement_percentage = improvement * 100

        # Samples that improved
        improved_samples = (improvement > 0).sum()
        total_samples = len(improvement)

        metrics = {
            'improvement_ratio_mean': float(np.mean(improvement)),
            'improvement_ratio_median': float(np.median(improvement)),
            'improvement_percentage_mean': float(np.mean(improvement_percentage)),
            'improved_samples': int(improved_samples),
            'total_samples': int(total_samples),
            'improvement_rate': float(improved_samples / total_samples),
        }

        return metrics


class QualityMetrics:
    """
    Specialized metrics for quality prediction
    """

    @staticmethod
    def compute_rul_metrics(predictions: np.ndarray,
                           targets: np.ndarray,
                           threshold: float = 100.0,
                           prefix: str = 'rul_') -> Dict[str, float]:
        """
        Compute RUL-specific metrics

        Args:
            predictions: Predicted RUL values [n_samples]
            targets: Ground truth RUL values [n_samples]
            threshold: Threshold for early warning (seconds)
            prefix: Prefix for metric names

        Returns:
            Dictionary with RUL metrics
        """
        # Standard regression metrics
        metrics = RegressionMetrics.compute(predictions, targets, prefix=prefix)

        # Early warning accuracy
        pred_warning = predictions < threshold
        true_warning = targets < threshold

        warning_accuracy = accuracy_score(true_warning, pred_warning)

        metrics[f'{prefix}warning_accuracy'] = float(warning_accuracy)

        return metrics

    @staticmethod
    def compute_quality_score_metrics(predictions: np.ndarray,
                                     targets: np.ndarray,
                                     threshold: float = 0.5,
                                     prefix: str = 'quality_') -> Dict[str, float]:
        """
        Compute quality score-specific metrics

        Args:
            predictions: Predicted quality scores [n_samples]
            targets: Ground truth quality scores [n_samples]
            threshold: Threshold for good quality
            prefix: Prefix for metric names

        Returns:
            Dictionary with quality score metrics
        """
        # Standard regression metrics
        metrics = RegressionMetrics.compute(predictions, targets, prefix=prefix)

        # Binary classification based on threshold
        pred_good = predictions > threshold
        true_good = targets > threshold

        # Accuracy
        accuracy = accuracy_score(true_good, pred_good)

        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_good, pred_good, average='binary', zero_division=0
        )

        metrics[f'{prefix}binary_accuracy'] = float(accuracy)
        metrics[f'{prefix}binary_precision'] = float(precision)
        metrics[f'{prefix}binary_recall'] = float(recall)
        metrics[f'{prefix}binary_f1'] = float(f1)

        return metrics


class UnifiedMetrics:
    """
    Unified metrics computation for all tasks
    """

    @staticmethod
    def compute_all(predictions: Dict[str, np.ndarray],
                   targets: Dict[str, np.ndarray],
                   num_fault_classes: int = 4) -> Dict[str, float]:
        """
        Compute metrics for all tasks

        Args:
            predictions: Dictionary of predictions
            targets: Dictionary of targets
            num_fault_classes: Number of fault classes

        Returns:
            Dictionary with all metrics
        """
        all_metrics = {}

        # Quality prediction metrics
        if 'rul' in predictions and 'rul' in targets:
            rul_metrics = QualityMetrics.compute_rul_metrics(
                predictions['rul'], targets['rul']
            )
            all_metrics.update(rul_metrics)

        if 'temperature' in predictions and 'temperature' in targets:
            temp_metrics = RegressionMetrics.compute(
                predictions['temperature'], targets['temperature'],
                prefix='temperature_'
            )
            all_metrics.update(temp_metrics)

        if 'vibration_x' in predictions and 'vibration_x' in targets:
            vib_x_metrics = RegressionMetrics.compute(
                predictions['vibration_x'], targets['vibration_x'],
                prefix='vibration_x_'
            )
            all_metrics.update(vib_x_metrics)

        if 'vibration_y' in predictions and 'vibration_y' in targets:
            vib_y_metrics = RegressionMetrics.compute(
                predictions['vibration_y'], targets['vibration_y'],
                prefix='vibration_y_'
            )
            all_metrics.update(vib_y_metrics)

        if 'quality_score' in predictions and 'quality_score' in targets:
            quality_metrics = QualityMetrics.compute_quality_score_metrics(
                predictions['quality_score'], targets['quality_score']
            )
            all_metrics.update(quality_metrics)

        # Fault classification metrics
        if 'fault_pred' in predictions and 'fault_label' in targets:
            fault_metrics = ClassificationMetrics.compute(
                predictions['fault_pred'],
                targets['fault_label'],
                num_classes=num_fault_classes,
                prefix='fault_'
            )
            all_metrics.update(fault_metrics)

        # Trajectory correction metrics
        if ('displacement_x' in predictions and 'displacement_x' in targets and
            'displacement_y' in predictions and 'displacement_y' in targets and
            'displacement_z' in predictions and 'displacement_z' in targets):

            pred_disp = np.stack([
                predictions['displacement_x'].flatten(),
                predictions['displacement_y'].flatten(),
                predictions['displacement_z'].flatten()
            ], axis=1)

            target_disp = np.stack([
                targets['displacement_x'].flatten(),
                targets['displacement_y'].flatten(),
                targets['displacement_z'].flatten()
            ], axis=1)

            traj_metrics = TrajectoryMetrics.compute_displacement_error(
                pred_disp, target_disp, prefix='trajectory_'
            )
            all_metrics.update(traj_metrics)

        return all_metrics

    @staticmethod
    def format_metrics(metrics: Dict[str, float],
                      precision: int = 4) -> str:
        """
        Format metrics for printing

        Args:
            metrics: Dictionary of metrics
            precision: Number of decimal places

        Returns:
            Formatted string
        """
        lines = ["=" * 80, "Evaluation Metrics", "=" * 80]

        # Group metrics by prefix
        groups = {}
        for key, value in metrics.items():
            prefix = key.split('_')[0]
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append((key, value))

        # Print each group
        for group_name in sorted(groups.keys()):
            lines.append(f"\n{group_name.upper()}:")
            for key, value in sorted(groups[group_name]):
                lines.append(f"  {key}: {value:.{precision}f}")

        lines.append("=" * 80)

        return "\n".join(lines)


def compute_model_metrics(model: torch.nn.Module,
                         data_loader: torch.utils.data.DataLoader,
                         device: str = 'cpu',
                         num_fault_classes: int = 4) -> Dict[str, float]:
    """
    Compute metrics for a model on a dataset

    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to run on
        num_fault_classes: Number of fault classes

    Returns:
        Dictionary with all metrics
    """
    model.eval()
    model.to(device)

    all_predictions = {}
    all_targets = {}

    with torch.no_grad():
        for batch in data_loader:
            # Move to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass
            outputs = model(inputs['features'])

            # Collect predictions
            for key, value in outputs.items():
                if key not in all_predictions:
                    all_predictions[key] = []
                all_predictions[key].append(value.cpu().numpy())

            # Collect targets
            for key in ['rul', 'temperature', 'vibration_x', 'vibration_y',
                      'quality_score', 'fault_label', 'displacement_x',
                      'displacement_y', 'displacement_z']:
                if key in batch:
                    if key not in all_targets:
                        all_targets[key] = []
                    all_targets[key].append(batch[key].cpu().numpy() if
                                          isinstance(batch[key], torch.Tensor) else batch[key])

    # Concatenate all batches
    for key in all_predictions:
        all_predictions[key] = np.concatenate(all_predictions[key], axis=0)

    for key in all_targets:
        all_targets[key] = np.concatenate(all_targets[key], axis=0)

    # Compute metrics
    metrics = UnifiedMetrics.compute_all(all_predictions, all_targets, num_fault_classes)

    return metrics
