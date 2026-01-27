"""
Visualization tools for model evaluation and results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch


class ResultVisualizer:
    """
    Visualize model predictions and evaluation results
    """

    def __init__(self, save_dir: str = 'results/figures'):
        """
        Initialize visualizer

        Args:
            save_dir: Directory to save figures
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10

    def save_figure(self, name: str, dpi: int = 300):
        """
        Save current figure

        Args:
            name: Figure filename
            dpi: Resolution
        """
        save_path = self.save_dir / name
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"Saved figure to {save_path}")

    def plot_regression_results(self,
                               predictions: np.ndarray,
                               targets: np.ndarray,
                               title: str = "Regression Results",
                               xlabel: str = "Ground Truth",
                               ylabel: str = "Prediction",
                               name: str = "regression.png"):
        """
        Plot regression results (scatter plot)

        Args:
            predictions: Predicted values
            targets: Ground truth values
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            name: Figure filename
        """
        # Flatten if needed
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        if targets.ndim > 1:
            targets = targets.flatten()

        fig, ax = plt.subplots(figsize=(8, 8))

        # Scatter plot
        ax.scatter(targets, predictions, alpha=0.5, s=20)

        # Perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(targets, predictions)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               verticalalignment='top', fontsize=12)

        plt.tight_layout()
        self.save_figure(name)

    def plot_residuals(self,
                      predictions: np.ndarray,
                      targets: np.ndarray,
                      title: str = "Residual Plot",
                      name: str = "residuals.png"):
        """
        Plot residual distribution

        Args:
            predictions: Predicted values
            targets: Ground truth values
            title: Plot title
            name: Figure filename
        """
        # Flatten if needed
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        if targets.ndim > 1:
            targets = targets.flatten()

        residuals = predictions - targets

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Residual vs Target
        axes[0].scatter(targets, residuals, alpha=0.5, s=20)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Ground Truth', fontsize=12)
        axes[0].set_ylabel('Residual', fontsize=12)
        axes[0].set_title('Residuals vs Ground Truth', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Residual distribution
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residual', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Residual Distribution', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        self.save_figure(name)

    def plot_confusion_matrix(self,
                             predictions: np.ndarray,
                             targets: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             title: str = "Confusion Matrix",
                             name: str = "confusion_matrix.png"):
        """
        Plot confusion matrix

        Args:
            predictions: Predicted class labels
            targets: Ground truth class labels
            class_names: List of class names
            title: Plot title
            name: Figure filename
        """
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(targets, predictions)

        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'}, ax=ax)

        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        self.save_figure(name)

    def plot_training_history(self,
                             history: Dict[str, List[float]],
                             title: str = "Training History",
                             name: str = "training_history.png"):
        """
        Plot training history

        Args:
            history: Dictionary with training history
            title: Plot title
            name: Figure filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss
        if 'train_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=11)
        axes[0, 0].set_ylabel('Loss', fontsize=11)
        axes[0, 0].set_title('Loss', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Quality metrics
        if 'quality_rmse' in history:
            axes[0, 1].plot(history['quality_rmse'], label='Quality RMSE', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=11)
        axes[0, 1].set_ylabel('RMSE', fontsize=11)
        axes[0, 1].set_title('Quality Prediction RMSE', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Fault accuracy
        if 'fault_accuracy' in history:
            axes[1, 0].plot(history['fault_accuracy'], label='Fault Accuracy', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('Accuracy', fontsize=11)
        axes[1, 0].set_title('Fault Classification Accuracy', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Trajectory error
        if 'trajectory_error' in history:
            axes[1, 1].plot(history['trajectory_error'], label='Trajectory Error', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=11)
        axes[1, 1].set_ylabel('Error (mm)', fontsize=11)
        axes[1, 1].set_title('Trajectory Correction Error', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        self.save_figure(name)

    def plot_trajectory_comparison(self,
                                   original_trajectory: np.ndarray,
                                   corrected_trajectory: np.ndarray,
                                   target_trajectory: np.ndarray,
                                   title: str = "Trajectory Comparison",
                                   name: str = "trajectory_comparison.png"):
        """
        Plot trajectory comparison

        Args:
            original_trajectory: Original trajectory [n_points, 3]
            corrected_trajectory: Corrected trajectory [n_points, 3]
            target_trajectory: Target/ideal trajectory [n_points, 3]
            title: Plot title
            name: Figure filename
        """
        fig = plt.figure(figsize=(14, 6))

        # 3D plot
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')

        ax1.plot(target_trajectory[:, 0], target_trajectory[:, 1], target_trajectory[:, 2],
                'g-', label='Target', linewidth=2, alpha=0.7)
        ax1.plot(original_trajectory[:, 0], original_trajectory[:, 1], original_trajectory[:, 2],
                'r--', label='Original', linewidth=1.5, alpha=0.7)
        ax1.plot(corrected_trajectory[:, 0], corrected_trajectory[:, 1], corrected_trajectory[:, 2],
                'b-', label='Corrected', linewidth=2, alpha=0.7)

        ax1.set_xlabel('X (mm)', fontsize=11)
        ax1.set_ylabel('Y (mm)', fontsize=11)
        ax1.set_zlabel('Z (mm)', fontsize=11)
        ax1.set_title('3D Trajectory', fontsize=12, fontweight='bold')
        ax1.legend()

        # 2D top view
        ax2 = fig.add_subplot(1, 2, 2)

        ax2.plot(target_trajectory[:, 0], target_trajectory[:, 1],
                'g-', label='Target', linewidth=2, alpha=0.7)
        ax2.plot(original_trajectory[:, 0], original_trajectory[:, 1],
                'r--', label='Original', linewidth=1.5, alpha=0.7)
        ax2.plot(corrected_trajectory[:, 0], corrected_trajectory[:, 1],
                'b-', label='Corrected', linewidth=2, alpha=0.7)

        ax2.set_xlabel('X (mm)', fontsize=11)
        ax2.set_ylabel('Y (mm)', fontsize=11)
        ax2.set_title('Top View (X-Y Plane)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        self.save_figure(name)

    def plot_error_distribution(self,
                               errors: Dict[str, np.ndarray],
                               title: str = "Error Distribution",
                               name: str = "error_distribution.png"):
        """
        Plot error distribution for multiple metrics

        Args:
            errors: Dictionary of errors {name: error_array}
            title: Plot title
            name: Figure filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, (error_name, error_values) in enumerate(errors.items()):
            if idx >= len(axes):
                break

            # Flatten if needed
            if error_values.ndim > 1:
                error_values = error_values.flatten()

            axes[idx].hist(error_values, bins=50, edgecolor='black', alpha=0.7)
            axes[idx].axvline(x=0, color='r', linestyle='--', lw=2)
            axes[idx].set_xlabel('Error', fontsize=11)
            axes[idx].set_ylabel('Frequency', fontsize=11)
            axes[idx].set_title(error_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

        # Remove empty subplots
        for idx in range(len(errors), len(axes)):
            fig.delaxes(axes[idx])

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        self.save_figure(name)

    def plot_time_series_prediction(self,
                                   ground_truth: np.ndarray,
                                   prediction: np.ndarray,
                                   title: str = "Time Series Prediction",
                                   name: str = "time_series.png"):
        """
        Plot time series prediction

        Args:
            ground_truth: Ground truth values [n_timesteps]
            prediction: Predicted values [n_timesteps]
            title: Plot title
            name: Figure filename
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        timesteps = np.arange(len(ground_truth))

        ax.plot(timesteps, ground_truth, 'g-', label='Ground Truth', linewidth=2, alpha=0.7)
        ax.plot(timesteps, prediction, 'r--', label='Prediction', linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_figure(name)

    def plot_per_class_metrics(self,
                              metrics: Dict[str, float],
                              num_classes: int = 4,
                              title: str = "Per-Class Metrics",
                              name: str = "per_class_metrics.png"):
        """
        Plot per-class metrics

        Args:
            metrics: Dictionary of metrics
            num_classes: Number of classes
            title: Plot title
            name: Figure filename
        """
        # Extract per-class metrics
        class_names = []
        precisions = []
        recalls = []
        f1_scores = []

        for i in range(num_classes):
            if f'fault_class{i}_precision' in metrics:
                class_names.append(f'Class {i}')
                precisions.append(metrics[f'fault_class{i}_precision'])
                recalls.append(metrics[f'fault_class{i}_recall'])
                f1_scores.append(metrics[f'fault_class{i}_f1'])

        if not class_names:
            print("No per-class metrics found")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(class_names))
        width = 0.25

        ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        self.save_figure(name)

    def create_evaluation_report(self,
                                predictions: Dict[str, np.ndarray],
                                targets: Dict[str, np.ndarray],
                                metrics: Dict[str, float],
                                save_path: Optional[str] = None):
        """
        Create comprehensive evaluation report with all visualizations

        Args:
            predictions: Dictionary of predictions
            targets: Dictionary of targets
            metrics: Dictionary of computed metrics
            save_path: Optional custom save path
        """
        print("Generating evaluation report...")

        # Quality prediction plots
        if 'rul' in predictions and 'rul' in targets:
            self.plot_regression_results(
                predictions['rul'], targets['rul'],
                title="RUL Prediction", xlabel="True RUL (s)", ylabel="Predicted RUL (s)",
                name="quality_rul_prediction.png"
            )
            self.plot_residuals(
                predictions['rul'], targets['rul'],
                title="RUL Residuals", name="quality_rul_residuals.png"
            )

        if 'quality_score' in predictions and 'quality_score' in targets:
            self.plot_regression_results(
                predictions['quality_score'], targets['quality_score'],
                title="Quality Score Prediction", xlabel="True Score", ylabel="Predicted Score",
                name="quality_score_prediction.png"
            )

        # Fault classification plots
        if 'fault_pred' in predictions and 'fault_label' in targets:
            class_names = ['Normal', 'Nozzle Clog', 'Mechanical Loose', 'Motor Fault']
            self.plot_confusion_matrix(
                predictions['fault_pred'].flatten(),
                targets['fault_label'].flatten(),
                class_names=class_names,
                title="Fault Classification Confusion Matrix",
                name="fault_confusion_matrix.png"
            )

        # Trajectory correction plots
        if ('displacement_x' in predictions and 'displacement_x' in targets and
            'displacement_y' in predictions and 'displacement_y' in targets):

            for axis in ['x', 'y', 'z']:
                if f'displacement_{axis}' in predictions:
                    self.plot_regression_results(
                        predictions[f'displacement_{axis}'],
                        targets[f'displacement_{axis}'],
                        title=f"Trajectory Correction ({axis.upper()}-axis)",
                        xlabel=f"True Displacement (mm)", ylabel=f"Predicted Displacement (mm)",
                        name=f"trajectory_{axis}_prediction.png"
                    )

        # Error distribution
        errors = {}
        for key in ['rul', 'temperature', 'displacement_x', 'displacement_y', 'displacement_z']:
            if key in predictions and key in targets:
                error = predictions[key] - targets[key]
                errors[key] = error

        if errors:
            self.plot_error_distribution(errors, name="error_distributions.png")

        print(f"Evaluation report generated and saved to {self.save_dir}")
