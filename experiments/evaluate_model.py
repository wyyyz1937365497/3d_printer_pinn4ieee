"""
Model evaluation script

Evaluate trained model and generate comprehensive report
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from config import get_config
from models import UnifiedPINNSeq3D
from evaluation.metrics import (
    UnifiedMetrics,
    compute_model_metrics,
    RegressionMetrics,
    ClassificationMetrics,
    TrajectoryMetrics,
    QualityMetrics
)
from evaluation.visualizer import ResultVisualizer
from utils import set_seed, setup_logger


class ModelEvaluator:
    """
    Comprehensive model evaluator
    """

    def __init__(self,
                 model_path: str,
                 config_preset: str = 'unified',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize evaluator

        Args:
            model_path: Path to model checkpoint
            config_preset: Configuration preset name
            device: Device to run evaluation on
        """
        self.device = torch.device(device)
        self.logger = setup_logger('model_evaluation', log_dir='logs')

        # Load configuration
        self.config = get_config(preset=config_preset)
        self.logger.info(f"Loaded configuration preset: {config_preset}")

        # Load model
        self.logger.info(f"Loading model from {model_path}")
        self.model = UnifiedPINNSeq3D(self.config)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.logger.info(f"Model loaded successfully")
        self.logger.info(f"Model parameters: {self.model.get_num_params():,}")

        # Initialize visualizer
        self.visualizer = ResultVisualizer(save_dir='results/figures')

        # Store results
        self.all_predictions = {}
        self.all_targets = {}
        self.metrics = {}

    def evaluate_dataloader(self, data_loader: DataLoader) -> dict:
        """
        Evaluate model on a dataloader

        Args:
            data_loader: Data loader to evaluate

        Returns:
            Dictionary with metrics
        """
        self.logger.info("Evaluating on dataset...")
        self.logger.info(f"Dataset size: {len(data_loader.dataset)}")
        self.logger.info(f"Number of batches: {len(data_loader)}")

        self.all_predictions = {}
        self.all_targets = {}

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Move to device
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # Forward pass
                outputs = self.model(inputs['features'])

                # Collect predictions
                for key, value in outputs.items():
                    if key == 'encoded':
                        continue
                    if key not in self.all_predictions:
                        self.all_predictions[key] = []
                    self.all_predictions[key].append(value.cpu().numpy())

                # Collect targets
                target_keys = ['rul', 'temperature', 'vibration_x', 'vibration_y',
                             'quality_score', 'fault_label', 'displacement_x',
                             'displacement_y', 'displacement_z']
                for key in target_keys:
                    if key in batch:
                        if key not in self.all_targets:
                            self.all_targets[key] = []
                        value = batch[key]
                        if isinstance(value, torch.Tensor):
                            value = value.cpu().numpy()
                        self.all_targets[key].append(value)

                if (batch_idx + 1) % 10 == 0:
                    self.logger.info(f"Processed {batch_idx + 1}/{len(data_loader)} batches")

        # Concatenate all batches
        for key in self.all_predictions:
            self.all_predictions[key] = np.concatenate(self.all_predictions[key], axis=0)

        for key in self.all_targets:
            self.all_targets[key] = np.concatenate(self.all_targets[key], axis=0)

        # Compute metrics
        self.metrics = UnifiedMetrics.compute_all(
            self.all_predictions,
            self.all_targets,
            num_fault_classes=self.config.data.num_fault_classes
        )

        return self.metrics

    def print_metrics(self):
        """Print formatted metrics"""
        metrics_str = UnifiedMetrics.format_metrics(self.metrics, precision=4)
        self.logger.info(f"\n{metrics_str}")
        print(metrics_str)

    def save_metrics(self, save_path: str = 'results/metrics.txt'):
        """
        Save metrics to file

        Args:
            save_path: Path to save metrics
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            f.write(UnifiedMetrics.format_metrics(self.metrics, precision=6))

        self.logger.info(f"Metrics saved to {save_path}")

    def generate_visualizations(self):
        """Generate visualization report"""
        self.logger.info("Generating visualizations...")

        self.visualizer.create_evaluation_report(
            self.all_predictions,
            self.all_targets,
            self.metrics
        )

        self.logger.info("Visualizations saved to results/figures/")

    def generate_summary_report(self, save_path: str = 'results/summary_report.txt'):
        """
        Generate summary report

        Args:
            save_path: Path to save report
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        report_lines = [
            "="*80,
            "MODEL EVALUATION SUMMARY REPORT",
            "="*80,
            "",
            f"Model: {self.config.experiment_name}",
            f"Device: {self.device}",
            f"Model Parameters: {self.model.get_num_params():,}",
            "",
            "="*80,
            "QUALITY PREDICTION METRICS",
            "="*80,
        ]

        # RUL metrics
        if 'rul_rmse' in self.metrics:
            report_lines.extend([
                "",
                "RUL (Remaining Useful Life):",
                f"  RMSE: {self.metrics['rul_rmse']:.4f} seconds",
                f"  MAE: {self.metrics['rul_mae']:.4f} seconds",
                f"  R²: {self.metrics['rul_r2']:.4f}",
            ])

        # Temperature metrics
        if 'temperature_rmse' in self.metrics:
            report_lines.extend([
                "",
                "Temperature Prediction:",
                f"  RMSE: {self.metrics['temperature_rmse']:.4f} °C",
                f"  MAE: {self.metrics['temperature_mae']:.4f} °C",
                f"  R²: {self.metrics['temperature_r2']:.4f}",
            ])

        # Vibration metrics
        if 'vibration_x_rmse' in self.metrics:
            report_lines.extend([
                "",
                "Vibration Prediction (X-axis):",
                f"  RMSE: {self.metrics['vibration_x_rmse']:.6f} mm",
                f"  MAE: {self.metrics['vibration_x_mae']:.6f} mm",
            ])

        if 'vibration_y_rmse' in self.metrics:
            report_lines.extend([
                "",
                "Vibration Prediction (Y-axis):",
                f"  RMSE: {self.metrics['vibration_y_rmse']:.6f} mm",
                f"  MAE: {self.metrics['vibration_y_mae']:.6f} mm",
            ])

        # Quality score metrics
        if 'quality_mse' in self.metrics:
            report_lines.extend([
                "",
                "Quality Score:",
                f"  MSE: {self.metrics['quality_mse']:.6f}",
                f"  Binary Accuracy: {self.metrics['quality_binary_accuracy']:.4f}",
                f"  Binary F1: {self.metrics['quality_binary_f1']:.4f}",
            ])

        report_lines.extend([
            "",
            "="*80,
            "FAULT CLASSIFICATION METRICS",
            "="*80,
        ])

        if 'fault_accuracy' in self.metrics:
            report_lines.extend([
                "",
                f"Overall Accuracy: {self.metrics['fault_accuracy']:.4f}",
                f"Overall Precision: {self.metrics['fault_precision']:.4f}",
                f"Overall Recall: {self.metrics['fault_recall']:.4f}",
                f"Overall F1-Score: {self.metrics['fault_f1']:.4f}",
                "",
                "Per-Class Metrics:",
            ])

            for i in range(self.config.data.num_fault_classes):
                class_names = ['Normal', 'Nozzle Clog', 'Mechanical Loose', 'Motor Fault']
                if i < len(class_names):
                    report_lines.append(f"  {class_names[i]}:")
                if f'fault_class{i}_precision' in self.metrics:
                    report_lines.append(f"    Precision: {self.metrics[f'fault_class{i}_precision']:.4f}")
                if f'fault_class{i}_recall' in self.metrics:
                    report_lines.append(f"    Recall: {self.metrics[f'fault_class{i}_recall']:.4f}")
                if f'fault_class{i}_f1' in self.metrics:
                    report_lines.append(f"    F1-Score: {self.metrics[f'fault_class{i}_f1']:.4f}")

        report_lines.extend([
            "",
            "="*80,
            "TRAJECTORY CORRECTION METRICS",
            "="*80,
        ])

        if 'trajectory_error_magnitude_mean' in self.metrics:
            report_lines.extend([
                "",
                "Displacement Error:",
                f"  Mean Error Magnitude: {self.metrics['trajectory_error_magnitude_mean']:.6f} mm",
                f"  Std Error Magnitude: {self.metrics['trajectory_error_magnitude_std']:.6f} mm",
                f"  Max Error Magnitude: {self.metrics['trajectory_error_magnitude_max']:.6f} mm",
                f"  Median Error Magnitude: {self.metrics['trajectory_error_magnitude_median']:.6f} mm",
            ])

        if 'trajectory_error_x_mean' in self.metrics:
            report_lines.extend([
                "",
                "Per-Axis Error:",
                f"  X-axis: {self.metrics['trajectory_error_x_mean']:.6f} mm (mean), "
                f"{self.metrics['trajectory_error_x_rmse']:.6f} mm (RMSE)",
                f"  Y-axis: {self.metrics['trajectory_error_y_mean']:.6f} mm (mean), "
                f"{self.metrics['trajectory_error_y_rmse']:.6f} mm (RMSE)",
                f"  Z-axis: {self.metrics['trajectory_error_z_mean']:.6f} mm (mean), "
                f"{self.metrics['trajectory_error_z_rmse']:.6f} mm (RMSE)",
            ])

        report_lines.extend([
            "",
            "="*80,
            "SUMMARY",
            "="*80,
        ])

        # Key performance indicators
        report_lines.extend([
            "",
            "Key Performance Indicators:",
        ])

        if 'rul_rmse' in self.metrics:
            report_lines.append(f"  ✓ RUL Prediction RMSE: {self.metrics['rul_rmse']:.2f} seconds")
        if 'temperature_rmse' in self.metrics:
            report_lines.append(f"  ✓ Temperature RMSE: {self.metrics['temperature_rmse']:.2f} °C")
        if 'fault_accuracy' in self.metrics:
            report_lines.append(f"  ✓ Fault Classification Accuracy: {self.metrics['fault_accuracy']*100:.2f}%")
        if 'trajectory_error_magnitude_mean' in self.metrics:
            report_lines.append(f"  ✓ Trajectory Error: {self.metrics['trajectory_error_magnitude_mean']:.4f} mm")

        report_lines.extend([
            "",
            "="*80,
        ])

        # Write report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report_lines))

        self.logger.info(f"Summary report saved to {save_path}")

        # Also print to console
        print('\n'.join(report_lines))


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate trained model')

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to evaluation data (if None, uses synthetic data)')
    parser.add_argument('--config_preset', type=str, default='unified',
                       choices=['unified', 'quality', 'trajectory', 'fast', 'research'],
                       help='Configuration preset')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        config_preset=args.config_preset,
        device=args.device
    )

    # Load data
    if args.data_path is None:
        # Use synthetic data for demo
        print("No data path provided, using synthetic data for demonstration...")

        from torch.utils.data import Dataset

        class SyntheticDataset(Dataset):
            def __init__(self, num_samples=1000, seq_len=200, num_features=12):
                self.num_samples = num_samples
                self.features = torch.randn(num_samples, seq_len, num_features)
                self.targets = {
                    'rul': torch.rand(num_samples, 1) * 1000,
                    'temperature': torch.rand(num_samples, 1) * 50 + 200,
                    'vibration_x': torch.randn(num_samples, 1) * 0.1,
                    'vibration_y': torch.randn(num_samples, 1) * 0.1,
                    'quality_score': torch.rand(num_samples, 1),
                    'fault_label': torch.randint(0, 4, (num_samples,)),
                    'displacement_x': torch.randn(num_samples, 1) * 0.01,
                    'displacement_y': torch.randn(num_samples, 1) * 0.01,
                    'displacement_z': torch.randn(num_samples, 1) * 0.001,
                }

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return {
                    'features': self.features[idx],
                    **{k: v[idx] for k, v in self.targets.items()}
                }

        dataset = SyntheticDataset(num_samples=1000)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    else:
        # Load real data (implement this based on your data format)
        raise NotImplementedError("Please implement data loading for your specific format")

    # Evaluate
    metrics = evaluator.evaluate_dataloader(data_loader)

    # Print metrics
    evaluator.print_metrics()

    # Save metrics
    evaluator.save_metrics()

    # Generate visualizations
    evaluator.generate_visualizations()

    # Generate summary report
    evaluator.generate_summary_report()

    print("\n✅ Evaluation completed successfully!")
    print(f"📊 Metrics saved to: results/metrics.txt")
    print(f"📈 Figures saved to: results/figures/")
    print(f"📝 Summary report saved to: results/summary_report.txt")


if __name__ == '__main__':
    main()
