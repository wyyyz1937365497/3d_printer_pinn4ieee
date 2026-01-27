"""
Complete evaluation pipeline

Run comprehensive evaluation including metrics, visualizations, and benchmark comparison
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from config import get_config
from models import UnifiedPINNSeq3D
from evaluation.metrics import UnifiedMetrics, compute_model_metrics
from evaluation.visualizer import ResultVisualizer
from evaluation.benchmark import BenchmarkComparison
from utils import set_seed, setup_logger


class EvaluationPipeline:
    """
    Complete evaluation pipeline
    """

    def __init__(self,
                 model_path: str,
                 config_preset: str = 'unified',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 output_dir: str = 'results'):
        """
        Initialize evaluation pipeline

        Args:
            model_path: Path to model checkpoint
            config_preset: Configuration preset
            device: Device to use
            output_dir: Output directory for results
        """
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = setup_logger('evaluation_pipeline', log_dir=str(self.output_dir / 'logs'))

        # Load configuration and model
        self.config = get_config(preset=config_preset)
        self.model = self._load_model(model_path)

        # Initialize components
        self.visualizer = ResultVisualizer(save_dir=str(self.output_dir / 'figures'))
        self.benchmark = BenchmarkComparison(save_dir=str(self.output_dir / 'benchmarks'))

        # Store results
        self.predictions = {}
        self.targets = {}
        self.metrics = {}

        self.logger.info("Evaluation pipeline initialized")

    def _load_model(self, model_path: str) -> UnifiedPINNSeq3D:
        """Load model from checkpoint"""
        self.logger.info(f"Loading model from {model_path}")

        model = UnifiedPINNSeq3D(self.config)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        self.logger.info(f"Model loaded: {model.get_num_params():,} parameters")
        return model

    def collect_predictions(self, data_loader: DataLoader):
        """
        Collect predictions and targets from dataloader

        Args:
            data_loader: Data loader
        """
        self.logger.info("Collecting predictions...")

        self.predictions = {}
        self.targets = {}

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else k
                         for k, v in batch.items()}

                # Forward pass
                outputs = self.model(inputs['features'])

                # Collect predictions
                for key, value in outputs.items():
                    if key == 'encoded':
                        continue
                    if key not in self.predictions:
                        self.predictions[key] = []
                    self.predictions[key].append(value.cpu().numpy())

                # Collect targets
                target_keys = ['rul', 'temperature', 'vibration_x', 'vibration_y',
                             'quality_score', 'fault_label', 'displacement_x',
                             'displacement_y', 'displacement_z']
                for key in target_keys:
                    if key in batch:
                        if key not in self.targets:
                            self.targets[key] = []
                        value = batch[key]
                        if isinstance(value, torch.Tensor):
                            value = value.cpu().numpy()
                        self.targets[key].append(value)

                if (batch_idx + 1) % 10 == 0:
                    self.logger.info(f"Processed {batch_idx + 1}/{len(data_loader)} batches")

        # Concatenate
        for key in self.predictions:
            self.predictions[key] = np.concatenate(self.predictions[key], axis=0)

        for key in self.targets:
            self.targets[key] = np.concatenate(self.targets[key], axis=0)

        self.logger.info("Predictions collected")

    def compute_metrics(self):
        """Compute all metrics"""
        self.logger.info("Computing metrics...")

        self.metrics = UnifiedMetrics.compute_all(
            self.predictions,
            self.targets,
            num_fault_classes=self.config.data.num_fault_classes
        )

        self.logger.info("Metrics computed")

    def save_metrics(self):
        """Save metrics to file"""
        metrics_file = self.output_dir / 'metrics.txt'
        metrics_file.write_text(UnifiedMetrics.format_metrics(self.metrics, precision=6))

        # Also save as JSON
        import json
        json_file = self.output_dir / 'metrics.json'
        with open(json_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

        self.logger.info(f"Metrics saved to {metrics_file} and {json_file}")

    def generate_visualizations(self):
        """Generate all visualizations"""
        self.logger.info("Generating visualizations...")

        self.visualizer.create_evaluation_report(
            self.predictions,
            self.targets,
            self.metrics
        )

        self.logger.info("Visualizations generated")

    def run_benchmark_comparison(self, train_targets: dict = None):
        """Run benchmark comparison"""
        self.logger.info("Running benchmark comparison...")

        self.benchmark.generate_comparison_report(
            self.predictions,
            self.targets,
            train_targets
        )

        self.logger.info("Benchmark comparison completed")

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        self.logger.info("Generating summary report...")

        report_lines = [
            "="*80,
            "COMPREHENSIVE EVALUATION REPORT",
            "="*80,
            "",
            f"Model: {self.config.experiment_name}",
            f"Device: {self.device}",
            f"Parameters: {self.model.get_num_params():,}",
            "",
            "="*80,
            "QUALITY PREDICTION",
            "="*80,
        ]

        # RUL
        if 'rul_rmse' in self.metrics:
            report_lines.extend([
                "\nRUL Prediction:",
                f"  RMSE: {self.metrics['rul_rmse']:.4f} s",
                f"  MAE: {self.metrics['rul_mae']:.4f} s",
                f"  R²: {self.metrics['rul_r2']:.4f}",
            ])

        # Temperature
        if 'temperature_rmse' in self.metrics:
            report_lines.extend([
                "\nTemperature Prediction:",
                f"  RMSE: {self.metrics['temperature_rmse']:.4f} °C",
                f"  MAE: {self.metrics['temperature_mae']:.4f} °C",
                f"  R²: {self.metrics['temperature_r2']:.4f}",
            ])

        # Quality Score
        if 'quality_binary_accuracy' in self.metrics:
            report_lines.extend([
                "\nQuality Score:",
                f"  Binary Accuracy: {self.metrics['quality_binary_accuracy']:.4f}",
                f"  Binary F1: {self.metrics['quality_binary_f1']:.4f}",
            ])

        report_lines.extend([
            "",
            "="*80,
            "FAULT CLASSIFICATION",
            "="*80,
        ])

        if 'fault_accuracy' in self.metrics:
            report_lines.extend([
                f"\nOverall Accuracy: {self.metrics['fault_accuracy']:.4f}",
                f"Overall F1: {self.metrics['fault_f1']:.4f}",
                "",
                "Per-Class Performance:",
            ])

            for i in range(4):
                if f'fault_class{i}_f1' in self.metrics:
                    report_lines.append(
                        f"  Class {i}: F1={self.metrics[f'fault_class{i}_f1']:.4f}, "
                        f"Precision={self.metrics[f'fault_class{i}_precision']:.4f}, "
                        f"Recall={self.metrics[f'fault_class{i}_recall']:.4f}"
                    )

        report_lines.extend([
            "",
            "="*80,
            "TRAJECTORY CORRECTION",
            "="*80,
        ])

        if 'trajectory_error_magnitude_mean' in self.metrics:
            report_lines.extend([
                f"\nMean Error: {self.metrics['trajectory_error_magnitude_mean']:.6f} mm",
                f"Max Error: {self.metrics['trajectory_error_magnitude_max']:.6f} mm",
                f"Median Error: {self.metrics['trajectory_error_magnitude_median']:.6f} mm",
            ])

        report_lines.extend([
            "",
            "="*80,
            "KEY FINDINGS",
            "="*80,
        ])

        # Key findings
        if 'rul_rmse' in self.metrics and self.metrics['rul_rmse'] < 50:
            report_lines.append("✓ RUL prediction error < 50 seconds (good)")
        if 'temperature_rmse' in self.metrics and self.metrics['temperature_rmse'] < 1.0:
            report_lines.append("✓ Temperature prediction error < 1°C (excellent)")
        if 'fault_accuracy' in self.metrics and self.metrics['fault_accuracy'] > 0.95:
            report_lines.append(f"✓ Fault classification accuracy > 95% (excellent)")
        if 'trajectory_error_magnitude_mean' in self.metrics and self.metrics['trajectory_error_magnitude_mean'] < 0.01:
            report_lines.append("✓ Trajectory error < 0.01mm (excellent)")

        report_lines.extend(["", "="*80])

        # Save report
        report_file = self.output_dir / 'summary_report.txt'
        report_file.write_text('\n'.join(report_lines))

        self.logger.info(f"Summary report saved to {report_file}")

        # Print to console
        print('\n'.join(report_lines))

    def run_full_pipeline(self,
                         test_loader: DataLoader,
                         train_targets: dict = None):
        """
        Run complete evaluation pipeline

        Args:
            test_loader: Test data loader
            train_targets: Optional training targets for baseline comparison
        """
        self.logger.info("Starting full evaluation pipeline...")
        print("\n" + "="*80)
        print("EVALUATION PIPELINE")
        print("="*80 + "\n")

        # Step 1: Collect predictions
        print("Step 1/5: Collecting predictions...")
        self.collect_predictions(test_loader)

        # Step 2: Compute metrics
        print("Step 2/5: Computing metrics...")
        self.compute_metrics()

        # Step 3: Save metrics
        print("Step 3/5: Saving metrics...")
        self.save_metrics()

        # Step 4: Generate visualizations
        print("Step 4/5: Generating visualizations...")
        self.generate_visualizations()

        # Step 5: Benchmark comparison
        print("Step 5/5: Running benchmark comparison...")
        self.run_benchmark_comparison(train_targets)

        # Generate summary
        print("\nGenerating summary report...")
        self.generate_summary_report()

        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nResults saved to: {self.output_dir}")
        print(f"  - Metrics: {self.output_dir / 'metrics.txt'}")
        print(f"  - Figures: {self.output_dir / 'figures'}")
        print(f"  - Benchmarks: {self.output_dir / 'benchmarks'}")
        print(f"  - Summary: {self.output_dir / 'summary_report.txt'}\n")


def create_synthetic_dataloader(num_samples=1000, batch_size=64):
    """Create synthetic dataloader for testing"""
    class SyntheticDataset(Dataset):
        def __init__(self, num_samples, seq_len=200, num_features=12):
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
            return len(self.features)

        def __getitem__(self, idx):
            return {
                'features': self.features[idx],
                **{k: v[idx] for k, v in self.targets.items()}
            }

    dataset = SyntheticDataset(num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Complete evaluation pipeline')

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config_preset', type=str, default='unified',
                       choices=['unified', 'quality', 'trajectory'],
                       help='Configuration preset')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to evaluation data (if None, uses synthetic)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create pipeline
    pipeline = EvaluationPipeline(
        model_path=args.model_path,
        config_preset=args.config_preset,
        device=args.device,
        output_dir=args.output_dir
    )

    # Load data
    if args.data_path is None:
        print("Using synthetic data for demonstration...")
        test_loader = create_synthetic_dataloader(num_samples=1000, batch_size=args.batch_size)
        train_targets = None
    else:
        # Load your real data here
        raise NotImplementedError("Please implement data loading for your format")

    # Run pipeline
    pipeline.run_full_pipeline(test_loader, train_targets)


if __name__ == '__main__':
    main()
