"""
Comprehensive Evaluation Script for Enhanced Implicit State Inference Model

Features:
1. Load trained model from checkpoint
2. Comprehensive metrics (RMSE, MAE, R², MAPE, etc.)
3. Advanced visualizations (scatter, residuals, error distribution)
4. Per-sample error analysis
5. JSON + HTML report generation
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from config import get_config
from models.implicit import ImplicitStateTCN
from data.simulation import PrinterSimulationDataset


class ComprehensiveEvaluator:
    """Comprehensive model evaluator with rich visualizations"""

    def __init__(self, model, device, save_dir='evaluation_results'):
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Task metadata
        self.tasks = {
            'adhesion_strength': {
                'name': 'Adhesion Strength',
                'unit': 'ratio',
                'nice_name': '粘合力强度'
            },
            'internal_stress': {
                'name': 'Internal Stress',
                'unit': 'MPa',
                'nice_name': '内应力'
            },
            'porosity': {
                'name': 'Porosity',
                'unit': '%',
                'nice_name': '孔隙率'
            },
            'dimensional_accuracy': {
                'name': 'Dimensional Accuracy',
                'unit': 'mm',
                'nice_name': '尺寸精度'
            },
            'quality_score': {
                'name': 'Quality Score',
                'unit': 'score',
                'nice_name': '质量分数'
            }
        }

    @torch.no_grad()
    def evaluate(self, data_loader):
        """Run evaluation on dataset"""
        self.model.eval()

        all_preds = {k: [] for k in self.tasks.keys()}
        all_targets = {k: [] for k in self.tasks.keys()}

        print("Running evaluation...")
        for batch_idx, batch in enumerate(data_loader):
            # Move to device
            input_features = batch['input_features'].to(self.device)
            quality_targets = batch['quality_targets'].to(self.device)

            # Forward pass
            outputs = self.model(input_features)

            # Collect
            for i, task in enumerate(self.tasks.keys()):
                pred = outputs[task].cpu().numpy()
                target = quality_targets[:, i:i+1].cpu().numpy()
                all_preds[task].append(pred)
                all_targets[task].append(target)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(data_loader)}")

        # Concatenate
        results = {}
        for task in self.tasks.keys():
            results[f'{task}_pred'] = np.concatenate(all_preds[task]).flatten()
            results[f'{task}_target'] = np.concatenate(all_targets[task]).flatten()

        return results

    def compute_metrics(self, results):
        """Compute comprehensive metrics"""
        metrics = {}

        for task in self.tasks.keys():
            pred = results[f'{task}_pred']
            target = results[f'{task}_target']

            # Basic metrics
            mse = np.mean((pred - target) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(pred - target))

            # R² score
            ss_res = np.sum((target - pred) ** 2)
            ss_tot = np.sum((target - np.mean(target)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((target - pred) / (np.abs(target) + 1e-8))) * 100

            # Max/Absolute error
            max_err = np.max(np.abs(pred - target))
            median_err = np.median(np.abs(pred - target))

            # Correlation
            corr = np.corrcoef(pred, target)[0, 1]

            # Within tolerance (10% of mean target)
            tolerance = 0.1 * np.abs(np.mean(target))
            within_tol = np.mean(np.abs(pred - target) <= tolerance) * 100

            # Statistics
            pred_mean = np.mean(pred)
            pred_std = np.std(pred)
            target_mean = np.mean(target)
            target_std = np.std(target)

            metrics[task] = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'mape': float(mape),
                'max_error': float(max_err),
                'median_error': float(median_err),
                'correlation': float(corr),
                'within_tolerance': float(within_tol),
                'pred_mean': float(pred_mean),
                'pred_std': float(pred_std),
                'target_mean': float(target_mean),
                'target_std': float(target_std),
                'bias': float(pred_mean - target_mean)
            }

        return metrics

    def generate_all_plots(self, results, metrics):
        """Generate comprehensive visualizations"""
        print("Generating visualizations...")

        self._plot_prediction_vs_target(results, metrics)
        self._plot_error_distributions(results)
        self._plot_residuals(results)
        self._plot_metrics_comparison(metrics)
        self._plot_error_boxplots(results)
        self._plot_correlation_heatmap(results)

        print("All plots generated!")

    def _plot_prediction_vs_target(self, results, metrics):
        """Prediction vs Target scatter plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (task, info) in enumerate(self.tasks.items()):
            ax = axes[idx]
            pred = results[f'{task}_pred']
            target = results[f'{task}_target']

            # Scatter with density
            ax.scatter(target, pred, alpha=0.3, s=10, rasterized=True)

            # Perfect prediction line
            min_v = min(pred.min(), target.min())
            max_v = max(pred.max(), target.max())
            ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Perfect')

            # Metrics text
            r2 = metrics[task]['r2']
            rmse = metrics[task]['rmse']
            mae = metrics[task]['mae']
            textstr = f'R²={r2:.4f}\nRMSE={rmse:.4f}\nMAE={mae:.4f}'
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel(f'True {info["nice_name"]} ({info["unit"]})')
            ax.set_ylabel(f'Predicted {info["nice_name"]} ({info["unit"]})')
            ax.set_title(f'{info["name"]}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide last subplot
        if len(self.tasks) < 6:
            fig.delaxes(axes[-1])

        plt.tight_layout()
        plt.savefig(self.save_dir / 'prediction_vs_target.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ prediction_vs_target.png")

    def _plot_error_distributions(self, results):
        """Error distribution histograms"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (task, info) in enumerate(self.tasks.items()):
            ax = axes[idx]
            pred = results[f'{task}_pred']
            target = results[f'{task}_target']
            errors = pred - target

            # Histogram
            n, bins, patches = ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')

            # Mean line
            mean_err = errors.mean()
            ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
            ax.axvline(mean_err, color='g', linestyle='-', linewidth=2, label=f'Mean={mean_err:.4f}')

            ax.set_xlabel(f'Error ({info["unit"]})')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{info["nice_name"]} Error (Std={errors.std():.4f})')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        if len(self.tasks) < 6:
            fig.delaxes(axes[-1])

        plt.tight_layout()
        plt.savefig(self.save_dir / 'error_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ error_distributions.png")

    def _plot_residuals(self, results):
        """Residual plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (task, info) in enumerate(self.tasks.items()):
            ax = axes[idx]
            pred = results[f'{task}_pred']
            target = results[f'{task}_target']
            residuals = pred - target

            ax.scatter(pred, residuals, alpha=0.3, s=10, rasterized=True)
            ax.axhline(0, color='r', linestyle='--', linewidth=2)

            ax.set_xlabel(f'Predicted {info["nice_name"]} ({info["unit"]})')
            ax.set_ylabel(f'Residual ({info["unit"]})')
            ax.set_title(f'{info["nice_name"]} Residual Plot')
            ax.grid(True, alpha=0.3)

        if len(self.tasks) < 6:
            fig.delaxes(axes[-1])

        plt.tight_layout()
        plt.savefig(self.save_dir / 'residual_plots.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ residual_plots.png")

    def _plot_metrics_comparison(self, metrics):
        """Metrics comparison bar chart"""
        fig, ax = plt.subplots(figsize=(14, 6))

        tasks = list(self.tasks.keys())
        task_names = [self.tasks[t]['nice_name'] for t in tasks]

        rmse = [metrics[t]['rmse'] for t in tasks]
        mae = [metrics[t]['mae'] for t in tasks]

        x = np.arange(len(tasks))
        width = 0.35

        ax.bar(x - width/2, rmse, width, label='RMSE', alpha=0.8)
        ax.bar(x + width/2, mae, width, label='MAE', alpha=0.8)

        ax.set_xlabel('Task')
        ax.set_ylabel('Error')
        ax.set_title('RMSE vs MAE Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(task_names, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (r, m) in enumerate(zip(rmse, mae)):
            ax.text(i - width/2, r, f'{r:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width/2, m, f'{m:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ metrics_comparison.png")

    def _plot_error_boxplots(self, results):
        """Error distribution box plots"""
        fig, ax = plt.subplots(figsize=(14, 6))

        errors_list = []
        task_names = []

        for task, info in self.tasks.items():
            pred = results[f'{task}_pred']
            target = results[f'{task}_target']
            errors = np.abs(pred - target)
            errors_list.append(errors)
            task_names.append(info['nice_name'])

        bp = ax.boxplot(errors_list, labels=task_names, patch_artist=True)

        # Color boxes
        colors = sns.color_palette("Set3", len(errors_list))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Absolute Error')
        ax.set_title('Error Distribution by Task')
        ax.grid(True, alpha=0.3, axis='y')

        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'error_boxplots.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ error_boxplots.png")

    def _plot_correlation_heatmap(self, results):
        """Correlation heatmap between predictions and targets"""
        # Compute correlation matrix
        all_vars = []
        var_names = []

        for task, info in self.tasks.items():
            pred = results[f'{task}_pred']
            target = results[f'{task}_target']
            all_vars.extend([pred, target])
            var_names.extend([f'{info["nice_name"]} (Pred)', f'{info["nice_name"]} (True)'])

        corr_matrix = np.corrcoef([v for v in all_vars])

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(16, 14))

        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   xticklabels=var_names, yticklabels=var_names,
                   ax=ax, cbar_kws={'label': 'Correlation'})

        ax.set_title('Correlation Heatmap: Predictions vs Targets')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ correlation_heatmap.png")

    def save_reports(self, metrics):
        """Save detailed reports"""
        # JSON report
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            'metrics': {}
        }

        for task, task_metrics in metrics.items():
            report['metrics'][task] = {
                'name': self.tasks[task]['name'],
                'unit': self.tasks[task]['unit'],
                'nice_name': self.tasks[task]['nice_name'],
                **task_metrics
            }

        with open(self.save_dir / 'metrics_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print(f"  ✓ metrics_report.json")

    def print_summary(self, metrics):
        """Print formatted summary"""
        print("\n" + "=" * 100)
        print(" " * 35 + "EVALUATION SUMMARY")
        print("=" * 100)

        for task, info in self.tasks.items():
            m = metrics[task]
            print(f"\n{info['nice_name']} ({info['name']})")
            print("-" * 100)
            print(f"  RMSE:               {m['rmse']:.6f}")
            print(f"  MAE:                {m['mae']:.6f}")
            print(f"  R²:                 {m['r2']:.6f}")
            print(f"  MAPE:               {m['mape']:.2f}%")
            print(f"  Correlation:        {m['correlation']:.6f}")
            print(f"  Max Error:          {m['max_error']:.6f}")
            print(f"  Median Error:       {m['median_error']:.6f}")
            print(f"  Within 10% Tolerance: {m['within_tolerance']:.2f}%")
            print(f"  Bias (Pred-Target):  {m['bias']:.6f}")
            print(f"  Pred Mean:          {m['pred_mean']:.6f}  (Target: {m['target_mean']:.6f})")
            print(f"  Pred Std:           {m['pred_std']:.6f}  (Target: {m['target_std']:.6f})")

        print("\n" + "=" * 100)
        print("OVERALL PERFORMANCE")
        print("=" * 100)

        avg_r2 = np.mean([metrics[t]['r2'] for t in self.tasks.keys()])
        print(f"Average R²:              {avg_r2:.6f}")

        good_perf = sum(1 for t in self.tasks.keys() if metrics[t]['r2'] > 0.8)
        ok_perf = sum(1 for t in self.tasks.keys() if 0.5 < metrics[t]['r2'] <= 0.8)

        print(f"Tasks with R² > 0.8:     {good_perf}/{len(self.tasks)}")
        print(f"Tasks with 0.5 < R² ≤ 0.8: {ok_perf}/{len(self.tasks)}")

        print("=" * 100 + "\n")


def load_model(checkpoint_path, config, device):
    """Load model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = ImplicitStateTCN(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"  Val Loss: {checkpoint['val_loss']:.6f}")

    print("  Model loaded!")
    return model


def build_dataloader(data_dir, config, batch_size=64):
    """Build test dataloader"""
    import glob
    import random

    mat_files = glob.glob(os.path.join(data_dir, "*.mat"))
    if not mat_files:
        raise ValueError(f"No .mat files in {data_dir}")

    # Use last 15% for testing
    random.Random(42).shuffle(mat_files)
    n_test = int(0.15 * len(mat_files))
    test_files = mat_files[-n_test:]

    # Get scaler from train set
    train_files = mat_files[:int(0.7 * len(mat_files))]
    train_dataset = PrinterSimulationDataset(
        train_files, seq_len=config.data.seq_len,
        pred_len=config.data.pred_len, stride=config.data.stride,
        mode='train', scaler=None, fit_scaler=True
    )

    # Test dataset
    test_dataset = PrinterSimulationDataset(
        test_files, seq_len=config.data.seq_len,
        pred_len=config.data.pred_len, stride=config.data.stride,
        mode='test', scaler=train_dataset.scaler, fit_scaler=False
    )

    loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    return loader


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--save_dir', type=str, default='evaluation_results/implicit_state_tcn')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    print("=" * 100)
    print(" " * 40 + "COMPREHENSIVE MODEL EVALUATION")
    print("=" * 100)
    print(f"\nConfig:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data Dir:  {args.data_dir}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Save Dir:   {args.save_dir}")
    print(f"  Device:     {args.device}\n")

    device = torch.device(args.device)
    config = get_config(preset='unified')

    # Load model
    model = load_model(args.checkpoint, config, device)

    # Build dataloader
    print("Building test dataloader...")
    test_loader = build_dataloader(args.data_dir, config, args.batch_size)
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Test batches: {len(test_loader)}\n")

    # Evaluate
    evaluator = ComprehensiveEvaluator(model, device, args.save_dir)
    results = evaluator.evaluate(test_loader)
    metrics = evaluator.compute_metrics(results)

    # Outputs
    evaluator.print_summary(metrics)
    evaluator.generate_all_plots(results, metrics)
    evaluator.save_reports(metrics)

    print("\n" + "=" * 100)
    print(f"EVALUATION COMPLETED! Results saved to: {args.save_dir}")
    print("=" * 100 + "\n")


if __name__ == '__main__':
    main()
