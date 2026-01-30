"""
Evaluation Script for Trajectory Error Correction Model

Evaluates the trained model's ability to predict trajectory errors
in X and Y directions for real-time correction.
"""

import os
import sys
import argparse
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from scipy import stats

from config import get_config
from models.trajectory import TrajectoryErrorTransformer
from data.simulation.dataset import PrinterSimulationDataset


def build_test_dataloader(data_dir, config, batch_size=256, num_workers=4):
    """Build test dataloader"""
    import glob

    # Handle wildcard patterns
    mat_files = []
    expanded_paths = glob.glob(data_dir)

    if expanded_paths:
        for path in expanded_paths:
            if os.path.isfile(path) and path.endswith('.mat'):
                mat_files.append(path)
            elif os.path.isdir(path):
                mat_files.extend(glob.glob(os.path.join(path, "*.mat")))

    if not mat_files:
        if os.path.isfile(data_dir) and data_dir.endswith('.mat'):
            mat_files = [data_dir]
        elif os.path.isdir(data_dir):
            mat_files = glob.glob(os.path.join(data_dir, "*.mat"))

    if not mat_files:
        raise ValueError(f"No .mat files found in {data_dir}")

    # Split: 70% train, 15% val, 15% test
    import random
    random.shuffle(mat_files)
    n_train = int(0.7 * len(mat_files))
    n_val = int(0.15 * len(mat_files))

    test_files = mat_files[n_train + n_val:]

    print(f"Loading test data from {len(test_files)} files...")

    # Load scaler from training data first
    train_files = mat_files[:n_train]
    train_dataset_temp = PrinterSimulationDataset(
        train_files,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        stride=config.data.stride,
        mode='train',
        include_trajectory=True
    )

    # Create test dataset with training scaler
    test_dataset = PrinterSimulationDataset(
        test_files,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        stride=config.data.stride,
        mode='test',
        scaler=train_dataset_temp.scaler,
        include_trajectory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return test_loader


@torch.no_grad()
def evaluate_model(model, test_loader, device):
    """Evaluate model and collect predictions"""
    model.eval()

    all_predictions_x = []
    all_predictions_y = []
    all_targets_x = []
    all_targets_y = []

    total_loss = 0
    total_samples = 0

    criterion = nn.MSELoss()

    for batch_idx, batch in enumerate(test_loader):
        input_features = batch['input_features'].to(device, non_blocking=True)
        trajectory_targets = batch['trajectory_targets'].to(device, non_blocking=True)

        # Forward pass
        outputs = model(input_features)

        # Extract predictions and targets
        pred_x = outputs['error_x'].cpu().numpy()
        pred_y = outputs['error_y'].cpu().numpy()

        target_x = trajectory_targets[:, :, 0:1].cpu().numpy()
        target_y = trajectory_targets[:, :, 1:2].cpu().numpy()

        all_predictions_x.append(pred_x)
        all_predictions_y.append(pred_y)
        all_targets_x.append(target_x)
        all_targets_y.append(target_y)

        # Compute loss
        loss_x = criterion(outputs['error_x'], trajectory_targets[:, :, 0:1].to(device))
        loss_y = criterion(outputs['error_y'], trajectory_targets[:, :, 1:2].to(device))
        total_loss += (loss_x + loss_y).item() * input_features.size(0)
        total_samples += input_features.size(0)

        if (batch_idx + 1) % 20 == 0:
            print(f"  Evaluated batch {batch_idx + 1}/{len(test_loader)}")

    # Concatenate all predictions
    all_predictions_x = np.concatenate(all_predictions_x, axis=0).flatten()
    all_predictions_y = np.concatenate(all_predictions_y, axis=0).flatten()
    all_targets_x = np.concatenate(all_targets_x, axis=0).flatten()
    all_targets_y = np.concatenate(all_targets_y, axis=0).flatten()

    avg_loss = total_loss / total_samples

    return (all_predictions_x, all_predictions_y,
            all_targets_x, all_targets_y, avg_loss)


def compute_trajectory_metrics(pred_x, pred_y, target_x, target_y):
    """Compute evaluation metrics for trajectory prediction"""

    metrics = {}

    for axis, pred, target in [('X', pred_x, target_x), ('Y', pred_y, target_y)]:
        # Remove NaN/Inf
        mask = np.isfinite(pred) & np.isfinite(target)
        pred_clean = pred[mask]
        target_clean = target[mask]

        if len(pred_clean) == 0:
            print(f"Warning: All predictions/targets are invalid for {axis}")
            continue

        # R² score
        r2 = stats.linregress(target_clean, pred_clean).rvalue ** 2

        # MAE, RMSE
        mae = np.mean(np.abs(pred_clean - target_clean))
        rmse = np.sqrt(np.mean((pred_clean - target_clean) ** 2))

        # Correlation coefficient
        corr = np.corrcoef(target_clean, pred_clean)[0, 1]

        # Relative metrics
        target_range = target_clean.max() - target_clean.min()
        mae_normalized = mae / target_range if target_range > 0 else float('inf')

        metrics[axis] = {
            'r2': float(r2) if not np.isnan(r2) else 0.0,
            'mae': float(mae),
            'rmse': float(rmse),
            'correlation': float(corr) if not np.isnan(corr) else 0.0,
            'mae_normalized': float(mae_normalized),
            'target_mean': float(target_clean.mean()),
            'target_std': float(target_clean.std()),
            'pred_mean': float(pred_clean.mean()),
            'pred_std': float(pred_clean.std()),
            'target_range': float(target_range),
            'num_samples': int(len(pred_clean))
        }

    return metrics


def print_evaluation_report(metrics, total_loss):
    """Print comprehensive evaluation report"""

    print("\n" + "=" * 80)
    print("TRAJECTORY ERROR CORRECTION MODEL EVALUATION")
    print("=" * 80)

    print(f"\nOverall Test Loss: {total_loss:.6f}")

    print("\n" + "-" * 80)
    print("PER-AXIS METRICS")
    print("-" * 80)

    for axis, metrics_dict in metrics.items():
        print(f"\n{axis}-Axis Error Prediction:")
        print(f"  R² (Coefficient of Determination): {metrics_dict['r2']:.4f}")
        print(f"  Correlation Coefficient:            {metrics_dict['correlation']:.4f}")
        print(f"  MAE (Mean Absolute Error):          {metrics_dict['mae']:.6f}")
        print(f"  RMSE (Root Mean Squared Error):     {metrics_dict['rmse']:.6f}")
        print(f"  Normalized MAE:                     {metrics_dict['mae_normalized']:.4f}")
        print(f"\n  Statistics:")
        print(f"    Target mean ± std:                {metrics_dict['target_mean']:.6f} ± {metrics_dict['target_std']:.6f}")
        print(f"    Prediction mean ± std:            {metrics_dict['pred_mean']:.6f} ± {metrics_dict['pred_std']:.6f}")
        print(f"    Target range:                     {metrics_dict['target_range']:.6f}")
        print(f"    Samples:                          {metrics_dict['num_samples']:,}")

    # Quality assessment
    print("\n" + "-" * 80)
    print("QUALITY ASSESSMENT")
    print("-" * 80)

    avg_r2 = np.mean([m['r2'] for m in metrics.values()])
    avg_corr = np.mean([m['correlation'] for m in metrics.values()])
    avg_mae_norm = np.mean([m['mae_normalized'] for m in metrics.values()])

    print(f"\nAverage Performance:")
    print(f"  R²:           {avg_r2:.4f}")
    print(f"  Correlation:  {avg_corr:.4f}")
    print(f"  Norm. MAE:    {avg_mae_norm:.4f}")

    # Quality rating
    if avg_r2 >= 0.8:
        rating = "EXCELLENT - Model predicts trajectory errors very accurately"
    elif avg_r2 >= 0.6:
        rating = "GOOD - Model learns most trajectory error patterns"
    elif avg_r2 >= 0.4:
        rating = "MODERATE - Model learns some trajectory error patterns"
    elif avg_r2 >= 0.2:
        rating = "WEAK - Model struggles to learn trajectory errors"
    else:
        rating = "POOR - Model does not learn meaningful trajectory patterns"

    print(f"\nOverall Rating: {rating}")

    # Per-axis ratings
    print("\nPer-Axis Ratings:")
    for axis, metrics_dict in metrics.items():
        r2 = metrics_dict['r2']
        if r2 >= 0.8:
            level = "Excellent"
        elif r2 >= 0.6:
            level = "Good"
        elif r2 >= 0.4:
            level = "Moderate"
        elif r2 >= 0.2:
            level = "Weak"
        else:
            level = "Poor"

        print(f"  {axis}-Axis: {level} (R²={r2:.4f})")

    # Correction capability assessment
    print("\n" + "-" * 80)
    print("CORRECTION CAPABILITY")
    print("-" * 80)

    avg_mae = np.mean([m['mae'] for m in metrics.values()])

    print(f"\nAverage absolute error: {avg_mae:.6f}")
    print(f"This means the model can predict trajectory errors with ±{avg_mae:.4f} accuracy")

    if avg_mae < 0.01:
        capability = "HIGH - Suitable for precision correction (±0.01 accuracy)"
    elif avg_mae < 0.05:
        capability = "GOOD - Suitable for moderate correction (±0.05 accuracy)"
    elif avg_mae < 0.1:
        capability = "MODERATE - Suitable for coarse correction (±0.1 accuracy)"
    else:
        capability = "LIMITED - May require additional filtering or manual correction"

    print(f"\nCorrection Capability: {capability}")

    print("\n" + "=" * 80)

    return metrics


def save_metrics(metrics, total_loss, output_path):
    """Save metrics to JSON file"""
    output_data = {
        'total_test_loss': float(total_loss),
        'metrics': metrics
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nMetrics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Trajectory Error Correction Model')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/trajectory_correction/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data_simulation_*/')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str,
                       default='evaluation_results/trajectory_model/metrics.json')

    args = parser.parse_args()

    print("=" * 80)
    print("EVALUATING TRAJECTORY ERROR CORRECTION MODEL")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {args.device}")
    print()

    # Load config
    config = get_config(preset='trajectory')

    device = torch.device(args.device)

    # Build test dataloader
    print("Building test dataloader...")
    test_loader = build_test_dataloader(
        args.data_dir, config, args.batch_size, args.num_workers
    )
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Test batches: {len(test_loader)}")
    print()

    # Create model
    print("Creating model...")
    model = TrajectoryErrorTransformer(config).to(device)

    # Load checkpoint
    if os.path.exists(args.checkpoint):
        print(f"  Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Handle torch.compile prefix
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            print("  Detected torch.compile checkpoint, removing prefix...")
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('_orig_mod.'):
                    new_state_dict[k[len('_orig_mod.'):]] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
        print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"  Checkpoint validation loss: {checkpoint.get('val_loss', 'unknown'):.6f}")
        if 'val_x_error' in checkpoint:
            print(f"  Checkpoint X error: {checkpoint.get('val_x_error'):.6f}")
            print(f"  Checkpoint Y error: {checkpoint.get('val_y_error'):.6f}")
    else:
        print(f"  Warning: Checkpoint not found at {args.checkpoint}")
        print("  Evaluating with randomly initialized model...")

    print()

    # Evaluate
    print("Evaluating model...")
    pred_x, pred_y, target_x, target_y, total_loss = evaluate_model(
        model, test_loader, device
    )

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_trajectory_metrics(pred_x, pred_y, target_x, target_y)

    # Print report
    metrics = print_evaluation_report(metrics, total_loss)

    # Save metrics
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, total_loss, output_path)


if __name__ == '__main__':
    main()
