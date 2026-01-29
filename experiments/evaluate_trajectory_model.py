"""
Evaluate trajectory correction model
"""

import os
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import get_config
from models import TrajectoryErrorTransformer
from data.simulation import PrinterSimulationDataset


def load_checkpoint(model, model_path, device):
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(model_path, map_location=device)
    except Exception:
        from torch.serialization import safe_globals
        from config.base_config import DataConfig, TrainingConfig, ModelConfig, PhysicsConfig, BaseConfig
        with safe_globals([DataConfig, TrainingConfig, ModelConfig, PhysicsConfig, BaseConfig]):
            ckpt = torch.load(model_path, map_location=device)

    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)


def main():
    parser = argparse.ArgumentParser(description='Evaluate trajectory correction model')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    config = get_config(preset='unified')

    import glob
    data_path = Path(args.data_path)
    if data_path.is_dir():
        mat_files = glob.glob(str(data_path / '*.mat'))
    else:
        candidates = glob.glob(args.data_path)
        mat_files = []
        for p in candidates:
            p_path = Path(p)
            if p_path.is_dir():
                mat_files.extend(glob.glob(str(p_path / '*.mat')))
            elif p_path.is_file() and p_path.suffix.lower() == '.mat':
                mat_files.append(str(p_path))

    if not mat_files:
        raise ValueError(f'No .mat files found for data_path: {args.data_path}')

    dataset = PrinterSimulationDataset(
        mat_files,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        stride=config.data.stride,
        mode='test',
        fit_scaler=True
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=config.num_workers)

    model = TrajectoryErrorTransformer(config).to(device)
    load_checkpoint(model, args.model_path, device)
    model.eval()

    abs_errors = []
    sq_errors = []
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch['input_features'])

            target = batch['trajectory_targets']  # [B, pred_len, 2]
            target_len = target.size(1)

            if 'displacement_x_seq' in outputs and 'displacement_y_seq' in outputs:
                pred_x = outputs['displacement_x_seq']
                pred_y = outputs['displacement_y_seq']
                if pred_x.size(1) != target_len:
                    pred_x = pred_x[:, -target_len:, :]
                if pred_y.size(1) != target_len:
                    pred_y = pred_y[:, -target_len:, :]
                pred = torch.cat([pred_x, pred_y], dim=2)  # [B, pred_len, 2]
            elif 'error_x' in outputs and 'error_y' in outputs:
                pred = torch.cat([outputs['error_x'], outputs['error_y']], dim=1).unsqueeze(1)
                pred = pred.repeat(1, target_len, 1)
            else:
                raise ValueError('No trajectory outputs found in model outputs.')

            err = pred - target
            abs_errors.append(err.abs().reshape(-1, 2).cpu().numpy())
            sq_errors.append((err ** 2).reshape(-1, 2).cpu().numpy())
            preds_all.append(pred.reshape(-1, 2).cpu().numpy())
            targets_all.append(target.reshape(-1, 2).cpu().numpy())

    abs_errors = np.concatenate(abs_errors, axis=0)
    sq_errors = np.concatenate(sq_errors, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)

    mae = abs_errors.mean(axis=0)
    rmse = np.sqrt(sq_errors.mean(axis=0))

    print('Trajectory Evaluation Metrics')
    print(f'MAE  (x, y): {mae[0]:.6f}, {mae[1]:.6f}')
    print(f'RMSE (x, y): {rmse[0]:.6f}, {rmse[1]:.6f}')

    # Save metrics table
    results_dir = Path('results')
    figures_dir = results_dir / 'figures'
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([
        {'metric': 'MAE_x', 'value': float(mae[0])},
        {'metric': 'MAE_y', 'value': float(mae[1])},
        {'metric': 'RMSE_x', 'value': float(rmse[0])},
        {'metric': 'RMSE_y', 'value': float(rmse[1])},
    ])
    metrics_df.to_csv(results_dir / 'trajectory_metrics.csv', index=False)

    # Configure matplotlib fonts (avoid missing glyph warnings)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # Plot 1: Pred vs Target scatter (sampled)
    sample_idx = np.random.choice(len(preds_all), size=min(5000, len(preds_all)), replace=False)
    pred_sample = preds_all[sample_idx]
    target_sample = targets_all[sample_idx]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(target_sample[:, 0], pred_sample[:, 0], s=6, alpha=0.4)
    plt.plot([target_sample[:, 0].min(), target_sample[:, 0].max()],
             [target_sample[:, 0].min(), target_sample[:, 0].max()], 'r--')
    plt.title('Displacement X: Pred vs Target')
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(target_sample[:, 1], pred_sample[:, 1], s=6, alpha=0.4)
    plt.plot([target_sample[:, 1].min(), target_sample[:, 1].max()],
             [target_sample[:, 1].min(), target_sample[:, 1].max()], 'r--')
    plt.title('Displacement Y: Pred vs Target')
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / 'trajectory_pred_vs_target.png', dpi=200)
    plt.close()

    # Plot 2: Error histogram
    plt.figure(figsize=(8, 4))
    plt.hist(abs_errors[:, 0], bins=60, alpha=0.7, label='|err_x|')
    plt.hist(abs_errors[:, 1], bins=60, alpha=0.7, label='|err_y|')
    plt.title('Trajectory Absolute Error Distribution')
    plt.xlabel('Absolute Error')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'trajectory_error_hist.png', dpi=200)
    plt.close()

    print(f'Metrics saved to: {results_dir / "trajectory_metrics.csv"}')
    print(f'Figures saved to: {figures_dir}')


if __name__ == '__main__':
    main()
