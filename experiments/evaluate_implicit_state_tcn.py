"""
Evaluate implicit state inference model (TCN) with visualizations.
"""

import os
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import matplotlib.pyplot as plt

from config import get_config
from models import ImplicitStateTCN
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


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot


def main():
    parser = argparse.ArgumentParser(description='Evaluate implicit state model (TCN)')
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

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    model = ImplicitStateTCN(config).to(device)
    load_checkpoint(model, args.model_path, device)
    model.eval()

    preds = {
        'adhesion_strength': [],
        'internal_stress': [],
        'porosity': [],
        'dimensional_accuracy': [],
        'quality_score': [],
    }
    targets = {k: [] for k in preds.keys()}

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch['input_features'])
            qt = batch['quality_targets']

            targets['adhesion_strength'].append(qt[:, 0:1].cpu().numpy())
            targets['internal_stress'].append(qt[:, 1:2].cpu().numpy())
            targets['porosity'].append(qt[:, 2:3].cpu().numpy())
            targets['dimensional_accuracy'].append(qt[:, 3:4].cpu().numpy())
            targets['quality_score'].append(qt[:, 4:5].cpu().numpy())

            for k in preds.keys():
                preds[k].append(outputs[k].detach().cpu().numpy())

    for k in preds.keys():
        preds[k] = np.concatenate(preds[k], axis=0).reshape(-1)
        targets[k] = np.concatenate(targets[k], axis=0).reshape(-1)

    metrics = []
    for k in preds.keys():
        y_true = targets[k]
        y_pred = preds[k]
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        r2 = r2_score(y_true, y_pred)
        metrics.append({'metric': f'{k}_mae', 'value': float(mae)})
        metrics.append({'metric': f'{k}_rmse', 'value': float(rmse)})
        metrics.append({'metric': f'{k}_r2', 'value': float(r2)})

    results_dir = Path('results')
    figures_dir = results_dir / 'figures'
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    pd.DataFrame(metrics).to_csv(results_dir / 'implicit_state_tcn_metrics.csv', index=False)

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    keys = list(preds.keys())
    for i, k in enumerate(keys):
        ax = axes[i]
        ax.scatter(targets[k], preds[k], s=6, alpha=0.4)
        min_v = min(targets[k].min(), preds[k].min())
        max_v = max(targets[k].max(), preds[k].max())
        ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=1)
        ax.set_title(f'{k}: Pred vs Target')
        ax.set_xlabel('Target')
        ax.set_ylabel('Prediction')
        ax.grid(True, alpha=0.3)

    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig(figures_dir / 'implicit_state_tcn_pred_vs_target.png', dpi=200)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for i, k in enumerate(keys):
        ax = axes[i]
        err = np.abs(targets[k] - preds[k])
        ax.hist(err, bins=60, alpha=0.7)
        ax.set_title(f'{k}: |Error|')
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)

    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig(figures_dir / 'implicit_state_tcn_error_hist.png', dpi=200)
    plt.close()

    print(f'Metrics saved to: {results_dir / "implicit_state_tcn_metrics.csv"}')
    print(f'Figures saved to: {figures_dir}')


if __name__ == '__main__':
    main()
