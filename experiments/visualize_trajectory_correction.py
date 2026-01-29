"""
Visualize full-layer trajectory heatmaps (before/after correction).
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
    parser = argparse.ArgumentParser(description='Visualize trajectory correction')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--file_index', type=int, default=0)
    parser.add_argument('--stride', type=int, default=None)
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

    file_index = max(0, min(args.file_index, len(dataset.data_list) - 1))
    data = dataset.data_list[file_index]

    # Build full-layer input features
    feature_names = config.data.input_features
    lengths = [len(np.squeeze(data[f])) for f in feature_names]
    lengths += [len(np.squeeze(data['x_ref'])), len(np.squeeze(data['y_ref'])),
                len(np.squeeze(data['error_x'])), len(np.squeeze(data['error_y']))]
    n = int(min(lengths))

    x_ref = np.squeeze(data['x_ref'])[:n]
    y_ref = np.squeeze(data['y_ref'])[:n]
    err_x = np.squeeze(data['error_x'])[:n]
    err_y = np.squeeze(data['error_y'])[:n]

    feat_list = []
    for f in feature_names:
        feat_list.append(np.squeeze(data[f])[:n])
    input_features = np.stack(feat_list, axis=1)  # [n, 12]

    # Normalize input features using dataset scaler
    if dataset.scaler is not None:
        input_features = dataset.scaler.transform(input_features)

    seq_len = config.data.seq_len
    pred_len = config.data.pred_len
    stride = args.stride if args.stride is not None else config.data.stride

    # Predict error sequence over the full layer
    model = TrajectoryErrorTransformer(config).to(device)
    load_checkpoint(model, args.model_path, device)
    model.eval()

    pred_sum = np.zeros((n, 2), dtype=np.float32)
    pred_count = np.zeros((n, 1), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n - seq_len - pred_len + 1, stride):
            seq = input_features[i:i + seq_len]
            inp = torch.FloatTensor(seq).unsqueeze(0).to(device)
            outputs = model(inp)

            if 'displacement_x_seq' in outputs and 'displacement_y_seq' in outputs:
                pred_x = outputs['displacement_x_seq'][:, -pred_len:, :].squeeze(0).squeeze(-1).cpu().numpy()
                pred_y = outputs['displacement_y_seq'][:, -pred_len:, :].squeeze(0).squeeze(-1).cpu().numpy()
            elif 'error_x' in outputs and 'error_y' in outputs:
                pred_x = outputs['error_x'].squeeze(0).cpu().numpy().repeat(pred_len)
                pred_y = outputs['error_y'].squeeze(0).cpu().numpy().repeat(pred_len)
            else:
                raise ValueError('No trajectory outputs found in model outputs.')

            start = i + seq_len
            end = start + pred_len
            pred_seq = np.stack([pred_x, pred_y], axis=1)
            pred_sum[start:end] += pred_seq
            pred_count[start:end] += 1

    pred_count[pred_count == 0] = 1
    pred_err = pred_sum / pred_count  # [n, 2]

    # Build error magnitudes
    err_mag = np.sqrt(err_x ** 2 + err_y ** 2)
    corr_err_x = err_x - pred_err[:, 0]
    corr_err_y = err_y - pred_err[:, 1]
    corr_err_mag = np.sqrt(corr_err_x ** 2 + corr_err_y ** 2)

    # Plot heatmaps for full layer
    results_dir = Path('results')
    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    sc1 = axes[0].scatter(x_ref, y_ref, c=err_mag, cmap='hot', s=2, alpha=0.7)
    axes[0].set_title('Simulated Error Heatmap (Before Correction)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].axis('equal')
    plt.colorbar(sc1, ax=axes[0], label='Error (mm)')

    sc2 = axes[1].scatter(x_ref, y_ref, c=corr_err_mag, cmap='hot', s=2, alpha=0.7)
    axes[1].set_title('Corrected Error Heatmap (After Correction)')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].axis('equal')
    plt.colorbar(sc2, ax=axes[1], label='Error (mm)')

    plt.tight_layout()
    out_path = figures_dir / 'trajectory_error_heatmap_compare.png'
    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
