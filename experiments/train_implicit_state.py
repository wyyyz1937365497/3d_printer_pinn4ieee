"""
Train implicit state inference model

Refactored from previous_code/3D_printer_pinn_transformer for paper-ready training.
"""

import os
import sys
import argparse
import random
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from config import get_config
from models import ImplicitStateTransformer
from data.simulation import PrinterSimulationDataset
from training.losses import MultiTaskLoss
from utils import set_seed


def build_loaders(data_dir, config, batch_size):
    import glob

    mat_files = glob.glob(os.path.join(data_dir, "*.mat"))
    if not mat_files:
        raise ValueError(f"No .mat files found in {data_dir}")

    random.shuffle(mat_files)
    n_train = int(0.7 * len(mat_files))
    n_val = int(0.15 * len(mat_files))

    train_files = mat_files[:n_train]
    val_files = mat_files[n_train:n_train + n_val]

    train_dataset = PrinterSimulationDataset(
        train_files,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        stride=config.data.stride,
        mode='train',
        scaler=None,
        fit_scaler=True
    )

    val_dataset = PrinterSimulationDataset(
        val_files,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        stride=config.data.stride,
        mode='val',
        scaler=train_dataset.scaler,
        fit_scaler=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.training.num_workers
    )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train implicit state inference model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .mat files')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    set_seed(args.seed)

    config = get_config(preset='unified')
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.lr

    device = torch.device(args.device)

    train_loader, val_loader = build_loaders(args.data_dir, config, args.batch_size)

    model = ImplicitStateTransformer(config).to(device)

    criterion = MultiTaskLoss(
        lambda_quality=1.0,
        lambda_fault=0.0,
        lambda_trajectory=0.0,
        lambda_physics=0.0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.training.weight_decay)

    best_val = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = model(batch['input_features'])

            quality_targets = batch['quality_targets']
            targets = {
                'adhesion_strength': quality_targets[:, 0:1],
                'internal_stress': quality_targets[:, 1:2],
                'porosity': quality_targets[:, 2:3],
                'dimensional_accuracy': quality_targets[:, 3:4],
                'quality_score': quality_targets[:, 4:5],
            }

            losses = criterion(outputs, targets)
            loss = losses['total']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(batch['input_features'])

                quality_targets = batch['quality_targets']
                targets = {
                    'adhesion_strength': quality_targets[:, 0:1],
                    'internal_stress': quality_targets[:, 1:2],
                    'porosity': quality_targets[:, 2:3],
                    'dimensional_accuracy': quality_targets[:, 3:4],
                    'quality_score': quality_targets[:, 4:5],
                }

                losses = criterion(outputs, targets)
                val_loss += losses['total'].item()

        val_loss = val_loss / max(1, len(val_loader))

        print(f"Epoch {epoch+1}/{args.epochs} | Train: {avg_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt_dir = Path('checkpoints/implicit_state')
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({'model_state_dict': model.state_dict()}, ckpt_dir / 'best_model.pth')


if __name__ == '__main__':
    main()
