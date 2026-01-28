"""
Quick Start Training Script for 3D Printer PINN

This is a minimal training script to get started quickly.
For full training options, see train_unified_model.py

Usage:
    python quick_train.py --data_dir <path>

Author: 3D Printer PINN Project
Date: 2026-01-28
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import UnifiedPINNSeq3D
from data.simulation import PrinterSimulationDataset
from training import Trainer
from config import get_config


def quick_train(data_dir, epochs=10, batch_size=32):
    """
    Quick training pipeline

    Args:
        data_dir: Directory containing MATLAB .mat files
        epochs: Number of training epochs
        batch_size: Batch size
    """
    print("=" * 60)
    print("3D Printer PINN - Quick Training")
    print("=" * 60)
    print()

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()

    # Load configuration
    config = get_config(preset='unified')
    config.training.batch_size = batch_size
    config.training.num_epochs = epochs
    config.training.device = device

    # Create datasets
    print("Loading datasets...")

    # For simplicity, use a single directory and auto-split
    import glob
    import random

    mat_files = glob.glob(os.path.join(data_dir, "*.mat"))

    if not mat_files:
        raise ValueError(f"No .mat files found in {data_dir}")

    # Shuffle and split
    random.shuffle(mat_files)
    n_train = int(0.7 * len(mat_files))
    n_val = int(0.15 * len(mat_files))

    train_files = mat_files[:n_train]
    val_files = mat_files[n_train:n_train + n_val]
    test_files = mat_files[n_train + n_val:]

    print(f"  Train: {len(train_files)} files")
    print(f"  Val:   {len(val_files)} files")
    print(f"  Test:  {len(test_files)} files")
    print()

    # Create datasets
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

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print()

    # Create model
    print("Creating model...")
    model = UnifiedPINNSeq3D(config.model).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    print()

    # Create trainer
    trainer = Trainer(model, config)

    # Train
    print("Starting training...")
    print()

    history = trainer.train(
        train_loader,
        val_loader,
        epochs=epochs,
        save_dir='checkpoints/quick_train'
    )

    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)

    return history


def main():
    parser = argparse.ArgumentParser(
        description='Quick training for 3D Printer PINN'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing MATLAB .mat simulation files'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )

    args = parser.parse_args()

    quick_train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
