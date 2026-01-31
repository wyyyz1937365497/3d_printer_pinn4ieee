"""
Training Script for Trajectory Error Correction Model

This script trains a Transformer-based model to predict trajectory errors
in 3D printing, enabling real-time correction of deviations.

Task: Trajectory Error Prediction
- Input: Command trajectory + environmental parameters
- Output: Predicted error in X and Y directions
- Goal: Enable proactive trajectory correction
"""

import os
import sys
import argparse
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from config import get_config
from models.trajectory import TrajectoryErrorTransformer
from data.simulation.dataset import PrinterSimulationDataset
from utils import set_seed


class TrajectoryTrainer:
    """Trainer for trajectory error correction model"""

    def __init__(self, model, criterion, optimizer, scheduler,
                 device, config, use_amp=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

    def train_epoch(self, train_loader, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_x_error = 0
        total_y_error = 0
        num_batches = len(train_loader)

        start_time = time.time()
        last_log_time = start_time

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            input_features = batch['input_features'].to(self.device, non_blocking=True)
            trajectory_targets = batch['trajectory_targets'].to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(input_features)
                    loss = self.criterion(outputs, trajectory_targets)

                # Backward
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(input_features)
                loss = self.criterion(outputs, trajectory_targets)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            # Metrics
            total_loss += loss.item() * input_features.size(0)

            # Per-axis errors - support both naming conventions, use last timestep
            if 'error_x' in outputs:
                x_error = nn.functional.l1_loss(outputs['error_x'][:, -1:, :], trajectory_targets[:, :, 0:1])
                y_error = nn.functional.l1_loss(outputs['error_y'][:, -1:, :], trajectory_targets[:, :, 1:2])
            else:
                x_error = nn.functional.l1_loss(outputs['displacement_x_seq'][:, -1:, :], trajectory_targets[:, :, 0:1])
                y_error = nn.functional.l1_loss(outputs['displacement_y_seq'][:, -1:, :], trajectory_targets[:, :, 1:2])
            total_x_error += x_error.item() * input_features.size(0)
            total_y_error += y_error.item() * input_features.size(0)

            # Progress logging
            current_time = time.time()
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches:
                elapsed = current_time - last_log_time
                samples_processed = (batch_idx + 1) * input_features.size(0)
                throughput = int(20 * input_features.size(0) / elapsed) if (batch_idx + 1) % 20 == 0 else 0

                print(f"    Batch {batch_idx + 1:4d}/{num_batches} | "
                      f"Loss: {loss.item():8.6f} | "
                      f"X-Err: {x_error.item():.6f} | "
                      f"Y-Err: {y_error.item():.6f} | "
                      f"Throughput: {throughput:6d} samples/s")

                last_log_time = current_time

        avg_loss = total_loss / len(train_loader.dataset)
        avg_x_error = total_x_error / len(train_loader.dataset)
        avg_y_error = total_y_error / len(train_loader.dataset)

        return avg_loss, avg_x_error, avg_y_error

    @torch.no_grad()
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_x_error = 0
        total_y_error = 0

        for batch in val_loader:
            input_features = batch['input_features'].to(self.device, non_blocking=True)
            trajectory_targets = batch['trajectory_targets'].to(self.device, non_blocking=True)

            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(input_features)
                    loss = self.criterion(outputs, trajectory_targets)
            else:
                outputs = self.model(input_features)
                loss = self.criterion(outputs, trajectory_targets)

            total_loss += loss.item() * input_features.size(0)

            # Per-axis errors - support both naming conventions, use last timestep
            if 'error_x' in outputs:
                x_error = nn.functional.l1_loss(outputs['error_x'][:, -1:, :], trajectory_targets[:, :, 0:1])
                y_error = nn.functional.l1_loss(outputs['error_y'][:, -1:, :], trajectory_targets[:, :, 1:2])
            else:
                x_error = nn.functional.l1_loss(outputs['displacement_x_seq'][:, -1:, :], trajectory_targets[:, :, 0:1])
                y_error = nn.functional.l1_loss(outputs['displacement_y_seq'][:, -1:, :], trajectory_targets[:, :, 1:2])
            total_x_error += x_error.item() * input_features.size(0)
            total_y_error += y_error.item() * input_features.size(0)

        avg_loss = total_loss / len(val_loader.dataset)
        avg_x_error = total_x_error / len(val_loader.dataset)
        avg_y_error = total_y_error / len(val_loader.dataset)

        return avg_loss, avg_x_error, avg_y_error


class TrajectoryLoss(nn.Module):
    """Combined loss for trajectory error prediction"""

    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha  # Weight for X error
        self.beta = beta    # Weight for Y error

    def forward(self, predictions, targets):
        """
        Compute combined trajectory loss

        Args:
            predictions: Dict with 'error_x'/'error_y' or 'displacement_x_seq'/'displacement_y_seq' [B, T, 1]
            targets: Trajectory targets [B, pred_len, 2]
        """
        # Support both naming conventions
        # Use last timestep of prediction sequence to match target
        if 'error_x' in predictions:
            pred_x = predictions['error_x'][:, -1:, :]
            pred_y = predictions['error_y'][:, -1:, :]
        else:
            pred_x = predictions['displacement_x_seq'][:, -1:, :]
            pred_y = predictions['displacement_y_seq'][:, -1:, :]

        loss_x = nn.functional.mse_loss(pred_x, targets[:, :, 0:1])
        loss_y = nn.functional.mse_loss(pred_y, targets[:, :, 1:2])

        total_loss = self.alpha * loss_x + self.beta * loss_y
        return total_loss


def build_dataloaders(data_dir, config, batch_size, num_workers=4):
    """Build train and validation dataloaders"""
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

    # Split files
    import random
    random.shuffle(mat_files)
    n_train = int(0.7 * len(mat_files))
    n_val = int(0.15 * len(mat_files))

    train_files = mat_files[:n_train]
    val_files = mat_files[n_train:n_train + n_val]

    print(f"Loading data from {len(mat_files)} files...")
    print(f"  Train: {len(train_files)} files")
    print(f"  Val: {len(val_files)} files")

    train_dataset = PrinterSimulationDataset(
        train_files,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        stride=config.data.stride,
        mode='train',
        include_trajectory=True
    )

    val_dataset = PrinterSimulationDataset(
        val_files,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        stride=config.data.stride,
        mode='val',
        scaler=train_dataset.scaler,
        include_trajectory=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train Trajectory Error Correction Model')
    parser.add_argument('--data_dir', type=str, default='data_simulation_*/')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_torch_compile', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    set_seed(args.seed)

    print("=" * 80)
    print("TRAJECTORY ERROR CORRECTION MODEL TRAINING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Task: Trajectory error prediction (X, Y)")
    print(f"  Model: TrajectoryErrorTransformer")
    print(f"  Mixed precision (AMP): True")
    print(f"  Data workers: {args.num_workers}")
    print(f"  torch.compile: {args.use_torch_compile}")
    print(f"  Device: {args.device}")
    print()

    # Load config
    config = get_config(preset='trajectory')
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.lr

    device = torch.device(args.device)

    # Build dataloaders
    print("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(
        args.data_dir, config, args.batch_size, args.num_workers
    )
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print()

    # Create model
    print("Creating trajectory error correction model...")
    model = TrajectoryErrorTransformer(config).to(device)

    model_info = model.get_model_info()
    print(f"  Model type: {model_info['model_type']}")
    print(f"  Parameters: {model_info['num_parameters']:,}")

    if args.use_torch_compile:
        try:
            print("  Compiling model with torch.compile...")
            model = torch.compile(model, mode='reduce-overhead')
            print("  Model compiled successfully!")
        except Exception as e:
            print(f"  Warning: torch.compile failed: {e}")
    print()

    # Create loss function
    criterion = TrajectoryLoss(alpha=1.0, beta=1.0)
    print("  Loss: MSE on (error_x, error_y)")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.training.weight_decay,
        fused=device.type == 'cuda'
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # Create trainer
    trainer = TrajectoryTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        use_amp=True
    )

    # Training loop
    print("Starting training...")
    print("=" * 80)
    print()

    best_val_loss = float('inf')
    ckpt_dir = Path('checkpoints/trajectory_correction')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        print("  Training:")
        train_loss, train_x_err, train_y_err = trainer.train_epoch(train_loader, epoch)

        # Validate
        print("  Validation:")
        val_loss, val_x_err, val_y_err = trainer.validate(val_loader)

        # Scheduler step
        scheduler.step()

        # Summary
        epoch_time = time.time() - epoch_start_time

        print(f"\n  Summary:")
        print(f"    Train | Loss: {train_loss:.6f} | X-Err: {train_x_err:.6f} | Y-Err: {train_y_err:.6f}")
        print(f"    Val   | Loss: {val_loss:.6f} | X-Err: {val_x_err:.6f} | Y-Err: {val_y_err:.6f}")
        print(f"    Time: {epoch_time:.2f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_x_error': val_x_err,
                'val_y_error': val_y_err,
            }, ckpt_dir / 'best_model.pth')
            print(f"    [*] Saved best model (val_loss: {val_loss:.6f})")

        print()

    total_time = time.time() - total_start_time
    print("=" * 80)
    print(f"Training completed!")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {ckpt_dir / 'best_model.pth'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
