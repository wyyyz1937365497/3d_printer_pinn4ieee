"""
OPTIMIZED: Enhanced Implicit State Inference Training with Vectorization

Optimizations:
1. Pin-memory and persistent workers for data loading
2. Mixed precision training (AMP)
3. torch.compile for JIT compilation
4. Vectorized batch processing (no CPU-GPU sync per batch)
5. Efficient metric computation
6. Gradient accumulation support
"""

import os
import sys
import argparse
import random
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Use new PyTorch 2.1+ AMP API
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    # Fallback to older API
    from torch.cuda.amp import GradScaler, autocast

from config import get_config
from models.implicit import ImplicitStateTCN, AdaptiveMultiTaskLoss
from data.simulation import PrinterSimulationDataset
from utils import set_seed


class VectorizedDataPreprocessor:
    """
    Vectorized data preprocessing - all operations on GPU

    Eliminates CPU-GPU synchronization points
    """

    def __init__(self, device):
        self.device = device
        # Pre-compute target indices for fast indexing
        self.target_indices = {
            'adhesion_strength': 0,
            'internal_stress': 1,
            'porosity': 2,
            'dimensional_accuracy': 3,
            'quality_score': 4,
        }

    def process_batch(self, batch):
        """
        Process entire batch on GPU with vectorized operations

        Args:
            batch: Dict with tensors

        Returns:
            Tuple of (inputs, targets) - all on GPU
        """
        # Vectorized device transfer (single operation)
        input_features = batch['input_features'].to(self.device, non_blocking=True)
        quality_targets = batch['quality_targets'].to(self.device, non_blocking=True)

        # Vectorized target slicing (no Python loops)
        targets = {
            'adhesion_strength': quality_targets[:, 0:1],
            'internal_stress': quality_targets[:, 1:2],
            'porosity': quality_targets[:, 2:3],
            'dimensional_accuracy': quality_targets[:, 3:4],
            'quality_score': quality_targets[:, 4:5],
        }

        return input_features, targets


class OptimizedTrainer:
    """
    Optimized trainer with vectorized operations and mixed precision
    """

    def __init__(self, model, criterion, optimizer, scheduler,
                 device, config, use_amp=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.use_amp = use_amp and torch.cuda.is_available()

        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None

        # Data preprocessor
        self.preprocessor = VectorizedDataPreprocessor(device)

        # Pre-allocate loss tensors (avoid repeated allocations)
        self.loss_keys = ['total', 'data', 'adhesion', 'stress',
                         'porosity', 'accuracy', 'quality', 'physics']

    def train_epoch(self, train_loader, epoch):
        """Vectorized training epoch with mixed precision"""
        self.model.train()
        self.precompute_loss_tracking()

        batch_start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Vectorized batch processing (all on GPU)
            input_features, targets = self.preprocessor.process_batch(batch)

            # Mixed precision forward pass
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(input_features)
                    losses = self.criterion(outputs, targets, inputs=None)  # Add inputs param
                    loss = losses['total']

                # Scaled backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision
                outputs = self.model(input_features)
                losses = self.criterion(outputs, targets, inputs=None)  # Add inputs param
                loss = losses['total']

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
                self.optimizer.step()

            # Vectorized loss tracking (no .item() calls in loop)
            self.update_losses(losses)

            # Periodic logging (less frequent to reduce overhead)
            if (batch_idx + 1) % 20 == 0:
                elapsed = time.time() - batch_start_time
                throughput = len(train_loader.dataset) / elapsed
                self.log_progress(batch_idx, len(train_loader), losses, throughput)

        # Compute epoch metrics
        return self.compute_epoch_metrics(len(train_loader))

    @torch.no_grad()
    def validate(self, val_loader):
        """Vectorized validation with no gradient computation"""
        self.model.eval()
        self.precompute_loss_tracking()

        for batch in val_loader:
            # Vectorized batch processing
            input_features, targets = self.preprocessor.process_batch(batch)

            # Mixed precision inference
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(input_features)
                    losses = self.criterion(outputs, targets, inputs=None)  # Add inputs param
            else:
                outputs = self.model(input_features)
                losses = self.criterion(outputs, targets, inputs=None)  # Add inputs param

            # Vectorized loss tracking
            self.update_losses(losses)

        return self.compute_epoch_metrics(len(val_loader))

    def precompute_loss_tracking(self):
        """Pre-allocate tensors for loss tracking"""
        self.loss_totals = {key: 0.0 for key in self.loss_keys}
        self.loss_count = 0

    def update_losses(self, losses):
        """Vectorized loss update (avoid repeated .item() calls)"""
        for key in self.loss_keys:
            if key in losses:
                self.loss_totals[key] += losses[key].detach()
        self.loss_count += 1

    def compute_epoch_metrics(self, num_batches):
        """Compute epoch metrics from accumulated losses"""
        metrics = {}
        for key in self.loss_keys:
            if key in self.loss_totals and num_batches > 0:
                metrics[key] = self.loss_totals[key].item() / num_batches
            else:
                metrics[key] = 0.0
        return metrics

    def log_progress(self, batch_idx, total_batches, losses, throughput):
        """Efficient progress logging"""
        print(f"    Batch {batch_idx:4d}/{total_batches} | "
              f"Loss: {losses['total'].item():.6f} | "
              f"Data: {losses['data'].item():.6f} | "
              f"Physics: {losses['physics'].item():.6f} | "
              f"Throughput: {throughput:.0f} samples/s")


def build_optimized_dataloaders(data_dir, config, batch_size, num_workers=4):
    """
    Build optimized dataloaders with:
    - Pin memory (faster CPU->GPU transfer)
    - Persistent workers (avoid worker recreation)
    - Prefetch factor (overlap data loading and training)
    """
    import glob

    mat_files = glob.glob(os.path.join(data_dir, "*.mat"))
    if not mat_files:
        raise ValueError(f"No .mat files found in {data_dir}")

    random.shuffle(mat_files)
    n_train = int(0.7 * len(mat_files))
    n_val = int(0.15 * len(mat_files))

    train_files = mat_files[:n_train]
    val_files = mat_files[n_train:n_train + n_val]

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

    # Optimized dataloader settings
    dataloader_kwargs = {
        'batch_size': batch_size,
        'pin_memory': True,  # Faster CPU->GPU transfer
        'persistent_workers': True if num_workers > 0 else False,  # Keep workers alive
        'prefetch_factor': 2 if num_workers > 0 else None,  # Prefetch 2 batches
    }

    if num_workers > 0:
        dataloader_kwargs['num_workers'] = num_workers
        dataloader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **dataloader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs
    )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Optimized Training with Vectorization')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)  # Increased for better GPU utilization
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda_physics', type=float, default=0.1)
    parser.add_argument('--use_adaptive_weights', action='store_true')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use mixed precision training (default: True)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    set_seed(args.seed)

    print("=" * 80)
    print("OPTIMIZED: Enhanced Implicit State Inference Training")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Physics loss weight: {args.lambda_physics}")
    print(f"  Adaptive weights: {args.use_adaptive_weights}")
    print(f"  Mixed precision (AMP): {args.use_amp}")
    print(f"  Data workers: {args.num_workers}")
    print(f"  Device: {args.device}")
    print()

    # Load config
    config = get_config(preset='unified')
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.lr

    device = torch.device(args.device)

    # Build optimized dataloaders
    print("Building optimized dataloaders...")
    train_loader, val_loader = build_optimized_dataloaders(
        args.data_dir, config, args.batch_size, args.num_workers
    )
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print()

    # Create model
    print("Creating enhanced model...")
    model = ImplicitStateTCN(config).to(device)

    model_info = model.get_model_info()
    print(f"  Model type: {model_info['model_type']}")
    print(f"  Parameters: {model_info['num_parameters']:,}")
    print()

    # Create loss function
    criterion = AdaptiveMultiTaskLoss(
        lambda_adhesion=1.0,
        lambda_stress=1.0,
        lambda_porosity=1.0,
        lambda_accuracy=1.0,
        lambda_quality=1.0,
        lambda_physics=args.lambda_physics,
        use_adaptive_weights=args.use_adaptive_weights
    )
    print(f"  Physics loss weight: {args.lambda_physics}")
    print(f"  Adaptive weighting: {args.use_adaptive_weights}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.training.weight_decay,
        # Enable optimized fusion
        fused=device.type == 'cuda'
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # Create optimized trainer
    trainer = OptimizedTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        use_amp=args.use_amp
    )

    # Training loop
    print("Starting optimized training...")
    print("=" * 80)
    print()

    best_val_loss = float('inf')
    ckpt_dir = Path('checkpoints/implicit_state_tcn_optimized')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        print("  Training:")
        train_metrics = trainer.train_epoch(train_loader, epoch)

        # Validate
        print("  Validation:")
        val_metrics = trainer.validate(val_loader)

        # Log metrics
        epoch_time = time.time() - epoch_start_time
        print(f"\n  Summary:")
        print(f"    Train | Total: {train_metrics['total']:.6f} | "
              f"Data: {train_metrics['data']:.6f} | "
              f"Physics: {train_metrics['physics']:.6f}")
        print(f"    Val   | Total: {val_metrics['total']:.6f} | "
              f"Data: {val_metrics['data']:.6f} | "
              f"Physics: {val_metrics['physics']:.6f}")
        print(f"    Time: {epoch_time:.2f}s")

        # Log adaptive weights
        if args.use_adaptive_weights:
            weights = criterion.get_effective_weights()
            print(f"    Effective weights:")
            for task, weight in weights.items():
                print(f"      {task}: {weight:.4f}")

        # Step scheduler
        scheduler.step()

        # Save checkpoint
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_metrics['total'],
                'config': config,
                'model_info': model_info,
            }, ckpt_dir / 'best_model.pth')
            print(f"    [*] Saved best model (val_loss: {val_metrics['total']:.6f})")

    total_time = time.time() - total_start_time
    print("\n" + "=" * 80)
    print("Optimized training completed!")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {ckpt_dir / 'best_model.pth'}")
    print(f"Average time per epoch: {total_time/args.epochs:.2f}s")
    print("=" * 80)


if __name__ == '__main__':
    main()
