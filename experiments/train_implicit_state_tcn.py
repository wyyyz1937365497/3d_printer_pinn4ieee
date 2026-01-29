"""
Train Enhanced Implicit State Inference Model with PINN

Features:
1. Enhanced TCN with LayerNorm, Attention Pooling, Skip Connections
2. Physics-constrained outputs
3. Adaptive multi-task loss with physics-informed regularization
4. Automatic loss weighting based on task uncertainty
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
from models.implicit import ImplicitStateTCN, AdaptiveMultiTaskLoss
from data.simulation import PrinterSimulationDataset
from utils import set_seed


def build_loaders(data_dir, config, batch_size):
    """Build train and validation dataloaders"""
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
        num_workers=config.num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    return train_loader, val_loader


def extract_physics_inputs(batch, device):
    """
    Extract input features needed for physics-informed loss

    Args:
        batch: Data batch from dataloader
        device: torch device

    Returns:
        Dictionary with physics inputs
    """
    # Get input features: [B, T, F]
    input_features = batch['input_features']

    # We need to extract specific features from the input_features
    # Assuming the input_features contains:
    # - Temperature-related features
    # - Acceleration features
    # etc.

    # For now, we'll extract what we can from the batch
    # Note: The exact feature indices depend on your data structure

    physics_inputs = {}

    # Try to get interface temperature from input features
    # This depends on your feature ordering
    # You may need to adjust indices based on your actual data

    # For demonstration, assuming temperature is one of the features
    if input_features.size(-1) > 5:  # If we have temperature features
        # Extract temperature-related feature (adjust index as needed)
        physics_inputs['T_interface'] = input_features[..., 5:6]  # Adjust index

    # Extract acceleration magnitude
    if input_features.size(-1) > 10:
        # Assuming acceleration features are at certain indices
        # You may compute magnitude from x, y, z components
        physics_inputs['acceleration'] = input_features[..., 9:10]  # Adjust index

    # Move to device
    physics_inputs = {k: v.to(device) for k, v in physics_inputs.items()}

    return physics_inputs


def train_epoch(model, train_loader, criterion, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_data_loss = 0.0
    total_physics_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        outputs = model(batch['input_features'])

        # Prepare targets
        quality_targets = batch['quality_targets']
        targets = {
            'adhesion_strength': quality_targets[:, 0:1],
            'internal_stress': quality_targets[:, 1:2],
            'porosity': quality_targets[:, 2:3],
            'dimensional_accuracy': quality_targets[:, 3:4],
            'quality_score': quality_targets[:, 4:5],
        }

        # Extract physics inputs (not used in current physics loss design, pass None)
        # physics_inputs = extract_physics_inputs(batch, device)

        # Compute loss with physics constraints
        losses = criterion(outputs, targets, inputs=None)  # Pass None since we don't use inputs
        loss = losses['total']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
        optimizer.step()

        # Track losses
        total_loss += loss.item()
        total_data_loss += losses['data'].item()
        total_physics_loss += losses['physics'].item()

        # Log progress
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.6f} | "
                  f"Data: {losses['data'].item():.6f} | "
                  f"Physics: {losses['physics'].item():.6f}")

    n_batches = len(train_loader)
    return total_loss / n_batches, total_data_loss / n_batches, total_physics_loss / n_batches


def validate(model, val_loader, criterion, device, config):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    total_data_loss = 0.0
    total_physics_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass
            outputs = model(batch['input_features'])

            # Prepare targets
            quality_targets = batch['quality_targets']
            targets = {
                'adhesion_strength': quality_targets[:, 0:1],
                'internal_stress': quality_targets[:, 1:2],
                'porosity': quality_targets[:, 2:3],
                'dimensional_accuracy': quality_targets[:, 3:4],
                'quality_score': quality_targets[:, 4:5],
            }

            # Extract physics inputs (not used, pass None)
            # physics_inputs = extract_physics_inputs(batch, device)

            # Compute loss
            losses = criterion(outputs, targets, inputs=None)
            total_loss += losses['total'].item()
            total_data_loss += losses['data'].item()
            total_physics_loss += losses['physics'].item()

    n_batches = len(val_loader)
    return total_loss / n_batches, total_data_loss / n_batches, total_physics_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description='Train Enhanced Implicit State Inference Model with PINN')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing .mat files')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--lambda_physics', type=float, default=0.1,
                       help='Physics loss weight (0 to disable)')
    parser.add_argument('--use_adaptive_weights', action='store_true',
                       help='Use adaptive loss weighting (homoscedastic uncertainty)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    set_seed(args.seed)

    print("=" * 80)
    print("Enhanced Implicit State Inference Training with PINN")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Physics loss weight: {args.lambda_physics}")
    print(f"  Adaptive weights: {args.use_adaptive_weights}")
    print(f"  Device: {args.device}")
    print()

    # Load config
    config = get_config(preset='unified')
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.lr

    device = torch.device(args.device)

    # Build dataloaders
    print("Building dataloaders...")
    train_loader, val_loader = build_loaders(args.data_dir, config, args.batch_size)
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print()

    # Create model
    print("Creating enhanced model...")
    model = ImplicitStateTCN(config).to(device)

    model_info = model.get_model_info()
    print(f"  Model type: {model_info['model_type']}")
    print(f"  Parameters: {model_info['num_parameters']:,}")
    print(f"  Features: {', '.join(model_info['features'])}")
    print()

    # Create loss function with PINN
    print("Creating adaptive multi-task loss with physics constraints...")
    criterion = AdaptiveMultiTaskLoss(
        lambda_adhesion=1.0,
        lambda_stress=1.0,
        lambda_porosity=1.0,
        lambda_accuracy=1.0,
        lambda_quality=1.0,
        lambda_physics=args.lambda_physics,
        use_adaptive_weights=args.use_adaptive_weights
    )
    print(f"  Physics loss enabled: {args.lambda_physics > 0}")
    print(f"  Adaptive weighting: {args.use_adaptive_weights}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.training.weight_decay
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # Training loop
    print("Starting training...")
    print("=" * 80)
    print()

    best_val_loss = float('inf')
    ckpt_dir = Path('checkpoints/implicit_state_tcn_enhanced')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_loss, train_data, train_physics = train_epoch(
            model, train_loader, criterion, optimizer, device, config
        )

        print(f"\n  Train | Loss: {train_loss:.6f} | "
              f"Data: {train_data:.6f} | Physics: {train_physics:.6f}")

        # Validate
        val_loss, val_data, val_physics = validate(
            model, val_loader, criterion, device, config
        )

        print(f"  Val   | Loss: {val_loss:.6f} | "
              f"Data: {val_data:.6f} | Physics: {val_physics:.6f}")

        # Log adaptive weights
        if args.use_adaptive_weights:
            weights = criterion.get_effective_weights()
            print(f"  Effective weights:")
            for task, weight in weights.items():
                print(f"    {task}: {weight:.4f}")

        # Step scheduler
        scheduler.step()

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, ckpt_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {ckpt_dir / 'best_model.pth'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
