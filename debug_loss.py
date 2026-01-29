#!/usr/bin/env python
"""
Debug script to test loss function with actual data
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from training.losses import MultiTaskLoss

# Create sample batch
batch_size = 2
seq_len = 200
pred_len = 50

# Mock predictions from model
predictions = {
    'displacement_x': torch.randn(batch_size, 1, 1) * 0.1,
    'displacement_y': torch.randn(batch_size, 1, 1) * 0.1,
    'displacement_z': torch.randn(batch_size, 1, 1) * 0.1,
    'adhesion_strength': torch.randn(batch_size, 1) * 10,
    'internal_stress': torch.randn(batch_size, 1) * 10,
    'porosity': torch.sigmoid(torch.randn(batch_size, 1)) * 100,
    'dimensional_accuracy': torch.randn(batch_size, 1) * 0.1,
    'quality_score': torch.sigmoid(torch.randn(batch_size, 1)),
    'displacement_x_seq': torch.randn(batch_size, seq_len, 1) * 0.1,
    'displacement_y_seq': torch.randn(batch_size, seq_len, 1) * 0.1,
    'displacement_z_seq': torch.randn(batch_size, seq_len, 1) * 0.1,
}

# Mock targets
targets = {
    'displacement_x': torch.randn(batch_size, 1, 1) * 0.1,
    'displacement_y': torch.randn(batch_size, 1, 1) * 0.1,
    'displacement_z': torch.randn(batch_size, 1, 1) * 0.1,
    'quality_score': torch.sigmoid(torch.randn(batch_size, 1)),
}

# Create loss function
criterion = MultiTaskLoss(
    lambda_quality=1.0,
    lambda_fault=1.0,
    lambda_trajectory=1.0,
    lambda_physics=0.1
)

# Compute losses
losses = criterion(predictions, targets, None)

print("Loss computation results:")
print(f"  Total loss: {losses['total'].item():.6f}")
print(f"  Quality loss: {losses['quality'].item():.6f}")
print(f"  Fault loss: {losses['fault'].item():.6f}")
print(f"  Trajectory loss: {losses['trajectory'].item():.6f}")
print(f"  Physics loss: {losses['physics'].item():.6f}")

print("\nPredictions shape check:")
for key, val in predictions.items():
    if isinstance(val, torch.Tensor):
        print(f"  {key}: {val.shape}")

print("\nTargets shape check:")
for key, val in targets.items():
    if isinstance(val, torch.Tensor):
        print(f"  {key}: {val.shape}")
