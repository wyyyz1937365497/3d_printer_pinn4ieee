"""
Training script for the unified PINN-Seq3D model
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from config import get_config
from models import UnifiedPINNSeq3D
from training import Trainer
from utils import set_seed, setup_logger


class Synthetic3DPrinterDataset(Dataset):
    """
    Synthetic dataset for demonstration purposes

    In production, replace this with your actual data loading logic
    """

    def __init__(self, num_samples=1000, seq_len=200, num_features=12):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_features = num_features

        # Generate synthetic data
        self.features = torch.randn(num_samples, seq_len, num_features)

        # Generate synthetic targets
        self.targets = {
            'rul': torch.rand(num_samples, 1) * 1000,  # RUL in seconds
            'temperature': torch.rand(num_samples, 1) * 50 + 200,  # 200-250°C
            'vibration_x': torch.randn(num_samples, 1) * 0.1,
            'vibration_y': torch.randn(num_samples, 1) * 0.1,
            'quality_score': torch.rand(num_samples, 1),
            'fault_label': torch.randint(0, 4, (num_samples,)),
            'displacement_x': torch.randn(num_samples, 1) * 0.01,
            'displacement_y': torch.randn(num_samples, 1) * 0.01,
            'displacement_z': torch.randn(num_samples, 1) * 0.001,
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return features and targets as a dictionary
        batch = {
            'features': self.features[idx],
            **{k: v[idx] for k, v in self.targets.items()}
        }
        return batch


def create_data_loaders(config):
    """
    Create train, validation, and test data loaders

    Args:
        config: Configuration object

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = Synthetic3DPrinterDataset(
        num_samples=5000,
        seq_len=config.data.seq_len,
        num_features=config.data.num_features
    )

    val_dataset = Synthetic3DPrinterDataset(
        num_samples=1000,
        seq_len=config.data.seq_len,
        num_features=config.data.num_features
    )

    test_dataset = Synthetic3DPrinterDataset(
        num_samples=1000,
        seq_len=config.data.seq_len,
        num_features=config.data.num_features
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )

    return train_loader, val_loader, test_loader


def main():
    """
    Main training function
    """
    # Setup
    config = get_config(preset='unified')
    set_seed(config.seed)
    logger = setup_logger('unified_model_training', log_dir=config.training.log_dir)

    logger.info("Starting unified model training")
    logger.info(f"Device: {config.device}")
    logger.info(f"Configuration: {config.experiment_name}")

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config)

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    logger.info("Creating model...")
    model = UnifiedPINNSeq3D(config)

    logger.info(f"Model parameters: {model.get_num_params():,}")
    logger.info(f"Trainable parameters: {model.get_num_trainable_params():,}")

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    # Train model
    logger.info("Starting training loop...")
    trainer.train()

    # Test model
    logger.info("Testing model...")
    test_metrics = trainer.test()

    logger.info("Training completed successfully!")
    logger.info(f"Final test metrics: {test_metrics}")


if __name__ == '__main__':
    main()
