"""
Quick Start Training Script

A simplified training script for users who want to get started quickly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from tqdm import tqdm

from config import get_config
from models import UnifiedPINNSeq3D
from training import Trainer
from training.losses import MultiTaskLoss
from utils import set_seed, setup_logger


class SimpleDataset(Dataset):
    """
    Simple dataset for loading preprocessed data
    """

    def __init__(self, data_path: str):
        """
        Initialize dataset

        Args:
            data_path: Path to pickle file with preprocessed data
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        print(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Features: [seq_len, num_features]
        features = torch.tensor(sample['features'], dtype=torch.float32)

        # Targets
        targets = {}
        quality_metrics = sample['targets']

        # Map to expected format
        target_mapping = {
            'adhesion_strength': 'adhesion_strength',
            'internal_stress': 'internal_stress',
            'porosity': 'porosity',
            'dimensional_accuracy': 'dimensional_accuracy',
            'quality_score': 'quality_score',
        }

        for new_key, old_key in target_mapping.items():
            if old_key in quality_metrics:
                value = quality_metrics[old_key]
                targets[new_key] = torch.tensor([[value]], dtype=torch.float32)

        # For fault classification (random for now if not in data)
        if 'fault_label' not in targets:
            targets['fault_label'] = torch.randint(0, 4, (1,), dtype=torch.long)

        # For trajectory correction (random for now if not in data)
        if 'displacement_x' not in targets:
            targets['displacement_x'] = torch.randn(1, 1) * 0.01
            targets['displacement_y'] = torch.randn(1, 1) * 0.01
            targets['displacement_z'] = torch.randn(1, 1) * 0.001

        return {
            'features': features,
            **targets
        }


def check_data_quality(dataloader: DataLoader, name: str = "Dataset"):
    """
    Check data quality and print statistics

    Args:
        dataloader: Data loader to check
        name: Name of the dataset
    """
    print(f"\n{'='*60}")
    print(f"Checking {name} Quality")
    print(f"{'='*60}")

    all_features = []
    all_targets = {key: [] for key in [
        'adhesion_strength', 'internal_stress', 'porosity',
        'dimensional_accuracy', 'quality_score'
    ]}

    for batch in dataloader:
        all_features.append(batch['features'].numpy())
        for key in all_targets.keys():
            if key in batch:
                all_targets[key].append(batch[key].numpy())

    # Concatenate
    all_features = np.concatenate(all_features, axis=0)
    for key in all_targets.keys():
        if all_targets[key]:
            all_targets[key] = np.concatenate(all_targets[key], axis=0).flatten()

    # Print statistics
    print(f"\nFeatures shape: {all_features.shape}")
    print(f"Feature range: [{all_features.min():.3f}, {all_features.max():.3f}]")
    print(f"Feature mean: {all_features.mean():.3f}")
    print(f"Feature std: {all_features.std():.3f}")

    print(f"\nTargets Statistics:")
    for key, values in all_targets.items():
        if values is not None and len(values) > 0:
            print(f"  {key}:")
            print(f"    Mean: {values.mean():.3f}")
            print(f"    Std: {values.std():.3f}")
            print(f"    Range: [{values.min():.3f}, {values.max():.3f}]")

    print(f"{'='*60}\n")


def main():
    """Main training function"""
    print("\n" + "="*80)
    print("QUICK START TRAINING")
    print("="*80 + "\n")

    # Configuration
    config = get_config(preset='unified')

    # Update configuration for quick start
    config.training.num_epochs = 50  # Fewer epochs for quick start
    config.training.batch_size = 16  # Smaller batch size
    config.training.save_every = 10
    config.training.early_stopping_patience = 10

    print(f"Configuration:")
    print(f"  Model: {config.experiment_name}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Device: {config.device}")
    print(f"  Learning rate: {config.training.learning_rate}")

    # Set seed
    set_seed(config.seed)

    # Create data loaders
    data_dir = Path('data/processed')

    if not (data_dir / 'train_data.pkl').exists():
        print("\n❌ Error: Preprocessed data not found!")
        print("\nPlease follow these steps:")
        print("1. Collect sensor data:")
        print("   python data/scripts/collect_sensor_data.py --mode manual")
        print("\n2. Record quality test results:")
        print("   python data/scripts/record_quality_test.py --mode interactive")
        print("\n3. Pair and preprocess data:")
        print("   python data/scripts/pair_and_preprocess.py")
        print("\nThen run this script again.")
        return

    print("\n📁 Loading preprocessed data...")

    train_dataset = SimpleDataset(str(data_dir / 'train_data.pkl'))
    val_dataset = SimpleDataset(str(data_dir / 'val_data.pkl'))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Check data quality
    check_data_quality(train_loader, "Training Data")
    check_data_quality(val_loader, "Validation Data")

    # Create model
    print("\n🧠 Creating model...")
    model = UnifiedPINNSeq3D(config)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")

    # Create trainer
    print("\n🏋️ Setting up trainer...")
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Train
    print("\n🚀 Starting training...")
    print("="*80 + "\n")

    try:
        trainer.train()
        print("\n" + "="*80)
        print("✅ Training completed successfully!")
        print("="*80)

        print(f"\n📁 Model saved to: {config.training.checkpoint_dir}/{config.experiment_name}/")
        print(f"\n🎯 Next steps:")
        print(f"1. Evaluate model:")
        print(f"   python experiments/evaluate_model.py \\")
        print(f"       --model_path checkpoints/{config.experiment_name}/best_model.pth")
        print(f"\n2. Use model for prediction:")
        print(f"   python examples/implicit_quality_examples.py")

    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        print(f"Checkpoint saved at: {config.training.checkpoint_dir}")

    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

        print("\n🔧 Troubleshooting:")
        print("1. Check if data format is correct")
        print("2. Reduce batch size if running out of memory")
        print("3. Reduce sequence length if still having memory issues")
        print("4. Check for NaN values in data")


if __name__ == '__main__':
    main()
