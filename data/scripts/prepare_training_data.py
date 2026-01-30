"""
Data Preparation Pipeline for 3D Printer PINN Training

This script prepares MATLAB simulation data for training:
1. Converts .mat files to HDF5 format
2. Splits into train/val/test sets
3. Creates data loaders
4. Saves preprocessed data for training

Usage:
    python prepare_training_data.py --mat_dir <path> --output_dir <path>

Author: 3D Printer PINN Project
Date: 2026-01-28
"""

import os
import sys
import argparse
import glob
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.simulation import PrinterSimulationDataset, create_dataloaders
from utils import save_data


def convert_matlab_to_hdf5(mat_dir, output_dir):
    """
    Convert all MATLAB .mat files in a directory to HDF5 format

    Args:
        mat_dir: Directory containing .mat files
        output_dir: Output directory for HDF5 files
    """
    import scipy.io as sio
    import h5py

    mat_files = glob.glob(os.path.join(mat_dir, "*.mat"))

    if not mat_files:
        print(f"Warning: No .mat files found in {mat_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Converting {len(mat_files)} MATLAB files to HDF5...")

    for mat_file in mat_files:
        try:
            # Load MATLAB file
            mat_data = sio.loadmat(mat_file)

            if 'simulation_data' not in mat_data:
                print(f"  Skipping {mat_file} (no simulation_data)")
                continue

            sim_data = mat_data['simulation_data'][0, 0]

            # Create HDF5 file
            basename = os.path.splitext(os.path.basename(mat_file))[0]
            h5_file = os.path.join(output_dir, f"{basename}.h5")

            with h5py.File(h5_file, 'w') as f:
                # Save all fields as datasets
                field_names = sim_data.dtype.names

                for field in field_names:
                    value = sim_data[field][0, 0]

                    if isinstance(value, np.ndarray):
                        value = value.squeeze()

                    if isinstance(value, np.ndarray) and value.dtype.kind in ['i', 'u', 'f', 'b']:
                        f.create_dataset(field, data=value, compression='gzip')
                    elif isinstance(value, (int, float)):
                        f.create_dataset(field, data=value)

            print(f"  Converted: {basename}.mat → {basename}.h5")

        except Exception as e:
            print(f"  Error converting {mat_file}: {e}")

    print(f"Conversion complete! HDF5 files saved to {output_dir}")


def prepare_dataset_splits(data_dir,
                          train_ratio=0.7,
                          val_ratio=0.15,
                          test_ratio=0.15):
    """
    Organize data into train/val/test splits

    Args:
        data_dir: Directory containing HDF5 files
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data

    Returns:
        Dictionary with train/val/test file lists
    """
    import random

    h5_files = glob.glob(os.path.join(data_dir, "*.h5"))

    if not h5_files:
        raise ValueError(f"No HDF5 files found in {data_dir}")

    # Shuffle files
    random.shuffle(h5_files)

    # Calculate split indices
    n_files = len(h5_files)
    train_end = int(n_files * train_ratio)
    val_end = train_end + int(n_files * val_ratio)

    splits = {
        'train': h5_files[:train_end],
        'val': h5_files[train_end:val_end],
        'test': h5_files[val_end:]
    }

    print(f"Dataset splits:")
    print(f"  Train: {len(splits['train'])} files")
    print(f"  Val:   {len(splits['val'])} files")
    print(f"  Test:  {len(splits['test'])} files")

    return splits


def create_dataloaders_from_splits(splits,
                                   batch_size=64,
                                   seq_len=200,
                                   pred_len=50,
                                   stride=10):
    """
    Create PyTorch dataloaders from file splits

    Args:
        splits: Dictionary with train/val/test file lists
        batch_size: Batch size
        seq_len: Input sequence length
        pred_len: Prediction sequence length
        stride: Stride between sequences

    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    from torch.utils.data import DataLoader

    # Create datasets
    train_dataset = PrinterSimulationDataset(
        splits['train'],
        seq_len=seq_len,
        pred_len=pred_len,
        stride=stride,
        mode='train',
        scaler=None,
        fit_scaler=True
    )

    val_dataset = PrinterSimulationDataset(
        splits['val'],
        seq_len=seq_len,
        pred_len=pred_len,
        stride=stride,
        mode='val',
        scaler=train_dataset.scaler,
        fit_scaler=False
    )

    test_dataset = PrinterSimulationDataset(
        splits['test'],
        seq_len=seq_len,
        pred_len=pred_len,
        stride=stride,
        mode='test',
        scaler=train_dataset.scaler,
        fit_scaler=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_dataset.scaler


def save_preprocessed_data(train_loader, val_loader, test_loader, scaler, output_dir):
    """
    Save preprocessed data (scaler, dataset info)

    Args:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        test_loader: Test dataloader
        scaler: Fitted StandardScaler
        output_dir: Output directory
    """
    import pickle

    os.makedirs(output_dir, exist_ok=True)

    # Save scaler
    scaler_file = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)

    # Save dataset info
    dataset_info = {
        'train_size': len(train_loader.dataset),
        'val_size': len(val_loader.dataset),
        'test_size': len(test_loader.dataset),
        'num_features': 12,
        'seq_len': train_loader.dataset.seq_len,
        'pred_len': train_loader.dataset.pred_len,
        'input_features': train_loader.dataset.INPUT_FEATURES,
        'output_trajectory': train_loader.dataset.OUTPUT_TRAJECTORY,
        'output_quality': train_loader.dataset.OUTPUT_QUALITY,
    }

    info_file = os.path.join(output_dir, 'dataset_info.pkl')
    with open(info_file, 'wb') as f:
        pickle.dump(dataset_info, f)

    print(f"\nPreprocessed data saved to {output_dir}:")
    print(f"  scaler.pkl: Fitted StandardScaler")
    print(f"  dataset_info.pkl: Dataset metadata")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare MATLAB simulation data for PINN training'
    )

    parser.add_argument(
        '--mat_dir',
        type=str,
        required=True,
        help='Directory containing MATLAB .mat files'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='Output directory for preprocessed data'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for dataloaders'
    )

    parser.add_argument(
        '--seq_len',
        type=int,
        default=200,
        help='Input sequence length'
    )

    parser.add_argument(
        '--pred_len',
        type=int,
        default=50,
        help='Prediction sequence length'
    )

    parser.add_argument(
        '--stride',
        type=int,
        default=10,
        help='Stride between sequences'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("3D Printer PINN - Data Preparation Pipeline")
    print("=" * 60)
    print()

    # Step 1: Convert MATLAB to HDF5
    print("Step 1: Converting MATLAB files to HDF5 format...")
    h5_dir = os.path.join(args.output_dir, 'hdf5')
    convert_matlab_to_hdf5(args.mat_dir, h5_dir)
    print()

    # Step 2: Split data into train/val/test
    print("Step 2: Splitting data into train/val/test sets...")
    splits = prepare_dataset_splits(h5_dir)
    print()

    # Step 3: Create dataloaders
    print("Step 3: Creating PyTorch dataloaders...")
    train_loader, val_loader, test_loader, scaler = create_dataloaders_from_splits(
        splits,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.stride
    )
    print()

    # Step 4: Save preprocessed data
    print("Step 4: Saving preprocessed data...")
    save_preprocessed_data(
        train_loader, val_loader, test_loader, scaler,
        args.output_dir
    )
    print()

    print("=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Train model: python experiments/train_unified_model.py")
    print("  2. Evaluate: python experiments/evaluate_model.py")
    print()


if __name__ == '__main__':
    main()
