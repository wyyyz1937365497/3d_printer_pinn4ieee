"""
Data Preparation Pipeline for 3D Printer PINN Training (Fixed for MATLAB v7.3)

This script prepares MATLAB simulation data for training:
1. Converts MATLAB v7.3 .mat files (HDF5) to HDF5 format
2. Splits into train/val/test sets
3. Creates sequence-based training samples
4. Saves preprocessed data for training

Usage:
    python prepare_training_data_v2.py --mat_dir <path> --output_dir <path>

Author: 3D Printer PINN Project
Date: 2026-01-31
"""

import os
import sys
import argparse
import glob
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def convert_matlab_to_hdf5(mat_dir, output_dir):
    """
    Convert all MATLAB v7.3 .mat files (HDF5) to clean HDF5 format

    Args:
        mat_dir: Directory containing .mat files
        output_dir: Output directory for HDF5 files
    """
    mat_files = glob.glob(os.path.join(mat_dir, "*.mat"))

    if not mat_files:
        print(f"Warning: No .mat files found in {mat_dir}")
        return 0

    os.makedirs(output_dir, exist_ok=True)

    print(f"Converting {len(mat_files)} MATLAB v7.3 files to HDF5...")

    converted = 0
    for mat_file in tqdm(mat_files, desc="Converting"):
        try:
            # Use h5py to read MATLAB v7.3 files
            with h5py.File(mat_file, 'r') as mat:
                if 'simulation_data' not in mat:
                    print(f"  Warning: {mat_file} has no 'simulation_data' field")
                    continue

                # Extract simulation_data (it's a HDF5 group)
                sim_data = mat['simulation_data']

                # Create output HDF5 file
                basename = os.path.splitext(os.path.basename(mat_file))[0]
                h5_file = os.path.join(output_dir, f"{basename}.h5")

                with h5py.File(h5_file, 'w') as f:
                    # Copy all datasets from simulation_data
                    def copy_dataset(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            # Create dataset with same data
                            data = obj[()]
                            # Squeeze to remove single dimensions
                            if isinstance(data, np.ndarray):
                                data = data.squeeze()

                            # Create dataset - only compress if data is large enough
                            if isinstance(data, np.ndarray) and data.size > 1:
                                f.create_dataset(name, data=data, compression='gzip')
                            elif isinstance(data, np.ndarray) and data.size == 1:
                                # Single-element array - don't compress
                                f.create_dataset(name, data=data.squeeze())
                            else:
                                # Scalar - don't compress
                                f.create_dataset(name, data=data)

                    # Visit all items in simulation_data
                    sim_data.visititems(copy_dataset)

                converted += 1

        except Exception as e:
            print(f"  Error converting {mat_file}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Conversion complete! {converted}/{len(mat_files)} files converted to {output_dir}")
    return converted


def prepare_dataset_splits(h5_dir,
                          output_dir,
                          train_ratio=0.8,
                          val_ratio=0.1,
                          test_ratio=0.1):
    """
    Organize HDF5 files into train/val/test splits

    Args:
        h5_dir: Directory containing HDF5 files
        output_dir: Base output directory
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data

    Returns:
        Dictionary with split file lists
    """
    h5_files = glob.glob(os.path.join(h5_dir, "*.h5"))

    if not h5_files:
        raise ValueError(f"No HDF5 files found in {h5_dir}")

    # Shuffle files for random split
    np.random.shuffle(h5_files)

    # Calculate split indices
    n_total = len(h5_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Split files
    train_files = h5_files[:n_train]
    val_files = h5_files[n_train:n_train + n_val]
    test_files = h5_files[n_train + n_val:]

    # Save split lists
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    os.makedirs(output_dir, exist_ok=True)

    for split_name, file_list in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Save file list
        list_file = os.path.join(split_dir, f"{split_name}_files.txt")
        with open(list_file, 'w') as f:
            for file_path in file_list:
                f.write(f"{file_path}\n")

        print(f"  {split_name.capitalize()}: {len(file_list)} files")

    return splits


def create_sequence_samples(file_lists,
                           output_dir,
                           sequence_length=128,
                           stride=4):
    """
    Create sequence-based samples from HDF5 files

    Args:
        file_lists: Dictionary with train/val/test file lists
        output_dir: Base output directory
        sequence_length: Length of each sequence
        stride: Stride for sampling sequences

    Returns:
        Statistics about created samples
    """
    stats = {}

    for split_name, file_list in file_lists.items():
        print(f"\nProcessing {split_name} split...")

        all_sequences = []
        all_targets = []

        for file_path in tqdm(file_list, desc=f"Creating {split_name} sequences"):
            try:
                with h5py.File(file_path, 'r') as f:
                    # Load trajectory data
                    x_ref = f['x_ref'][()]
                    y_ref = f['y_ref'][:]
                    z_ref = f['z_ref'][:]

                    vx_ref = f['vx_ref'][:]
                    vy_ref = f['vy_ref'][:]
                    vz_ref = f['vz_ref'][:]

                    ax_ref = f['ax_ref'][:]
                    ay_ref = f['ay_ref'][:]
                    az_ref = f['az_ref'][:]

                    error_x = f['error_x'][:]
                    error_y = f['error_y'][:]

                    # Ensure all arrays are 1D
                    x_ref = x_ref.squeeze()
                    y_ref = y_ref.squeeze()
                    z_ref = z_ref.squeeze()
                    vx_ref = vx_ref.squeeze()
                    vy_ref = vy_ref.squeeze()
                    vz_ref = vz_ref.squeeze()
                    ax_ref = ax_ref.squeeze()
                    ay_ref = ay_ref.squeeze()
                    az_ref = az_ref.squeeze()
                    error_x = error_x.squeeze()
                    error_y = error_y.squeeze()

                    n_points = len(x_ref)

                    # Create sequences
                    for i in range(0, n_points - sequence_length + 1, stride):
                        # Extract sequence
                        seq_x = x_ref[i:i+sequence_length]
                        seq_y = y_ref[i:i+sequence_length]
                        seq_z = z_ref[i:i+sequence_length]
                        seq_vx = vx_ref[i:i+sequence_length]
                        seq_vy = vy_ref[i:i+sequence_length]
                        seq_vz = vz_ref[i:i+sequence_length]
                        seq_ax = ax_ref[i:i+sequence_length]
                        seq_ay = ay_ref[i:i+sequence_length]
                        seq_az = az_ref[i:i+sequence_length]

                        # Stack features: [x, y, z, vx, vy, vz, ax, ay, az]
                        features = np.stack([seq_x, seq_y, seq_z,
                                            seq_vx, seq_vy, seq_vz,
                                            seq_ax, seq_ay, seq_az], axis=1)

                        # Target: [error_x, error_y]
                        target = np.stack([error_x[i:i+sequence_length],
                                          error_y[i:i+sequence_length]], axis=1)

                        all_sequences.append(features)
                        all_targets.append(target)

            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
                continue

        # Convert to numpy arrays
        if all_sequences:
            all_sequences = np.array(all_sequences, dtype=np.float32)
            all_targets = np.array(all_targets, dtype=np.float32)

            # Save to disk
            split_dir = os.path.join(output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)

            seq_file = os.path.join(split_dir, f"{split_name}_sequences.npy")
            target_file = os.path.join(split_dir, f"{split_name}_targets.npy")

            np.save(seq_file, all_sequences)
            np.save(target_file, all_targets)

            stats[split_name] = {
                'n_samples': len(all_sequences),
                'sequence_length': sequence_length,
                'feature_dim': all_sequences.shape[2],
                'target_dim': all_targets.shape[2]
            }

            print(f"  ✓ {split_name.capitalize()}: {len(all_sequences):,} samples")
            print(f"    Features: {all_sequences.shape}")
            print(f"    Targets: {all_targets.shape}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Prepare MATLAB v7.3 simulation data for training'
    )
    parser.add_argument('--mat_dir', type=str, required=True,
                       help='Directory containing MATLAB .mat files (supports wildcards)')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--sequence_length', type=int, default=128,
                       help='Length of each sequence')
    parser.add_argument('--stride', type=int, default=4,
                       help='Stride for sampling sequences')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Ratio of validation data')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Ratio of test data')

    args = parser.parse_args()

    print("=" * 80)
    print("3D Printer PINN - Data Preparation Pipeline (MATLAB v7.3)")
    print("=" * 80)
    print()

    # Step 1: Convert MATLAB files to HDF5
    print("Step 1: Converting MATLAB v7.3 files to HDF5 format...")

    # Handle wildcards in mat_dir
    mat_dirs = glob.glob(args.mat_dir)
    if not mat_dirs:
        print(f"Error: No directories found matching {args.mat_dir}")
        return

    h5_dir = os.path.join(args.output_dir, 'hdf5')
    os.makedirs(h5_dir, exist_ok=True)

    total_converted = 0
    for mat_dir in mat_dirs:
        print(f"\nProcessing directory: {mat_dir}")
        n_converted = convert_matlab_to_hdf5(mat_dir, h5_dir)
        total_converted += n_converted

    if total_converted == 0:
        print("Error: No files were converted!")
        return

    print(f"\n✓ Total: {total_converted} files converted")

    # Step 2: Split into train/val/test
    print("\nStep 2: Splitting data into train/val/test sets...")
    try:
        file_lists = prepare_dataset_splits(
            h5_dir,
            args.output_dir,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Step 3: Create sequence samples
    print("\nStep 3: Creating sequence-based training samples...")
    print(f"  Sequence length: {args.sequence_length}")
    print(f"  Stride: {args.stride}")

    stats = create_sequence_samples(
        file_lists,
        args.output_dir,
        args.sequence_length,
        args.stride
    )

    # Summary
    print("\n" + "=" * 80)
    print("Data Preparation Complete!")
    print("=" * 80)

    total_samples = sum(s['n_samples'] for s in stats.values())
    print(f"\nTotal samples created: {total_samples:,}")

    for split_name, split_stats in stats.items():
        print(f"  {split_name.capitalize()}: {split_stats['n_samples']:,} samples")

    print(f"\nData saved to: {args.output_dir}")
    print("\nYou can now train the model with:")
    print(f"  python experiments/train_trajectory_model.py --data_root {args.output_dir}")


if __name__ == '__main__':
    main()
