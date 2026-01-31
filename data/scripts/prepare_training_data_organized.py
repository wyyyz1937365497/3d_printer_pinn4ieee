"""
Data Preparation Pipeline - Organized by Source (Fixed for Duplicate Names)

This script properly handles data from multiple gcode files by keeping them organized.

Usage:
    python prepare_training_data_organized.py --mat_dir "data_simulation_*" --output_dir data/processed

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
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def convert_matlab_to_hdf5_organized(mat_dir, output_base_dir):
    """
    Convert MATLAB files to HDF5, organized by source directory

    Args:
        mat_dir: Directory containing .mat files
        output_base_dir: Base output directory
    """
    mat_files = glob.glob(os.path.join(mat_dir, "*.mat"))

    if not mat_files:
        print(f"Warning: No .mat files found in {mat_dir}")
        return 0, None

    # Create a unique subdirectory for this source
    source_name = os.path.basename(mat_dir)
    # Clean up the name to make it filesystem-friendly
    source_clean = source_name.replace('data_simulation_', '').replace('_sampled_', '_').replace('_PLA_', '')
    output_dir = os.path.join(output_base_dir, 'hdf5_organized', source_clean)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Converting {len(mat_files)} files from {source_clean}...")

    converted = 0
    sample_counts = []

    for mat_file in tqdm(mat_files, desc=f"Converting {source_clean}"):
        try:
            # Use h5py to read MATLAB v7.3 files
            with h5py.File(mat_file, 'r') as mat:
                if 'simulation_data' not in mat:
                    continue

                sim_data = mat['simulation_data']

                # Extract layer info for validation
                try:
                    if 'layer_num' in sim_data:
                        layer_num = int(sim_data['layer_num'][0])
                        n_points = sim_data['x_ref'].shape[0]
                        sample_counts.append((layer_num, n_points))
                except:
                    pass

                # Create output HDF5 file with unique name
                basename = os.path.splitext(os.path.basename(mat_file))[0]
                # Add source prefix to avoid collisions
                h5_filename = f"{source_clean}_{basename}.h5"
                h5_file = os.path.join(output_dir, h5_filename)

                with h5py.File(h5_file, 'w') as f:
                    # Copy all datasets
                    def copy_dataset(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            data = obj[()]
                            if isinstance(data, np.ndarray):
                                data = data.squeeze()

                            # Create dataset - only compress if data is large enough
                            if isinstance(data, np.ndarray) and data.size > 1:
                                f.create_dataset(name, data=data, compression='gzip')
                            elif isinstance(data, np.ndarray) and data.size == 1:
                                f.create_dataset(name, data=data.squeeze())
                            else:
                                f.create_dataset(name, data=data)

                    sim_data.visititems(copy_dataset)

                converted += 1

        except Exception as e:
            print(f"  Error converting {mat_file}: {e}")
            continue

    print(f"  Converted {converted}/{len(mat_files)} files")
    return converted, output_dir


def create_sequences_from_multiple_sources(source_dirs, output_base_dir,
                                          sequence_length=128, stride=4,
                                          train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Create training sequences from multiple source directories

    Args:
        source_dirs: List of directories containing HDF5 files
        output_base_dir: Base output directory
        sequence_length: Length of each sequence
        stride: Stride for sampling
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    """
    print(f"\nCreating sequences from {len(source_dirs)} sources...")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Stride: {stride}")

    # Collect all HDF5 files from all sources
    all_files = []
    for source_dir in source_dirs:
        h5_files = glob.glob(os.path.join(source_dir, "*.h5"))
        all_files.extend(h5_files)

    print(f"  Total HDF5 files: {len(all_files)}")

    # Shuffle for random split
    np.random.shuffle(all_files)

    # Split into train/val/test
    n_total = len(all_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]

    print(f"  Train files: {len(train_files)}")
    print(f"  Val files: {len(val_files)}")
    print(f"  Test files: {len(test_files)}")

    # Process each split
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    stats = {}

    for split_name, file_list in splits.items():
        print(f"\nProcessing {split_name} split ({len(file_list)} files)...")

        all_sequences = []
        all_targets = []

        for file_path in tqdm(file_list, desc=f"Creating {split_name}"):
            try:
                with h5py.File(file_path, 'r') as f:
                    # Check required fields
                    required_fields = ['x_ref', 'y_ref', 'z_ref', 'vx_ref', 'vy_ref', 'vz_ref',
                                      'ax_ref', 'ay_ref', 'az_ref', 'error_x', 'error_y']

                    if not all(field in f for field in required_fields):
                        continue

                    # Load data
                    x_ref = f['x_ref'][()].squeeze()
                    y_ref = f['y_ref'][()].squeeze()
                    z_ref = f['z_ref'][()].squeeze()
                    vx_ref = f['vx_ref'][()].squeeze()
                    vy_ref = f['vy_ref'][()].squeeze()
                    vz_ref = f['vz_ref'][()].squeeze()
                    ax_ref = f['ax_ref'][()].squeeze()
                    ay_ref = f['ay_ref'][()].squeeze()
                    az_ref = f['az_ref'][()].squeeze()
                    error_x = f['error_x'][()].squeeze()
                    error_y = f['error_y'][()].squeeze()

                    n_points = len(x_ref)

                    # Create sequences
                    for i in range(0, n_points - sequence_length + 1, stride):
                        # Features
                        features = np.stack([
                            x_ref[i:i+sequence_length],
                            y_ref[i:i+sequence_length],
                            z_ref[i:i+sequence_length],
                            vx_ref[i:i+sequence_length],
                            vy_ref[i:i+sequence_length],
                            vz_ref[i:i+sequence_length],
                            ax_ref[i:i+sequence_length],
                            ay_ref[i:i+sequence_length],
                            az_ref[i:i+sequence_length]
                        ], axis=1)

                        # Targets
                        targets = np.stack([
                            error_x[i:i+sequence_length],
                            error_y[i:i+sequence_length]
                        ], axis=1)

                        all_sequences.append(features)
                        all_targets.append(targets)

            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
                continue

        # Save to disk
        if all_sequences:
            all_sequences = np.array(all_sequences, dtype=np.float32)
            all_targets = np.array(all_targets, dtype=np.float32)

            split_dir = os.path.join(output_base_dir, split_name)
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

            print(f"  Saved {len(all_sequences):,} samples")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Prepare MATLAB simulation data (organized by source)'
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
    print("3D Printer PINN - Data Preparation (Organized)")
    print("=" * 80)
    print()

    # Clean up old processed data
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
        print(f"Cleaned up old output directory: {args.output_dir}\n")

    # Step 1: Convert all MATLAB files
    print("Step 1: Converting MATLAB files to HDF5 (organized by source)...")

    mat_dirs = glob.glob(args.mat_dir)
    if not mat_dirs:
        print(f"Error: No directories found matching {args.mat_dir}")
        return

    source_dirs = []
    total_converted = 0

    for mat_dir in mat_dirs:
        n_converted, output_dir = convert_matlab_to_hdf5_organized(mat_dir, args.output_dir)
        if n_converted > 0 and output_dir:
            source_dirs.append(output_dir)
            total_converted += n_converted

    if total_converted == 0:
        print("Error: No files were converted!")
        return

    print(f"\nTotal: {total_converted} files converted from {len(source_dirs)} sources")

    # Step 2: Create sequences
    print("\nStep 2: Creating sequence-based training samples...")

    stats = create_sequences_from_multiple_sources(
        source_dirs,
        args.output_dir,
        args.sequence_length,
        args.stride,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
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
