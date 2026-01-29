"""
Convert MATLAB simulation data to trajectory features for Transformer+BiLSTM model

Key design:
- Input: Trajectory features only (NO parameters like max_accel, fan, etc.)
- Output: Displacement vectors (error_x, error_y)

Author: 3D Printer PINN Project
Date: 2026-01-28
"""

import os
import numpy as np
import h5py
import glob
from pathlib import Path
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
import argparse


def compute_curvature(x, y, vx, vy, ax, ay):
    """
    Compute curvature at each point

    Formula: κ = |v × a| / |v|³
             = |vx*ay - vy*ax| / (vx² + vy²)^(3/2)
    """
    numerator = np.abs(vx * ay - vy * ax)
    denominator = np.power(vx**2 + vy**2, 1.5) + 1e-8
    curvature = numerator / denominator
    return curvature


def compute_jerk(jx, jy, jz):
    """Compute jerk magnitude"""
    return np.sqrt(jx**2 + jy**2 + jz**2)


def extract_trajectory_features(trajectory_data, simulation_data):
    """
    Extract trajectory features for Transformer+BiLSTM input

    Input: trajectory_data, simulation_data from MATLAB
    Output: features [T, feature_dim], targets [T, 2]
    """

    # Basic kinematic data
    x = trajectory_data['x_ref'].flatten()
    y = trajectory_data['y_ref'].flatten()
    z = trajectory_data['z_ref'].flatten()

    vx = trajectory_data['vx_ref'].flatten()
    vy = trajectory_data['vy_ref'].flatten()
    vz = trajectory_data['vz_ref'].flatten()
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)

    ax = trajectory_data['ax_ref'].flatten()
    ay = trajectory_data['ay_ref'].flatten()
    az = trajectory_data['az_ref'].flatten()
    a_mag = np.sqrt(ax**2 + ay**2 + az**2)

    jx = trajectory_data['jx_ref'].flatten()
    jy = trajectory_data['jy_ref'].flatten()
    jz = trajectory_data['jz_ref'].flatten()
    jerk_mag = compute_jerk(jx, jy, jz)

    # Computed features
    curvature = compute_curvature(x, y, vx, vy, ax, ay)

    # Velocity direction (normalized)
    vx_norm = np.zeros_like(vx)
    vy_norm = np.zeros_like(vy)
    valid_v = v_mag > 1e-8
    vx_norm[valid_v] = vx[valid_v] / v_mag[valid_v]
    vy_norm[valid_v] = vy[valid_v] / v_mag[valid_v]

    # Acceleration direction (normalized)
    ax_norm = np.zeros_like(ax)
    ay_norm = np.zeros_like(ay)
    valid_a = a_mag > 1e-8
    ax_norm[valid_a] = ax[valid_a] / a_mag[valid_a]
    ay_norm[valid_a] = ay[valid_a] / a_mag[valid_a]

    # Next position (shifted by 1)
    next_x = np.roll(x, -1)
    next_y = np.roll(y, -1)
    # Handle last point
    next_x[-1] = x[-1]
    next_y[-1] = y[-1]

    # Previous position (for bidirectional context)
    prev_x = np.roll(x, 1)
    prev_y = np.roll(y, 1)
    # Handle first point
    prev_x[0] = x[0]
    prev_y[0] = y[0]

    # Distance to next point
    dist_next = np.sqrt((next_x - x)**2 + (next_y - y)**2)

    # Distance from previous point
    dist_prev = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)

    # Speed change (acceleration magnitude smoothed)
    speed_change = np.gradient(v_mag)

    # Direction change (angle between consecutive velocity vectors)
    dot_product = vx[:-1] * np.roll(vx, -1)[:-1] + vy[:-1] * np.roll(vy, -1)[:-1]
    v_mag_prev = v_mag[:-1]
    v_mag_next = np.roll(v_mag, -1)[:-1]
    cos_angle = np.zeros_like(v_mag)
    valid_angle = (v_mag_prev > 1e-8) & (v_mag_next > 1e-8)
    cos_angle[:-1][valid_angle] = dot_product[valid_angle] / (v_mag_prev[valid_angle] * v_mag_next[valid_angle])
    cos_angle[-1] = cos_angle[-2]
    direction_change = np.arccos(np.clip(cos_angle, -1, 1))

    # Is corner flag (high curvature)
    is_corner = (curvature > 0.1).astype(float)

    # Is extruding
    is_extruding = trajectory_data['is_extruding'].flatten().astype(float)

    # Time
    time = trajectory_data['time'].flatten()

    # Combine all features
    features = np.stack([
        x, y, z,                      # Position (3)
        vx, vy, vz, v_mag,            # Velocity (4)
        ax, ay, az, a_mag,            # Acceleration (4)
        jx, jy, jz, jerk_mag,         # Jerk (4)
        curvature,                    # Curvature (1)
        vx_norm, vy_norm,             # Velocity direction (2)
        ax_norm, ay_norm,             # Acceleration direction (2)
        next_x - x, next_y - y,       # Next position delta (2)
        dist_next, dist_prev,         # Distance (2)
        speed_change,                 # Speed change (1)
        direction_change,             # Direction change (1)
        is_corner,                    # Corner flag (1)
        is_extruding,                 # Extruding flag (1)
        time,                         # Time (1)
    ], axis=-1)

    # Targets: displacement vectors (error)
    error_x = simulation_data['error_x'].flatten()
    error_y = simulation_data['error_y'].flatten()
    targets = np.stack([error_x, error_y], axis=-1)

    return features, targets


def normalize_features(features, method='standard'):
    """
    Normalize features

    Args:
        features: [N, feature_dim]
        method: 'standard' (z-score) or 'minmax' or 'robust'

    Returns:
        normalized_features, scaler_params
    """
    scaler_params = {}

    if method == 'standard':
        # Z-score normalization
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        std[std < 1e-8] = 1.0  # Avoid division by zero
        normalized = (features - mean) / std
        scaler_params['mean'] = mean
        scaler_params['std'] = std

    elif method == 'minmax':
        # Min-max normalization to [-1, 1]
        min_val = np.min(features, axis=0, keepdims=True)
        max_val = np.max(features, axis=0, keepdims=True)
        range_val = max_val - min_val
        range_val[range_val < 1e-8] = 1.0
        normalized = 2 * (features - min_val) / range_val - 1
        scaler_params['min'] = min_val
        scaler_params['max'] = max_val

    elif method == 'robust':
        # Robust normalization (median and IQR)
        median = np.median(features, axis=0, keepdims=True)
        q75 = np.percentile(features, 75, axis=0, keepdims=True)
        q25 = np.percentile(features, 25, axis=0, keepdims=True)
        iqr = q75 - q25
        iqr[iqr < 1e-8] = 1.0
        normalized = (features - median) / iqr
        scaler_params['median'] = median
        scaler_params['iqr'] = iqr

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, scaler_params


def load_matlab_file(mat_file):
    """
    Load MATLAB .mat file (supports both v7.3 HDF5 format and older formats)

    Returns: dictionary with simulation data
    """
    # Try h5py first (for v7.3 format)
    try:
        with h5py.File(mat_file, 'r') as f:
            if 'simulation_data' in f:
                # v7.3 format - HDF5
                sim_data_group = f['simulation_data']
                trajectory_dict = {}

                # Extract all datasets from the group
                def extract_items(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        # Convert dataset to numpy array
                        trajectory_dict[name] = obj[:]

                sim_data_group.visititems(extract_items)
                return trajectory_dict
    except Exception as e:
        # Not a v7.3 HDF5 file, try scipy
        pass

    # Try scipy.io.loadmat (for older formats)
    try:
        mat_data = loadmat(mat_file)
        simulation_data = mat_data['simulation_data'][0, 0]

        # Extract trajectory data
        field_names = simulation_data.dtype.names
        trajectory_dict = {}
        for field in field_names:
            if field not in ['params']:
                trajectory_dict[field] = simulation_data[field]

        return trajectory_dict
    except Exception as e:
        raise ValueError(f"Failed to load {mat_file}: {e}")


def process_single_file(mat_file, normalize=True, norm_method='standard'):
    """
    Process a single MATLAB .mat file

    Returns:
        features: [T, feature_dim] - trajectory features
        targets: [T, 2] - displacement vectors
        metadata: dict with file info
    """
    print(f"  Processing: {os.path.basename(mat_file)}")

    # Load MATLAB file (supports both v7.3 and older formats)
    try:
        trajectory_dict = load_matlab_file(mat_file)

        # Extract features and targets
        features, targets = extract_trajectory_features(trajectory_dict, trajectory_dict)

        # Normalize features
        if normalize:
            features, scaler_params = normalize_features(features, method=norm_method)
        else:
            scaler_params = None

        # Metadata
        metadata = {
            'n_points': features.shape[0],
            'feature_dim': features.shape[1],
            'file_name': os.path.basename(mat_file),
        }

        print(f"    ✓ Extracted {features.shape[0]} points")
        return features, targets, metadata, scaler_params

    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def convert_matlab_to_hdf5(matlab_files, output_h5, normalize=True, norm_method='standard'):
    """
    Convert multiple MATLAB .mat files to HDF5 format

    Args:
        matlab_files: list of .mat file paths
        output_h5: output HDF5 file path
        normalize: whether to normalize features
        norm_method: normalization method ('standard', 'minmax', 'robust')
    """
    print(f"\nConverting {len(matlab_files)} MATLAB files to HDF5...")
    print(f"Output: {output_h5}")
    print(f"Normalization: {normalize} ({norm_method})")
    print()

    all_features = []
    all_targets = []
    all_metadata = []

    for i, mat_file in enumerate(matlab_files):
        features, targets, metadata, _ = process_single_file(mat_file, normalize, norm_method)

        if features is not None:
            all_features.append(features)
            all_targets.append(targets)
            all_metadata.append(metadata)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(matlab_files)} files")

    if not all_features:
        print("  ✗ No valid files found!")
        return

    # Concatenate all data
    print("\nConcatenating data...")
    all_features = np.concatenate(all_features, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Statistics
    total_points = all_features.shape[0]
    feature_dim = all_features.shape[1]
    target_dim = all_targets.shape[1]

    print(f"\nDataset statistics:")
    print(f"  Total samples: {total_points:,}")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Target dimension: {target_dim}")

    # Target statistics
    error_x = all_targets[:, 0]
    error_y = all_targets[:, 1]
    error_mag = np.sqrt(error_x**2 + error_y**2)

    print(f"\nTarget (error) statistics:")
    print(f"  X error: mean={np.mean(error_x):.6f}, std={np.std(error_x):.6f}, " +
          f"min={np.min(error_x):.6f}, max={np.max(error_x):.6f}")
    print(f"  Y error: mean={np.mean(error_y):.6f}, std={np.std(error_y):.6f}, " +
          f"min={np.min(error_y):.6f}, max={np.max(error_y):.6f}")
    print(f"  Error magnitude: mean={np.mean(error_mag):.6f}, std={np.std(error_mag):.6f}, " +
          f"min={np.min(error_mag):.6f}, max={np.max(error_mag):.6f}")

    # Save to HDF5
    print(f"\nSaving to HDF5...")
    with h5py.File(output_h5, 'w') as f:
        # Create datasets
        f.create_dataset('features', data=all_features, compression='gzip')
        f.create_dataset('targets', data=all_targets, compression='gzip')

        # Metadata
        f.attrs['n_samples'] = total_points
        f.attrs['feature_dim'] = feature_dim
        f.attrs['target_dim'] = target_dim
        f.attrs['n_files'] = len(all_metadata)

        # Feature names (for reference)
        feature_names = [
            'x', 'y', 'z',                      # Position (3)
            'vx', 'vy', 'vz', 'v_mag',         # Velocity (4)
            'ax', 'ay', 'az', 'a_mag',         # Acceleration (4)
            'jx', 'jy', 'jz', 'jerk_mag',      # Jerk (4)
            'curvature',                        # Curvature (1)
            'vx_norm', 'vy_norm',              # Velocity direction (2)
            'ax_norm', 'ay_norm',              # Acceleration direction (2)
            'dx_next', 'dy_next',              # Next position delta (2)
            'dist_next', 'dist_prev',          # Distance (2)
            'speed_change',                    # Speed change (1)
            'direction_change',                # Direction change (1)
            'is_corner',                       # Corner flag (1)
            'is_extruding',                    # Extruding flag (1)
            'time',                            # Time (1)
        ]
        f.create_dataset('feature_names', data=np.bytes_(feature_names))

        # Store metadata
        for i, meta in enumerate(all_metadata):
            grp = f.create_group(f'file_{i:04d}')
            for key, value in meta.items():
                grp.attrs[key] = value

    print(f"  ✓ Saved: {output_h5}")
    print(f"  File size: {os.path.getsize(output_h5) / 1024 / 1024:.2f} MB")
    print(f"\n✓ Conversion complete!")


def main():
    parser = argparse.ArgumentParser(description='Convert MATLAB simulation data to HDF5')
    parser.add_argument('input_dirs', nargs='+',
                       help='Input directories containing .mat files')
    parser.add_argument('-o', '--output', default='trajectory_data.h5',
                       help='Output HDF5 file (default: trajectory_data.h5)')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize features (default: True)')
    parser.add_argument('--no-normalize', action='store_false', dest='normalize',
                       help='Do not normalize features')
    parser.add_argument('--norm-method', choices=['standard', 'minmax', 'robust'],
                       default='standard', help='Normalization method (default: standard)')

    args = parser.parse_args()

    # Find all .mat files
    matlab_files = []
    for input_dir in args.input_dirs:
        pattern = os.path.join(input_dir, '*.mat')
        matlab_files.extend(glob.glob(pattern))

    if not matlab_files:
        print("Error: No .mat files found in input directories!")
        return

    print(f"Found {len(matlab_files)} .mat files")

    # Convert
    convert_matlab_to_hdf5(
        matlab_files,
        args.output,
        normalize=args.normalize,
        norm_method=args.norm_method
    )


if __name__ == '__main__':
    main()
