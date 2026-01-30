"""
Quick test script to verify the data conversion

Usage:
    python test_conversion.py
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_sample_data(h5_file, output_dir='test_output'):
    """Visualize a sample of the converted data"""

    Path(output_dir).mkdir(exist_ok=True)

    with h5py.File(h5_file, 'r') as f:
        features = f['features'][:]
        targets = f['targets'][:]
        feature_names = [name.decode() for name in f['feature_names'][:]]

    print(f"Data shape: features={features.shape}, targets={targets.shape}")
    print(f"Feature names ({len(feature_names)}): {feature_names}")

    # Extract a sample trajectory (first 1000 points)
    sample_size = min(1000, len(features))

    # Plot 1: Reference trajectory
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Trajectory (X-Y)
    ax = axes[0, 0]
    ax.plot(features[:sample_size, 0], features[:sample_size, 1], 'b-', alpha=0.5, linewidth=0.5, label='Reference')
    ax.plot(features[:sample_size, 0] + targets[:sample_size, 0],
            features[:sample_size, 1] + targets[:sample_size, 1], 'r-', alpha=0.5, linewidth=0.5, label='Actual')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Reference vs Actual Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Error vectors
    ax = axes[0, 1]
    error_mag = np.sqrt(targets[:, 0]**2 + targets[:, 1]**2)
    ax.plot(error_mag[:sample_size], 'g-', linewidth=0.5)
    ax.axhline(np.mean(error_mag), color='r', linestyle='--', label=f'Mean: {np.mean(error_mag):.6f}')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Error magnitude (mm)')
    ax.set_title('Error Magnitude over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Speed profile
    ax = axes[1, 0]
    v_mag = features[:sample_size, 6]  # v_mag is at index 6
    ax.plot(v_mag, 'b-', linewidth=0.5)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Speed (mm/s)')
    ax.set_title('Speed Profile')
    ax.grid(True, alpha=0.3)

    # Acceleration magnitude
    ax = axes[1, 1]
    a_mag = features[:sample_size, 10]  # a_mag is at index 10
    ax.plot(a_mag, 'r-', linewidth=0.5)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Acceleration (mm/s²)')
    ax.set_title('Acceleration Profile')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/trajectory_overview.png', dpi=150)
    print(f"✓ Saved: {output_dir}/trajectory_overview.png")

    # Plot 2: Feature distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Speed vs Error
    ax = axes[0, 0]
    ax.scatter(features[:sample_size, 6], error_mag[:sample_size], alpha=0.3, s=1)
    ax.set_xlabel('Speed (mm/s)')
    ax.set_ylabel('Error magnitude (mm)')
    ax.set_title('Speed vs Error')
    ax.grid(True, alpha=0.3)

    # Acceleration vs Error
    ax = axes[0, 1]
    ax.scatter(features[:sample_size, 10], error_mag[:sample_size], alpha=0.3, s=1)
    ax.set_xlabel('Acceleration (mm/s²)')
    ax.set_ylabel('Error magnitude (mm)')
    ax.set_title('Acceleration vs Error')
    ax.grid(True, alpha=0.3)

    # Curvature vs Error
    ax = axes[1, 0]
    curvature = features[:sample_size, 15]  # curvature is at index 15
    ax.scatter(curvature, error_mag[:sample_size], alpha=0.3, s=1)
    ax.set_xlabel('Curvature (1/mm)')
    ax.set_ylabel('Error magnitude (mm)')
    ax.set_title('Curvature vs Error')
    ax.grid(True, alpha=0.3)

    # Direction change vs Error
    ax = axes[1, 1]
    direction_change = features[:sample_size, 25]  # direction_change is at index 25
    ax.scatter(direction_change, error_mag[:sample_size], alpha=0.3, s=1)
    ax.set_xlabel('Direction change (rad)')
    ax.set_ylabel('Error magnitude (mm)')
    ax.set_title('Direction Change vs Error')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_correlations.png', dpi=150)
    print(f"✓ Saved: {output_dir}/feature_correlations.png")

    # Plot 3: Error heat map (trajectory with error color-coded)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Color by error magnitude
    scatter = ax.scatter(features[:sample_size, 0], features[:sample_size, 1],
                       c=error_mag[:sample_size], cmap='hot', s=1, alpha=0.5)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Trajectory with Error Magnitude (Color)')
    ax.axis('equal')
    plt.colorbar(scatter, ax=ax, label='Error (mm)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_heatmap.png', dpi=150)
    print(f"✓ Saved: {output_dir}/error_heatmap.png")

    print(f"\n✓ Visualization complete! Check {output_dir}/")


def check_data_integrity(h5_file):
    """Check if the converted data is correct"""

    print("Checking data integrity...")

    with h5py.File(h5_file, 'r') as f:
        features = f['features'][:]
        targets = f['targets'][:]

        # Check shapes
        assert len(features) == len(targets), "Features and targets length mismatch!"
        assert features.shape[1] == 29, f"Expected 29 features, got {features.shape[1]}"
        assert targets.shape[1] == 2, f"Expected 2 targets, got {targets.shape[1]}"

        # Check for NaN/Inf
        assert not np.any(np.isnan(features)), "Features contain NaN!"
        assert not np.any(np.isinf(features)), "Features contain Inf!"
        assert not np.any(np.isnan(targets)), "Targets contain NaN!"

        # Print statistics
        print(f"\n✓ Data integrity check passed!")
        print(f"  Samples: {len(features):,}")
        print(f"  Features: {features.shape[1]}")
        print(f"  Targets: {targets.shape[1]}")

        # Error statistics
        error_x = targets[:, 0]
        error_y = targets[:, 1]
        error_mag = np.sqrt(error_x**2 + error_y**2)

        print(f"\nError statistics:")
        print(f"  X: mean={np.mean(error_x):.6f}, std={np.std(error_x):.6f}, " +
              f"range=[{np.min(error_x):.6f}, {np.max(error_x):.6f}]")
        print(f"  Y: mean={np.mean(error_y):.6f}, std={np.std(error_y):.6f}, " +
              f"range=[{np.min(error_y):.6f}, {np.max(error_y):.6f}]")
        print(f"  Mag: mean={np.mean(error_mag):.6f}, std={np.std(error_mag):.6f}, " +
              f"range=[{np.min(error_mag):.6f}, {np.max(error_mag):.6f}]")

        # Feature statistics (sample)
        print(f"\nFeature statistics (sample):")
        for i, name in enumerate(['x', 'y', 'z', 'v_mag', 'a_mag', 'curvature']):
            if i < 6:
                idx = [0, 1, 2, 6, 10, 15][i]
                print(f"  {name}: mean={np.mean(features[:, idx]):.3f}, " +
                      f"std={np.std(features[:, idx]):.3f}")

        return True


def main():
    import sys

    h5_file = 'trajectory_data.h5'

    if not Path(h5_file).exists():
        print(f"Error: {h5_file} not found!")
        print("\nPlease run the conversion first:")
        print("  python matlab_simulation/convert_to_trajectory_features.py data_simulation_* -o trajectory_data.h5")
        sys.exit(1)

    # Check integrity
    check_data_integrity(h5_file)

    # Visualize
    print("\nGenerating visualizations...")
    visualize_sample_data(h5_file)

    print(f"\n✓ All checks passed!")


if __name__ == '__main__':
    main()
