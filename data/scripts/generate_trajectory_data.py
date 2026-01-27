"""
Generate synthetic trajectory data for corner correction

This script generates synthetic trajectory data with corner errors.
In production, replace this with actual printer data.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Tuple
import argparse


def generate_trajectory_data(
    num_sequences: int = 5000,
    seq_len: int = 10,
    num_features: int = 4,
    corner_ratio: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate trajectory data with corner errors

    Args:
        num_sequences: Number of sequences to generate
        seq_len: Sequence length
        num_features: Number of features per time step
        corner_ratio: Ratio of sharp corner sequences

    Returns:
        features: Feature array [num_sequences, seq_len, num_features]
        targets: Target displacement array [num_sequences, 3]
    """
    print(f"Generating {num_sequences} trajectory sequences...")

    features = np.zeros((num_sequences, seq_len, num_features))
    targets = np.zeros((num_sequences, 3))  # dx, dy, dz

    for i in range(num_sequences):
        # Decide if this is a sharp corner or rounded corner
        is_sharp_corner = np.random.rand() < corner_ratio

        # Starting position
        x, y, z = 100.0, 100.0, 0.2

        # Velocity and direction
        velocity = np.random.uniform(20, 50)  # mm/s
        direction = np.random.uniform(0, 2 * np.pi)  # radians

        # Physical parameters
        weight = np.random.uniform(1, 5)  # kg
        stiffness = np.random.uniform(1000, 5000)  # N/mm

        # Corner radius
        if is_sharp_corner:
            corner_radius = np.random.uniform(1, 5)  # mm (sharp corner)
        else:
            corner_radius = np.random.uniform(10, 20)  # mm (rounded corner)

        for t in range(seq_len):
            # Update direction (simulate corner)
            if t == seq_len // 2:
                # Turn
                turn_angle = np.random.uniform(np.pi / 4, np.pi / 2)  # 45-90 degrees
                if np.random.rand() < 0.5:
                    direction += turn_angle
                else:
                    direction -= turn_angle

            # Calculate velocity components
            vx = velocity * np.cos(direction)
            vy = velocity * np.sin(direction)
            vz = 0.0  # Assume flat layer

            # Position
            features[i, t, 0] = x
            features[i, t, 1] = y
            features[i, t, 2] = z

            # Speed and direction
            features[i, t, 3] = velocity

            # Update position
            x += vx * 0.01  # 10ms timestep
            y += vy * 0.01

        # Calculate displacement error based on physics
        # Error ∝ velocity² × weight / stiffness
        base_error = (velocity ** 2) * weight / stiffness

        # Sharp corners have amplified error
        if is_sharp_corner:
            error_multiplier = 1.5
        else:
            error_multiplier = 1.0

        # Add direction-dependent error
        error_x = error_multiplier * base_error * np.cos(direction)
        error_y = error_multiplier * base_error * np.sin(direction)
        error_z = 0.1 * base_error  # Small Z error

        # Add noise
        noise_level = 0.01
        error_x += noise_level * np.random.randn()
        error_y += noise_level * np.random.randn()
        error_z += noise_level * np.random.randn()

        targets[i, 0] = error_x
        targets[i, 1] = error_y
        targets[i, 2] = error_z

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_sequences} sequences...")

    print(f"Trajectory data generation completed!")
    return features, targets


def save_trajectory_data(features: np.ndarray, targets: np.ndarray, output_dir: str):
    """
    Save trajectory data to files

    Args:
        features: Feature array
        targets: Target array
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Split into train/val/test
    num_samples = features.shape[0]
    train_end = int(0.7 * num_samples)
    val_end = int(0.85 * num_samples)

    splits = {
        'train': (0, train_end),
        'val': (train_end, val_end),
        'test': (val_end, num_samples),
    }

    for split_name, (start, end) in splits.items():
        split_features = features[start:end]
        split_targets = targets[start:end]

        data = {
            'features': split_features,
            'targets': split_targets,
        }

        output_file = output_path / f"trajectory_{split_name}_data.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved {split_name} trajectory data to {output_file}")
        print(f"  Samples: {end - start}")
        print(f"  Features shape: {split_features.shape}")


def main():
    parser = argparse.ArgumentParser(description='Generate trajectory data')
    parser.add_argument('--num_sequences', type=int, default=5000, help='Number of sequences')
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length')
    parser.add_argument('--num_features', type=int, default=4, help='Number of features')
    parser.add_argument('--corner_ratio', type=float, default=0.3, help='Ratio of sharp corners')
    parser.add_argument('--output_dir', type=str, default='data/processed', help='Output directory')

    args = parser.parse_args()

    # Generate data
    features, targets = generate_trajectory_data(
        num_sequences=args.num_sequences,
        seq_len=args.seq_len,
        num_features=args.num_features,
        corner_ratio=args.corner_ratio
    )

    # Save data
    save_trajectory_data(features, targets, args.output_dir)

    print("\nTrajectory data generation completed successfully!")
    print(f"Total sequences: {args.num_sequences}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
