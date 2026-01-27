"""
Generate synthetic physics-based 3D printer data

This script generates synthetic sensor data for training the quality prediction model.
In production, you would replace this with your actual data collection system.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple
import argparse


def generate_physics_data(
    num_samples: int = 10000,
    seq_len: int = 200,
    sampling_rate: int = 1000,
    noise_level: float = 0.05
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate physics-based synthetic 3D printer data

    Args:
        num_samples: Number of samples to generate
        seq_len: Sequence length (time steps)
        sampling_rate: Sampling rate in Hz
        noise_level: Noise level to add

    Returns:
        features: Sensor data array [num_samples, seq_len, num_features]
        targets: Dictionary of target arrays
    """
    print(f"Generating {num_samples} samples with sequence length {seq_len}...")

    # Time vector
    t = np.linspace(0, seq_len / sampling_rate, seq_len)

    # Initialize feature arrays
    # Features: [temp_nozzle, temp_bed, vib_x, vib_y, vib_z,
    #            current_x, current_y, current_z, pressure,
    #            pos_x, pos_y, pos_z]
    num_features = 12
    features = np.zeros((num_samples, seq_len, num_features))

    # Initialize target arrays
    targets = {
        'rul': np.zeros((num_samples, 1)),
        'temperature': np.zeros((num_samples, 1)),
        'vibration_x': np.zeros((num_samples, 1)),
        'vibration_y': np.zeros((num_samples, 1)),
        'quality_score': np.zeros((num_samples, 1)),
        'fault_label': np.zeros((num_samples,), dtype=int),
    }

    # Generate samples
    for i in range(num_samples):
        # Random fault type (0: normal, 1: nozzle clog, 2: mechanical loose, 3: motor fault)
        fault_type = np.random.choice([0, 1, 2, 3], p=[0.7, 0.1, 0.1, 0.1])
        targets['fault_label'][i] = fault_type

        # Temperature dynamics (heat equation)
        base_temp = 220.0  # Base nozzle temperature
        if fault_type == 1:  # Nozzle clog
            temp_rise = 15.0 * (1 - np.exp(-t / 50))
        else:
            temp_rise = 2.0 * np.sin(2 * np.pi * 0.1 * t)

        temp_nozzle = base_temp + temp_rise + noise_level * np.random.randn(seq_len)
        temp_bed = 60.0 + 0.5 * temp_rise + noise_level * np.random.randn(seq_len)

        # Vibration dynamics (mass-spring-damper)
        if fault_type == 2:  # Mechanical loose
            vib_amplitude = 0.15
        elif fault_type == 3:  # Motor fault
            vib_amplitude = 0.10
        else:
            vib_amplitude = 0.02

        vibration_x = vib_amplitude * np.sin(2 * np.pi * 5 * t) + noise_level * 0.5 * np.random.randn(seq_len)
        vibration_y = vib_amplitude * np.cos(2 * np.pi * 7 * t) + noise_level * 0.5 * np.random.randn(seq_len)
        vibration_z = 0.5 * vib_amplitude * np.sin(2 * np.pi * 3 * t) + noise_level * 0.5 * np.random.randn(seq_len)

        # Motor current (correlated with vibration and acceleration)
        current_x = 1.0 + 0.3 * np.abs(vibration_x) + 0.1 * np.random.randn(seq_len)
        current_y = 1.0 + 0.3 * np.abs(vibration_y) + 0.1 * np.random.randn(seq_len)
        current_z = 0.5 + 0.2 * np.abs(vibration_z) + 0.1 * np.random.randn(seq_len)

        # Pressure (extrusion pressure)
        if fault_type == 1:  # Nozzle clog
            pressure_base = 5.0 + 2.0 * (1 - np.exp(-t / 30))
        else:
            pressure_base = 5.0 + 0.5 * np.sin(2 * np.pi * 0.2 * t)

        pressure = pressure_base + noise_level * np.random.randn(seq_len)

        # Position (printing path)
        pos_x = 100.0 + 50.0 * np.sin(2 * np.pi * 0.05 * t)
        pos_y = 100.0 + 50.0 * np.cos(2 * np.pi * 0.05 * t)
        pos_z = 0.2 * t / t[-1]  # Layer height

        # Combine features
        features[i, :, 0] = temp_nozzle
        features[i, :, 1] = temp_bed
        features[i, :, 2] = vibration_x
        features[i, :, 3] = vibration_y
        features[i, :, 4] = vibration_z
        features[i, :, 5] = current_x
        features[i, :, 6] = current_y
        features[i, :, 7] = current_z
        features[i, :, 8] = pressure
        features[i, :, 9] = pos_x
        features[i, :, 10] = pos_y
        features[i, :, 11] = pos_z

        # Generate targets
        if fault_type == 0:  # Normal
            rul = 1000.0  # No impending failure
            quality_score = 0.8 + 0.2 * np.random.rand()
        else:
            # RUL decreases as fault progresses
            fault_progress = np.random.rand()
            rul = (1 - fault_progress) * 500
            quality_score = 0.3 + 0.5 * (1 - fault_progress) + 0.2 * np.random.rand()

        targets['rul'][i] = rul
        targets['temperature'][i] = temp_nozzle[-1]  # Last temperature
        targets['vibration_x'][i] = vibration_x[-1]  # Last vibration
        targets['vibration_y'][i] = vibration_y[-1]
        targets['quality_score'][i] = np.clip(quality_score, 0, 1)

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")

    print(f"Data generation completed!")
    return features, targets


def save_data(features: np.ndarray, targets: Dict[str, np.ndarray], output_dir: str):
    """
    Save generated data to files

    Args:
        features: Feature array
        targets: Target dictionary
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
        split_targets = {k: v[start:end] for k, v in targets.items()}

        data = {
            'features': split_features,
            'targets': split_targets,
        }

        output_file = output_path / f"{split_name}_data.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved {split_name} data to {output_file}")
        print(f"  Samples: {end - start}")
        print(f"  Features shape: {split_features.shape}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic physics data')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--seq_len', type=int, default=200, help='Sequence length')
    parser.add_argument('--sampling_rate', type=int, default=1000, help='Sampling rate (Hz)')
    parser.add_argument('--noise_level', type=float, default=0.05, help='Noise level')
    parser.add_argument('--output_dir', type=str, default='data/processed', help='Output directory')

    args = parser.parse_args()

    # Generate data
    features, targets = generate_physics_data(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        sampling_rate=args.sampling_rate,
        noise_level=args.noise_level
    )

    # Save data
    save_data(features, targets, args.output_dir)

    print("\nData generation completed successfully!")
    print(f"Total samples: {args.num_samples}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Number of features: {features.shape[2]}")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
