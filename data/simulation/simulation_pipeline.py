"""
Complete physics-based simulation pipeline

This script generates synthetic sensor data and quality metrics
based on realistic physics models of 3D printing.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import json
import argparse

from .thermal_model import (
    SimulationParameters,
    ThermalSimulationModel,
    VibrationSimulationModel,
    MotorCurrentSimulationModel,
    TrajectorySimulationModel,
)
from .quality_formation_model import (
    InterlayerAdhesionModel,
    InternalStressModel,
    PorosityFormationModel,
    DimensionalAccuracyModel,
    QualityScoreModel,
)


@dataclass
class PrintConfiguration:
    """
    Configuration for a single print simulation
    """
    sample_id: str
    nozzle_temp: float = 220.0          # °C
    bed_temp: float = 60.0               # °C
    print_speed: float = 50.0            # mm/s
    layer_height: float = 0.2            # mm
    print_duration: float = 30.0         # minutes
    pattern_type: str = 'rectilinear'    # 'rectilinear', 'grid', 'circle'

    # Advanced parameters
    fan_speed: float = 100.0             # %
    cooling_factor: float = 1.0          # Multiplier for cooling
    material_type: str = 'PLA'           # PLA, ABS, PETG, etc.


class CompletePrintSimulation:
    """
    Complete physics-based simulation of 3D printing

    Integrates:
    - Thermal model
    - Vibration model
    - Motor model
    - Trajectory model
    - Quality formation models
    """

    def __init__(self, params: SimulationParameters = None):
        """
        Initialize simulation

        Args:
            params: Physical parameters
        """
        self.params = params or SimulationParameters()

        # Initialize sub-models
        self.thermal_model = ThermalSimulationModel(self.params)
        self.vibration_model = VibrationSimulationModel(self.params)
        self.motor_model = MotorCurrentSimulationModel(self.params)
        self.trajectory_model = TrajectorySimulationModel(self.params)

        # Quality formation models
        self.adhesion_model = InterlayerAdhesionModel(self.params)
        self.stress_model = InternalStressModel(self.params)
        self.porosity_model = PorosityFormationModel(self.params)
        self.accuracy_model = DimensionalAccuracyModel(self.params)
        self.quality_score_model = QualityScoreModel()

    def simulate_print(self, config: PrintConfiguration) -> Dict:
        """
        Simulate a complete print and generate sensor data + quality metrics

        Args:
            config: Print configuration

        Returns:
            Dictionary with sensor data and quality metrics
        """
        print(f"\nSimulating print: {config.sample_id}")
        print(f"  Parameters: T={config.nozzle_temp}°C, v={config.print_speed}mm/s, h={config.layer_height}mm")

        # Step 1: Simulate thermal history
        thermal_data = self.thermal_model.simulate_printing_process(
            print_duration=config.print_duration,
            time_step=0.1,  # 10Hz sampling for thermal
            nozzle_temp=config.nozzle_temp,
            bed_temp=config.bed_temp,
            print_speed=config.print_speed,
            layer_height=config.layer_height,
        )

        # Step 2: Simulate trajectory (position and velocity)
        trajectory_data = self.trajectory_model.simulate_trajectory(
            print_duration=config.print_duration,
            time_step=0.1,
            print_speed=config.print_speed,
            pattern_type=config.pattern_type,
        )

        # Resample thermal data to match trajectory
        # (Assuming same sampling rate for simplicity)
        assert len(thermal_data['time']) == len(trajectory_data['time'])

        # Step 3: Simulate vibration
        vibration_data = self.vibration_model.simulate_vibration(
            thermal_stress=thermal_data['thermal_stress'],
            print_speed=config.print_speed,
            time_array=thermal_data['time'],
        )

        # Step 4: Simulate motor current
        current_data = self.motor_model.simulate_current(
            velocity=trajectory_data['print_speed'],
            load_mass=1.0,  # kg
        )

        # Step 5: Compute quality metrics based on process history
        quality_metrics = self._compute_quality_metrics(
            config, thermal_data, vibration_data
        )

        # Step 6: Compile sensor data
        sensor_data = {
            'sample_id': config.sample_id,
            'print_parameters': {
                'nozzle_temp': config.nozzle_temp,
                'bed_temp': config.bed_temp,
                'print_speed': config.print_speed,
                'layer_height': config.layer_height,
                'pattern_type': config.pattern_type,
                'material': config.material_type,
            },
            'time': thermal_data['time'],
            'nozzle_temp': thermal_data['nozzle_temp'],
            'bed_temp': thermal_data['bed_temp'],
            'part_temp': thermal_data['part_temp'],
            'cooling_rate': thermal_data['cooling_rate'],
            'thermal_stress': thermal_data['thermal_stress'],
            'vibration_x': vibration_data['vibration_x'],
            'vibration_y': vibration_data['vibration_y'],
            'vibration_z': vibration_data['vibration_z'],
            'vibration_magnitude': vibration_data['vibration_magnitude'],
            'motor_current_x': current_data['motor_current_x'],
            'motor_current_y': current_data['motor_current_y'],
            'motor_current_z': current_data['motor_current_z'],
            'position_x': trajectory_data['position_x'],
            'position_y': trajectory_data['position_y'],
            'position_z': trajectory_data['position_z'],
            'velocity_x': trajectory_data['velocity_x'],
            'velocity_y': trajectory_data['velocity_y'],
            'velocity_z': trajectory_data['velocity_z'],
            'print_speed': trajectory_data['print_speed'],
        }

        return {
            'sensor_data': sensor_data,
            'quality_metrics': quality_metrics,
        }

    def _compute_quality_metrics(self,
                                config: PrintConfiguration,
                                thermal_data: Dict,
                                vibration_data: Dict) -> Dict:
        """
        Compute quality metrics from process history

        Args:
            config: Print configuration
            thermal_data: Thermal simulation data
            vibration_data: Vibration data

        Returns:
            Quality metrics dictionary
        """
        # Interlayer adhesion strength
        adhesion_strength = self.adhesion_model.compute_adhesion_strength(
            temperature_history=thermal_data['part_temp'],
            cooling_rate=thermal_data['cooling_rate'],
            contact_pressure=0.5,  # MPa
        )

        # Internal stress
        internal_stress = self.stress_model.compute_internal_stress(
            temperature_history=thermal_data['part_temp'],
            cooling_rate=thermal_data['cooling_rate'],
            layer_count=int(config.print_duration * 10),  # Estimate
        )

        # Porosity
        porosity = self.porosity_model.compute_porosity(
            temperature_history=thermal_data['part_temp'],
            vibration_magnitude=vibration_data['vibration_magnitude'],
            print_speed=config.print_speed,
            layer_height=config.layer_height,
        )

        # Dimensional accuracy
        dimensional_error = self.accuracy_model.compute_dimensional_error(
            temperature_history=thermal_data['part_temp'],
            internal_stress=internal_stress,
            feature_size=20.0,  # mm
        )

        # Overall quality score
        quality_score = self.quality_score_model.compute_quality_score(
            adhesion_strength=adhesion_strength,
            internal_stress=internal_stress,
            porosity=porosity,
            dimensional_error=dimensional_error,
            feature_size=20.0,
        )

        return {
            'sample_id': config.sample_id,
            'adhesion_strength': float(adhesion_strength),
            'internal_stress': float(internal_stress),
            'porosity': float(porosity),
            'dimensional_accuracy': float(dimensional_error),
            'quality_score': float(quality_score),
        }


class SimulationDatasetGenerator:
    """
    Generate dataset with varied printing parameters

    Creates diverse dataset by varying:
    - Temperature
    - Speed
    - Layer height
    - Pattern type
    """

    def __init__(self, params: SimulationParameters = None):
        """Initialize dataset generator"""
        self.params = params or SimulationParameters()
        self.simulator = CompletePrintSimulation(self.params)

    def generate_dataset(self,
                         num_samples: int = 1000,
                         output_dir: str = 'data/raw',
                         parameter_ranges: Dict = None) -> List[Dict]:
        """
        Generate complete dataset

        Args:
            num_samples: Number of samples to generate
            output_dir: Output directory
            parameter_ranges: Ranges for parameters (optional)

        Returns:
            List of simulation results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Default parameter ranges
        if parameter_ranges is None:
            parameter_ranges = {
                'nozzle_temp': (190, 240),      # °C
                'bed_temp': (55, 65),             # °C
                'print_speed': (30, 80),          # mm/s
                'layer_height': (0.1, 0.3),       # mm
                'pattern_type': ['rectilinear', 'grid', 'circle'],
                'print_duration': (10, 60),       # minutes
            }

        print(f"\nGenerating {num_samples} simulation samples...")
        print(f"Parameter ranges:")
        for key, value in parameter_ranges.items():
            print(f"  {key}: {value}")

        results = []

        for i in range(num_samples):
            # Sample parameters
            config = self._sample_configuration(i, parameter_ranges)

            # Simulate print
            try:
                result = self.simulator.simulate_print(config)
                results.append(result)

                # Save sensor data
                self._save_sensor_data(result['sensor_data'], output_path)

                # Save quality data
                self._save_quality_data(result['quality_metrics'], output_path)

                if (i + 1) % 100 == 0:
                    print(f"  Generated {i + 1}/{num_samples} samples")

            except Exception as e:
                print(f"  Warning: Failed to generate sample {i}: {e}")

        print(f"\n✅ Generated {len(results)} samples")
        print(f"   Sensor data: {output_path}")
        print(f"   Quality data: {output_path}/quality_data/")

        return results

    def _sample_configuration(self, idx: int, ranges: Dict) -> PrintConfiguration:
        """Sample a random configuration"""
        config = PrintConfiguration(
            sample_id=f'sim_print_{idx:04d}',
            nozzle_temp=np.random.uniform(*ranges['nozzle_temp']),
            bed_temp=np.random.uniform(*ranges['bed_temp']),
            print_speed=np.random.uniform(*ranges['print_speed']),
            layer_height=np.random.uniform(*ranges['layer_height']),
            pattern_type=np.random.choice(ranges['pattern_type']),
            print_duration=np.random.uniform(*ranges['print_duration']),
        )

        return config

    def _save_sensor_data(self, sensor_data: Dict, output_dir: Path):
        """Save sensor data to file"""
        sample_id = sensor_data['sample_id']

        # Convert to numpy format for efficiency
        data_arrays = {}
        for key, value in sensor_data.items():
            if key == 'sample_id' or key == 'print_parameters':
                continue
            if isinstance(value, np.ndarray):
                data_arrays[key] = value

        # Save as compressed numpy
        npz_file = output_dir / f'{sample_id}_sensor_data.npz'
        np.savez_compressed(npz_file, **data_arrays)

        # Also save metadata as JSON
        metadata = {
            'sample_id': sensor_data['sample_id'],
            'print_parameters': sensor_data['print_parameters'],
            'num_samples': len(sensor_data['time']),
            'sampling_rate': 10,  # Hz
            'duration_minutes': sensor_data['time'][-1] / 60,
        }

        json_file = output_dir / f'{sample_id}_metadata.json'
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _save_quality_data(self, quality_metrics: Dict, output_dir: Path):
        """Save quality metrics to file"""
        sample_id = quality_metrics['sample_id']

        quality_dir = output_dir / 'quality_data'
        quality_dir.mkdir(exist_ok=True)

        quality_data = {
            'sample_id': sample_id,
            'test_date': '2024-01-27',  # Simulation date
            'quality_metrics': {
                'adhesion_strength': quality_metrics['adhesion_strength'],
                'internal_stress': quality_metrics['internal_stress'],
                'porosity': quality_metrics['porosity'],
                'dimensional_accuracy': quality_metrics['dimensional_accuracy'],
                'quality_score': quality_metrics['quality_score'],
            },
            'test_info': {
                'test_method': 'physics_simulation',
                'simulation': True,
            },
        }

        json_file = quality_dir / f'{sample_id}_quality_data.json'
        with open(json_file, 'w') as f:
            json.dump(quality_data, f, indent=2)


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Generate physics-based simulation data')

    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                       help='Output directory')
    parser.add_argument('--min_temp', type=float, default=190,
                       help='Minimum nozzle temperature (°C)')
    parser.add_argument('--max_temp', type=float, default=240,
                       help='Maximum nozzle temperature (°C)')
    parser.add_argument('--min_speed', type=float, default=30,
                       help='Minimum print speed (mm/s)')
    parser.add_argument('--max_speed', type=float, default=80,
                       help='Maximum print speed (mm/s)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Create parameter ranges
    parameter_ranges = {
        'nozzle_temp': (args.min_temp, args.max_temp),
        'bed_temp': (55, 65),
        'print_speed': (args.min_speed, args.max_speed),
        'layer_height': (0.1, 0.3),
        'pattern_type': ['rectilinear', 'grid', 'circle'],
        'print_duration': (10, 60),
    }

    # Generate dataset
    generator = SimulationDatasetGenerator()

    results = generator.generate_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        parameter_ranges=parameter_ranges
    )

    print(f"\n✅ Simulation complete!")
    print(f"   Generated {len(results)} samples")
    print(f"   Next step: python data/scripts/pair_and_preprocess.py")


if __name__ == '__main__':
    main()
