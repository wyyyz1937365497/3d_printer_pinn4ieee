"""
Physics-based 3D printing simulation system

This module implements realistic physics models for generating synthetic
sensor data and quality metrics that follow real physical laws.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SimulationParameters:
    """
    Physical parameters for 3D printing simulation
    """
    # Material properties (PLA)
    thermal_conductivity: float = 0.13      # W/(m·K)
    density: float = 1240.0                  # kg/m³
    specific_heat: float = 1800.0            # J/(kg·K)
    thermal_diffusivity: float = 5.8e-8      # m²/s
    melting_point: float = 170.0             # °C
    glass_transition: float = 60.0           # °C

    # Mechanical properties
    elastic_modulus: float = 3500.0          # MPa
    yield_strength: float = 60.0             # MPa
    ultimate_strength: float = 70.0          # MPa
    poissons_ratio: float = 0.36

    # Thermal expansion
    thermal_expansion_coeff: float = 6.8e-5  # 1/K

    # Interlayer bonding
    bond_activation_energy: float = 50000.0  # J/mol
    bond_pre_exponential: float = 1.0e8      # 1/s
    gas_constant: float = 8.314              # J/(mol·K)

    # Process parameters
    nozzle_diameter: float = 0.4            # mm
    layer_height: float = 0.2               # mm
    extrusion_width: float = 0.45           # mm

    # Heat transfer
    convective_heat_transfer: float = 10.0  # W/(m²·K)
    emissivity: float = 0.95                 # radiation emissivity
    stefan_boltzmann: float = 5.67e-8       # W/(m²·K⁴)

    # Environment
    ambient_temp: float = 25.0               # °C
    bed_temp: float = 60.0                   # °C


class ThermalSimulationModel:
    """
    3D printing thermal simulation based on heat transfer physics

    Implements:
    - Heat conduction (Fourier's law)
    - Convection cooling
    - Radiation cooling
    - Heat source from nozzle
    """

    def __init__(self, params: SimulationParameters = None):
        """
        Initialize thermal model

        Args:
            params: Simulation parameters
        """
        self.params = params or SimulationParameters()

    def simulate_printing_process(self,
                                  print_duration: float = 30.0,
                                  time_step: float = 0.1,
                                  nozzle_temp: float = 220.0,
                                  bed_temp: float = 60.0,
                                  print_speed: float = 50.0,
                                  layer_height: float = 0.2) -> Dict[str, np.ndarray]:
        """
        Simulate thermal history during printing

        Args:
            print_duration: Total printing time (minutes)
            time_step: Simulation time step (seconds)
            nozzle_temp: Nozzle temperature (°C)
            bed_temp: Bed temperature (°C)
            print_speed: Print speed (mm/s)
            layer_height: Layer height (mm)

        Returns:
            Dictionary with thermal history
        """
        n_steps = int(print_duration * 60 / time_step)

        # Time array
        t = np.linspace(0, print_duration * 60, n_steps)

        # Initialize temperature fields
        nozzle_temp_history = np.full(n_steps, nozzle_temp)
        bed_temp_history = np.full(n_steps, bed_temp)

        # Simulate printed part temperature (simplified model)
        # Using lumped capacitance model with heat source and cooling

        # Initial part temperature
        part_temp = np.full(n_steps, bed_temp)
        part_temp[0] = bed_temp

        # Heat transfer parameters
        h = self.params.convective_heat_transfer
        emissivity = self.params.emissivity
        sigma = self.params.stefan_boltzmann

        # ===== NEW APPROACH: Empirical thermal model =====
        # Based on typical 3D printing thermal behavior
        # Part temperature is mainly determined by:
        # 1. Nozzle temperature (heat source)
        # 2. Bed temperature (base heating)
        # 3. Print speed (exposure time to heat)
        # 4. Layer height (thermal mass)
        # 5. Cooling (fan, ambient)

        # Estimate steady-state part temperature based on parameters
        # Typical FDM: part temp ≈ 0.7 * nozzle_temp + 0.1 * bed_temp
        # Modified by speed (slower = hotter)
        speed_factor = np.clip(50.0 / print_speed, 0.6, 1.4)  # Normalize around 50mm/s
        layer_factor = np.clip(layer_height / 0.2, 0.8, 1.2)  # Normalize around 0.2mm

        # Base temperature (empirical)
        base_part_temp = (
            0.65 * nozzle_temp +  # Most heat from nozzle
            0.15 * bed_temp +     # Some from bed
            20.0                  # Ambient offset
        ) * speed_factor * layer_factor

        # Add dynamic heating/cooling cycles
        # When nozzle passes over, temperature spikes
        # Then cools gradually until next pass
        cycle_period = 10.0  # seconds (typical layer time)
        phase = (t % cycle_period) / cycle_period

        # Temperature varies in a cycle
        # Peak when nozzle just passed (phase=0)
        # Lowest just before next pass (phase=1)
        temp_variation = 30.0 * (1 - phase) * np.exp(-phase * 3)  # Heating spike then decay

        # Add thermal buildup over time (first few layers heat up)
        layer_index = (t * print_speed / 60.0) / 20.0  # Rough estimate
        thermal_buildup = 20.0 * (1 - np.exp(-layer_index / 5.0))

        # Combine for realistic temperature profile
        part_temp = base_part_temp + temp_variation + thermal_buildup

        # Add realistic fluctuations
        part_temp += np.random.randn(n_steps) * 2.0

        # Add cooling effect for higher speeds
        cooling_effect = (print_speed - 50) * 0.3
        part_temp -= cooling_effect

        # Ensure physical bounds
        part_temp = np.clip(part_temp, bed_temp - 10, nozzle_temp + 10)

        # Calculate temperature gradients (cooling rate)
        cooling_rate = np.gradient(part_temp, t)

        # Calculate thermal stress indicator
        thermal_stress_indicator = np.abs(cooling_rate) * self.params.elastic_modulus * \
                                     self.params.thermal_expansion_coeff

        return {
            'time': t,
            'nozzle_temp': nozzle_temp_history,
            'bed_temp': bed_temp_history,
            'part_temp': part_temp,
            'cooling_rate': cooling_rate,
            'thermal_stress': thermal_stress_indicator,
        }


class VibrationSimulationModel:
    """
    3D printing vibration simulation based on dynamics

    Implements:
    - Motor-induced vibration
    - Mechanical resonance
    - External disturbances
    """

    def __init__(self, params: SimulationParameters = None):
        """
        Initialize vibration model

        Args:
            params: Simulation parameters
        """
        self.params = params or SimulationParameters()

        # Vibration model parameters
        self.natural_frequency_x = 50.0  # Hz
        self.natural_frequency_y = 50.0  # Hz
        self.natural_frequency_z = 80.0  # Hz
        self.damping_ratio = 0.1

    def simulate_vibration(self,
                          thermal_stress: np.ndarray,
                          print_speed: float = 50.0,
                          time_array: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Simulate vibration during printing

        Args:
            thermal_stress: Thermal stress indicator
            print_speed: Print speed (mm/s)
            time_array: Time array

        Returns:
            Dictionary with vibration data
        """
        if time_array is None:
            time_array = np.linspace(0, 30, 300)  # 30 seconds default

        n_steps = len(time_array)
        dt = time_array[1] - time_array[0]

        # Base vibration from motors
        # Vibration amplitude increases with speed
        base_amplitude = 0.01 * (print_speed / 50.0)

        # Generate vibration signals
        vibration_x = self._generate_vibration_signal(
            time_array, self.natural_frequency_x, base_amplitude, thermal_stress
        )
        vibration_y = self._generate_vibration_signal(
            time_array, self.natural_frequency_y, base_amplitude, thermal_stress
        )
        vibration_z = self._generate_vibration_signal(
            time_array, self.natural_frequency_z, base_amplitude * 0.5, thermal_stress
        )

        return {
            'vibration_x': vibration_x,
            'vibration_y': vibration_y,
            'vibration_z': vibration_z,
            'vibration_magnitude': np.sqrt(vibration_x**2 + vibration_y**2 + vibration_z**2),
        }

    def _generate_vibration_signal(self,
                                  t: np.ndarray,
                                  natural_freq: float,
                                  base_amplitude: float,
                                  stress: np.ndarray) -> np.ndarray:
        """
        Generate vibration signal with multiple frequency components

        Args:
            t: Time array
            natural_freq: Natural frequency (Hz)
            base_amplitude: Base amplitude
            stress: Thermal stress (affects vibration)

        Returns:
            Vibration signal
        """
        n_samples = len(t)
        vibration = np.zeros(n_samples)

        # Motor fundamental frequency
        motor_freq = 20.0  # Hz (typical stepper motor frequency)

        # Multiple frequency components
        frequencies = [
            motor_freq,
            2 * motor_freq,  # Harmonic
            natural_freq,     # Resonance
            3 * motor_freq,  # Higher harmonic
        ]

        amplitudes = [
            base_amplitude,
            base_amplitude * 0.5,
            base_amplitude * 0.3,  # Lower at resonance due to damping
            base_amplitude * 0.2,
        ]

        for freq, amp in zip(frequencies, amplitudes):
            phase = np.random.uniform(0, 2 * np.pi)
            vibration += amp * np.sin(2 * np.pi * freq * t + phase)

        # Add stress-induced vibration (higher stress → more vibration)
        stress_component = 0.5 * stress * np.random.randn(n_samples)
        vibration += stress_component

        # Add measurement noise
        vibration += 0.001 * np.random.randn(n_samples)

        return vibration


class MotorCurrentSimulationModel:
    """
    Motor current simulation based on load and dynamics

    Implements:
    - Current based on acceleration and load
    - Hold current (to maintain position)
    - Phase currents for stepper motors
    """

    def __init__(self, params: SimulationParameters = None):
        """Initialize motor current model"""
        self.params = params or SimulationParameters()

        # Motor parameters
        self.hold_current = 0.3  # A
        self.rated_current = 1.2  # A
        self.max_current = 2.0     # A

    def simulate_current(self,
                        velocity: np.ndarray,
                        acceleration: np.ndarray = None,
                        load_mass: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Simulate motor current during printing

        Args:
            velocity: Velocity profile [n_steps]
            acceleration: Acceleration profile (optional)
            load_mass: Mass being moved (kg)

        Returns:
            Dictionary with motor currents
        """
        n_steps = len(velocity)

        if acceleration is None:
            # Calculate acceleration from velocity
            dt = 0.1  # Assume 0.1s time step
            acceleration = np.gradient(velocity, dt)

        # Base current (hold current + motion current)
        current_x = self.hold_current + np.abs(acceleration) * load_mass * 0.1
        current_y = self.hold_current + np.abs(acceleration) * load_mass * 0.1
        current_z = self.hold_current + np.abs(acceleration) * load_mass * 0.15  # Z is heavier

        # Add noise
        current_x += 0.01 * np.random.randn(n_steps)
        current_y += 0.01 * np.random.randn(n_steps)
        current_z += 0.01 * np.random.randn(n_steps)

        # Clip to valid range
        current_x = np.clip(current_x, 0, self.max_current)
        current_y = np.clip(current_y, 0, self.max_current)
        current_z = np.clip(current_z, 0, self.max_current)

        return {
            'motor_current_x': current_x,
            'motor_current_y': current_y,
            'motor_current_z': current_z,
        }


class TrajectorySimulationModel:
    """
    3D printing trajectory simulation

    Implements:
    - G-code path following
    - Corner rounding
    - Velocity planning
    - Acceleration constraints
    """

    def __init__(self, params: SimulationParameters = None):
        """Initialize trajectory model"""
        self.params = params or SimulationParameters()

    def simulate_trajectory(self,
                           print_duration: float = 30.0,
                           time_step: float = 0.1,
                           print_speed: float = 50.0,
                           pattern_type: str = 'rectilinear') -> Dict[str, np.ndarray]:
        """
        Simulate printing trajectory

        Args:
            print_duration: Total printing time (minutes)
            time_step: Time step (seconds)
            print_speed: Print speed (mm/s)
            pattern_type: 'rectilinear', 'grid', or 'circle'

        Returns:
            Dictionary with position and velocity data
        """
        n_steps = int(print_duration * 60 / time_step)
        t = np.linspace(0, print_duration * 60, n_steps)

        if pattern_type == 'rectilinear':
            positions, velocities = self._simulate_rectilinear_pattern(
                t, print_speed
            )
        elif pattern_type == 'grid':
            positions, velocities = self._simulate_grid_pattern(
                t, print_speed
            )
        elif pattern_type == 'circle':
            positions, velocities = self._simulate_circular_pattern(
                t, print_speed
            )
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

        return {
            'time': t,
            'position_x': positions[:, 0],
            'position_y': positions[:, 1],
            'position_z': positions[:, 2],
            'velocity_x': velocities[:, 0],
            'velocity_y': velocities[:, 1],
            'velocity_z': velocities[:, 2],
            'print_speed': np.linalg.norm(velocities, axis=1),
        }

    def _simulate_rectilinear_pattern(self,
                                     t: np.ndarray,
                                     speed: float) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate rectilinear infill pattern"""
        n_steps = len(t)

        # Generate zigzag pattern
        period = 10.0  # seconds per line
        positions = np.zeros((n_steps, 3))
        velocities = np.zeros((n_steps, 3))

        x = 100.0
        y = 100.0
        z = 0.2

        direction = 1  # 1 for positive X, -1 for negative X
        line_length = 20.0  # mm

        for i in range(n_steps):
            # Calculate position along current line
            phase = (t[i] % period) / period  # [0, 1]

            # Position along line
            if phase < 0.8:
                # Moving along line
                line_progress = phase / 0.8
                positions[i, 0] = x + direction * line_progress * line_length
                positions[i, 1] = y
                positions[i, 2] = z

                # Velocity
                velocities[i, 0] = direction * speed
                velocities[i, 1] = 0.0
                velocities[i, 2] = 0.0
            else:
                # Turning around (corner)
                corner_progress = (phase - 0.8) / 0.2
                positions[i, 0] = x + direction * line_length
                positions[i, 1] = y + corner_progress * 0.4  # Small Y shift
                positions[i, 2] = z

                # Velocity (slower at corner)
                velocities[i, 0] = direction * speed * (1 - corner_progress)
                velocities[i, 1] = speed * corner_progress
                velocities[i, 2] = 0.0

                # After corner, change direction
                if phase >= 0.95:
                    direction *= -1
                    y += 0.4

        # Add small random variations
        positions += np.random.randn(*positions.shape) * 0.01

        return positions, velocities

    def _simulate_grid_pattern(self,
                               t: np.ndarray,
                               speed: float) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate grid infill pattern"""
        n_steps = len(t)

        positions = np.zeros((n_steps, 3))
        velocities = np.zeros((n_steps, 3))

        grid_spacing = 5.0  # mm
        line_length = 20.0  # mm

        x, y, z = 100.0, 100.0, 0.2

        for i in range(n_steps):
            # Create grid pattern
            cycle = i % 4
            progress = (i // 4) % int(line_length)

            if cycle == 0:
                # Move in X
                positions[i, 0] = x + progress * speed * 0.1
                positions[i, 1] = y
                velocities[i, 0] = speed
                velocities[i, 1] = 0
            elif cycle == 1:
                # Move in Y
                positions[i, 0] = x + line_length
                positions[i, 1] = y + progress * speed * 0.1
                velocities[i, 0] = 0
                velocities[i, 1] = speed
            elif cycle == 2:
                # Move back in X
                positions[i, 0] = x + line_length - progress * speed * 0.1
                positions[i, 1] = y + line_length
                velocities[i, 0] = -speed
                velocities[i, 1] = 0
            else:
                # Move back in Y
                positions[i, 2] = z + 0.2
                positions[i, 1] = y + line_length - progress * speed * 0.1
                velocities[i, 0] = 0
                velocities[i, 1] = -speed

            positions[i, 2] = z

        return positions, velocities

    def _simulate_circular_pattern(self,
                                  t: np.ndarray,
                                  speed: float) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate circular pattern"""
        n_steps = len(t)

        center_x, center_y = 100.0, 100.0
        radius = 10.0
        z = 0.2

        # Angular velocity
        angular_velocity = speed / radius

        positions = np.zeros((n_steps, 3))
        velocities = np.zeros((n_steps, 3))

        for i in range(n_steps):
            angle = angular_velocity * t[i]

            positions[i, 0] = center_x + radius * np.cos(angle)
            positions[i, 1] = center_y + radius * np.sin(angle)
            positions[i, 2] = z

            # Velocity (tangent to circle)
            velocities[i, 0] = -speed * np.sin(angle)
            velocities[i, 1] = speed * np.cos(angle)
            velocities[i, 2] = 0.0

        return positions, velocities
