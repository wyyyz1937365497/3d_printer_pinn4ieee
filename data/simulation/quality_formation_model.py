"""
Quality parameter formation models

These models simulate how quality parameters (adhesion, stress, porosity, etc.)
form based on the printing process conditions and physics.
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

from .thermal_model import SimulationParameters


class InterlayerAdhesionModel:
    """
    Model for interlayer adhesion strength formation

    Based on:
    - Molecular diffusion theory
    - Thermal activation
    - Contact time and pressure
    """

    def __init__(self, params: SimulationParameters = None):
        """Initialize adhesion model"""
        self.params = params or SimulationParameters()

    def compute_adhesion_strength(self,
                                 temperature_history: np.ndarray,
                                 cooling_rate: np.ndarray,
                                 contact_pressure: float = 0.5) -> float:
        """
        Compute final interlayer adhesion strength

        Args:
            temperature_history: Temperature during bonding (°C)
            cooling_rate: Cooling rate (°C/s)
            contact_pressure: Contact pressure (MPa)

        Returns:
            Adhesion strength (MPa)
        """
        # Base material strength
        base_strength = 30.0  # MPa (for PLA)

        # Thermal conditions
        avg_temp = np.mean(temperature_history)
        max_temp = np.max(temperature_history)
        min_temp = np.min(temperature_history)
        temp_range = max_temp - min_temp

        avg_cooling_rate = np.mean(np.abs(cooling_rate))

        # Time above melting point (critical for bonding)
        time_above_melt = np.sum(temperature_history > self.params.melting_point)
        total_time = len(temperature_history) * 0.1  # Assuming 0.1s timestep
        melt_fraction = time_above_melt / total_time

        # Molecular diffusion factor (Arrhenius equation)
        # D = D0 * exp(-Ea / RT)
        T_kelvin = avg_temp + 273.15
        activation_energy = self.params.bond_activation_energy  # J/mol
        gas_constant = self.params.gas_constant

        diffusion_factor = np.exp(-activation_energy / (gas_constant * T_kelvin))

        # Temperature effect on adhesion
        # Optimal temperature window: 200-230°C for PLA
        temp_optimal = 220.0
        temp_deviation = np.abs(avg_temp - temp_optimal)
        temp_factor = np.exp(-(temp_deviation / 20.0) ** 2)

        # Cooling rate effect (rapid cooling reduces bonding)
        # Optimal cooling: slow and controlled
        cooling_factor = 1.0 / (1.0 + avg_cooling_rate / 5.0)

        # Pressure effect (higher pressure improves contact)
        pressure_factor = 1.0 + contact_pressure * 0.5

        # Time effect (longer bonding time improves strength)
        time_factor = 1.0 - np.exp(-melt_fraction * 5)

        # Combine all factors
        adhesion_strength = (
            base_strength *
            diffusion_factor * 1000 *  # Scale to reasonable value
            temp_factor *
            cooling_factor *
            pressure_factor *
            time_factor
        )

        # Add some variability (material inhomogeneity)
        adhesion_strength *= np.random.normal(1.0, 0.05)

        # Physical constraint: must be positive
        adhesion_strength = max(5.0, min(adhesion_strength, 40.0))

        return adhesion_strength


class InternalStressModel:
    """
    Model for internal/residual stress formation

    Based on:
    - Thermal stress from temperature gradients
    - Shrinkage stress
    - Layer interaction constraints
    """

    def __init__(self, params: SimulationParameters = None):
        """Initialize stress model"""
        self.params = params or SimulationParameters()

    def compute_internal_stress(self,
                              temperature_history: np.ndarray,
                              cooling_rate: np.ndarray,
                              layer_count: int = 100) -> float:
        """
        Compute final internal stress

        Args:
            temperature_history: Temperature history (°C)
            cooling_rate: Cooling rate (°C/s)
            layer_count: Number of layers

        Returns:
            Internal stress (MPa)
        """
        # Temperature range
        max_temp = np.max(temperature_history)
        min_temp = np.min(temperature_history)
        temp_range = max_temp - min_temp

        # Average cooling rate
        avg_cooling_rate = np.mean(np.abs(cooling_rate))

        # Thermal stress (from temperature gradient)
        # σ = E * α * ΔT
        E = self.params.elastic_modulus  # MPa
        alpha = self.params.thermal_expansion_coeff  # 1/K

        thermal_stress = E * alpha * temp_range

        # Cooling rate effect (rapid cooling → high stress)
        cooling_stress = avg_cooling_rate * E * 0.5

        # Layer constraint effect (stress accumulation)
        # More layers → more constraint → more stress
        layer_factor = 1.0 + 0.01 * layer_count

        # Stress relaxation (time dependent)
        # Not implemented for simplicity

        # Total stress
        total_stress = (
            thermal_stress +
            cooling_stress
        ) * layer_factor

        # Add variability
        total_stress *= np.random.normal(1.0, 0.08)

        # Physical constraint
        total_stress = max(1.0, min(total_stress, 30.0))

        return total_stress


class PorosityFormationModel:
    """
    Model for porosity formation during printing

    Based on:
    - Insufficient fusion
    - Gas entrapment
    - Shrinkage voids
    """

    def __init__(self, params: SimulationParameters = None):
        """Initialize porosity model"""
        self.params = params or SimulationParameters()

    def compute_porosity(self,
                        temperature_history: np.ndarray,
                        vibration_magnitude: np.ndarray,
                        print_speed: float,
                        layer_height: float = 0.2) -> float:
        """
        Compute final porosity percentage

        Args:
            temperature_history: Temperature history (°C)
            vibration_magnitude: Vibration magnitude
            print_speed: Print speed (mm/s)
            layer_height: Layer height (mm)

        Returns:
            Porosity percentage (%)
        """
        # Temperature factor (low temp → poor fusion → high porosity)
        avg_temp = np.mean(temperature_history)
        min_temp = np.min(temperature_history)

        temp_deficit = max(0, self.params.melting_point - min_temp)
        temp_factor = temp_deficit / self.params.melting_point

        # Speed factor (high speed → less fusion time → higher porosity)
        # Reference speed: 50 mm/s
        speed_factor = np.clip((print_speed - 30) / 70, 0, 1)

        # Layer height factor (tall layers → poor bonding → higher porosity)
        layer_height_factor = np.clip((layer_height - 0.1) / 0.3, 0, 1)

        # Vibration factor (high vibration → voids and gaps)
        avg_vibration = np.mean(vibration_magnitude)
        vibration_factor = np.clip(avg_vibration / 0.5, 0, 1)

        # Base porosity contributions
        porosity_from_temp = 8.0 * temp_factor
        porosity_from_speed = 4.0 * speed_factor
        porosity_from_layer = 3.0 * layer_height_factor
        porosity_from_vibration = 2.0 * vibration_factor

        total_porosity = (
            porosity_from_temp +
            porosity_from_speed +
            porosity_from_layer +
            porosity_from_vibration
        )

        # Add variability
        total_porosity *= np.random.normal(1.0, 0.1)

        # Physical constraints
        total_porosity = max(0.0, min(total_porosity, 20.0))

        return total_porosity


class DimensionalAccuracyModel:
    """
    Model for dimensional accuracy

    Based on:
    - Thermal expansion
    - Shrinkage
    - Stress-induced warping
    """

    def __init__(self, params: SimulationParameters = None):
        """Initialize dimensional accuracy model"""
        self.params = params or SimulationParameters()

    def compute_dimensional_error(self,
                                  temperature_history: np.ndarray,
                                  internal_stress: float,
                                  feature_size: float = 20.0) -> float:
        """
        Compute dimensional error

        Args:
            temperature_history: Temperature history (°C)
            internal_stress: Internal stress (MPa)
            feature_size: Nominal feature size (mm)

        Returns:
            Dimensional error (mm)
        """
        # Thermal expansion
        avg_temp = np.mean(temperature_history)
        room_temp = self.params.ambient_temp

        delta_T = avg_temp - room_temp
        thermal_strain = self.params.thermal_expansion_coeff * delta_T

        # Shrinkage (material dependent)
        # PLA typically shrinks 0.2-0.3%
        shrinkage_strain = 0.0025

        # Stress-induced warping
        warping_strain = 1e-4 * internal_stress

        # Total strain
        total_strain = thermal_strain + shrinkage_strain + warping_strain

        # Convert to absolute error
        dimensional_error = abs(total_strain) * feature_size

        # Add variability
        dimensional_error *= np.random.normal(1.0, 0.05)

        # Physical constraint
        dimensional_error = max(0.0, min(dimensional_error, 0.5))

        return dimensional_error


class QualityScoreModel:
    """
    Overall quality score model

    Combines individual quality metrics into an overall score [0, 1]
    """

    def __init__(self):
        """Initialize quality score model"""
        pass

    def compute_quality_score(self,
                            adhesion_strength: float,
                            internal_stress: float,
                            porosity: float,
                            dimensional_error: float,
                            feature_size: float = 20.0) -> float:
        """
        Compute overall quality score [0, 1]

        Args:
            adhesion_strength: Interlayer adhesion (MPa)
            internal_stress: Internal stress (MPa)
            porosity: Porosity (%)
            dimensional_error: Dimensional error (mm)
            feature_size: Nominal feature size (mm)

        Returns:
            Quality score [0, 1]
        """
        # Individual component scores

        # Adhesion score (higher is better, target > 25 MPa)
        adhesion_score = 1.0 / (1.0 + np.exp(-(adhesion_strength - 20) / 5))

        # Stress score (lower is better, target < 15 MPa)
        stress_score = 1.0 / (1.0 + np.exp(-(15 - internal_stress) / 5))

        # Porosity score (lower is better, target < 5%)
        porosity_score = 1.0 / (1.0 + np.exp(-(5 - porosity) / 2))

        # Accuracy score (lower error is better, target < 0.1mm)
        relative_error = dimensional_error / feature_size
        accuracy_score = 1.0 / (1.0 + np.exp(-(0.005 - relative_error) / 0.002))

        # Weighted combination
        quality_score = (
            0.35 * adhesion_score +  # 35% weight
            0.25 * stress_score +    # 25% weight
            0.20 * porosity_score +   # 20% weight
            0.20 * accuracy_score    # 20% weight
        )

        # Add small variability
        quality_score *= np.random.normal(1.0, 0.02)
        quality_score = np.clip(quality_score, 0.0, 1.0)

        return quality_score
