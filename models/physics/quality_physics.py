"""
Physics-based models for connecting observable sensor data to implicit quality parameters

This module implements physical models that relate:
- Observable: Temperature, vibration, motor current, pressure, etc.
- Unobservable: Interlayer adhesion, internal stress, porosity, etc.

Key Physical Principles:
1. Thermal-adhesion relationship: Temperature affects interlayer bonding
2. Stress accumulation: Rapid thermal changes create residual stress
3. Porosity formation: Insufficient fusion creates voids
4. Dimensional accuracy: Thermal expansion/contraction affects dimensions
"""

import torch
import torch.nn as nn
import numpy as np


class ThermalAdhesionModel(nn.Module):
    """
    Physics model relating thermal history to interlayer adhesion strength

    Physical principle:
    - Adhesion strength depends on thermal energy available for bonding
    - Higher temperature → better molecular diffusion → stronger adhesion
    - But excessive temperature can cause degradation

    Model: σ_adhesion = f(T_history, cooling_rate, time_above_melt)
    """

    def __init__(self):
        super().__init__()

        # Physical parameters
        self.melt_temperature = 200.0  # °C, typical PLA/ABS
        self.optimal_bond_temp = 220.0  # °C
        self.degradation_temp = 260.0  # °C

    def forward(self,
                temperature_history: torch.Tensor,
                cooling_rate: torch.Tensor) -> torch.Tensor:
        """
        Predict adhesion strength from thermal history

        Args:
            temperature_history: Temperature over time [batch, seq_len]
            cooling_rate: Rate of temperature change [batch, seq_len]

        Returns:
            Adhesion strength prediction [batch, 1]
        """
        # Time above melt temperature (bonding time)
        above_melt = (temperature_history - self.melt_temperature).clamp(min=0)
        bonding_time = above_melt.sum(dim=1, keepdim=True)  # Integrate over time

        # Temperature penalty for overheating
        max_temp = temperature_history.max(dim=1, keepdim=True)[0]
        overheating_penalty = torch.relu(max_temp - self.degradation_temp)

        # Optimal temperature window
        temp_deviation = torch.abs(temperature_history.mean(dim=1, keepdim=True) -
                                   self.optimal_bond_temp)

        # Adhesion strength model (MPa)
        # Base strength + thermal bonding factor - overheating penalty
        adhesion_strength = (
            20.0 +  # Base strength (MPa)
            0.1 * bonding_time -  # Thermal bonding contribution
            0.5 * overheating_penalty -  # Overheating reduces strength
            0.05 * temp_deviation  # Temperature instability reduces strength
        )

        # Physical constraint: adhesion strength must be positive
        adhesion_strength = torch.relu(adhesion_strength)

        return adhesion_strength


class StressAccumulationModel(nn.Module):
    """
    Physics model relating thermal gradients to internal stress accumulation

    Physical principle:
    - Rapid cooling → high thermal gradients → high residual stress
    - Non-uniform temperature creates differential contraction
    - Stress accumulates layer by layer

    Model: σ_stress = f(ΔT, cooling_rate, layer_count, material_properties)
    """

    def __init__(self):
        super().__init__()

        # Material properties (PLA-like)
        self.thermal_expansion_coeff = 6.8e-5  # 1/°C
        self.elastic_modulus = 3500.0  # MPa

    def forward(self,
                temperature_history: torch.Tensor,
                cooling_rate: torch.Tensor) -> torch.Tensor:
        """
        Predict internal stress from thermal history

        Args:
            temperature_history: Temperature over time [batch, seq_len]
            cooling_rate: Rate of temperature change [batch, seq_len]

        Returns:
            Internal stress prediction [batch, 1] (MPa)
        """
        # Temperature range (max - min)
        temp_range = (temperature_history.max(dim=1, keepdim=True)[0] -
                     temperature_history.min(dim=1, keepdim=True)[0])

        # Average cooling rate
        avg_cooling_rate = cooling_rate.abs().mean(dim=1, keepdim=True)

        # Thermal stress model
        # σ = E × α × ΔT
        # Stress increases with:
        # 1. Larger temperature differences
        # 2. Faster cooling rates
        # 3. More layers (accumulation)

        thermal_stress = (
            self.elastic_modulus *
            self.thermal_expansion_coeff *
            temp_range *
            (1 + 0.1 * avg_cooling_rate)  # Cooling rate multiplier
        )

        # Stress cannot be negative (compressive stress modeled separately)
        thermal_stress = torch.relu(thermal_stress)

        return thermal_stress


class PorosityFormationModel(nn.Module):
    """
    Physics model relating process conditions to porosity formation

    Physical principle:
    - Insufficient fusion → gaps between layers → porosity
    - Low temperature or high speed reduces interlayer diffusion
    - Vibration can cause void formation

    Model: Porosity = f(temperature, speed, vibration, pressure)
    """

    def __init__(self):
        super().__init__()

        # Critical parameters
        self.min_fusion_temp = 200.0  # °C
        self.optimal_pressure = 0.5  # MPa (extrusion pressure)

    def forward(self,
                temperature: torch.Tensor,
                vibration: torch.Tensor,
                print_speed: torch.Tensor) -> torch.Tensor:
        """
        Predict porosity from process conditions

        Args:
            temperature: Average temperature [batch, 1] (°C)
            vibration: Vibration magnitude [batch, 1] (mm/s²)
            print_speed: Print speed [batch, 1] (mm/s)

        Returns:
            Porosity prediction [batch, 1] (%)
        """
        # Temperature factor (low temp → high porosity)
        temp_deficit = torch.relu(self.min_fusion_temp - temperature)
        temp_factor = temp_deficit / self.min_fusion_temp  # [0, 1]

        # Speed factor (high speed → less time for fusion → higher porosity)
        speed_factor = torch.tanh(print_speed / 100.0)  # Normalize to [0, 1]

        # Vibration factor (high vibration → void formation)
        vib_factor = torch.tanh(vibration / 10.0)  # Normalize to [0, 1]

        # Base porosity from process factors
        porosity = (
            5.0 * temp_factor +      # Up to 5% from temperature issues
            3.0 * speed_factor +     # Up to 3% from high speed
            2.0 * vib_factor         # Up to 2% from vibration
        )

        # Physical constraint: porosity must be non-negative
        porosity = torch.relu(porosity)

        return porosity


class DimensionalAccuracyModel(nn.Module):
    """
    Physics model relating thermal effects to dimensional accuracy

    Physical principle:
    - Thermal expansion during printing → oversized features
    - Contraction during cooling → undersized features
    - Residual stress causes warping

    Model: Error = f(ΔT, cooling_rate, stress, shrinkage_factor)
    """

    def __init__(self):
        super().__init__()

        # Material properties
        self.thermal_expansion_coeff = 6.8e-5  # 1/°C
        self.shrinkage_factor = 0.002  # 0.2% typical shrinkage

    def forward(self,
                temperature_history: torch.Tensor,
                internal_stress: torch.Tensor) -> torch.Tensor:
        """
        Predict dimensional error from thermal history

        Args:
            temperature_history: Temperature over time [batch, seq_len]
            internal_stress: Predicted internal stress [batch, 1] (MPa)

        Returns:
            Dimensional error [batch, 1] (mm)
        """
        # Average temperature
        avg_temp = temperature_history.mean(dim=1, keepdim=True)

        # Room temperature (reference)
        room_temp = 25.0  # °C

        # Thermal expansion/contraction
        temp_diff = avg_temp - room_temp
        thermal_error = self.thermal_expansion_coeff * temp_diff  # Strain

        # Shrinkage contribution (percentage)
        shrinkage_error = self.shrinkage_factor

        # Stress-induced warping (approximately)
        warping_error = 1e-4 * internal_stress  # Small contribution

        # Total dimensional error (mm per mm of feature size)
        total_error = torch.abs(thermal_error) + shrinkage_error + warping_error

        return total_error


class PhysicsInformedQualityPredictor(nn.Module):
    """
    Complete physics-informed model for predicting implicit quality parameters

    This module combines individual physics models to predict:
    1. Interlayer Adhesion Strength
    2. Internal Stress
    3. Porosity
    4. Dimensional Accuracy
    5. Overall Quality Score

    Key Innovation:
    - Uses observable sensor data to predict unobservable quality parameters
    - Embeds physical constraints into neural network
    - Provides interpretable predictions based on physics
    """

    def __init__(self):
        super().__init__()

        # Individual physics models
        self.adhesion_model = ThermalAdhesionModel()
        self.stress_model = StressAccumulationModel()
        self.porosity_model = PorosityFormationModel()
        self.accuracy_model = DimensionalAccuracyModel()

    def forward(self,
                temperature: torch.Tensor,
                vibration: torch.Tensor,
                print_speed: torch.Tensor,
                pressure: torch.Tensor = None) -> dict:
        """
        Predict all implicit quality parameters

        Args:
            temperature: Temperature history [batch, seq_len]
            vibration: Vibration data [batch, seq_len, 3] (x, y, z)
            print_speed: Print speed [batch, seq_len]
            pressure: Extrusion pressure [batch, seq_len] (optional)

        Returns:
            Dictionary with quality predictions:
            - adhesion_strength: Interlayer adhesion (MPa)
            - internal_stress: Residual stress (MPa)
            - porosity: Void fraction (%)
            - dimensional_accuracy: Dimensional error (mm)
            - quality_score: Overall quality [0, 1]
        """
        # Compute cooling rate (derivative of temperature)
        cooling_rate = torch.diff(temperature, dim=1)
        # Pad to match original length
        cooling_rate = torch.cat([cooling_rate, cooling_rate[:, -1:]], dim=1)

        # Average vibration magnitude
        vibration_mag = torch.sqrt((vibration ** 2).sum(dim=-1))

        # Average print speed
        avg_speed = print_speed.mean(dim=1, keepdim=True)

        # Average temperature
        avg_temp = temperature.mean(dim=1, keepdim=True)

        # Predict individual quality parameters
        adhesion = self.adhesion_model(temperature, cooling_rate)
        stress = self.stress_model(temperature, cooling_rate)
        porosity = self.porosity_model(avg_temp, vibration_mag, avg_speed)
        accuracy = self.accuracy_model(temperature, stress)

        # Compute overall quality score
        # Higher adhesion → better quality
        # Lower stress → better quality
        # Lower porosity → better quality
        # Lower dimensional error → better quality

        quality_score = (
            torch.sigmoid(adhesion / 30.0) * 0.35 +  # Adhesion contribution (35%)
            torch.sigmoid(-stress / 10.0) * 0.25 +  # Low stress is good (25%)
            torch.sigmoid(-porosity / 5.0) * 0.20 +  # Low porosity is good (20%)
            torch.sigmoid(-accuracy / 0.01) * 0.20   # Low error is good (20%)
        )

        return {
            'adhesion_strength': adhesion,
            'internal_stress': stress,
            'porosity': porosity,
            'dimensional_accuracy': accuracy,
            'quality_score': quality_score,
        }

    def compute_physics_constraints(self,
                                  predictions: dict,
                                  inputs: dict) -> dict:
        """
        Compute physics-based constraints for loss function

        These constraints ensure predictions follow physical laws

        Args:
            predictions: Model predictions
            inputs: Input sensor data

        Returns:
            Dictionary with constraint violations (lower is better)
        """
        constraints = {}

        # Constraint 1: Adhesion must decrease with insufficient temperature
        # Constraint 2: Stress must increase with cooling rate
        # Constraint 3: Porosity must be non-negative
        # Constraint 4: All values must be in physically realistic ranges

        # Get inputs
        temperature = inputs.get('temperature', None)
        if temperature is not None:
            avg_temp = temperature.mean(dim=1)

            # Adhesion-temperature correlation
            # Higher temp → better adhesion (up to degradation point)
            expected_adhesion_trend = torch.sign(avg_temp - 200.0)
            actual_adhesion = predictions['adhesion_strength']

            # This is a simplified constraint
            # In practice, you'd use more sophisticated physics
            constraints['adhesion_temp_consistency'] = torch.abs(
                expected_adhesion_trend * actual_adhesion
            ).mean()

        # Non-negativity constraints
        constraints['porosity_nonnegative'] = torch.relu(
            -predictions['porosity']
        ).mean()

        constraints['stress_nonnegative'] = torch.relu(
            -predictions['internal_stress']
        ).mean()

        # Range constraints
        constraints['porosity_max'] = torch.relu(
            predictions['porosity'] - 20.0  # Max 20% porosity
        ).mean()

        constraints['adhesion_min'] = torch.relu(
            5.0 - predictions['adhesion_strength']  # Min 5 MPa
        ).mean()

        return constraints
