"""
Physics computation utilities for PINN constraints
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict


def compute_gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient dy/dx using automatic differentiation

    Args:
        y: Output tensor
        x: Input tensor

    Returns:
        Gradient tensor dy/dx
    """
    grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    return grad


def compute_thermal_loss(temperature: torch.Tensor,
                        coordinates: torch.Tensor,
                        time: torch.Tensor,
                        diffusivity: float = 1.0,
                        heat_source: torch.Tensor = None) -> torch.Tensor:
    """
    Compute thermal physics loss: ∂T/∂t = α∇²T + Q

    Args:
        temperature: Temperature field [batch, seq_len, 1]
        coordinates: Spatial coordinates [batch, seq_len, 3] (x, y, z)
        time: Time values [batch, seq_len, 1]
        diffusivity: Thermal diffusivity coefficient
        heat_source: Heat source term [batch, seq_len, 1]

    Returns:
        Thermal physics loss
    """
    # Time derivative: ∂T/∂t
    dT_dt = compute_gradient(temperature, time)

    # Spatial gradients: ∂T/∂x, ∂T/∂y, ∂T/∂z
    dT_dx = compute_gradient(temperature, coordinates[..., 0:1])
    dT_dy = compute_gradient(temperature, coordinates[..., 1:2])
    dT_dz = compute_gradient(temperature, coordinates[..., 2:3])

    # Second derivatives: ∂²T/∂x², ∂²T/∂y², ∂²T/∂z²
    d2T_dx2 = compute_gradient(dT_dx, coordinates[..., 0:1])
    d2T_dy2 = compute_gradient(dT_dy, coordinates[..., 1:2])
    d2T_dz2 = compute_gradient(dT_dz, coordinates[..., 2:3])

    # Laplacian: ∇²T = ∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²
    laplacian_T = d2T_dx2 + d2T_dy2 + d2T_dz2

    # Heat equation residual: ∂T/∂t - α∇²T - Q
    if heat_source is None:
        heat_source = torch.zeros_like(temperature)

    residual = dT_dt - diffusivity * laplacian_T - heat_source

    # MSE of residual
    loss = torch.mean(residual ** 2)

    return loss


def compute_vibration_loss(displacement: torch.Tensor,
                          velocity: torch.Tensor,
                          time: torch.Tensor,
                          mass: float = 1.0,
                          damping: float = 0.5,
                          stiffness: float = 10.0,
                          force: torch.Tensor = None) -> torch.Tensor:
    """
    Compute vibration dynamics loss: m·d²x/dt² + c·dx/dt + k·x = F

    Args:
        displacement: Displacement [batch, seq_len, 3]
        velocity: Velocity [batch, seq_len, 3]
        time: Time values [batch, seq_len, 1]
        mass: Mass coefficient
        damping: Damping coefficient
        stiffness: Stiffness coefficient
        force: External force [batch, seq_len, 3]

    Returns:
        Vibration physics loss
    """
    # Acceleration: d²x/dt² = dv/dt
    acceleration = compute_gradient(velocity, time)

    # Damped harmonic oscillator residual: m·a + c·v + k·x - F
    if force is None:
        force = torch.zeros_like(displacement)

    residual = mass * acceleration + damping * velocity + stiffness * displacement - force

    # MSE of residual
    loss = torch.mean(residual ** 2)

    return loss


def compute_energy_loss(energy: torch.Tensor,
                       input_power: torch.Tensor,
                       output_power: torch.Tensor,
                       time: torch.Tensor,
                       loss_coefficient: float = 0.1) -> torch.Tensor:
    """
    Compute energy conservation loss: dE/dt = P_in - P_out - P_loss

    Args:
        energy: System energy [batch, seq_len, 1]
        input_power: Input power [batch, seq_len, 1]
        output_power: Output power [batch, seq_len, 1]
        time: Time values [batch, seq_len, 1]
        loss_coefficient: Power loss coefficient

    Returns:
        Energy conservation loss
    """
    # Time derivative: dE/dt
    dE_dt = compute_gradient(energy, time)

    # Power loss: P_loss = loss_coefficient * E
    power_loss = loss_coefficient * energy

    # Energy conservation residual: dE/dt - (P_in - P_out - P_loss)
    residual = dE_dt - (input_power - output_power - power_loss)

    # MSE of residual
    loss = torch.mean(residual ** 2)

    return loss


def compute_motor_coupling_loss(motor_current: torch.Tensor,
                               acceleration: torch.Tensor,
                               vibration: torch.Tensor,
                               coupling_coeff: float = 1.0) -> torch.Tensor:
    """
    Compute motor-vibration coupling loss: I_motor ∝ acceleration + vibration_load

    Args:
        motor_current: Motor current [batch, seq_len, 3]
        acceleration: Acceleration [batch, seq_len, 3]
        vibration: Vibration load [batch, seq_len, 3]
        coupling_coeff: Coupling coefficient

    Returns:
        Motor coupling loss
    """
    # Expected current: I_expected = coupling_coeff * (acceleration + vibration)
    expected_current = coupling_coeff * (acceleration + vibration)

    # Residual: I_motor - I_expected
    residual = motor_current - expected_current

    # MSE of residual
    loss = torch.mean(residual ** 2)

    return loss


def compute_physics_loss(predictions: Dict[str, torch.Tensor],
                        targets: Dict[str, torch.Tensor],
                        inputs: Dict[str, torch.Tensor],
                        physics_config: Dict) -> torch.Tensor:
    """
    Compute combined physics loss from all physics constraints

    Args:
        predictions: Model predictions dictionary
        targets: Target values dictionary
        inputs: Input features dictionary
        physics_config: Physics configuration parameters

    Returns:
        Combined physics loss
    """
    total_loss = 0.0
    loss_weights = {
        'thermal': 1.0,
        'vibration': 1.0,
        'energy': 1.0,
        'motor': 1.0,
    }

    # Thermal loss
    if 'temperature' in predictions and 'coordinates' in inputs:
        thermal_loss = compute_thermal_loss(
            predictions['temperature'],
            inputs['coordinates'],
            inputs['time'],
            diffusivity=physics_config.get('thermal_diffusivity', 1.0)
        )
        total_loss += loss_weights['thermal'] * thermal_loss

    # Vibration loss
    if 'displacement' in predictions and 'velocity' in inputs:
        vibration_loss = compute_vibration_loss(
            predictions['displacement'],
            inputs['velocity'],
            inputs['time'],
            mass=physics_config.get('mass', 1.0),
            damping=physics_config.get('damping', 0.5),
            stiffness=physics_config.get('stiffness', 10.0)
        )
        total_loss += loss_weights['vibration'] * vibration_loss

    # Energy loss
    if 'energy' in predictions:
        energy_loss = compute_energy_loss(
            predictions['energy'],
            inputs.get('input_power', torch.zeros_like(predictions['energy'])),
            inputs.get('output_power', torch.zeros_like(predictions['energy'])),
            inputs['time'],
            loss_coefficient=physics_config.get('energy_loss_weight', 0.1)
        )
        total_loss += loss_weights['energy'] * energy_loss

    # Motor coupling loss
    if 'motor_current' in predictions and 'acceleration' in inputs:
        motor_loss = compute_motor_coupling_loss(
            predictions['motor_current'],
            inputs['acceleration'],
            inputs.get('vibration', torch.zeros_like(inputs['acceleration'])),
            coupling_coeff=physics_config.get('motor_coupling_weight', 1.0)
        )
        total_loss += loss_weights['motor'] * motor_loss

    return total_loss
