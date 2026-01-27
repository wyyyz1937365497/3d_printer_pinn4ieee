"""
Examples: Implicit Quality Parameter Prediction

This script demonstrates how to use the model to predict implicit quality parameters
that cannot be directly measured during printing, such as:
- Interlayer adhesion strength (层间粘合力)
- Internal stress (内应力)
- Porosity (孔隙率)
- Dimensional accuracy (尺寸精度)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from config import get_config
from models import UnifiedPINNSeq3D
from inference import UnifiedPredictor


# ============================================================================
# Example 1: Predict Implicit Quality Parameters
# ============================================================================

def example_predict_implicit_quality():
    """
    Example 1: Predict implicit quality parameters from sensor data
    """
    print("\n" + "="*80)
    print("Example 1: Predict Implicit Quality Parameters")
    print("="*80)

    # Load model
    config = get_config(preset='unified')
    model = UnifiedPINNSeq3D(config)
    predictor = UnifiedPredictor(model, config, device='cpu')

    # Simulate sensor data (observable parameters)
    # In real application, this comes from actual sensors
    seq_len = 200
    num_features = 12

    # Create realistic sensor data
    sensor_data = np.zeros((seq_len, num_features))

    # Temperature: 200-230°C (typical PLA printing range)
    sensor_data[:, 0] = np.linspace(210, 220, seq_len)  # Nozzle temp
    sensor_data[:, 1] = np.linspace(55, 60, seq_len)     # Bed temp

    # Vibration: small random variations
    sensor_data[:, 3:6] = np.random.randn(seq_len, 3) * 0.1  # X, Y, Z vibration

    # Motor current
    sensor_data[:, 6:9] = np.random.randn(seq_len, 3) * 0.05 + 0.5

    # Print speed
    sensor_data[:, 9] = 50.0  # mm/s

    # Position
    sensor_data[:, 10:12] = np.random.randn(seq_len, 2) * 10 + 100

    # Add batch dimension
    sensor_data = sensor_data[np.newaxis, :]  # [1, seq_len, num_features]

    print("\n🔍 Observable Sensor Data (Input):")
    print(f"  Temperature range: {sensor_data[0, :, 0].min():.1f} - {sensor_data[0, :, 0].max():.1f} °C")
    print(f"  Vibration RMS: {np.sqrt((sensor_data[0, :, 3:6]**2).mean()):.4f} mm/s²")
    print(f"  Print speed: {sensor_data[0, 0, 9]:.1f} mm/s")

    # Predict quality
    results = predictor.predict(sensor_data)

    print("\n🎯 Predicted Implicit Quality Parameters (Output):")
    print(f"  层间粘合力 (Adhesion Strength): {results['quality']['adhesion_strength'][0][0]:.2f} MPa")
    print(f"  内应力 (Internal Stress): {results['quality']['internal_stress'][0][0]:.2f} MPa")
    print(f"  孔隙率 (Porosity): {results['quality']['porosity'][0][0]:.2f}%")
    print(f"  尺寸精度 (Dimensional Accuracy Error): {results['quality']['dimensional_accuracy'][0][0]:.4f} mm")
    print(f"  综合质量评分 (Quality Score): {results['quality']['quality_score'][0][0]:.3f}")

    print("\n💡 Interpretation:")
    adhesion = results['quality']['adhesion_strength'][0][0]
    stress = results['quality']['internal_stress'][0][0]
    porosity = results['quality']['porosity'][0][0]

    if adhesion > 20.0:
        print(f"  ✓ 层间粘合力优秀 ({adhesion:.2f} MPa > 20 MPa)")
    else:
        print(f"  ✗ 层间粘合力不足 ({adhesion:.2f} MPa < 20 MPa)")

    if stress < 15.0:
        print(f"  ✓ 内应力在可接受范围 ({stress:.2f} MPa < 15 MPa)")
    else:
        print(f"  ✗ 内应力过大 ({stress:.2f} MPa > 15 MPa)")

    if porosity < 5.0:
        print(f"  ✓ 孔隙率低 ({porosity:.2f}% < 5%)")
    else:
        print(f"  ✗ 孔隙率过高 ({porosity:.2f}% > 5%)")


# ============================================================================
# Example 2: Early Stopping Decision
# ============================================================================

def example_early_stopping_decision():
    """
    Example 2: Use quality prediction for early stopping decision
    """
    print("\n" + "="*80)
    print("Example 2: Early Stopping Decision Based on Quality Prediction")
    print("="*80)

    config = get_config(preset='quality')
    model = UnifiedPINNSeq3D(config)
    predictor = UnifiedPredictor(model, config, device='cpu')

    # Simulate printing at different stages
    print("\n📊 Monitoring printing progress...\n")

    for progress in [20, 40, 60, 80]:
        print(f"Printing progress: {progress}%")

        # Generate sensor data (degrading quality over time)
        sensor_data = np.random.randn(200, 12)

        # Simulate quality degradation
        if progress >= 60:
            # Add some anomalies
            sensor_data[:, 0] -= (progress - 60) * 0.5  # Temperature dropping
            sensor_data[:, 3] += (progress - 60) * 0.01  # Vibration increasing

        # Predict quality
        quality = predictor.predict_quality_only(sensor_data)

        adhesion = quality['adhesion_strength'][0][0]
        stress = quality['internal_stress'][0][0]
        porosity = quality['porosity'][0][0]
        quality_score = quality['quality_score'][0][0]

        print(f"  Predicted quality:")
        print(f"    Adhesion: {adhesion:.2f} MPa")
        print(f"    Stress: {stress:.2f} MPa")
        print(f"    Porosity: {porosity:.2f}%")
        print(f"    Quality Score: {quality_score:.3f}")

        # Early stopping decision
        if quality_score < 0.4:
            print(f"\n  🛑 DECISION: STOP PRINTING at {progress}%")
            print(f"  Reason: Quality score too low ({quality_score:.3f} < 0.4)")
            print(f"  Estimated savings: {100 - progress}% of print time and material")
            break
        elif quality_score < 0.6:
            print(f"  ⚠️  WARNING: Quality degrading, monitor closely")
        else:
            print(f"  ✓ Quality acceptable, continue printing")

        print()


# ============================================================================
# Example 3: Process Parameter Optimization
# ============================================================================

def example_optimize_printing_parameters():
    """
    Example 3: Use quality prediction to optimize printing parameters
    """
    print("\n" + "="*80)
    print("Example 3: Process Parameter Optimization Based on Quality Prediction")
    print("="*80)

    config = get_config(preset='quality')
    model = UnifiedPINNSeq3D(config)
    predictor = UnifiedPredictor(model, config, device='cpu')

    print("\n🔧 Testing different printing parameters...\n")

    # Test different parameter combinations
    test_cases = [
        {"name": "Low Temperature", "temp": 190, "speed": 50},
        {"name": "Standard", "temp": 220, "speed": 50},
        {"name": "High Temperature", "temp": 240, "speed": 50},
        {"name": "High Speed", "temp": 220, "speed": 80},
    ]

    best_quality_score = 0
    best_params = None

    for case in test_cases:
        # Simulate sensor data with given parameters
        sensor_data = np.random.randn(200, 12)
        sensor_data[:, 0] = case["temp"]  # Set temperature
        sensor_data[:, 9] = case["speed"]  # Set speed

        # Predict quality
        quality = predictor.predict_quality_only(sensor_data)

        quality_score = quality['quality_score'][0][0]
        adhesion = quality['adhesion_strength'][0][0]
        stress = quality['internal_stress'][0][0]

        print(f"  {case['name']}:")
        print(f"    Temp: {case['temp']}°C, Speed: {case['speed']} mm/s")
        print(f"    Predicted Quality Score: {quality_score:.3f}")
        print(f"    Adhesion: {adhesion:.2f} MPa, Stress: {stress:.2f} MPa")

        if quality_score > best_quality_score:
            best_quality_score = quality_score
            best_params = case
            print(f"    ⭐ Best so far!")

        print()

    print(f"\n🏆 Recommended Parameters:")
    print(f"  {best_params['name']}")
    print(f"  Temperature: {best_params['temp']}°C")
    print(f"  Speed: {best_params['speed']} mm/s")
    print(f"  Expected Quality Score: {best_quality_score:.3f}")


# ============================================================================
# Example 4: Physics-Based Interpretation
# ============================================================================

def example_physics_interpretation():
    """
    Example 4: Interpret predictions using physics models
    """
    print("\n" + "="*80)
    print("Example 4: Physics-Based Interpretation of Quality Predictions")
    print("="*80)

    from models.physics.quality_physics import (
        ThermalAdhesionModel,
        StressAccumulationModel,
        PorosityFormationModel
    )

    # Initialize physics models
    adhesion_model = ThermalAdhesionModel()
    stress_model = StressAccumulationModel()
    porosity_model = PorosityFormationModel()

    # Create test data
    batch_size = 1
    seq_len = 200

    # Scenario 1: Good printing conditions
    print("\n📈 Scenario 1: Optimal Conditions")
    temp_good = torch.ones(batch_size, seq_len) * 220  # Optimal temp
    cooling_good = torch.randn(batch_size, seq_len) * 0.5

    adhesion_good = adhesion_model(temp_good, cooling_good)
    stress_good = stress_model(temp_good, cooling_good)

    print(f"  Temperature: 220°C (optimal)")
    print(f"  Predicted Adhesion: {adhesion_good[0][0]:.2f} MPa")
    print(f"  Predicted Stress: {stress_good[0][0]:.2f} MPa")
    print(f"  ✓ Strong bonding, low stress")

    # Scenario 2: Poor printing conditions
    print("\n📉 Scenario 2: Suboptimal Conditions")
    temp_poor = torch.ones(batch_size, seq_len) * 190  # Too cold
    cooling_poor = -torch.ones(batch_size, seq_len) * 2.0  # Fast cooling

    adhesion_poor = adhesion_model(temp_poor, cooling_poor)
    stress_poor = stress_model(temp_poor, cooling_poor)

    print(f"  Temperature: 190°C (too cold)")
    print(f"  Predicted Adhesion: {adhesion_poor[0][0]:.2f} MPa")
    print(f"  Predicted Stress: {stress_poor[0][0]:.2f} MPa")
    print(f"  ✗ Weak bonding, high stress")

    # Explain the physics
    print("\n💡 Physics Explanation:")
    print(f"  Temperature difference: {220 - 190}°C")
    print(f"  Adhesion reduction: {adhesion_good[0][0] - adhesion_poor[0][0]:.2f} MPa")
    print(f"  Stress increase: {stress_poor[0][0] - stress_good[0][0]:.2f} MPa")
    print(f"  → Lower temperature reduces molecular diffusion between layers")
    print(f"  → Faster cooling creates larger thermal gradients")
    print(f"  → Result: Poor interlayer bonding and high residual stress")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("IMPLICIT QUALITY PARAMETER PREDICTION EXAMPLES")
    print("="*80)
    print("\nThese examples demonstrate how to predict quality parameters")
    print("that CANNOT be directly measured during printing:")
    print("  • Interlayer Adhesion Strength (层间粘合力)")
    print("  • Internal Stress (内应力)")
    print("  • Porosity (孔隙率)")
    print("  • Dimensional Accuracy (尺寸精度)")

    examples = [
        ("Predict Implicit Quality Parameters", example_predict_implicit_quality),
        ("Early Stopping Decision", example_early_stopping_decision),
        ("Optimize Printing Parameters", example_optimize_printing_parameters),
        ("Physics-Based Interpretation", example_physics_interpretation),
    ]

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {name}: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
