"""
Implicit Quality Prediction Examples

This file contains examples for using the implicit quality prediction system
based on physics-informed neural networks.

NOTE: Training data for these models should be generated using the MATLAB
physics simulation system and converted using convert_matlab_to_python.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt

from config import get_config
from models import QualityDecoder, UnifiedPINNSeq3D
from training import Trainer
from inference import UnifiedPredictor


def example_physics_informed_quality_prediction():
    """Example: Using physics-informed neural networks for quality prediction"""
    print("\n" + "="*80)
    print("Physics-Informed Quality Prediction Example")
    print("="*80)
    
    config = get_config(preset='quality')
    
    # Create model
    model = QualityDecoder(config.model)
    
    # Example sensor data from simulated physics
    # NOTE: In practice, this would come from MATLAB physics simulation
    batch_size = 8
    seq_len = 200
    num_features = 12
    
    # Simulated sensor data (temperature, vibration, current, etc.)
    sensor_data = torch.randn(batch_size, seq_len, num_features)
    
    # Physics constraints (simulated from MATLAB thermal and vibration models)
    # These would normally come from converted MATLAB simulation data
    physics_constraints = {
        'thermal_constraint': torch.randn(batch_size, seq_len, 1) * 0.1,
        'vibration_constraint': torch.randn(batch_size, seq_len, 2) * 0.05,
        'material_flow_constraint': torch.randn(batch_size, seq_len, 1) * 0.2
    }
    
    # Forward pass with physics constraints
    with torch.no_grad():
        quality_predictions = model(
            sensor_data=sensor_data,
            physics_constraints=physics_constraints
        )
    
    print(f"Quality predictions shape: {quality_predictions['quality_score'].shape}")
    print(f"RUL predictions shape: {quality_predictions['rul'].shape}")
    print(f"Temperature predictions shape: {quality_predictions['temperature'].shape}")
    

def example_multi_physics_integration():
    """Example: Integrating multiple physics domains for quality assessment"""
    print("\n" + "="*80)
    print("Multi-Physics Integration Example")
    print("="*80)
    
    config = get_config(preset='unified')
    model = UnifiedPINNSeq3D(config)
    
    # Simulated multi-physics data (from MATLAB simulation)
    batch_size = 4
    seq_len = 150
    num_features = 12
    
    sensor_data = torch.randn(batch_size, seq_len, num_features)
    
    # In a real scenario, this would be loaded from MATLAB simulation data
    # that includes thermal, mechanical, and material flow physics
    print("Integrating multiple physics domains...")
    print("- Thermal physics (temperature fields, cooling rates)")
    print("- Mechanical physics (vibrations, resonance, belt stretch)")
    print("- Material physics (flow, adhesion, curing)")
    
    with torch.no_grad():
        outputs = model(sensor_data)
        
    print(f"\nModel outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")


def example_quality_early_stopping():
    """Example: Using quality prediction for early stopping during print"""
    print("\n" + "="*80)
    print("Quality-Based Early Stopping Example")
    print("="*80)
    
    config = get_config(preset='quality')
    model = QualityDecoder(config.model)
    predictor = UnifiedPredictor(model, config, device='cpu')
    
    print("Simulating print monitoring with early stopping...")
    
    # Simulate quality degradation over time (as would happen in a real print)
    for step in range(10):  # Simulate 10 monitoring steps
        # Generate increasingly degraded sensor data
        degradation_factor = step * 0.1
        sensor_data = torch.randn(1, 200, 12)
        # Increase temperature variation to simulate problems
        sensor_data[:, :, 0] += torch.randn(1, 200) * degradation_factor
        
        with torch.no_grad():
            quality_results = predictor.predict_quality_only(sensor_data.numpy())
        
        quality_score = float(quality_results['quality_score'][0][0])
        rul = float(quality_results['rul'][0][0])
        
        print(f"Step {step+1}: Quality={quality_score:.3f}, RUL={rul:.1f}s")
        
        # Early stopping condition
        if quality_score < 0.3:  # Threshold for stopping
            print(f"  >>> EARLY STOPPING: Quality below threshold!")
            break


def example_visualize_physics_correlations():
    """Example: Visualizing correlations between physics variables and quality"""
    print("\n" + "="*80)
    print("Physics-Quality Correlation Visualization")
    print("="*80)
    
    # Generate correlated data (would come from MATLAB simulation in practice)
    n_points = 500
    
    # Simulate relationships that would be found in physics simulation
    temperature_variance = np.random.normal(0, 0.5, n_points)
    vibration_magnitude = np.random.normal(0, 0.3, n_points)
    belt_stretch = np.random.normal(0, 0.2, n_points)
    
    # Simulated quality score based on physics variables
    quality_scores = (
        0.8 - 
        0.3 * np.abs(temperature_variance) * 0.5 - 
        0.3 * np.abs(vibration_magnitude) * 0.7 - 
        0.3 * np.abs(belt_stretch) * 0.6 +
        np.random.normal(0, 0.05, n_points)  # noise
    )
    quality_scores = np.clip(quality_scores, 0, 1)  # Keep in [0, 1]
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].scatter(temperature_variance, quality_scores, alpha=0.6)
    axes[0].set_xlabel('Temperature Variance')
    axes[0].set_ylabel('Quality Score')
    axes[0].set_title('Temperature vs Quality')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(vibration_magnitude, quality_scores, alpha=0.6, color='orange')
    axes[1].set_xlabel('Vibration Magnitude')
    axes[1].set_ylabel('Quality Score')
    axes[1].set_title('Vibration vs Quality')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].scatter(belt_stretch, quality_scores, alpha=0.6, color='green')
    axes[2].set_xlabel('Belt Stretch')
    axes[2].set_ylabel('Quality Score')
    axes[2].set_title('Mechanical Errors vs Quality')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('physics_quality_correlations.png', dpi=150, bbox_inches='tight')
    print("Correlation plot saved as 'physics_quality_correlations.png'")
    

def main():
    """Run all implicit quality examples"""
    print("\n" + "="*80)
    print("Implicit Quality Prediction Examples")
    print("NOTE: Uses MATLAB physics simulation data in practice")
    print("="*80)
    
    examples = [
        ("Physics-Informed Quality Prediction", example_physics_informed_quality_prediction),
        ("Multi-Physics Integration", example_multi_physics_integration),
        ("Quality Early Stopping", example_quality_early_stopping),
        ("Physics-Quality Correlations", example_visualize_physics_correlations),
    ]
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Error in {name}: {str(e)}")
    
    print("\n" + "="*80)
    print("All implicit quality examples completed!")
    print("="*80)


if __name__ == '__main__':
    main()
