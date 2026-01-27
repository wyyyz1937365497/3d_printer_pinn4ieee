"""
Usage examples for the 3D Printer PINN-Seq3D Framework

This file demonstrates how to use the framework for various tasks.
NOTE: This project now primarily uses MATLAB-based physics simulation for data generation
and training data preparation. Python scripts are used for model training and inference only.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from config import get_config
from models import UnifiedPINNSeq3D, QualityPredictionOnlyModel, TrajectoryCorrectionOnlyModel
from inference import UnifiedPredictor
from training import Trainer


# ============================================================================
# Example 1: Create and Use a Model
# ============================================================================

def example_create_model():
    """Example: Create a model and make predictions"""
    print("\n" + "="*80)
    print("Example 1: Create Model and Make Predictions")
    print("="*80)

    # Get configuration
    config = get_config(preset='unified')

    # Create model
    model = UnifiedPINNSeq3D(config)

    # Print model info
    info = model.get_model_info()
    print(f"\nModel Type: {info['model_type']}")
    print(f"Total Parameters: {info['num_parameters']:,}")
    print(f"Trainable Parameters: {info['num_trainable_parameters']:,}")

    # Create dummy input
    batch_size = 4
    seq_len = 200
    num_features = 12

    dummy_input = torch.randn(batch_size, seq_len, num_features)

    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_input)

    # Print outputs
    print(f"\nOutputs:")
    print(f"  RUL shape: {outputs['rul'].shape}")
    print(f"  Temperature shape: {outputs['temperature'].shape}")
    print(f"  Fault prediction shape: {outputs['fault_pred'].shape}")
    print(f"  Trajectory correction X shape: {outputs['displacement_x'].shape}")


# ============================================================================
# Example 2: Train a Model
# ============================================================================

def example_train_model():
    """Example: Train a model with MATLAB-simulated data"""
    print("\n" + "="*80)
    print("Example 2: Train Model with MATLAB-Simulated Data")
    print("="*80)

    from torch.utils.data import Dataset, DataLoader

    # Create dataset from MATLAB simulation data
    # NOTE: Real data would come from MATLAB simulations converted via convert_matlab_to_python.py
    class MatlabSimulationDataset(Dataset):
        def __init__(self, data_file='data/processed/training_data.h5', num_samples=1000):
            self.num_samples = num_samples
            
            # In a real scenario, we would load from MATLAB-generated data
            # For demo purposes, we still create synthetic data similar to MATLAB simulation
            self.features = torch.randn(num_samples, 200, 12)
            self.targets = {
                'rul': torch.rand(num_samples, 1) * 1000,
                'temperature': torch.rand(num_samples, 1) * 50 + 200,
                'vibration_x': torch.randn(num_samples, 1) * 0.1,
                'vibration_y': torch.randn(num_samples, 1) * 0.1,
                'quality_score': torch.rand(num_samples, 1),
                'fault_label': torch.randint(0, 4, (num_samples,)),
                'displacement_x': torch.randn(num_samples, 1) * 0.01,
                'displacement_y': torch.randn(num_samples, 1) * 0.01,
                'displacement_z': torch.randn(num_samples, 1) * 0.001,
            }

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return {
                'features': self.features[idx],
                **{k: v[idx] for k, v in self.targets.items()}
            }

    # Get configuration (fast training)
    config = get_config(preset='fast')
    config.training.num_epochs = 2  # Train for just 2 epochs for demo
    config.training.batch_size = 32

    # Create data loaders
    train_dataset = MatlabSimulationDataset(num_samples=500)
    val_dataset = MatlabSimulationDataset(num_samples=100)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model and trainer
    model = UnifiedPINNSeq3D(config)

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Train
    print("\nStarting training...")
    print("(Note: Real training would use MATLAB-simulated data from converted .mat files)")
    trainer.train()

    print("\nTraining completed!")


# ============================================================================
# Example 3: Use Trained Model for Inference
# ============================================================================

def example_inference():
    """Example: Use trained model for inference"""
    print("\n" + "="*80)
    print("Example 3: Inference with Trained Model")
    print("="*80)

    # Note: This example assumes you have a trained checkpoint
    # For demo purposes, we'll use the model directly

    config = get_config(preset='unified')
    model = UnifiedPINNSeq3D(config)

    # Create predictor
    predictor = UnifiedPredictor(model, config, device='cpu')

    # Create dummy sensor data
    sensor_data = np.randn(200, 12)  # [seq_len, num_features]

    # Make predictions
    print("\nMaking predictions...")

    # Full prediction
    results = predictor.predict(sensor_data)

    print(f"\nQuality Metrics:")
    print(f"  RUL: {results['quality']['rul'][0][0]:.2f} seconds")
    print(f"  Temperature: {results['quality']['temperature'][0][0]:.2f} °C")
    print(f"  Quality Score: {results['quality']['quality_score'][0][0]:.3f}")

    print(f"\nFault Classification:")
    print(f"  Predicted Class: {results['fault']['predicted_class'][0]}")
    print(f"  Probabilities: {results['fault']['probabilities'][0]}")

    print(f"\nTrajectory Correction:")
    print(f"  dx: {results['trajectory']['dx'][0][0]:.4f} mm")
    print(f"  dy: {results['trajectory']['dy'][0][0]:.4f} mm")
    print(f"  dz: {results['trajectory']['dz'][0][0]:.4f} mm")

    # Quality-only prediction (faster)
    quality = predictor.predict_quality_only(sensor_data)
    print(f"\nQuality-only prediction: RUL = {quality['rul'][0][0]:.2f} seconds")

    # Fault-only prediction (faster)
    fault = predictor.predict_fault_only(sensor_data)
    print(f"Fault-only prediction: Class = {fault['predicted_class'][0]}")

    # Trajectory-only prediction (faster)
    trajectory = predictor.predict_trajectory_only(sensor_data)
    print(f"Trajectory correction: dx = {trajectory['dx'][0][0]:.4f} mm")


# ============================================================================
# Example 4: Quality-Based Early Stopping
# ============================================================================

def example_early_stopping():
    """Example: Use quality prediction for early stopping decision"""
    print("\n" + "="*80)
    print("Example 4: Quality-Based Early Stopping")
    print("="*80)

    config = get_config(preset='quality')
    model = QualityPredictionOnlyModel(config)
    predictor = UnifiedPredictor(model, config, device='cpu')

    # Simulate monitoring over time
    print("\nSimulating print monitoring...")

    for timestep in range(5):
        # Simulate sensor data (degrading quality over time)
        quality_degradation = timestep * 0.15
        sensor_data = np.random.randn(200, 12)
        sensor_data[:, 0] += quality_degradation  # Degrading temperature

        # Predict quality
        quality = predictor.predict_quality_only(sensor_data)
        quality_score = quality['quality_score'][0][0]

        # Make decision
        should_stop = predictor.should_stop_printing(sensor_data, threshold=0.5)

        print(f"\nTimestep {timestep + 1}:")
        print(f"  Quality Score: {quality_score:.3f}")
        print(f"  RUL: {quality['rul'][0][0]:.2f} seconds")
        print(f"  Should Stop: {should_stop}")

        if should_stop:
            print(f"\n  >> STOP PRINTING at timestep {timestep + 1} due to low quality!")
            break


# ============================================================================
# Example 5: Trajectory Correction
# ============================================================================

def example_trajectory_correction():
    """Example: Apply trajectory correction to a printing position"""
    print("\n" + "="*80)
    print("Example 5: Trajectory Correction")
    print("="*80)

    config = get_config(preset='trajectory')
    model = TrajectoryCorrectionOnlyModel(config)
    predictor = UnifiedPredictor(model, config, device='cpu')

    # Current position
    current_position = np.array([100.0, 100.0, 0.2])  # x, y, z in mm

    # Sensor data
    sensor_data = np.random.randn(10, 4)  # Shorter sequence for trajectory

    # Get corrected position
    corrected_position = predictor.get_trajectory_correction(sensor_data, current_position)

    print(f"\nOriginal Position: [{current_position[0]:.3f}, {current_position[1]:.3f}, {current_position[2]:.3f}] mm")
    print(f"Corrected Position: [{corrected_position[0]:.3f}, {corrected_position[1]:.3f}, {corrected_position[2]:.3f}] mm")

    correction = corrected_position - current_position
    print(f"Correction Applied: [{correction[0]:.4f}, {correction[1]:.4f}, {correction[2]:.4f}] mm")


# ============================================================================
# Example 6: Custom Configuration
# ============================================================================

def example_custom_config():
    """Example: Use custom configuration"""
    print("\n" + "="*80)
    print("Example 6: Custom Configuration")
    print("="*80)

    # Create custom configuration
    config = get_config(
        preset='unified',
        experiment_name='my_custom_experiment',
        d_model=128,  # Smaller model
        num_heads=4,
        num_layers=4,
        learning_rate=5e-4,
        batch_size=128,
        lambda_quality=2.0,  # Emphasize quality prediction
        lambda_trajectory=0.5,  # De-emphasize trajectory
    )

    print(f"\nCustom Configuration:")
    print(f"  Experiment Name: {config.experiment_name}")
    print(f"  Model Dimension: {config.model.d_model}")
    print(f"  Number of Heads: {config.model.num_heads}")
    print(f"  Number of Layers: {config.model.num_layers}")
    print(f"  Learning Rate: {config.training.learning_rate}")
    print(f"  Batch Size: {config.training.batch_size}")
    print(f"  Lambda Quality: {config.lambda_quality}")
    print(f"  Lambda Trajectory: {config.lambda_trajectory}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("3D Printer PINN-Seq3D Framework - Usage Examples")
    print("="*80)

    examples = [
        ("Create Model", example_create_model),
        ("Train Model", example_train_model),
        ("Inference", example_inference),
        ("Early Stopping", example_early_stopping),
        ("Trajectory Correction", example_trajectory_correction),
        ("Custom Configuration", example_custom_config),
    ]

    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    # Run examples
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {name}: {str(e)}")

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
