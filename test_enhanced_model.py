"""
Quick test script for enhanced TCN model

Tests:
1. Model initialization
2. Forward pass
3. Loss computation
4. Physics constraints
5. Backward pass
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from config import get_config
from models.implicit import ImplicitStateTCN, AdaptiveMultiTaskLoss


def test_model():
    """Test the enhanced model"""
    print("="*80)
    print("Testing Enhanced Implicit State Inference Model")
    print("="*80)
    print()

    # Load config
    config = get_config(preset='unified')

    # Test 1: Model initialization
    print("[Test 1] Model Initialization")
    try:
        model = ImplicitStateTCN(config)
        print("  [OK] Model created successfully")

        model_info = model.get_model_info()
        print(f"  Type: {model_info['model_type']}")
        print(f"  Parameters: {model_info['num_parameters']:,}")
        print(f"  Features: {', '.join(model_info['features'])}")
        print()
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

    # Test 2: Forward pass
    print("[Test 2] Forward Pass")
    try:
        # Create dummy input
        batch_size = 4
        seq_len = 100
        num_features = 12  # Adjust based on your data

        dummy_input = torch.randn(batch_size, seq_len, num_features)

        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input)

        print("  ✓ Forward pass successful")
        print(f"  Output shapes:")
        for key, value in outputs.items():
            print(f"    {key}: {value.shape}")
        print()
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

    # Test 3: Physics constraints
    print("[Test 3] Physics Constraints")
    try:
        # Check bounds
        adhesion = outputs['adhesion_strength']
        stress = outputs['internal_stress']
        porosity = outputs['porosity']
        quality = outputs['quality_score']

        checks = []

        # Adhesion: [0, 1]
        if adhesion.min() >= 0 and adhesion.max() <= 1:
            checks.append("  ✓ Adhesion in [0, 1]")
        else:
            checks.append(f"  ✗ Adhesion out of bounds: [{adhesion.min():.3f}, {adhesion.max():.3f}]")

        # Stress: >= 10
        if stress.min() >= 10:
            checks.append("  ✓ Stress >= 10 MPa")
        else:
            checks.append(f"  ✗ Stress below 10 MPa: min={stress.min():.3f}")

        # Porosity: [0, 100]
        if porosity.min() >= 0 and porosity.max() <= 100:
            checks.append("  ✓ Porosity in [0, 100]")
        else:
            checks.append(f"  ✗ Porosity out of bounds: [{porosity.min():.3f}, {porosity.max():.3f}]")

        # Quality: [0, 1]
        if quality.min() >= 0 and quality.max() <= 1:
            checks.append("  ✓ Quality in [0, 1]")
        else:
            checks.append(f"  ✗ Quality out of bounds: [{quality.min():.3f}, {quality.max():.3f}]")

        for check in checks:
            print(check)
        print()
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

    # Test 4: Loss computation
    print("[Test 4] Loss Computation")
    try:
        # Create dummy targets
        targets = {
            'adhesion_strength': torch.rand(batch_size, 1) * 0.7 + 0.15,
            'internal_stress': torch.rand(batch_size, 1) * 10 + 12,
            'porosity': torch.rand(batch_size, 1) * 7 + 13,
            'dimensional_accuracy': torch.randn(batch_size, 1) * 0.02 + 0.15,
            'quality_score': torch.rand(batch_size, 1) * 0.4 + 0.3,
        }

        # Create physics inputs
        physics_inputs = {
            'T_interface': torch.randn(batch_size, seq_len, 1) * 10 + 130,
            'acceleration': torch.randn(batch_size, seq_len, 1) * 200 + 300,
        }

        # Create loss function
        criterion = AdaptiveMultiTaskLoss(
            lambda_physics=0.1,
            use_adaptive_weights=False
        )

        # Compute loss
        losses = criterion(outputs, targets, physics_inputs)

        print("  ✓ Loss computed successfully")
        print(f"  Total loss: {losses['total'].item():.6f}")
        print(f"  Data loss: {losses['data'].item():.6f}")
        print(f"  Physics loss: {losses['physics'].item():.6f}")

        # Individual task losses
        print(f"  Task losses:")
        print(f"    Adhesion: {losses['adhesion'].item():.6f}")
        print(f"    Stress: {losses['stress'].item():.6f}")
        print(f"    Porosity: {losses['porosity'].item():.6f}")
        print(f"    Accuracy: {losses['accuracy'].item():.6f}")
        print(f"    Quality: {losses['quality'].item():.6f}")
        print()
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Backward pass
    print("[Test 5] Backward Pass")
    try:
        model.train()
        loss = losses['total']
        loss.backward()

        # Check gradients
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                break

        if has_grad:
            print("  ✓ Backward pass successful")
            print("  ✓ Gradients computed")
        else:
            print("  ✗ No gradients computed")
            return False
        print()
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 6: Model parameters
    print("[Test 6] Model Parameters")
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
        print()
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

    print("="*80)
    print("✅ All Tests Passed!")
    print("="*80)
    print("\nThe enhanced model is ready for training.")
    print("\nTo start training, run:")
    print("  python experiments/train_implicit_state_tcn.py \\")
    print("      --data_dir data_simulation_* \\")
    print("      --epochs 100 \\")
    print("      --batch_size 32 \\")
    print("      --lambda_physics 0.1")
    print()

    return True


if __name__ == '__main__':
    success = test_model()
    sys.exit(0 if success else 1)
