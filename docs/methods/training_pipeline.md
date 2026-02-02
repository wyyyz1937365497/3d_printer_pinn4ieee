# Training Pipeline

**Purpose**: End-to-end workflow for training trajectory error prediction models.

---

## Overview

The training pipeline consists of five stages:

1. **Data collection** - MATLAB simulation
2. **Data preparation** - Convert to PyTorch format
3. **Model training** - Train neural network
4. **Model evaluation** - Test performance
5. **Model deployment** - Export for inference

---

## Stage 1: Data Collection

### MATLAB Simulation

```matlab
% Collect training data
collect_3dbenchy('sampled:5');   % ~10 layers
collect_bearing5('sampled:5');   % ~10 layers
collect_nautilus('sampled:5');   % ~10 layers
collect_boat('sampled:5');       % ~10 layers
```

**Expected output**:
- 40 layers × 2 minutes/layer = ~80 minutes
- ~40,000 sample points total
- Files in `data_simulation_*/` directories

### Verification

```matlab
% Verify data quality
test_firmware_effects_simple

% Expected output:
%   RMS error: ~140 μm ✅
%   Temperature range: 45-85 °C ✅
%   Adhesion ratio: 0.75-0.90 ✅
```

---

## Stage 2: Data Preparation

### Convert to Training Format

```bash
python data/scripts/prepare_training_data.py \
    --data_dirs data_simulation_* \
    --output_dir data/processed \
    --sequence_length 128 \
    --stride 4 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

### Output Structure

```
data/processed/
├── train/
│   ├── sample_00001.pt
│   ├── sample_00002.pt
│   └── ...
├── val/
│   └── ...
├── test/
│   └── ...
└── stats.json  % Normalization statistics
```

### Sample Format

Each `.pt` file contains:

```python
{
    'input_features': torch.Tensor,  # [sequence_length, 15]
    'trajectory_targets': torch.Tensor,  # [sequence_length, 2]
    'thermal_targets': torch.Tensor,     # [sequence_length, 3]
    'metadata': dict  # layer, model, params
}
```

### Data Statistics

Check generated data:

```bash
python -c "
from data.simulation.dataset import PrinterSimulationDataset
dataset = PrinterSimulationDataset('data/processed/train')
print(f'Total samples: {len(dataset)}')
print(f'Feature shape: {dataset[0][\"input_features\"].shape}')
print(f'Target shape: {dataset[0][\"trajectory_targets\"].shape}')
"
```

**Expected output**:
```
Total samples: ~50,000
Feature shape: torch.Size([128, 15])
Target shape: torch.Size([128, 2])
```

---

## Stage 3: Model Training

### Basic Training Command

```bash
python experiments/train_trajectory_model.py \
    --data_root data/processed \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --device cuda:0 \
    --experiment_name trajectory_correction_v1
```

### Full Training Command

```bash
python experiments/train_trajectory_model.py \
    --data_root data/processed \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --device cuda:0 \
    --experiment_name firmware_enhanced_v1 \
    --log_dir logs/firmware_v1 \
    --save_dir checkpoints/firmware_v1 \
    --mixed_precision \
    --gradient_clip 1.0 \
    --early_stopping_patience 15 \
    --lambda_physics 0.1
```

### Monitor Training

#### Terminal Output

```
Epoch 1/100
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
    Batch  20/156 | Loss: 0.023456 | X-Err: 0.012345 | Y-Err: 0.011234
    Batch  40/156 | Loss: 0.021234 | X-Err: 0.011234 | Y-Err: 0.010123
    ...
  ▶ Train Loss: 0.019876 | X-Err: 0.010234 | Y-Err: 0.009876
  ▶ Val Loss: 0.018765 | X-Err: 0.009876 | Y-Err: 0.009234
  ✓ Saved checkpoint: checkpoints/firmware_v1/best_model.pt
```

#### TensorBoard

```bash
tensorboard --logdir logs/firmware_v1 --port 6006
# Open http://localhost:6006
```

**Monitored metrics**:
- Training/validation loss curves
- X/Y axis error over time
- Learning rate schedule
- Gradient distribution

---

## Stage 4: Model Evaluation

### Basic Evaluation

```bash
python experiments/evaluate_trajectory_model.py \
    --checkpoint checkpoints/firmware_v1/best_model.pt \
    --data_root data/processed/test
```

### Comprehensive Evaluation

```bash
python experiments/evaluate_trajectory_model.py \
    --checkpoint checkpoints/firmware_v1/best_model.pt \
    --data_root data/processed/test \
    --output_dir results/evaluation \
    --visualize \
    --save_predictions \
    --generate_report
```

### Evaluation Output

**Console output**:
```
===========================================
Trajectory Error Model Evaluation
===========================================
Dataset: Test (5234 samples)

Overall Metrics:
  MAE:  0.0156 mm
  RMSE: 0.0223 mm
  R²:   0.8923

Per-Axis Metrics:
  X-Axis:
    MAE:  0.0145 mm
    RMSE: 0.0201 mm
    R²:   0.9012

  Y-Axis:
    MAE:  0.0167 mm
    RMSE: 0.0245 mm
    R²:   0.8834

Percentile Errors:
  50th (Median): 0.0123 mm
  90th:         0.0345 mm
  95th:         0.0456 mm
  99th:         0.0678 mm

✓ Model meets performance targets!
===========================================
```

**Generated files**:
```
results/evaluation/
├── metrics.json
├── trajectory_comparison.png
├── error_distribution.png
├── error_correlation.png
├── per_axis_error.png
└── sequence_prediction.png
```

---

## Stage 5: Model Deployment

### Export for Inference

```python
from models.trajectory import TrajectoryErrorTransformer
import torch

# Load best model
model = TrajectoryErrorTransformer.load_from_checkpoint(
    'checkpoints/firmware_v1/best_model.pt'
)
model.eval()

# TorchScript compilation for deployment
scripted_model = torch.jit.script(model)
scripted_model.save('models/trajectory_error_model.pt')

# Or ONNX export
torch.onnx.export(
    model,
    dummy_input,
    'models/trajectory_error_model.onnx',
    input_names=['input_features'],
    output_names=['error_x', 'error_y'],
    dynamic_axes={
        'input_features': {0: 'batch_size'},
        'error_x': {0: 'batch_size'},
        'error_y': {0: 'batch_size'}
    }
)
```

### Real-Time Inference Example

```python
import torch

# Load compiled model
model = torch.jit.load('models/trajectory_error_model.pt')
model.eval()

# Prepare input (single sample)
input_features = prepare_trajectory_features(gcode_line)
input_tensor = torch.from_numpy(input_features).unsqueeze(0).float()

# Predict
with torch.no_grad():
    start_time = time.time()
    predictions = model(input_tensor)
    inference_time = (time.time() - start_time) * 1000  # ms

# Extract errors
error_x = predictions['error_x'][0].numpy()  # [sequence_length]
error_y = predictions['error_y'][0].numpy()

# Apply correction
corrected_gcode = apply_correction(original_gcode, error_x, error_y)

print(f"Inference time: {inference_time:.2f} ms")
```

---

## Hyperparameter Tuning

### Grid Search

```bash
# Test different learning rates
for lr in 1e-5 1e-4 1e-3; do
    python train_trajectory_model.py --lr $lr --exp_name lr_${lr}
done

# Test different batch sizes
for bs in 16 32 64; do
    python train_trajectory_model.py --batch_size $bs --exp_name bs_${bs}
done
```

### Bayesian Optimization

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout = trial.suggest_uniform('dropout', 0.0, 0.3)

    # Train model
    metrics = train_model(lr, batch_size, dropout)

    return metrics['val_loss']

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print(f'Best trial: {study.best_trial.params}')
```

---

## Common Issues and Solutions

### Issue 1: CUDA Out of Memory

**Solution**:
```bash
# Reduce batch size
--batch_size 16  # or 8

# Or reduce sequence length
# In prepare_training_data.py: --sequence_length 64
```

### Issue 2: Model Not Converging

**Solution**:
```bash
# Lower learning rate
--lr 5e-5  # or 1e-5

# Enable learning rate finder
--lr_find

# Increase model capacity
--d_model 512  # or --num_layers 8
```

### Issue 3: Overfitting

**Solution**:
```bash
# Increase dropout
--dropout 0.2  # or 0.3

# Add data augmentation
--augment_noise --noise_level 0.01

# Add weight decay
--weight_decay 1e-3
```

### Issue 4: Underfitting

**Solution**:
```bash
# Increase model size
--d_model 512 --num_layers 8

# Train longer
--epochs 200

# Reduce regularization
--dropout 0.05 --weight_decay 1e-5
```

---

## Training Time Estimates

| Hardware | Batch Size | Per Epoch | Total (100 epochs) |
|----------|-----------|-----------|-------------------|
| CPU (i7-8700K) | 16 | ~15 min | ~25 hours |
| GPU (GTX 1080) | 32 | ~3 min | ~5 hours |
| GPU (RTX 3080) | 64 | ~1 min | ~2 hours |

**Note**: Actual time depends on dataset size and sequence length.

---

## Complete Workflow Example

```bash
#!/bin/bash
# complete_training_pipeline.sh

# Stage 1: Collect data (MATLAB)
echo "=== Stage 1: Data Collection ==="
matlab -batch "collect_all"

# Stage 2: Prepare data
echo "=== Stage 2: Data Preparation ==="
python data/scripts/prepare_training_data.py \
    --data_dirs data_simulation_* \
    --output_dir data/processed

# Stage 3: Train model
echo "=== Stage 3: Model Training ==="
python experiments/train_trajectory_model.py \
    --data_root data/processed \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --mixed_precision \
    --experiment_name production_model

# Stage 4: Evaluate model
echo "=== Stage 4: Model Evaluation ==="
python experiments/evaluate_trajectory_model.py \
    --checkpoint checkpoints/production_model/best_model.pt \
    --data_root data/processed/test \
    --visualize \
    --save_predictions

# Stage 5: Export model
echo "=== Stage 5: Model Export ==="
python experiments/export_model.py \
    --checkpoint checkpoints/production_model/best_model.pt \
    --output_dir models/ \
    --format torchscript

echo "=== Pipeline Complete ==="
```

---

## References

**See Also**:
- [Neural Network](neural_network.md) - Model architecture
- [Data Generation](data_generation.md) - Data collection
- [Experiments/Metrics](../experiments/metrics.md) - Evaluation metrics

**Related Documents**:
- [Previous]: [Neural Network](neural_network.md)
- [See Also]: [Experiments/Setup](../experiments/setup.md)
