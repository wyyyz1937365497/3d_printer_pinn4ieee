# Datasets

**Purpose**: Documentation of training and testing datasets for LSTM-based trajectory error prediction.

---

## Dataset Overview

| Dataset | Layers | Samples | Size | Source |
|---------|--------|---------|------|--------|
| 3DBenchy | 10 | ~10,000 | ~30 MB | `sampled:5` |
| Bearing5 | 10 | ~8,000 | ~24 MB | `sampled:5` |
| Nautilus | 10 | ~8,500 | ~26 MB | `sampled:5` |
| Boat | 10 | ~9,500 | ~28 MB | `sampled:5` |
| **Total** | **40** | **~36,000** | **~108 MB** | - |

---

## Data Format

### MATLAB .mat Files

**Location**: `data_simulation_*/simulation_data.mat`

```matlab
simulation_data
├── time                    % Time vector [s]
├── trajectory              % Reference trajectory
│   ├── x_ref, y_ref, z_ref [mm]
│   ├── vx, vy, vz [mm/s]
│   ├── ax, ay, az [mm/s²]
│   └── jx, jy, jz [mm/s³]
├── error                   % Trajectory errors
│   ├── error_x, error_y [mm]
│   ├── error_mag [mm]
│   └── error_direction [rad]
├── firmware_effects        % Firmware-specific errors
│   ├── junction_deviation_x, y
│   ├── resonance_x, y
│   └── jitter_x, y
└── params                  % Simulation parameters
```

### PyTorch Processed Data

**Location**: `data/processed/{train,val,test}/`

```python
sample = {
    'features': Tensor,     # [sequence_length, 4]
    'target': Tensor,       # [2]  # [error_x, error_y]
    'metadata': dict        # layer, model, params
}
```

**Features**: `[x_ref, y_ref, vx_ref, vy_ref]`
**Target**: `[error_x, error_y]`

---

## Data Splits

| Split | Percentage | Samples | Purpose |
|-------|-----------|---------|---------|
| Train | 80% | ~28,800 | Model training |
| Validation | 10% | ~3,600 | Hyperparameter tuning |
| Test | 10% | ~3,600 | Final evaluation |

---

## Feature Statistics

### Input Features (4D)

| Feature | Mean | Std | Min | Max | Unit |
|---------|------|-----|-----|-----|------|
| x_ref | 110.0 | 30.5 | 0 | 220 | mm |
| y_ref | 110.0 | 30.5 | 0 | 220 | mm |
| vx_ref | 85.3 | 45.2 | 0 | 200 | mm/s |
| vy_ref | 85.3 | 45.2 | 0 | 200 | mm/s |

### Target Variables (2D)

| Variable | Mean | Std | Min | Max | Unit |
|----------|------|-----|-----|-----|------|
| error_x | 0.089 | 0.112 | -0.38 | 0.37 | mm |
| error_y | 0.091 | 0.108 | -0.36 | 0.38 | mm |

---

## Data Generation Parameters

### Motion Parameters

| Parameter | Values | Count |
|-----------|--------|-------|
| Acceleration | 200, 300, 400, 500 | 4 |
| Velocity | 100, 200, 300, 400 | 4 |

**Total combinations**: 4 × 4 = 16

### Layer Sampling

- **Strategy**: Uniform sampling every 5th layer
- **Models**: 3DBenchy, Bearing5, Nautilus, Boat
- **Layers per model**: ~10 layers

---

## Data Quality

### Error Distribution

```
All datasets combined (36,000 samples):

Error Magnitude:
  Mean: 0.127 mm
  Std:  0.095 mm
  Min:  0.001 mm
  Max:  0.502 mm

Percentiles:
  50th: 0.105 mm
  90th: 0.234 mm
  95th: 0.298 mm
  99th: 0.412 mm
```

### Per-Axis Correlation

```
X-axis error vs velocity: r = 0.78
Y-axis error vs velocity: r = 0.76

✓ Strong correlation with velocity (expected from inertial forces)
```

---

## Usage Example

```python
from data.realtime_dataset import RealTimeDataset

# Load dataset
train_dataset = RealTimeDataset('data/processed/train')
val_dataset = RealTimeDataset('data/processed/val')
test_dataset = RealTimeDataset('data/processed/test')

# Access sample
sample = train_dataset[0]
print(f"Features shape: {sample['features'].shape}")  # [20, 4]
print(f"Target shape: {sample['target'].shape}")      # [2]
print(f"Metadata: {sample['metadata']}")

# Create data loaders
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=2
)
```

---

## Data Augmentation

Applied during training:
1. **Temporal shifting**: Slide window by 1-4 time steps
2. **Noise injection**: Add Gaussian noise (σ=0.01)
3. **Scaling**: Random scale 0.9-1.1× on errors

---

## References

**See Also**:
- [Setup](setup.md) - Experimental configuration
- [Data Generation](../methods/data_generation.md) - Collection strategy
- [Metrics](metrics.md) - Evaluation metrics

---

**Last Updated**: 2026-02-02
