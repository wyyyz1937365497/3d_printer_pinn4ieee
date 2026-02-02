# Data Generation Strategy

**Purpose**: Strategy for generating high-quality training data using firmware-enhanced simulation.

---

## Overview

The goal is to generate **large-scale, high-fidelity, fully-annotated** training data for trajectory error prediction networks. This document describes the optimized data generation strategy.

---

## Data Generation Philosophy

### Key Principle

**Quality over quantity**: Focus on representative layers with diverse parameter configurations rather than simulating all layers.

### Optimization Strategy

Instead of simulating all layers of each model:
1. Sample representative layers (every Nth layer)
2. Diverse parameter configurations per sampled layer
3. Data augmentation for additional diversity

**Result**: 30-40× more samples with 10× less computation time

---

## Layer Selection Strategy

### Strategy 1: Uniform Sampling (Recommended)

Sample every Nth layer:
- N = 5: ~10 layers from 48-layer model
- N = 10: ~5 layers from 48-layer model
- N = 20: ~2-3 layers from 48-layer model

**Usage**:
```matlab
collect_3dbenchy('sampled:5');   % Every 5th layer
collect_bearing5('sampled:5');
```

### Strategy 2: Representative Layers

Select layers representing different regions:
- Layer 1: First layer (bed adhesion)
- Layer 25: Middle layer (typical)
- Layer 48: Last layer (minimal support)

**Usage**:
```matlab
collect_data('model.gcode', [1, 25, 48], ...);
```

### Strategy 3: All Layers (Not Recommended)

Simulate every layer:
- Pros: Maximum data
- Cons: Computationally expensive (6-8 hours per model)

**Usage**:
```matlab
collect_3dbenchy('all');
```

---

## Parameter Configuration

### Base Configuration (Ender-3 V2)

| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| Max velocity | 500 | mm/s | Firmware |
| Max acceleration | 500 | mm/s² | Firmware |
| Jerk limit | 8-10 | mm/s³ | Firmware |
| Nozzle temp | 220 | °C | PLA |
| Bed temp | 60 | °C | PLA |
| Layer height | 0.2 | mm | Slicer |

### Parameter Sweep Strategy

For sampled layers, vary parameters to increase diversity:

| Parameter | Values | Count |
|-----------|--------|-------|
| Acceleration | 200, 300, 400, 500 | 4 |
| Velocity | 100, 200, 300, 400 | 4 |
| Fan speed | 0, 128, 255 | 3 |
| Ambient temp | 20, 25, 30 | 3 |

**Total combinations**: 4 × 4 × 3 × 3 = 144

**Implementation**:
```matlab
% Create parameter grid
accels = [200, 300, 400, 500];
velocities = [100, 200, 300, 400];
fans = [0, 128, 255];
temps = [20, 25, 30];

% Loop over all combinations
for a = accels
    for v = velocities
        for f = fans
            for t = temps
                params = physics_parameters();
                params.motion.max_accel = a;
                params.motion.max_velocity = v;
                params.heat_transfer.fan_speed = f;
                params.environment.ambient_temp = t;

                run_simulation('model.gcode', params);
            end
        end
    end
end
```

---

## Recommended Data Collection Plans

### Plan 1: Quick Validation (~30 minutes)

**Purpose**: Initial model testing

**Configuration**:
- 3DBenchy: sampled every 10 layers (~5 layers)
- Bearing: sampled every 10 layers (~5 layers)
- Nautilus: sampled every 10 layers (~5 layers)

**Output**: ~15 layers × 2 minutes/layer = ~30 minutes

**Estimated samples**: ~15,000 points

### Plan 2: Standard Training (~2 hours)

**Purpose**: Standard model training

**Configuration**:
- 3DBenchy: sampled every 5 layers (~10 layers)
- Bearing: sampled every 5 layers (~10 layers)
- Nautilus: sampled every 5 layers (~10 layers)
- Boat: sampled every 5 layers (~10 layers)

**Output**: ~40 layers × 2 minutes/layer = ~80 minutes

**Estimated samples**: ~40,000 points

### Plan 3: Complete Dataset (~6 hours)

**Purpose**: Production model

**Configuration**:
- All models: all layers (48 per model)

**Output**: ~200 layers × 2 minutes/layer = ~400 minutes

**Estimated samples**: ~200,000 points

---

## Data Augmentation

### Temporal Shifting

Shift the sequence window:

```python
# Original: [t0, t1, ..., t127]
# Shifted: [t5, t6, ..., t132]
# Multiple shifts from same layer
```

### Noise Injection

Add small noise to features:

```python
noise = np.random.normal(0, 0.01, features.shape)
augmented_features = features + noise
```

### Scaling

Apply random scaling to error magnitudes:

```python
scale = np.random.uniform(0.9, 1.1)
augmented_errors = errors * scale
```

---

## Output Data Format

### MATLAB .mat File

```matlab
simulation_data
├── time                    % Time vector [s]
├── trajectory              % Reference trajectory
│   ├── x_ref, y_ref, z_ref
│   ├── vx, vy, vz
│   ├── ax, ay, az
│   └── jx, jy, jz
├── error                   % Trajectory errors
│   ├── error_x, error_y
│   ├── error_mag
│   └── error_direction
├── thermal                 % Temperature field
│   ├── T_nozzle
│   ├── T_interface
│   ├── T_surface
│   └── cooling_rate
├── firmware_effects        % Firmware-specific errors
│   ├── junction_deviation_x, y
│   ├── resonance_x, y
│   └── jitter_x, y
└── params                  % Simulation parameters
```

### Python Training Data

**HDF5 format** (recommended):

```python
import h5py

with h5py.File('trajectory_data.h5', 'w') as f:
    f.create_dataset('features', data=X)  # [N, sequence_length, n_features]
    f.create_dataset('targets', data=y)    # [N, sequence_length, 2]
    f.create_dataset('params', data=params)  # Simulation parameters
    f.create_dataset('metadata', data=metadata)  # Layer, model, etc.
```

---

## Data Quality Checks

### Error Range Validation

```python
def check_data_quality(data):
    """Validate simulation output"""
    checks = {
        'error_magnitude': (0.05, 0.3),  # Mean error should be 50-300 μm
        'error_max': (0, 0.5),            # Max error < 0.5 mm
        'temperature': (20, 220),         # Within ambient to nozzle
        'adhesion': (0, 1),               # Strength ratio [0, 1]
    }

    for key, (min_val, max_val) in checks.items():
        if key in data:
            mean_val = np.mean(data[key])
            assert min_val <= mean_val <= max_val, \
                f"{key}: {mean_val} outside range [{min_val}, {max_val}]"

    print("✓ Data quality checks passed")
```

### Statistical Validation

```
Dataset: 3DBenchy, Layer 25
Samples: 2800

Error Statistics:
  X: mean=0.089 mm, std=0.112 mm
  Y: mean=0.091 mm, std=0.108 mm
  Mag: mean=0.127 mm, std=0.095 mm

Temperature:
  Nozzle: 220.0 °C (constant)
  Interface: 68.3 ± 5.2 °C
  Surface: 45.1 ± 3.8 °C

Adhesion:
  Strength ratio: 0.82 ± 0.11

✓ All statistics within expected range
```

---

## Usage Examples

### Basic Collection

```matlab
% Collect single model
collect_3dbenchy('sampled:5');
```

### Batch Collection

```matlab
% Collect all models
collect_all  % Collects all 4 models with sampled:5
```

### Custom Collection

```matlab
% Specify exact parameters
collect_data('test.gcode', 1:25, ...
    'GPU', 1, ...
    'UseFirmwareEffects', true, ...
    'OutputFile', 'custom_output');
```

---

## Firmware Effects Enhancement

All data collection should use firmware-enhanced simulation for realistic error magnitudes:

```matlab
% Enable firmware effects
simulation_data = run_simulation('model.gcode', ...
    'UseFirmwareEffects', true);
```

**Expected output** (firmware-enhanced):
- RMS error: ~140 μm ✅
- Max error: 80-150 μm ✅
- Matches real printer errors ✅

**Without firmware effects**:
- RMS error: ~50-80 μm ❌
- Underestimates real errors ❌

---

## Performance Tips

### GPU Acceleration

Always use GPU when available:

```matlab
% Check GPU availability
gpuDevice

% Enable in collection
collect_data(..., 'GPU', 1);
```

**Speedup**: 10-13× for large datasets

### Parallel Processing

Run multiple MATLAB instances for different models:

```
Instance 1: collect_3dbenchy('sampled:5')
Instance 2: collect_bearing5('sampled:5')
Instance 3: collect_nautilus('sampled:5')
```

### Resume from Checkpoint

Skip already-completed layers:

```matlab
collect_data(..., 'Resume', true);
```

---

## Data Statistics

### Per-Layer Statistics

Typical layer from 3DBenchy (Layer 25):

| Metric | Value | Notes |
|--------|-------|-------|
| Duration | ~2 minutes | Simulation time |
| Sample points | ~2000-3000 | Δt = 0.01s |
| Error (RMS) | 0.12-0.15 mm | Firmware-enhanced |
| Temperature range | 45-85 °C | Interface |
| Adhesion ratio | 0.75-0.90 | Layer bonding |

### Dataset Size Estimation

| Plan | Layers | Points/Layer | Total Points | File Size |
|------|--------|--------------|--------------|-----------|
| Quick validation | 15 | 2500 | 37,500 | ~50 MB |
| Standard training | 40 | 2500 | 100,000 | ~130 MB |
| Complete dataset | 200 | 2500 | 500,000 | ~650 MB |

---

## References

**See Also**:
- [Simulation System](simulation_system.md) - How simulation works
- [Firmware Effects](firmware_effects.md) - Error source details
- [Experiments/Datasets](../experiments/datasets.md) - Dataset documentation

**Related Documents**:
- [Previous]: [Firmware Effects](firmware_effects.md)
- [Next]: [Neural Network](neural_network.md)
