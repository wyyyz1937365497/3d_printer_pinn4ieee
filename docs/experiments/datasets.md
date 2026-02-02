# Datasets

**Purpose**: Documentation of training and testing datasets.

---

## Dataset Overview

| Dataset | Layers | Samples | Size | Source |
|---------|--------|---------|------|--------|
| 3DBenchy | 10 | ~10,000 | ~50 MB | `sampled:5` |
| Bearing5 | 10 | ~8,000 | ~40 MB | `sampled:5` |
| Nautilus | 10 | ~8,500 | ~42 MB | `sampled:5` |
| Boat | 10 | ~9,500 | ~48 MB | `sampled:5` |
| **Total** | **40** | **~36,000** | **~180 MB** | - |

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
├── thermal                 % Temperature field
│   ├── T_nozzle [°C]
│   ├── T_interface [°C]
│   └── T_surface [°C]
└── params                  % Simulation parameters
```

### PyTorch Processed Data

**Location**: `data/processed/{train,val,test}/`

```python
sample = {
    'input_features': Tensor,     # [sequence_length, 15]
    'trajectory_targets': Tensor, # [sequence_length, 2]
    'thermal_targets': Tensor,    # [sequence_length, 3]
    'metadata': dict               # layer, model, params
}
```

---

## Data Splits

| Split | Percentage | Samples | Purpose |
|-------|-----------|---------|---------|
| Train | 80% | ~28,800 | Model training |
| Validation | 10% | ~3,600 | Hyperparameter tuning |
| Test | 10% | ~3,600 | Final evaluation |

---

## Feature Statistics

### Input Features (15D)

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| x [mm] | 110.0 | 30.5 | 0 | 220 |
| y [mm] | 110.0 | 30.5 | 0 | 220 |
| z [mm] | 5.0 | 2.8 | 0.2 | 10 |
| v [mm/s] | 85.3 | 45.2 | 0 | 200 |
| a [mm/s²] | 120.5 | 85.3 | -500 | 500 |
| j [mm/s³] | 8.5 | 15.2 | -50 | 50 |
| curvature | 0.05 | 0.12 | 0 | 0.8 |

### Target Variables (2D)

| Variable | Mean | Std | Min | Max |
|----------|------|-----|-----|-----|
| error_x [mm] | 0.089 | 0.112 | -0.38 | 0.37 |
| error_y [mm] | 0.091 | 0.108 | -0.36 | 0.38 |

---

## References

**See Also**:
- [Setup](setup.md) - Experimental configuration
- [Data Generation](../methods/data_generation.md) - Collection strategy
