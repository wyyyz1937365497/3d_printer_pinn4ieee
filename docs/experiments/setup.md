# Experimental Setup

**Purpose**: Printer configuration, test models, and data collection methodology.

---

## Printer Configuration

### Ender-3 V2 Specifications

| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| **Build Volume** | 220 × 220 × 250 | mm | Official |
| **Nozzle Diameter** | 0.4 | mm | Standard |
| **Layer Height** | 0.2 | mm | Slicer setting |
| **Filament Diameter** | 1.75 | mm | Standard |

### Motion System

| Parameter | X-Axis | Y-Axis | Unit | Source |
|-----------|--------|--------|------|--------|
| Moving mass | 0.485 | 0.650 | kg | Measured |
| Belt type | GT2 6mm | GT2 6mm | - | Official |
| Pulley teeth | 20 | 20 | - | Official |
| Stiffness | 150,000 | 150,000 | N/m | Experimental |
| Damping | 25 | 25 | N·s/m | Estimated |

### Firmware Configuration (Marlin)

| Parameter | Value | Unit |
|-----------|-------|------|
| Max velocity | 500 | mm/s |
| Max acceleration | 500 | mm/s² |
| Jerk limit | 8-10 | mm/s³ |
| Junction deviation | 0.013 | mm |
| Microstepping | 16 | - |

---

## Material: PLA

### Mechanical Properties

| Parameter | Value | Unit |
|-----------|-------|------|
| Elastic modulus | 3.5 | GPa |
| Tensile strength | 60-70 | MPa |
| Yield strength | ~60 | MPa |

---

## Test Models

### Model 1: 3DBenchy

**Purpose**: Comprehensive geometric features test

**Statistics**:
- Layers: 48
- Print time: ~1h 28m
- Features: Overhangs, bridges, holes, curves, text

**File**: `test_gcode_files/3DBenchy_PLA_1h28m.gcode`

### Model 2: Bearing5

**Purpose**: Circular and geometric accuracy test

**Statistics**:
- Layers: ~50
- Print time: ~1h 15m
- Features: Concentric circles, smooth curves

**File**: `test_gcode_files/Bearing5_1h15m.gcode`

### Model 3: Nautilus

**Purpose**: Complex spiral structure

**Statistics**:
- Layers: ~50
- Print time: ~1h 20m
- Features: Spiral, varying layer height

**File**: `test_gcode_files/Nautilus_1h20m.gcode`

### Model 4: Boat

**Purpose**: Large flat surface test

**Statistics**:
- Layers: ~50
- Print time: ~1h 30m
- Features: Hull, deck, flat surfaces

**File**: `test_gcode_files/Boat_1h30m.gcode`

---

## Data Collection Methodology

### Layer Selection Strategy

#### Strategy 1: Uniform Sampling (Primary)

Sample every Nth layer to capture diversity without redundancy:
- `sampled:5` - Every 5th layer (recommended)
- `sampled:10` - Every 10th layer (quick validation)
- `all` - All layers (complete dataset, not recommended)

**Example**:
```matlab
collect_3dbenchy('sampled:5');  % 10 layers from 48
```

#### Strategy 2: Representative Layers

Select specific layers representing different regions:
- Layer 1: First layer (different dynamics, lower speeds)
- Layer 25: Middle layer (typical conditions)
- Layer 48: Last layer (minimal support, higher vibration)

### Data Collection Commands

**Single model**:
```matlab
collect_3dbenchy('sampled:5');
```

**All models**:
```matlab
collect_all  % Collects all 4 models with 'sampled:5'
```

**Custom range**:
```matlab
collect_data('model.gcode', 1:25, ...
    'GPU', 1, ...
    'UseFirmwareEffects', true);
```

---

## G-code Processing

### Trajectory Reconstruction

From G-code waypoints to dense time series:
1. Parse G-code commands (G0, G1)
2. Apply motion constraints (v_max, a_max, j_max)
3. Generate velocity profile (S-curve)
4. Integrate position → velocity → acceleration → jerk
5. Output at Δt = 0.01s (100 Hz sampling)

### Output Time Series

Each layer generates:
- Time vector: ~2000-3000 points
- Position: x, y, z [mm]
- Velocity: vx, vy, vz [mm/s]
- Acceleration: ax, ay, az [mm/s²]
- Jerk: jx, jy, jz [mm/s³]

---

## Firmware Effects

### Enabled Features

All data collection uses **firmware-enhanced simulation**:
- ✅ Junction deviation
- ✅ Microstep resonance
- ✅ Timer jitter

**Result**: RMS error ≈ 140 μm (matches real printer)

### Validation

```matlab
% Test firmware effects
test_firmware_effects_simple

% Expected output:
%   RMS error: 140.5 μm ✅
%   Within target range (80-150 μm) ✅
```

---

## Data Quality Assurance

### Error Range Validation

After simulation, verify:

```python
check_error_ranges(simulation_data)
```

**Expected ranges**:
- Mean error: 0.08-0.15 mm (80-150 μm)
- Max error: < 0.5 mm

### Statistical Validation

Check distribution:

```
Error Distribution:
  < 50 μm:      2.6%
  50-100 μm:   52.8%  ← Main peak
  100-150 μm:  33.4%  ← Secondary peak
  > 150 μm:    11.3%

✓ 86.2% in target range (50-150 μm)
```

---

## Computational Resources

### Hardware Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8 GB
- GPU: None (CPU mode, 10× slower)
- Storage: 10 GB

**Recommended**:
- CPU: 8 cores
- RAM: 16 GB
- GPU: GTX 1080 or better (8 GB VRAM)
- Storage: 20 GB SSD

**Ideal**:
- CPU: 16 cores
- RAM: 32 GB
- GPU: RTX 3080 or better (10+ GB VRAM)
- Storage: 50 GB NVMe SSD

### Software Requirements

- **MATLAB R2023b** (or later)
  - Parallel Computing Toolbox (optional, for GPU)
- **Python 3.8+**
  - PyTorch 1.10+
  - NumPy, SciPy, Pandas
  - Matplotlib (visualization)
  - HDF5 (data storage)

---

## Environment Setup

### MATLAB Setup

```matlab
% Add simulation to path
addpath('simulation')

% Configure GPU (if available)
gpuDevice(1);

% Test simulation
test_firmware_effects_simple
```

### Python Setup

```bash
# Create conda environment
conda create -n 3dprint python=3.10
conda activate 3dprint

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy scipy pandas
pip install matplotlib seaborn
pip install h5py
pip install tensorboard

# Verify installation
python -c "import torch; print(torch.__version__)"
```

---

## Data Collection Checklist

### Before Starting

- [ ] G-code files in `test_gcode_files/`
- [ ] MATLAB configured with simulation in path
- [ ] GPU available (check `gpuDevice`)
- [ ] Sufficient disk space (> 10 GB)
- [ ] Firmware effects tested and validated

### During Collection

- [ ] Monitor simulation progress
- [ ] Check intermediate results (first few layers)
- [ ] Verify error ranges in target (80-150 μm RMS)
- [ ] Check GPU memory usage

### After Collection

- [ ] Verify all `.mat` files created
- [ ] Check file sizes (typically 1-5 MB per layer)
- [ ] Run quality checks on sample data
- [ ] Backup data before proceeding to training

---

## Common Issues

### Issue 1: Simulation Too Slow

**Symptoms**: Single layer takes > 5 minutes

**Solutions**:
1. Enable GPU: `collect_data(..., 'GPU', 1)`
2. Reduce sequence length: `trajectory_options.time_step = 0.02`
3. Use fewer layers: `'sampled:10'` instead of `'sampled:5'`

### Issue 2: Error Range Incorrect

**Symptoms**: RMS error < 50 μm or > 200 μm

**Solutions**:
1. Verify firmware effects enabled: `'UseFirmwareEffects', true`
2. Check parameters: `physics_parameters()`
3. Test with `test_firmware_effects_simple`

### Issue 3: Out of Memory

**Symptoms**: MATLAB/Python crashes or becomes unresponsive

**Solutions**:
1. Reduce layers: Fewer simultaneous simulations
2. Reduce batch size: For Python training
3. Close other applications

---

## References

**See Also**:
- [Simulation System](../methods/simulation_system.md) - How simulation works
- [Firmware Effects](../methods/firmware_effects.md) - Error sources
- [Data Generation](../methods/data_generation.md) - Collection strategy

**Related Documents**:
- [Next]: [Datasets](datasets.md)
- [Previous]: [Simulation System](../methods/simulation_system.md)
