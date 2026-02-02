# Simulation System Architecture

**Purpose**: MATLAB-based high-fidelity simulation system for FDM 3D printer trajectory errors.

---

## System Overview

```
G-code file
    ↓
[Trajectory Reconstruction] → Dense time series (x,y,z,v,a,j)
    ↓
[Dynamics Simulation] → Trajectory errors (error_x, error_y, F_inertia, F_elastic)
    ↓
[Thermal Field Simulation] → Temperature field (T_nozzle, T_interface, T_surface)
    ↓
[Adhesion Strength Calculation] → Bonding quality (adhesion_strength)
    ↓
[Data Fusion] → Complete training dataset
```

---

## Module 1: Trajectory Reconstruction

**Function**: `reconstruct_trajectory(gcode_file, params)`

### Purpose
Reconstruct the actual nozzle trajectory from G-code waypoints by modeling the firmware motion planner.

### Algorithm

1. Parse G-code to extract waypoints
2. Apply motion constraints (v_max, a_max, j_max)
3. Generate velocity profile (trapezoidal or S-curve)
4. Integrate to get position over time
5. Output dense time series {t, r, v, a, j}

### S-Curve Velocity Profile

For smooth motion:

$$
v(t) = \begin{cases}
\frac{1}{2}j_{\max}t^2 & 0 \leq t < t_1 \\
v_1 + a_{\max}(t-t_1) & t_1 \leq t < t_2 \\
v_{\max} - \frac{1}{2}j_{\max}(t_{\text{acc}}-t)^2 & t_2 \leq t < t_{\text{acc}} \\
v_{\max} & t_{\text{acc}} \leq t < t_{\text{dec}} \\
\text{(symmetric deceleration)}
\end{cases}
$$

### Output Format

```matlab
trajectory_data.time           % Time vector [s]
trajectory_data.x_ref          % Reference X [mm]
trajectory_data.y_ref          % Reference Y [mm]
trajectory_data.z_ref          % Reference Z [mm]
trajectory_data.vx             % X velocity [mm/s]
trajectory_data.vy             % Y velocity [mm/s]
trajectory_data.vz             % Z velocity [mm/s]
trajectory_data.ax             % X acceleration [mm/s²]
trajectory_data.ay             % Y acceleration [mm/s²]
trajectory_data.az             % Z acceleration [mm/s²]
trajectory_data.jx             % X jerk [mm/s³]
trajectory_data.jy             % Y jerk [mm/s³]
trajectory_data.jz             % Z jerk [mm/s³]
```

---

## Module 2: Dynamics Simulation

**Function**: `simulate_trajectory_error(trajectory_data, params)`

### Purpose
Simulate the second-order mass-spring-damper system to predict trajectory errors.

### State Space Model

$$
\frac{d}{dt}\begin{bmatrix} x \\ \dot{x} \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ -\omega_n^2 & -2\zeta\omega_n \end{bmatrix} \begin{bmatrix} x \\ \dot{x} \end{bmatrix} + \begin{bmatrix} 0 \\ -1 \end{bmatrix} a_{\text{ref}}(t)
$$

### Numerical Method
RK4 integration with time step Δt = 0.01 s (100 Hz sampling)

### Output Format

```matlab
error_data.x_actual            % Actual X position [mm]
error_data.y_actual            % Actual Y position [mm]
error_data.error_x             % X error [mm]
error_data.error_y             % Y error [mm]
error_data.error_mag           % Error magnitude [mm]
error_data.F_inertia_x         % Inertial force X [N]
error_data.F_inertia_y         % Inertial force Y [N]
error_data.F_elastic_x         % Elastic force X [N]
error_data.F_elastic_y         % Elastic force Y [N]
```

---

## Module 3: Thermal Field Simulation

**Function**: `simulate_thermal_field(trajectory_data, params)`

### Purpose
Model the moving heat source and temperature evolution during printing.

### Moving Heat Source Model

$$T(x,y,t) = T_0 + \frac{Q}{2\pi k r} \exp\left(-\frac{r^2}{4\alpha t}\right)$$

where:
- $Q$ = heat input [W]
- $r$ = radial distance from nozzle [m]
- $\alpha$ = thermal diffusivity [m²/s]

### Simplified Point Tracking

Instead of full 3D solution, track nozzle position temperature:

1. **Nozzle heating phase**:
   $$T_{\text{after}} = T_{\text{prev}} + (T_{\text{nozzle}} - T_{\text{prev}})(1 - e^{-t/\tau_{\text{heat}}})$$

2. **Cooling phase** (Newton's law):
   $$T(t) = T_{\text{amb}} + (T_0 - T_{\text{amb}})e^{-t/\tau_{\text{cool}}}$$

3. **Layer accumulation** (weighted sum of recent layers):
   $$T_n = 0.7 \times T_{\text{cool}}(n) + 0.3 \times (0.5T_{n-1} + 0.3T_{n-2} + 0.2T_{n-3})$$

### Output Format

```matlab
thermal_data.T_nozzle           % Nozzle temperature [°C]
thermal_data.T_interface        % Interface temperature [°C]
thermal_data.T_surface          % Surface temperature [°C]
thermal_data.cooling_rate       % Cooling rate [°C/s]
thermal_data.T_gradient_z       % Z-gradient [°C/mm]
```

---

## Module 4: Adhesion Strength Calculation

**Function**: `calculate_adhesion_strength(thermal_data, params)`

### Purpose
Predict interlayer bonding strength using Wool-O'Connor model.

### Healing Model

$$\frac{\sigma}{\sigma_{\text{bulk}}} = 1 - \exp\left(-\frac{t}{\tau(T)}\right)$$

$$\tau(T) = \tau_0 \exp\left(\frac{E_a}{RT}\right)$$

### Implementation

```matlab
function strength = calculate_adhesion_strength(T_interface, t_layer, params)
    R = 8.314;  % J/(mol·K)
    T_K = T_interface + 273.15;
    tau = params.tau0 * exp(params.Ea / (R * T_K));
    strength = params.sigma_bulk * (1 - exp(-t_layer / tau));
end
```

### Output Format

```matlab
adhesion_data.strength_ratio    % Adhesion/bulk strength ratio [0-1]
adhesion_data.strength          % Absolute strength [MPa]
adhesion_data.T_effective       % Effective temperature [°C]
adhesion_data.healing_ratio     % Degree of healing [0-1]
```

---

## Data Integration

**Function**: `combine_results(trajectory_data, error_data, thermal_data, adhesion_data)`

### Output .mat File

```matlab
simulation_data = struct();
simulation_data.time = trajectory_data.time;
simulation_data.trajectory = trajectory_data;
simulation_data.error = error_data;
simulation_data.thermal = thermal_data;
simulation_data.adhesion = adhesion_data;
simulation_data.params = params;
```

---

## GPU Acceleration

### When to Use GPU
- Data points N > 1000
- Matrix operations dominate
- Memory transfer overhead acceptable

### Implementation

```matlab
% Setup GPU
gpuDevice(1);

% Transfer data to GPU
x_gpu = gpuArray(x_data);

% Vectorized operations
result = arrayfun(@compute_dynamics, x_gpu);

% Transfer back
result = gather(result);
```

### Performance Improvement
- 10K points: 4× speedup
- 100K points: 13× speedup

---

## Usage Example

```matlab
% Load parameters
params = physics_parameters();

% Run simulation
simulation_data = run_simulation('test_gcode_files/3DBenchy.gcode', ...
    'OutputFile', 'data_simulation_3DBenchy', ...
    'Layers', 1:50, ...
    'UseGPU', true, ...
    'UseFirmwareEffects', true);

% Access results
disp(['Mean error X: ' num2str(mean(simulation_data.error.error_x)) ' mm']);
disp(['Adhesion strength: ' num2str(mean(simulation_data.adhesion.strength_ratio))]);
```

---

## File Structure

```
simulation/
├── run_simulation.m                   % Main entry point
├── parse_gcode.m                      % G-code parser
├── reconstruct_trajectory.m            % Trajectory reconstruction
├── simulate_trajectory_error.m         % CPU dynamics
├── simulate_trajectory_error_gpu.m    % GPU dynamics
├── simulate_thermal_field.m           % Thermal simulation
├── calculate_adhesion_strength.m      % Adhesion calculation
├── combine_results.m                  % Data fusion
├── physics_parameters.m               % Parameter configuration
└── collect_data.m                     % Data collection wrapper
```

---

## References

**See Also**:
- [Firmware Effects](firmware_effects.md) - Junction deviation and resonance
- [Data Generation](data_generation.md) - Data collection strategy
- [Trajectory Dynamics](../theory/trajectory_dynamics.md) - Theory

**Related Documents**:
- [Next]: [Firmware Effects](firmware_effects.md)
- [Previous]: [Data Generation](data_generation.md)
