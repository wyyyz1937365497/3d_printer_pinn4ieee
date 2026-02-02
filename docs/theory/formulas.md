# Physics Formulas and Symbols Library

**Purpose**: Complete collection of physical models, equations, and symbols for FDM 3D printing research.

---

## Table of Contents

1. [Symbols and Notation](#symbols-and-notation)
2. [Trajectory Dynamics](#trajectory-dynamics)
3. [Heat Transfer](#heat-transfer)
4. [Adhesion Model](#adhesion-model)
5. [Error Metrics](#error-metrics)
6. [Material Properties](#material-properties)
7. [Firmware Effects](#firmware-effects)
8. [LaTeX Code Examples](#latex-code-examples)

---

## Symbols and Notation

### Kinematic Symbols

| Symbol | Meaning | Units | Typical Value |
|--------|---------|-------|---------------|
| $x, y, z$ | Position | mm | 0-220 |
| $\mathbf{r}$ | Position vector | mm | - |
| $v$ | Velocity magnitude | mm/s | 0-200 |
| $\mathbf{v}$ | Velocity vector | mm/s | - |
| $a$ | Acceleration magnitude | mm/s² | -500 to 500 |
| $\mathbf{a}$ | Acceleration vector | mm/s² | - |
| $j$ | Jerk magnitude | mm/s³ | -50 to 50 |
| $\kappa$ | Curvature | mm⁻¹ | 0-0.8 |

### Dynamic Symbols

| Symbol | Meaning | Units | Typical Value (X/Y) |
|--------|---------|-------|---------------------|
| $m$ | Mass | kg | 0.485 / 0.650 |
| $c$ | Damping coefficient | N·s/m | 25 |
| $k$ | Stiffness | N/m | 150,000 |
| $\omega_n$ | Natural frequency | rad/s | 556 / 480 |
| $f_n$ | Natural frequency | Hz | 88.5 / 76.5 |
| $\zeta$ | Damping ratio | - | 0.046 / 0.040 |
| $F$ | Force | N | - |

### Thermal Symbols

| Symbol | Meaning | Units | Typical Value |
|--------|---------|-------|---------------|
| $T$ | Temperature | °C | 20-220 |
| $\rho$ | Density | kg/m³ | 1240 |
| $c_p$ | Specific heat | J/(kg·K) | 1200 |
| $k$ | Thermal conductivity | W/(m·K) | 0.13 |
| $\alpha$ | Thermal diffusivity | m²/s | 8.7×10⁻⁸ |
| $h$ | Convection coefficient | W/(m²·K) | 10-44 |
| $\tau$ | Time constant | s | 6.76 |

### Adhesion Symbols

| Symbol | Meaning | Units | Typical Value |
|--------|---------|-------|---------------|
| $\sigma$ | Strength | MPa | 0-70 |
| $E_a$ | Activation energy | J/mol | 50,000 |
| $R$ | Gas constant | J/(mol·K) | 8.314 |
| $T_g$ | Glass transition | °C | 60 |
| $T_m$ | Melting point | °C | 171 |
| $H$ | Healing ratio | - | 0-1 |

---

## Trajectory Dynamics

### Second-Order System Equation

**Newton's Second Law**:
$$m\ddot{x} + c\dot{x} + kx = F(t)$$

**With inertial forcing**:
$$m\ddot{x} + c\dot{x} + kx = -m a_{\text{ref}}(t)$$

where:
- $x$ = position error [mm]
- $a_{\text{ref}}$ = reference acceleration [mm/s²]

### Transfer Function

$$H(s) = \frac{X(s)}{A_{\text{ref}}(s)} = \frac{-1}{s^2 + 2\zeta\omega_n s + \omega_n^2}$$

**Natural frequency**:
$$\omega_n = \sqrt{\frac{k}{m}}$$

**Damping ratio**:
$$\zeta = \frac{c}{2\sqrt{mk}}$$

### Time Domain Response

**Underdamped step response** ($\zeta < 1$):
$$x(t) = 1 - \frac{e^{-\zeta\omega_n t}}{\sqrt{1-\zeta^2}} \sin\left(\omega_d t + \phi\right)$$

**Damped natural frequency**:
$$\omega_d = \omega_n\sqrt{1-\zeta^2}$$

**Phase angle**:
$$\phi = \arccos(\zeta)$$

### Response Characteristics

**Rise time**:
$$t_r \approx \frac{1.8}{\omega_n}$$

**Settling time** (2% criterion):
$$t_s \approx \frac{4}{\zeta\omega_n}$$

**Peak time**:
$$t_p = \frac{\pi}{\omega_d}$$

**Overshoot**:
$$M_p = \exp\left(-\frac{\pi\zeta}{\sqrt{1-\zeta^2}}\right)$$

### Forces

**Inertial force**:
$$F_{\text{inertia}}(t) = -m a_{\text{ref}}(t)$$

**Elastic force**:
$$F_{\text{elastic}}(t) = -k \Delta x(t)$$

**Belt stiffness**:
$$k_{\text{belt}} = \frac{EA}{L}$$

where:
- $E$ = Young's modulus [Pa]
- $A$ = cross-sectional area [m²]
- $L$ = belt length [m]

### State Space Form

$$\frac{d}{dt}\begin{bmatrix} x \\ \dot{x} \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ -\omega_n^2 & -2\zeta\omega_n \end{bmatrix} \begin{bmatrix} x \\ \dot{x} \end{bmatrix} + \begin{bmatrix} 0 \\ -1 \end{bmatrix} a_{\text{ref}}(t)$$

Or compactly:
$$\dot{\mathbf{z}} = A\mathbf{z} + B\mathbf{u}(t)$$

### Error Calculations

**Error vector**:
$$\mathbf{e}(t) = \mathbf{r}_{\text{actual}}(t) - \mathbf{r}_{\text{ref}}(t) = [e_x(t), e_y(t)]^T$$

**Error magnitude**:
$$e_{\text{mag}}(t) = \sqrt{e_x(t)^2 + e_y(t)^2}$$

**Error direction**:
$$\theta_e(t) = \arctan2(e_y(t), e_x(t))$$

---

## Heat Transfer

### Heat Conduction Equation

**3D transient**:
$$\rho c_p \frac{\partial T}{\partial t} = k \nabla^2 T + \dot{q}_{\text{source}} - \dot{q}_{\text{cooling}}$$

**In Cartesian coordinates**:
$$\frac{\partial T}{\partial t} = \alpha\left(\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} + \frac{\partial^2 T}{\partial z^2}\right)$$

### Thermal Diffusivity

$$\alpha = \frac{k}{\rho c_p}$$

### Convection

**Newton's Law of Cooling**:
$$q_{\text{conv}} = h(T_{\text{surface}} - T_{\text{amb}})$$

**Cooling equation**:
$$\frac{dT}{dt} = -\frac{hA}{mc_p}(T - T_{\text{amb}})$$

**Solution**:
$$T(t) = T_{\text{amb}} + (T_0 - T_{\text{amb}})e^{-t/\tau}$$

where:
$$\tau = \frac{mc_p}{hA}$$

### Moving Heat Source

**Gaussian heat source**:
$$Q(x, y, z, t) = Q_0 \exp\left(-\frac{(x-x_0)^2 + (y-y_0)^2}{2\sigma^2}\right) \delta(z - z_0)$$

**Analytical solution** (point source):
$$T(x, y, t) = T_{\text{amb}} + \frac{Q_0}{2\pi k r} \exp\left(-\frac{r^2}{4\alpha t}\right)$$

where $r = \sqrt{(x-x_0)^2 + (y-y_0)^2}$

### Simplified Point Tracking

**Phase 1: Heating**:
$$\Delta T_{\text{heating}} = (T_{\text{nozzle}} - T_{\text{prev}}) \left(1 - e^{-t_{\text{print}}/\tau_{\text{heating}}}\right) e^{-n/20}$$

**Phase 2: Cooling**:
$$T_{\text{after cooling}} = T_{\text{amb}} + (T_{\text{after printing}} - T_{\text{amb}}) e^{-\Delta t/\tau_{\text{cooling}}}$$

**Phase 3: Diffusion from below**:
$$T_{\text{from below}} = 0.5 T(n-1) + 0.3 T(n-2) + 0.2 T(n-3)$$

**Final temperature**:
$$T_n = 0.7 \times T_{\text{after cooling}}(n) + 0.3 \times T_{\text{from below}}(n)$$

### Time Constants

**Heating time constant**:
$$\tau_{\text{heating}} = \frac{\rho c_p h_{\text{layer}}}{h_{\text{conv}}}$$

**Cooling time constant**:
$$\tau_{\text{cooling}} = \frac{\rho c_p}{h_{\text{conv}} (A/V)} = \frac{\rho c_p h_{\text{layer}}}{h_{\text{conv}}}$$

For thin layers, $\tau_{\text{heating}} \approx \tau_{\text{cooling}}$.

### Interface Temperature

$$T_{\text{interface}} = \frac{T_{\text{new}} + T_{\text{old}}}{2}$$

---

## Adhesion Model

### Wool-O'Connor Polymer Healing Model

**Healing ratio**:
$$H = H_\infty \exp\left(-\frac{E_a}{RT}\right) t^n$$

where:
- $H_\infty$ = maximum healing (≈1 for PLA)
- $E_a$ = activation energy [J/mol]
- $R$ = gas constant = 8.314 J/(mol·K)
- $T$ = absolute temperature [K]
- $t$ = healing time [s]
- $n$ = time exponent (≈0.5 for Fickian diffusion)

### Simplified Strength Model

**Adhesion strength ratio**:
$$\frac{\sigma_{\text{adh}}}{\sigma_{\text{bulk}}} = 1 - \exp\left(-\frac{t}{\tau(T)}\right)$$

**Temperature-dependent time constant**:
$$\tau(T) = \tau_0 \exp\left(\frac{E_a}{RT}\right)$$

**Complete model**:
$$\sigma_{\text{adh}} = \sigma_{\text{bulk}} \left[1 - \exp\left(-\frac{t}{\tau_0 \exp\left(\frac{E_a}{RT}\right)}\right)\right]$$

### Temperature Effects

**Below glass transition** ($T < T_g$):
$$\sigma_{\text{adh}} \approx 0$$

**Above glass transition** ($T > T_g$):
$$\sigma_{\text{adh}} = \sigma_{\text{bulk}} f(T) g(t)$$

**Temperature factor**:
$$f(T) = \begin{cases}
0 & T < T_g \\
\frac{T - T_g}{T_m - T_g} & T_g \leq T \leq T_m \\
1 & T > T_m
\end{cases}$$

**Time factor**:
$$g(t) = 1 - \exp\left(-\frac{t}{\tau(T)}\right)$$

### Critical Parameters for PLA

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Bulk strength | $\sigma_{\text{bulk}}$ | 70 | MPa |
| Glass transition | $T_g$ | 60 | °C |
| Melting point | $T_m$ | 171 | °C |
| Activation energy | $E_a$ | 50,000 | J/mol |
| Pre-exponential | $\tau_0$ | 0.01-0.1 | s |

---

## Error Metrics

### Primary Metrics

**Mean Absolute Error (MAE)**:
$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**Root Mean Square Error (RMSE)**:
$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**Coefficient of Determination (R²)**:
$$R^2 = 1 - \frac{\sum_i(y_i - \hat{y}_i)^2}{\sum_i(y_i - \bar{y})^2}$$

### Per-Axis Metrics

$$\text{MAE}_x = \frac{1}{n}\sum_{i=1}^{n}|e_{x,i}|$$

$$\text{MAE}_y = \frac{1}{n}\sum_{i=1}^{n}|e_{y,i}|$$

$$\text{RMSE}_x = \sqrt{\frac{1}{n}\sum_{i=1}^{n}e_{x,i}^2}$$

$$\text{RMSE}_y = \sqrt{\frac{1}{n}\sum_{i=1}^{n}e_{y,i}^2}$$

### Vector Error Metrics

**Magnitude error**:
$$e_{\text{mag},i} = \sqrt{e_{x,i}^2 + e_{y,i}^2}$$

**Mean magnitude error**:
$$\text{MAE}_{\text{mag}} = \frac{1}{n}\sum_{i=1}^{n}e_{\text{mag},i}$$

### Percentile Errors

$$\text{Error}_{p} = \text{percentile}(\{e_i\}, p)$$

Common percentiles:
- 50th (median)
- 90th
- 95th
- 99th

### Computational Metrics

**Inference time**:
$$t_{\text{inf}} = \text{time for single prediction}$$

**Throughput**:
$$\text{TP} = \frac{n_{\text{samples}}}{t_{\text{total}}}$$

---

## Material Properties

### PLA Properties

| Property | Symbol | Value | Unit |
|----------|--------|-------|------|
| Density | $\rho$ | 1240 | kg/m³ |
| Specific heat | $c_p$ | 1200 | J/(kg·K) |
| Thermal conductivity | $k$ | 0.13 | W/(m·K) |
| Thermal diffusivity | $\alpha$ | 8.7×10⁻⁸ | m²/s |
| Glass transition | $T_g$ | 60 | °C |
| Melting point | $T_m$ | 171 | °C |
| Elastic modulus | $E$ | 3.5 | GPa |
| Poisson's ratio | $\nu$ | 0.36 | - |
| Tensile strength | $\sigma_u$ | 70 | MPa |

### Verification

**Thermal diffusivity calculation**:
$$\alpha = \frac{k}{\rho c_p} = \frac{0.13}{1240 \times 1200} = 8.7 \times 10^{-8} \text{ m²/s}$$

---

## Firmware Effects

### Junction Deviation

**Maximum velocity at corner**:
$$v_{\max}^2 = \frac{a \cdot \text{JD} \cdot \sin(\theta/2)}{1 - \sin(\theta/2)}$$

where:
- $a$ = acceleration [mm/s²]
- JD = junction deviation parameter [mm]
- $\theta$ = corner angle [rad]

### Microstep Resonance

**Magnification factor**:
$$M(f) = \frac{Q}{\sqrt{(1-\omega^2)^2 + (2\zeta\omega)^2}}$$

where:
- $Q$ = quality factor
- $\omega = f/f_n$ = frequency ratio
- $\zeta$ = damping ratio

### Timer Jitter

**Jitter time**:
$$\Delta t_{\text{jitter}} = \frac{N_{\text{cycles}}}{f_{\text{CPU}}}$$

where:
- $N_{\text{cycles}}$ = interrupt cycles
- $f_{\text{CPU}}$ = CPU frequency [Hz]

---

## LaTeX Code Examples

### Second-Order System

```latex
% Equation of motion
\begin{equation}
m\ddot{x} + c\dot{x} + kx = -m a_{\text{ref}}(t)
\end{equation}

% Transfer function
\begin{equation}
H(s) = \frac{X(s)}{A_{\text{ref}}(s)} = \frac{-1}{s^2 + 2\zeta\omega_n s + \omega_n^2}
\end{equation}

% Natural frequency and damping ratio
\begin{align}
\omega_n &= \sqrt{\frac{k}{m}} \\
\zeta &= \frac{c}{2\sqrt{mk}}
\end{align}
```

### Heat Transfer

```latex
% Heat conduction equation
\begin{equation}
\frac{\partial T}{\partial t} = \alpha \nabla^2 T + Q_{\text{source}} - Q_{\text{cooling}}
\end{equation}

% Newton's cooling law
\begin{equation}
\frac{dT}{dt} = -\frac{hA}{mc_p}(T - T_{\text{amb}})
\end{equation}

% Solution
\begin{equation}
T(t) = T_{\text{amb}} + (T_0 - T_{\text{amb}})e^{-t/\tau}
\end{equation}
```

### Adhesion Model

```latex
% Wool-O'Connor model
\begin{equation}
\frac{\sigma}{\sigma_{\text{bulk}}} = 1 - \exp\left(-\frac{t}{\tau_0 \exp\left(\frac{E_a}{RT}\right)}\right)
\end{equation}

% Temperature-dependent time constant
\begin{equation}
\tau(T) = \tau_0 \exp\left(\frac{E_a}{RT}\right)
\end{equation}
```

### Error Metrics

```latex
% MAE and RMSE
\begin{align}
\text{MAE} &= \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i| \\
\text{RMSE} &= \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
\end{align}

% R-squared
\begin{equation}
R^2 = 1 - \frac{\sum_i(y_i - \hat{y}_i)^2}{\sum_i(y_i - \bar{y})^2}
\end{equation}
```

---

## References

**See Also**:
- [Trajectory Dynamics](trajectory_dynamics.md) - Detailed dynamics derivation
- [Thermal Model](thermal_model.md) - Complete heat transfer analysis
- [Adhesion Model](adhesion_model.md) - Polymer healing theory
- [Firmware Effects](../methods/firmware_effects.md) - Firmware-level error sources

**Related Documents**:
- [Previous]: [Thermal Model](thermal_model.md)
- [Next]: [Simulation System](../methods/simulation_system.md)
