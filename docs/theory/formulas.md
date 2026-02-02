# Physics Formulas and Symbols Library

**Purpose**: Complete collection of physical models, equations, and symbols for trajectory error prediction in FDM 3D printing.

---

## Table of Contents

1. [Symbols and Notation](#symbols-and-notation)
2. [Trajectory Dynamics](#trajectory-dynamics)
3. [Error Metrics](#error-metrics)
4. [Firmware Effects](#firmware-effects)
5. [LaTeX Code Examples](#latex-code-examples)

---

## Symbols and Notation

### Kinematic Symbols

| Symbol | Meaning | Units | Typical Value |
|--------|---------|-------|---------------|
| $x, y$ | Position in build plane | mm | 0-220 |
| $\mathbf{r}$ | Position vector | mm | - |
| $v_x, v_y$ | Velocity components | mm/s | -200 to 200 |
| $\mathbf{v}$ | Velocity vector | mm/s | - |
| $a$ | Acceleration magnitude | mm/s² | -500 to 500 |
| $\mathbf{a}$ | Acceleration vector | mm/s² | - |
| $\mathbf{e}$ | Error vector | mm | - |

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

### LSTM Network Symbols

| Symbol | Meaning | Units | Typical Value |
|--------|---------|-------|---------------|
| $\mathbf{h}_t$ | Hidden state at time t | - | 56-dimensional |
| $\mathbf{c}_t$ | Cell state at time t | - | 56-dimensional |
| $\mathbf{x}_t$ | Input at time t | - | 4-dimensional |
| $\mathbf{y}_t$ | Output (error prediction) | mm | 2-dimensional |
| $N_{params}$ | Number of parameters | - | ~38,000 |

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
- $m$ = effective mass [kg]
- $c$ = damping coefficient [N·s/m]
- $k$ = stiffness [N/m]

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

where:
- $\mathbf{z} = [x, \dot{x}]^T$ = state vector
- $A$ = system matrix
- $B$ = input matrix
- $\mathbf{u}(t) = a_{\text{ref}}(t)$ = input

### Error Calculations

**Error vector**:
$$\mathbf{e}(t) = \mathbf{r}_{\text{actual}}(t) - \mathbf{r}_{\text{ref}}(t) = [e_x(t), e_y(t)]^T$$

**Error magnitude**:
$$e_{\text{mag}}(t) = \sqrt{e_x(t)^2 + e_y(t)^2}$$

**Error direction**:
$$\theta_e(t) = \arctan2(e_y(t), e_x(t))$$

---

## Error Metrics

### Primary Metrics

**Mean Absolute Error (MAE)**:
$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**Root Mean Square Error (RMSE)**:
$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**Coefficient of Determination (R²)**:
$$R^2 = 1 - \frac{\sum_i(y_i - \hat{y}_i)^2}{\sum_i(y_i - \bar{y})^2}$$

where:
- $y_i$ = ground truth value
- $\hat{y}_i$ = predicted value
- $\bar{y}$ = mean of ground truth values

### Per-Axis Metrics

**X-axis errors**:
$$\text{MAE}_x = \frac{1}{n}\sum_{i=1}^{n}|e_{x,i}|$$

$$\text{RMSE}_x = \sqrt{\frac{1}{n}\sum_{i=1}^{n}e_{x,i}^2}$$

$$R^2_x = 1 - \frac{\sum_i e_{x,i}^2}{\sum_i (e_{x,i} - \bar{e}_x)^2}$$

**Y-axis errors**:
$$\text{MAE}_y = \frac{1}{n}\sum_{i=1}^{n}|e_{y,i}|$$

$$\text{RMSE}_y = \sqrt{\frac{1}{n}\sum_{i=1}^{n}e_{y,i}^2}$$

$$R^2_y = 1 - \frac{\sum_i e_{y,i}^2}{\sum_i (e_{y,i} - \bar{e}_y)^2}$$

### Vector Error Metrics

**Magnitude error**:
$$e_{\text{mag},i} = \sqrt{e_{x,i}^2 + e_{y,i}^2}$$

**Mean magnitude error**:
$$\text{MAE}_{\text{mag}} = \frac{1}{n}\sum_{i=1}^{n}e_{\text{mag},i}$$

**RMS magnitude error**:
$$\text{RMSE}_{\text{mag}} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}e_{\text{mag},i}^2}$$

### Percentile Errors

$$\text{Error}_{p} = \text{percentile}(\{|e_i|\}, p)$$

Common percentiles:
- **50th (median)**: Typical error magnitude
- **90th**: Worst 10% of errors
- **95th**: Worst 5% of errors
- **99th**: Extreme errors

### Correlation Coefficient

**Pearson correlation** (for each axis):
$$r = \frac{\sum_i (e_i - \bar{e})(\hat{e}_i - \bar{\hat{e}})}{\sqrt{\sum_i (e_i - \bar{e})^2 \sum_i (\hat{e}_i - \bar{\hat{e}})^2}}$$

where $e_i$ are true errors and $\hat{e}_i$ are predicted errors.

### Computational Metrics

**Inference time**:
$$t_{\text{inf}} = \text{time for single prediction [ms]}$$

**Throughput**:
$$\text{TP} = \frac{n_{\text{samples}}}{t_{\text{total}}} \text{ [inferences/sec]}$$

**Parameter efficiency**:
$$\text{Accuracy per parameter} = \frac{1}{\text{MAE} \times N_{params}}$$

---

## Firmware Effects

### Junction Deviation

**Maximum velocity at corner**:
$$v_{\max}^2 = \frac{a \cdot \text{JD} \cdot \sin(\theta/2)}{1 - \sin(\theta/2)}$$

where:
- $a$ = acceleration [mm/s²]
- JD = junction deviation parameter [mm]
- $\theta$ = corner angle [rad]

**Corner rounding error**:
$$e_{\text{corner}} \approx \frac{v^2}{a \cdot \text{JD}}$$

### Microstep Resonance

**Resonance frequency** (for 16× microstepping):
$$f_{\text{res}} = \frac{f_{\text{motor}}}{16}$$

where $f_{\text{motor}}$ = full-step frequency (~800 Hz for NEMA 17).

**Magnification factor**:
$$M(f) = \frac{Q}{\sqrt{(1-\omega^2)^2 + (2\zeta\omega)^2}}$$

where:
- $Q$ = quality factor (≈10 for stepper systems)
- $\omega = f/f_n$ = frequency ratio
- $\zeta$ = damping ratio

### Timer Jitter

**Jitter time**:
$$\Delta t_{\text{jitter}} = \frac{N_{\text{cycles}}}{f_{\text{CPU}}}$$

where:
- $N_{\text{cycles}}$ = interrupt service routine cycles
- $f_{\text{CPU}}$ = CPU frequency [Hz]

For STM32F103 at 72 MHz with 100-cycle ISR:
$$\Delta t_{\text{jitter}} = \frac{100}{72 \times 10^6} \approx 1.4 \text{ μs}$$

---

## LaTeX Code Examples

### Second-Order System

```latex
% Equation of motion with inertial forcing
\begin{equation}
m\ddot{x} + c\dot{x} + kx = -m a_{\text{ref}}(t)
\label{eq:second-order}
\end{equation}

% Transfer function
\begin{equation}
H(s) = \frac{X(s)}{A_{\text{ref}}(s)} = \frac{-1}{s^2 + 2\zeta\omega_n s + \omega_n^2}
\label{eq:transfer-function}
\end{equation}

% Natural frequency and damping ratio
\begin{align}
\omega_n &= \sqrt{\frac{k}{m}} \label{eq:natural-freq} \\
\zeta &= \frac{c}{2\sqrt{mk}} \label{eq:damping-ratio}
\end{align}

% Underdamped response
\begin{equation}
x(t) = 1 - \frac{e^{-\zeta\omega_n t}}{\sqrt{1-\zeta^2}} \sin\left(\omega_d t + \phi\right)
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

% Per-axis metrics
\begin{align}
\text{MAE}_x &= \frac{1}{n}\sum_{i=1}^{n}|e_{x,i}| \\
\text{MAE}_y &= \frac{1}{n}\sum_{i=1}^{n}|e_{y,i}| \\
\text{MAE}_{\text{mag}} &= \frac{1}{n}\sum_{i=1}^{n}\sqrt{e_{x,i}^2 + e_{y,i}^2}
\end{align}
```

### LSTM Network Equations

```latex
% LSTM cell equations
\begin{align}
\mathbf{i}_t &= \sigma(\mathbf{W}_{xi}\mathbf{x}_t + \mathbf{W}_{hi}\mathbf{h}_{t-1} + \mathbf{b}_i) \\
\mathbf{f}_t &= \sigma(\mathbf{W}_{xf}\mathbf{x}_t + \mathbf{W}_{hf}\mathbf{h}_{t-1} + \mathbf{b}_f) \\
\mathbf{o}_t &= \sigma(\mathbf{W}_{xo}\mathbf{x}_t + \mathbf{W}_{ho}\mathbf{h}_{t-1} + \mathbf{b}_o) \\
\tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_{xc}\mathbf{x}_t + \mathbf{W}_{hc}\mathbf{h}_{t-1} + \mathbf{b}_c) \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
\end{align}

% Final prediction
\begin{equation}
\hat{\mathbf{e}} = \mathbf{W}_o \mathbf{h}_T + \mathbf{b}_o
\end{equation}
```

### Firmware Effects

```latex
% Junction deviation
\begin{equation}
v_{\max}^2 = \frac{a \cdot \text{JD} \cdot \sin(\theta/2)}{1 - \sin(\theta/2)}
\end{equation}

% Microstep resonance
\begin{equation}
M(f) = \frac{Q}{\sqrt{(1-\omega^2)^2 + (2\zeta\omega)^2}}
\end{equation}

% Timer jitter
\begin{equation}
\Delta t_{\text{jitter}} = \frac{N_{\text{cycles}}}{f_{\text{CPU}}}
\end{equation}
```

---

## Material Properties (PLA)

### Mechanical Properties

| Property | Symbol | Value | Unit | Source |
|----------|--------|-------|------|--------|
| Elastic modulus | $E$ | 3.5 | GPa | Manufacturer spec |
| Poisson's ratio | $\nu$ | 0.36 | - | Literature |
| Tensile strength | $\sigma_u$ | 70 | MPa | Manufacturer spec |
| Density | $\rho$ | 1240 | kg/m³ | Manufacturer spec |

### Printer Dynamics (Ender-3 V2)

| Parameter | X-axis | Y-axis | Unit | Source |
|-----------|--------|--------|------|--------|
| Mass | 0.485 | 0.650 | kg | Measured |
| Damping | 25 | 25 | N·s/m | Estimated |
| Stiffness | 150,000 | 150,000 | N/m | GT2 belt specs |
| Natural freq | 556 | 480 | rad/s | Calculated |
| Damping ratio | 0.046 | 0.040 | - | Calculated |
| Resonance freq | 88.5 | 76.5 | Hz | Calculated |

### Belt Specifications (GT2)

| Parameter | Value | Unit |
|-----------|-------|------|
| Pitch | 2 | mm |
| Width | 6 | mm |
| Tooth count | 20 | - |
| Young's modulus | 2-3 | GPa |
| Steel reinforcement | Yes | - |

---

## References

**See Also**:
- [Trajectory Dynamics](trajectory_dynamics.md) - Detailed dynamics derivation and validation
- [Neural Network](../methods/neural_network.md) - LSTM architecture for error prediction
- [Firmware Effects](../methods/firmware_effects.md) - Firmware-level error sources
- [Error Metrics](../experiments/metrics.md) - Evaluation metrics and benchmarks

**Related Documents**:
- [Previous]: [Simulation System](../methods/simulation_system.md)
- [Next]: [Neural Network](../methods/neural_network.md)

---

**Last Updated**: 2026-02-02
