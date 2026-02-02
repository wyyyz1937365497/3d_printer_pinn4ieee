# Physics Formulas and Symbols Library

**Purpose**: Central repository for all physical formulas, symbols, and LaTeX code used in the paper.

---

## Notation and Units

### Coordinate System
- **Position**: $x, y, z$ [mm] - Cartesian coordinates
- **Velocity**: $v_x, v_y, v_z$ or $\dot{x}, \dot{y}, \dot{z}$ [mm/s]
- **Acceleration**: $a_x, a_y, a_z$ or $\ddot{x}, \ddot{y}, \ddot{z}$ [mm/s²]
- **Jerk**: $j_x, j_y, j_z$ or $\dddot{x}, \dddot{y}, \dddot{z}$ [mm/s³]

### Trajectory Error
- **Error Components**: $\epsilon_x, \epsilon_y$ [mm]
- **Error Magnitude**: $\epsilon = \sqrt{\epsilon_x^2 + \epsilon_y^2}$ [mm]
- **Error Direction**: $\theta_\epsilon = \arctan2(\epsilon_y, \epsilon_x)$ [rad]

### Forces
- **Inertial Force**: $F_{inertia}$ [N]
- **Elastic Force**: $F_{elastic}$ [N]
- **Total Force**: $F_{total}$ [N]

### System Parameters
- **Mass**: $m$ [kg]
- **Stiffness**: $k$ [N/m]
- **Damping**: $c$ [N·s/m]
- **Natural Frequency**: $\omega_n = \sqrt{k/m}$ [rad/s]
- **Damping Ratio**: $\zeta = \frac{c}{2\sqrt{mk}}$

### Thermal
- **Temperature**: $T$ [°C or K]
- **Nozzle Temperature**: $T_{nozzle}$ [°C]
- **Interface Temperature**: $T_{interface}$ [°C]
- **Ambient Temperature**: $T_{ambient}$ [°C]
- **Glass Transition Temperature**: $T_g$ [°C]
- **Heat Transfer Coefficient**: $h$ [W/(m²·K)]
- **Thermal Diffusivity**: $\alpha = \frac{k}{\rho c_p}$ [m²/s]

### Time
- **Time**: $t$ [s]
- **Time Step**: $\Delta t$ [s]
- **Interlayer Time**: $\Delta t_{layer}$ [s]

---

## Dynamics Equations

### Second-Order Mass-Spring-Damper System

**Equation of Motion**:

$$m\ddot{x} + c\dot{x} + kx = F(t)$$

**In Standard Form**:

$$\ddot{x} + 2\zeta\omega_n\dot{x} + \omega_n^2 x = \frac{F(t)}{m}$$

where:
- $m$ = effective mass [kg]
- $c$ = damping coefficient [N·s/m]
- $k$ = stiffness [N/m]
- $\omega_n = \sqrt{k/m}$ = natural frequency [rad/s]
- $\zeta = \frac{c}{2\sqrt{mk}}$ = damping ratio (dimensionless)

**Characteristic Equation**:

$$s^2 + 2\zeta\omega_n s + \omega_n^2 = 0$$

**Natural Frequency**:

$$f_n = \frac{\omega_n}{2\pi} = \frac{1}{2\pi}\sqrt{\frac{k}{m}} \quad \text{[Hz]}$$

**Settling Time** (for 2% criterion):

$$t_s = \frac{4}{\zeta\omega_n}$$

### Belt Stretch Model

**Elastic Deformation**:

$$\Delta x = \frac{F}{k_{belt}}$$

where $k_{belt}$ is the effective belt stiffness [N/m]

**GT2 Belt Stiffness**:

$$k_{belt} = \frac{EA}{L}$$

where:
- $E$ = Young's modulus of rubber [Pa]
- $A$ = cross-sectional area [m²]
- $L$ = belt length [m]

### Inertial Force

**Newton's Second Law**:

$$F_{inertia} = ma = m\ddot{x}$$

**Component Form**:

$$F_{inertia,x} = m \ddot{x}$$
$$F_{inertia,y} = m \ddot{y}$$

### Elastic Force

**Hooke's Law**:

$$F_{elastic} = -k \Delta x$$

**Component Form**:

$$F_{elastic,x} = -k_x \Delta x$$
$$F_{elastic,y} = -k_y \Delta y$$

---

## Thermal Models

### Moving Heat Source Model

**Temperature Distribution**:

$$T(x,y,t) = T_0 + \frac{Q}{2\pi k r} \exp\left(-\frac{r^2}{4\alpha t}\right)$$

where:
- $T_0$ = initial temperature [°C]
- $Q$ = heat input [W]
- $k$ = thermal conductivity [W/(m·K)]
- $\alpha$ = thermal diffusivity [m²/s]
- $r$ = radial distance from heat source [m]

### Heat Conduction Equation

$$\frac{\partial T}{\partial t} = \alpha \nabla^2 T$$

**In 3D Cartesian Coordinates**:

$$\frac{\partial T}{\partial t} = \alpha\left(\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} + \frac{\partial^2 T}{\partial z^2}\right)$$

### Convective Cooling

**Newton's Law of Cooling**:

$$\frac{dT}{dt} = -\frac{hA}{mc_p}(T - T_{ambient})$$

where:
- $h$ = convective heat transfer coefficient [W/(m²·K)]
- $A$ = surface area [m²]
- $m$ = mass [kg]
- $c_p$ = specific heat capacity [J/(kg·K)]

### Cooling Rate

**Instantaneous Cooling Rate**:

$$\text{Cooling Rate} = -\frac{dT}{dt} \quad \text{[°C/s]}$$

### Temperature Gradient

**Z-Direction Gradient**:

$$\nabla_z T = \frac{\partial T}{\partial z} \quad \text{[°C/mm]}$$

---

## Adhesion Strength Model

### Wool-OConnor Polymer Healing Model

**Degree of Healing**:

$$H = H_\infty \exp\left(-\frac{E_a}{RT}\right) t^n$$

where:
- $H$ = degree of healing (0-1)
- $H_\infty$ = maximum healing ratio
- $E_a$ = activation energy [J/mol]
- $R$ = gas constant = 8.314 J/(mol·K)
- $T$ = absolute temperature [K]
- $t$ = time [s]
- $n$ = time exponent (typically 0.5 for Fickian diffusion)

**Adhesion Strength**:

$$\sigma_{adh} = \sigma_{bulk} \left[1 - \exp\left(-\frac{t}{\tau(T)}\right)\right]$$

**Temperature-Dependent Time Constant**:

$$\tau(T) = \tau_0 \exp\left(\frac{E_a}{RT}\right)$$

where:
- $\sigma_{adh}$ = interlayer adhesion strength [Pa]
- $\sigma_{bulk}$ = bulk material strength [Pa]
- $\tau_0$ = pre-exponential factor [s]

---

## Firmware Effects Models

### Junction Deviation

**Marlin Junction Deviation**:

$$v_{corner} = \min\left(v_{max}, \sqrt{\frac{a_{max} \cdot d_{junction}}{\sqrt{2} - 1}}\right)$$

where:
- $v_{corner}$ = corner speed [mm/s]
- $v_{max}$ = maximum speed [mm/s]
- $a_{max}$ = maximum acceleration [mm/s²]
- $d_{junction}$ = junction deviation [mm]

### Microstepping Resonance

**Resonance Frequency**:

$$f_{res} = \frac{n_{microsteps} \cdot f_{step}}{2}$$

where:
- $n_{microsteps}$ = number of microsteps (e.g., 16 for 1/16 microstepping)
- $f_{step}$ = step frequency [Hz]

### Timer Jitter

**Timing Uncertainty**:

$$\Delta t_{jitter} \sim \mathcal{N}(0, \sigma_{timer}^2)$$

---

## Neural Network Equations

### LSTM/GRU Cell (if used)

**LSTM**:

$$\begin{aligned}
\mathbf{f}_t &= \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \\
\mathbf{i}_t &= \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \\
\tilde{\mathbf{C}}_t &= \tanh(\mathbf{W}_C \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C) \\
\mathbf{C}_t &= \mathbf{f}_t * \mathbf{C}_{t-1} + \mathbf{i}_t * \tilde{\mathbf{C}}_t \\
\mathbf{o}_t &= \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \\
\mathbf{h}_t &= \mathbf{o}_t * \tanh(\mathbf{C}_t)
\end{aligned}$$

### Feed-Forward Network

**Output**:

$$\mathbf{y} = \mathbf{W}_2 \cdot \phi(\mathbf{W}_1 \cdot \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

where $\phi$ is activation function (ReLU, tanh, etc.)

### Loss Function

**Mean Absolute Error (MAE)**:

$$\mathcal{L}_{MAE} = \frac{1}{N}\sum_{i=1}^{N}|\mathbf{y}_i - \hat{\mathbf{y}}_i|$$

**Root Mean Square Error (RMSE)**:

$$\mathcal{L}_{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\mathbf{y}_i - \hat{\mathbf{y}}_i)^2}$$

---

## Statistical Measures

### Error Metrics

**Mean Absolute Error (MAE)**:

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**Root Mean Square Error (RMSE)**:

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**Coefficient of Determination (R²)**:

$$R^2 = 1 - \frac{\sum_i(y_i - \hat{y}_i)^2}{\sum_i(y_i - \bar{y})^2}$$

**Maximum Error**:

$$\epsilon_{max} = \max_i |y_i - \hat{y}_i|$$

---

## Constants

### Physical Constants

- **Gas Constant**: $R = 8.314$ J/(mol·K)
- **Stefan-Boltzmann Constant**: $\sigma = 5.67 \times 10^{-8}$ W/(m²·K⁴)

### PLA Material Properties

- **Density**: $\rho = 1240$ kg/m³
- **Elastic Modulus**: $E = 3.5$ GPa
- **Glass Transition**: $T_g = 60$ °C
- **Melting Point**: $T_m = 171$ °C
- **Thermal Conductivity**: $k = 0.13$ W/(m·K)
- **Specific Heat**: $c_p = 1200$ J/(kg·K)

### Ender-3 V2 Parameters

- **Nozzle Temperature**: $T_{nozzle} = 220$ °C
- **Bed Temperature**: $T_{bed} = 60$ °C
- **Max Velocity**: $v_{max} = 500$ mm/s
- **Max Acceleration**: $a_{max} = 500$ mm/s²
- **Nozzle Diameter**: $d_{nozzle} = 0.4$ mm

---

## LaTeX Custom Commands

For use in LaTeX documents:

```tex
% Vectors and matrices
\newcommand{\vx}{\mathbf{x}}
\newcommand{\vy}{\mathbf{y}}
\newcommand{\vz}{\mathbf{z}}
\newcommand{\vv}{\mathbf{v}}
\newcommand{\va}{\mathbf{a}}
\newcommand{\F}{\mathbf{F}}

% Temperatures
\newcommand{\Tnozzle}{T_{\text{nozzle}}}
\newcommand{\Tinterface}{T_{\text{interface}}}
\newcommand{\Tambient}{T_{\text{ambient}}}
\newcommand{\Tg}{T_{\text{g}}}

% Errors
\newcommand{\ex}{\epsilon_x}
\newcommand{\ey}{\epsilon_y}
\newcommand{\emag}{\epsilon_{\text{mag}}}

% System parameters
\newcommand{\omegan}{\omega_n}
\newcommand{\zeta}{\zeta}

% Energy
\newcommand{\Ea}{E_a}
\newcommand{\Rgas}{R}

% Derivatives
\newcommand{\ddt}[1]{\frac{d#1}{dt}}
\newcommand{\ddtt}[1]{\frac{d^2#1}{dt^2}}

% Equation shortcuts
\newcommand{\beq}{\begin{equation}}
\newcommand{\eeq}{\end{equation}}
```

---

## Common Abbreviations

- **FDM**: Fused Deposition Modeling
- **PINN**: Physics-Informed Neural Network
- **RMS**: Root Mean Square
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **G-code**: Geometry code (printer control language)

---

## References to Key Equations

**Main dynamics**: {#eq:second-order}
**Belt stretch**: {#eq:belt-stretch}
**Adhesion model**: {#eq:adhesion}
**Moving heat source**: {#eq:heat-source}

---

**Last Updated**: 2026-02-01
**Maintained by**: 3D Printer PINN Project Team
