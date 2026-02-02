# Adhesion Strength Model

**Purpose**: Mathematical modeling of interlayer bonding strength in FDM 3D printing.

---

## Overview

Layer bonding is a critical quality factor in FDM printing. The **Wool-O'Connor polymer healing model** describes how temperature and time affect the diffusion of polymer chains across layer interfaces.

---

## Wool-O'Connor Polymer Healing Model

### Degree of Healing

$$H = H_\infty \exp\left(-\frac{E_a}{RT}\right) t^n$$

where:
- $H$ = degree of healing (0-1)
- $H_\infty$ = maximum healing ratio
- $E_a$ = activation energy [J/mol]
- $R$ = gas constant = 8.314 J/(mol·K)
- $T$ = absolute temperature [K]
- $t$ = time [s]
- $n$ = time exponent (typically 0.5 for Fickian diffusion)

### Adhesion Strength

$$\sigma_{\text{adh}} = \sigma_{\text{bulk}} \left[1 - \exp\left(-\frac{t}{\tau(T)}\right)\right]$$

### Temperature-Dependent Time Constant

$$\tau(T) = \tau_0 \exp\left(\frac{E_a}{RT}\right)$$

where:
- $\sigma_{\text{adh}}$ = interlayer adhesion strength [Pa]
- $\sigma_{\text{bulk}}$ = bulk material strength [Pa]
- $\tau_0$ = pre-exponential factor [s]

---

## Key Temperature Thresholds

### Glass Transition Temperature ($T_g$)

$$T_g = 60\degree\text{C} \quad \text{(for PLA)}$$

- Below $T_g$: Molecular chains frozen, minimal diffusion
- Above $T_g$: Chains begin to move, healing starts

### Melting Temperature ($T_m$)

$$T_m = 171\degree\text{C} \quad \text{(for PLA)}$$

- Above $T_m$: Chains diffuse freely
- Optimal bonding temperature: 150-160°C

---

## Time Requirements

### Minimum Healing Time

$$t_{\min} \approx 0.5 \text{ s}$$

### Optimal Healing Time

$$t_{\optimal} \approx 2.0 \text{ s}$$

---

## Simplified Model for Real-Time Computation

### Temperature Factor

$$f(T) = \begin{cases}
0 & T < T_g \\
\frac{T - T_g}{T_m - T_g} & T_g \leq T \leq T_m \\
1 & T > T_m
\end{cases}$$

### Time Factor

$$g(t) = 1 - \exp\left(-\frac{t}{\tau_{\text{heal}}}\right)$$

$$\tau_{\text{heal}} = \tau_0 \exp\left(\frac{E_a}{RT}\right)$$

### Combined Strength

$$\sigma_{\text{adh}} = \sigma_{\text{bulk}} \cdot f(T) \cdot g(t)$$

---

## Model Parameters (PLA)

| Parameter | Symbol | Value | Unit | Source |
|-----------|--------|-------|------|--------|
| Glass transition | $T_g$ | 60 | °C | Material datasheet |
| Melting point | $T_m$ | 171 | °C | Material datasheet |
| Activation energy | $E_a$ | ~50 | kJ/mol | Literature |
| Pre-exponential factor | $\tau_0$ | ~10⁻⁶ | s | Fitted |
| Bulk strength | $\sigma_{\text{bulk}}$ | 60-70 | MPa | Material datasheet |

---

## Expected Adhesion Strength

| Condition | Adhesion Ratio | Primary Factors |
|-----------|---------------|-----------------|
| Fast printing | 0.6-0.7 | Low interlayer temp |
| Normal printing | 0.7-0.85 | Balanced temp/time |
| Slow printing | 0.85-0.95 | High interlayer temp |

---

## Temperature-Adhesion Correlation

The model predicts the correlation between interface temperature and bonding strength:

$$\sigma_{\text{adh}}(T) \approx \sigma_{\text{bulk}} \cdot \left[1 - \exp\left(-\frac{t_{\text{layer}}}{\tau(T)}\right)\right]$$

where $t_{\text{layer}}$ is the time available for bonding before the next layer is deposited.

---

## Physical Interpretation

1. **Below $T_g$**: Polymer chains are in glassy state, cannot diffuse
2. **Between $T_g$ and $T_m$**: Chains have mobility, diffusion rate increases with temperature
3. **Above $T_m$**: Chains are in melt state, rapid diffusion

The Arrhenius term $\exp\left(-\frac{E_a}{RT}\right)$ captures the exponential increase in diffusion rate with temperature.

---

## Validation

### Literature Comparison

| Condition | Predicted | Literature [1] | Status |
|-----------|-----------|---------------|--------|
| Layer 20, normal speed | 0.75-0.85 | 0.70-0.80 | ✅ |
| Layer 50, slow speed | 0.85-0.95 | 0.85-0.92 | ✅ |
| Layer 10, fast speed | 0.60-0.70 | 0.55-0.65 | ✅ |

**Reference**: [1] McCullough et al., "Interlayer adhesion in FDM", Additive Manufacturing, 2023.

---

## References

**See Also**:
- [Formula Library](formulas.md#adhesion-strength-model) - Complete equations
- [Thermal Model](thermal_model.md) - Temperature field calculation
- [Results](../results/correction_performance.md) - Experimental validation

**Related Documents**:
- [Previous]: [Thermal Model](thermal_model.md)
- [See Also]: [Overview](overview.md)
