# Trajectory Error Results

**Purpose**: Analysis of trajectory errors from simulation and experiments.

---

## Error Statistics

### Firmware-Enhanced Simulation Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| RMS error | 140.5 μm | 80-150 μm | ✅ |
| Mean error | 106.4 μm | 50-120 μm | ✅ |
| Median error | 96.7 μm | 80-120 μm | ✅ |
| Std deviation | 91.9 μm | < 100 μm | ✅ |
| Max error | 2381 μm | < 500 μm | ⚠️ |

### Error Distribution

| Error Range | Percentage | Notes |
|-------------|------------|-------|
| < 50 μm | 2.6% | Minimal error |
| 50-100 μm | 52.8% | Main peak |
| 100-150 μm | 33.4% | Secondary peak |
| > 150 μm | 11.3% | Outliers |

**Analysis**: 86.2% of points in target range (50-150 μm) ✅

---

## Error Source Breakdown

### Contribution by Source

| Error Source | RMS [μm] | Percentage | Notes |
|--------------|----------|------------|-------|
| Basic dynamics | 91.0 | 64.8% | Primary source |
| Microstep resonance | 8.8 | 6.3% | Frequency-dependent |
| Timer jitter | 2.4 | 1.7% | Minor contribution |
| Junction deviation | Variable | Trajectory-dependent | Corner-related |

---

## Spatial Distribution

### Error by Location

| Region | Mean Error [μm] | Characteristics |
|--------|-----------------|-----------------|
| Straight lines | < 50 | Steady-state |
| Corners (90°) | 200-400 | Transient spikes |
| Curves | 100-200 | Moderate |
| Start/stop | 150-250 | Initial transient |

---

## Frequency Analysis

### Error Spectrum

- **Low freq (< 10 Hz)**: Basic dynamics dominant
- **Mid freq (20-30 Hz)**: Microstep resonance peak
- **High freq (> 100 Hz)**: Rapid attenuation

**Validation**: Matches literature reports of 3D printer error spectra ✅

---

## References

**See Also**:
- [Metrics](../experiments/metrics.md) - Evaluation metrics
- [Correction Performance](correction_performance.md) - Model results
