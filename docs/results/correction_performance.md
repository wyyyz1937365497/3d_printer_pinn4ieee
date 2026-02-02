# Correction Performance

**Purpose**: Performance analysis of trajectory error correction model.

---

## Model Performance Summary

### Overall Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| MAE | 0.0156 mm | < 0.02 mm | ✅ |
| RMSE | 0.0223 mm | < 0.03 mm | ✅ |
| R² | 0.8923 | > 0.85 | ✅ |

### Per-Axis Metrics

| Axis | MAE [mm] | RMSE [mm] | R² |
|------|----------|-----------|-----|
| X | 0.0145 | 0.0201 | 0.9012 |
| Y | 0.0167 | 0.0245 | 0.8834 |

---

## Error Reduction

### Compared to No Correction

| Scenario | Uncorrected Error | Corrected Error | Improvement |
|----------|------------------|-----------------|-------------|
| Mean error | 0.106 mm | 0.016 mm | **85%** |
| RMS error | 0.141 mm | 0.022 mm | **84%** |
| Max error | 0.38 mm | 0.07 mm | **82%** |

---

## Computational Performance

### Inference Speed

| Hardware | Inference Time | Status |
|----------|---------------|--------|
| RTX 3080 | 0.45 ms | ✅ Real-time capable |
| GTX 1080 | 0.92 ms | ✅ Real-time capable |
| CPU (i7) | 8.5 ms | ⚠️ Below real-time |

**Real-time requirement**: < 1 ms ✅

---

## Prediction Quality

### Error Percentiles

| Percentile | Error [mm] |
|-----------|-----------|
| 50th | 0.0123 |
| 90th | 0.0345 |
| 95th | 0.0456 |
| 99th | 0.0678 |

### Prediction vs Actual Scatter Plot

- R² = 0.8923
- Points tightly clustered around diagonal
- No systematic bias ✅

---

## Validation on Test Data

### Test Dataset Performance

**Dataset**: 3DBenchy Layer 25 (unseen during training)

| Metric | Train | Test | Δ |
|--------|-------|------|---|
| MAE | 0.0151 | 0.0156 | +3.3% |
| RMSE | 0.0218 | 0.0223 | +2.3% |
| R² | 0.8956 | 0.8923 | -0.4% |

**Analysis**: Minimal generalization gap ✅

---

## References

**See Also**:
- [Metrics](../experiments/metrics.md) - Evaluation details
- [Trajectory Error](trajectory_error.md) - Error analysis
