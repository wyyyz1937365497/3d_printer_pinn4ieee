# Evaluation Metrics

**Purpose**: Metrics for evaluating trajectory error prediction models.

---

## Primary Metrics

### Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**Target**: < 0.02 mm (20 μm)

### Root Mean Square Error (RMSE)

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**Target**: < 0.03 mm (30 μm)

### Coefficient of Determination (R²)

$$R^2 = 1 - \frac{\sum_i(y_i - \hat{y}_i)^2}{\sum_i(y_i - \bar{y})^2}$$

**Target**: > 0.85

---

## Per-Axis Metrics

| Metric | X-Axis Target | Y-Axis Target | Notes |
|--------|---------------|---------------|-------|
| MAE | < 0.02 mm | < 0.02 mm | Mean absolute error |
| RMSE | < 0.03 mm | < 0.03 mm | Root mean square |
| R² | > 0.85 | > 0.85 | Coefficient of determination |

---

## Error Distribution Metrics

### Percentile Errors

| Percentile | Target [mm] | Description |
|-----------|-------------|-------------|
| 50th (Median) | < 0.015 | Central tendency |
| 90th | < 0.035 | 90% of errors below this |
| 95th | < 0.045 | 95% of errors below this |
| 99th | < 0.070 | Almost all errors |

### Maximum Error

$$\epsilon_{\max} = \max_i |y_i - \hat{y}_i|$$

**Target**: < 0.1 mm (100 μm)

---

## Computational Metrics

### Inference Time

**Target**: < 1 ms per sample

Real-time correction requires:
- Process 128-timestep sequence
- Predict 2 error components
- Total time < 1 ms

### Throughput

**Target**: > 1000 samples/second (batch processing)

---

## Usage Example

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calculate metrics
mae_x = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
mae_y = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
rmse_x = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
rmse_y = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
r2_x = r2_score(y_true[:, 0], y_pred[:, 0])
r2_y = r2_score(y_true[:, 1], y_pred[:, 1])

# Magnitude error
error_mag = np.sqrt((y_true - y_pred)**2).sum(axis=1)
mae_mag = np.mean(error_mag)

print(f"MAE X: {mae_x:.4f} mm")
print(f"MAE Y: {mae_y:.4f} mm")
print(f"RMSE X: {rmse_x:.4f} mm")
print(f"RMSE Y: {rmse_y:.4f} mm")
print(f"R² X: {r2_x:.4f}")
print(f"R² Y: {r2_y:.4f}")
```

---

## References

**See Also**:
- [Setup](setup.md) - Experimental configuration
- [Training Pipeline](../methods/training_pipeline.md) - Evaluation process
