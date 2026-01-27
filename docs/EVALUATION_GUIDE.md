# Evaluation Module Documentation

## Overview

The evaluation module provides comprehensive tools for assessing model performance, generating visualizations, and comparing against baseline methods.

## 📊 Components

### 1. **Metrics Module** (`evaluation/metrics.py`)

Comprehensive evaluation metrics for all tasks:

#### **RegressionMetrics**
For quality prediction and trajectory correction:
- MSE, RMSE, MAE
- R² score
- MAPE (Mean Absolute Percentage Error)

#### **ClassificationMetrics**
For fault detection:
- Accuracy, Precision, Recall, F1-score
- Per-class metrics
- Confusion matrix
- AUC (with probability predictions)

#### **TrajectoryMetrics**
Specialized metrics for trajectory correction:
- Displacement error (magnitude, per-axis)
- Improvement ratio
- Error distribution statistics

#### **QualityMetrics**
Specialized metrics for quality prediction:
- RUL-specific metrics (warning accuracy)
- Quality score metrics (binary classification)

#### **UnifiedMetrics**
All-in-one metrics computation:
```python
from evaluation.metrics import UnifiedMetrics

metrics = UnifiedMetrics.compute_all(predictions, targets, num_fault_classes=4)
```

### 2. **Visualizer Module** (`evaluation/visualizer.py`)

Generate publication-quality figures:

#### **Available Visualizations**

1. **Regression Results** - Scatter plots with R²
2. **Residual Plots** - Error analysis
3. **Confusion Matrix** - Classification performance
4. **Training History** - Loss and metrics over epochs
5. **Trajectory Comparison** - 3D and 2D trajectory visualization
6. **Error Distribution** - Multi-panel error histograms
7. **Time Series Prediction** - Sequential prediction visualization
8. **Per-Class Metrics** - Bar charts for classification

#### **Usage Example**
```python
from evaluation.visualizer import ResultVisualizer

visualizer = ResultVisualizer(save_dir='results/figures')

# Generate single plot
visualizer.plot_regression_results(
    predictions, targets,
    title="RUL Prediction",
    name="rul_prediction.png"
)

# Generate full report
visualizer.create_evaluation_report(predictions, targets, metrics)
```

### 3. **Benchmark Module** (`evaluation/benchmark.py`)

Compare against baseline methods:

#### **Baseline Models**
- **Mean Predictor**: Predicts training mean
- **Last Value Predictor**: Predicts last training value
- **Random Predictor**: Random classification
- **Majority Class Predictor**: Most frequent class
- **Zero Correction**: No trajectory correction

#### **Usage Example**
```python
from evaluation.benchmark import BenchmarkComparison

benchmark = BenchmarkComparison(save_dir='results/benchmarks')

# Compare single task
results = benchmark.compare_regression(
    model_predictions,
    targets,
    train_targets,
    task_name='RUL_prediction'
)

# Generate full comparison report
benchmark.generate_comparison_report(
    model_predictions,
    targets,
    train_targets
)
```

### 4. **Evaluation Scripts**

#### **evaluate_model.py**
Basic model evaluation:
```bash
python experiments/evaluate_model.py \
    --model_path checkpoints/unified_model/best_model.pth \
    --config_preset unified \
    --batch_size 64 \
    --device cuda
```

#### **full_evaluation_pipeline.py**
Complete evaluation pipeline:
```bash
python experiments/full_evaluation_pipeline.py \
    --model_path checkpoints/unified_model/best_model.pth \
    --output_dir results \
    --device cuda
```

## 🚀 Quick Start

### Option 1: Using Evaluation Scripts

```bash
# Basic evaluation
python experiments/evaluate_model.py \
    --model_path checkpoints/unified_model/best_model.pth

# Full pipeline with visualizations and benchmarks
python experiments/full_evaluation_pipeline.py \
    --model_path checkpoints/unified_model/best_model.pth \
    --output_dir results/full_eval
```

### Option 2: Using Python API

```python
from experiments.evaluate_model import ModelEvaluator

# Create evaluator
evaluator = ModelEvaluator(
    model_path='checkpoints/unified_model/best_model.pth',
    config_preset='unified',
    device='cuda'
)

# Evaluate on dataloader
metrics = evaluator.evaluate_dataloader(test_loader)

# Print results
evaluator.print_metrics()

# Save results
evaluator.save_metrics('results/metrics.txt')
evaluator.generate_visualizations()
evaluator.generate_summary_report()
```

### Option 3: Using Pipeline

```python
from experiments.full_evaluation_pipeline import EvaluationPipeline

# Create pipeline
pipeline = EvaluationPipeline(
    model_path='checkpoints/unified_model/best_model.pth',
    output_dir='results'
)

# Run full evaluation
pipeline.run_full_pipeline(test_loader, train_targets)
```

## 📈 Output Structure

After running evaluation, you'll get:

```
results/
├── metrics.txt                    # Human-readable metrics
├── metrics.json                   # Machine-readable metrics
├── summary_report.txt             # Comprehensive summary
├── figures/                       # All visualizations
│   ├── quality_rul_prediction.png
│   ├── quality_rul_residuals.png
│   ├── fault_confusion_matrix.png
│   ├── trajectory_x_prediction.png
│   ├── error_distributions.png
│   └── ...
└── benchmarks/
    ├── full_comparison.json       # Comparison with baselines
    └── ...
```

## 📊 Understanding the Metrics

### Quality Prediction Metrics

**RUL (Remaining Useful Life):**
- **RMSE**: Root mean squared error (seconds)
  - Target: < 50 seconds
  - Excellent: < 30 seconds
- **R²**: Coefficient of determination
  - Target: > 0.9
  - Excellent: > 0.95

**Temperature:**
- **RMSE**: Temperature error (°C)
  - Target: < 1.0 °C
  - Excellent: < 0.5 °C

**Quality Score:**
- **Binary Accuracy**: Classification accuracy (good/bad)
  - Target: > 0.85
  - Excellent: > 0.95

### Fault Classification Metrics

**Overall Performance:**
- **Accuracy**: Overall correctness
  - Target: > 0.90
  - Excellent: > 0.95

**Per-Class Metrics:**
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
  - Target: > 0.85 for each class
  - Excellent: > 0.90 for each class

### Trajectory Correction Metrics

**Error Magnitude:**
- **Mean Error**: Average displacement error (mm)
  - Target: < 0.01 mm
  - Excellent: < 0.005 mm

**Per-Axis Error:**
- **RMSE**: Root mean squared error per axis
  - Target: < 0.01 mm for X and Y
  - Target: < 0.001 mm for Z

## 🔧 Customization

### Custom Metrics

```python
from evaluation.metrics import RegressionMetrics

# Compute custom metrics
custom_metrics = RegressionMetrics.compute(
    predictions=y_pred,
    targets=y_true,
    prefix='custom_'
)
```

### Custom Visualizations

```python
from evaluation.visualizer import ResultVisualizer

visualizer = ResultVisualizer(save_dir='results/custom')

# Create custom plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(x, y)
visualizer.save_figure('custom_plot.png')
```

### Custom Baselines

```python
from evaluation.benchmark import BaselineModel

# Create custom baseline
custom_baseline = BaselineModel.linear_regression_predictor(
    X_train, y_train, X_test
)
```

## 📝 Example: Complete Evaluation Workflow

```python
# 1. Load model and data
from experiments.full_evaluation_pipeline import EvaluationPipeline
from torch.utils.data import DataLoader

pipeline = EvaluationPipeline(
    model_path='checkpoints/best_model.pth',
    output_dir='results/final_evaluation'
)

# 2. Run full evaluation
pipeline.run_full_pipeline(
    test_loader=test_loader,
    train_targets=train_targets_dict  # Optional, for baseline comparison
)

# 3. Results are automatically saved to:
#    - results/final_evaluation/metrics.txt
#    - results/final_evaluation/metrics.json
#    - results/final_evaluation/summary_report.txt
#    - results/final_evaluation/figures/*.png
#    - results/final_evaluation/benchmarks/*.json
```

## 🎯 Best Practices

1. **Always evaluate on separate test set** - Never use training/validation data for final evaluation
2. **Use multiple metrics** - Different metrics capture different aspects of performance
3. **Visualize results** - Plots reveal patterns that numbers hide
4. **Compare against baselines** - Shows the real value of your model
5. **Report confidence intervals** - Run multiple evaluations for robust estimates

## 🐛 Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size
```python
evaluator = ModelEvaluator(
    model_path='model.pth',
    device='cuda'
)
# Use smaller batch size in data loader
```

### Issue: CUDA Out of Memory

**Solution**: Use CPU or gradient checkpointing
```python
evaluator = ModelEvaluator(
    model_path='model.pth',
    device='cpu'  # Force CPU
)
```

### Issue: Missing Metrics

**Solution**: Check if predictions contain all required fields
```python
print(pipeline.predictions.keys())  # Should see all output keys
print(pipeline.targets.keys())      # Should see all target keys
```

## 📚 Advanced Usage

### Cross-Validation Evaluation

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
all_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    # Train model...
    # Evaluate...
    all_metrics.append(metrics)

# Aggregate metrics
import numpy as np
mean_metrics = {
    key: np.mean([m[key] for m in all_metrics])
    for key in all_metrics[0].keys()
}
```

### Statistical Significance Testing

```python
from scipy.stats import ttest_rel

# Compare two models
model1_errors = ...
model2_errors = ...

t_stat, p_value = ttest_rel(model1_errors, model2_errors)
print(f"p-value: {p_value:.6f}")
```

## 📧 Support

For issues or questions:
1. Check this documentation
2. Check examples in `experiments/`
3. Open an issue on GitHub

---

Happy evaluating! 📊✨
