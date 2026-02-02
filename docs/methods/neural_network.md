# Neural Network Architecture

**Purpose**: Model architecture for real-time trajectory error prediction and correction.

---

## Overview

The neural network predicts **trajectory error sequences** (error_x, error_y) from input trajectory features, enabling real-time error compensation during printing.

### Network Type

**TrajectoryErrorTransformer**: A Transformer-based encoder-decoder architecture for sequence-to-sequence error prediction.

---

## Architecture Design

### Network Structure

```
Input Features [batch, seq_len, 15]
    ↓
┌─────────────────────────────────────┐
│     Positional Encoding              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│     Transformer Encoder             │
│     ┌───────────────────────────┐   │
│     │ Multi-Head Attention (×N)│   │
│     └───────────────────────────┘   │
│     ┌───────────────────────────┐   │
│     │ Feed Forward Network      │   │
│     └───────────────────────────┘   │
│              (×L layers)             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│     Sequence Decoder (LSTM)          │
│     ┌───────────────────────────┐   │
│     │ LSTM Layer                │   │
│     └───────────────────────────┘   │
│     ┌───────────────────────────┐   │
│     │ Attention Mechanism      │   │
│     └───────────────────────────┘   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│     Output Projection               │
│     Linear(2) → error_x, error_y    │
└─────────────────────────────────────┘
```

---

## Input Features (15 dimensions)

| Category | Features | Dimensions |
|----------|----------|------------|
| Position | x, y, z | 3 |
| Velocity | vx, vy, vz, v_mag | 4 |
| Acceleration | ax, ay, az, a_mag | 4 |
| Jerk | jx, jy, jz | 3 |
| Curvature | curvature | 1 |

**Total**: 15 features per time step

**Feature normalization**: Standard scaling (zero mean, unit variance)

---

## Model Configuration

### Architecture Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 256 | Transformer dimension |
| `nhead` | 8 | Number of attention heads |
| `num_layers` | 6 | Number of encoder layers |
| `dim_feedforward` | 1024 | FFN hidden dimension |
| `dropout` | 0.1 | Dropout rate |
| `sequence_length` | 128 | Input sequence length |
| `prediction_length` | 128 | Output sequence length |

### Network Size

**Total parameters**: ~5M

**Parameter breakdown**:
- Transformer encoder: ~4.2M
- LSTM decoder: ~0.6M
- Output projection: ~0.2M

---

## Model Components

### 1. Input Embedding

```python
class InputEmbedding(nn.Module):
    def __init__(self, in_features=15, d_model=256):
        self.linear = nn.Linear(in_features, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [batch, seq_len, in_features]
        x = self.linear(x)  # [batch, seq_len, d_model]
        x = self.norm(x)
        return x
```

### 2. Transformer Encoder

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=256,
    nhead=8,
    dim_feedforward=1024,
    dropout=0.1,
    activation='gelu'
)

self.encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=6
)
```

### 3. LSTM Decoder

```python
self.decoder = nn.LSTM(
    input_size=256,
    hidden_size=256,
    num_layers=2,
    dropout=0.1,
    batch_first=True
)
```

### 4. Output Projection

```python
self.output_layer = nn.Sequential(
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, 2)  # error_x, error_y
)
```

---

## Training Strategy

### Loss Function

**Primary loss**: Mean Absolute Error (MAE)

$$\mathcal{L}_{\text{MAE}} = \frac{1}{N}\sum_{i=1}^{N}|\mathbf{y}_i - \hat{\mathbf{y}}_i|$$

**Combined loss**:

$$\mathcal{L} = \mathcal{L}_{\text{MAE}}^x + \mathcal{L}_{\text{MAE}}^y + \lambda \mathcal{L}_{\text{physics}}$$

where $\mathcal{L}_{\text{physics}}$ enforces physical constraints (e.g., error-velocity correlation).

### Optimizer

**AdamW** with learning rate scheduling:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10
)
```

### Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Epochs | 100 | Max training epochs |
| Batch size | 32 | Samples per batch |
| Learning rate | 1e-4 | Initial learning rate |
| Early stopping | 15 epochs | Patience for validation loss |
| Mixed precision | Enabled | Faster training |
| Gradient clipping | 1.0 | Prevent exploding gradients |

---

## Real-Time Implementation

### Inference Optimization

For real-time correction (< 1 ms inference):

1. **Model optimization**:
   ```python
   model.eval()

   # TorchScript compilation
   scripted_model = torch.jit.script(model)
   scripted_model.save('trajectory_model.pt')
   ```

2. **Fixed batch size**: Use batch_size=1 for streaming

3. **Sequence sliding**: Process sequences with overlap

### Expected Performance

| Hardware | Inference Time | Batch Size |
|----------|---------------|------------|
| RTX 3080 | < 0.5 ms | 1 |
| GTX 1080 | < 1.0 ms | 1 |
| CPU (i7) | ~5-10 ms | 1 |

**Target**: < 1 ms for real-time correction ✅

---

## Model Evaluation

### Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| MAE (X-axis) | < 0.02 mm | Mean absolute error |
| MAE (Y-axis) | < 0.02 mm | Mean absolute error |
| RMSE (X-axis) | < 0.03 mm | Root mean square error |
| RMSE (Y-axis) | < 0.03 mm | Root mean square error |
| R² (X-axis) | > 0.85 | Coefficient of determination |
| R² (Y-axis) | > 0.85 | Coefficient of determination |

### Error Analysis

```python
# Per-axis error
mae_x = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
mae_y = mean_absolute_error(y_true[:, 1], y_pred[:, 1])

# Magnitude error
error_mag = sqrt((y_true[:, 0] - y_pred[:, 0])**2 +
                 (y_true[:, 1] - y_pred[:, 1])**2)
mae_mag = mean(error_mag)

# Coefficient of determination
r2_x = r2_score(y_true[:, 0], y_pred[:, 0])
r2_y = r2_score(y_true[:, 1], y_pred[:, 1])
```

---

## Usage Example

### Training

```bash
python experiments/train_trajectory_model.py \
    --data_root data/processed \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --device cuda:0 \
    --experiment_name trajectory_correction \
    --mixed_precision
```

### Evaluation

```bash
python experiments/evaluate_trajectory_model.py \
    --checkpoint checkpoints/trajectory_correction/best_model.pt \
    --data_root data/processed/test \
    --visualize \
    --save_predictions
```

### Inference

```python
from models.trajectory import TrajectoryErrorTransformer
import torch

# Load model
model = TrajectoryErrorTransformer.load_from_checkpoint(
    'checkpoints/trajectory_correction/best_model.pt'
)
model.eval()

# Prepare input
# input_features: [batch, sequence_length, 15]
input_features = prepare_trajectory_features(gcode_trajectory)

# Predict
with torch.no_grad():
    predictions = model(input_features)

# Extract errors
error_x = predictions['error_x']  # [batch, sequence_length, 1]
error_y = predictions['error_y']  # [batch, sequence_length, 1]

# Apply correction
corrected_x = nominal_x - error_x
corrected_y = nominal_y - error_y
```

---

## Model File Structure

```
models/
├── __init__.py
├── trajectory/
│   ├── __init__.py
│   ├── transformer.py           % Transformer encoder
│   ├── decoder.py               % LSTM decoder
│   └── model.py                 % Complete model
└── training/
    ├── loss.py                  % Loss functions
    ├── optimizer.py             % Optimizer setup
    └── scheduler.py             % Learning rate scheduling
```

---

## Architecture Alternatives

### Option 1: Pure Transformer

```
Encoder: Transformer
Decoder: Transformer (autoregressive)
```

**Pros**: Better long-range dependencies
**Cons**: Slower inference for real-time

### Option 2: LSTM Only

```
Encoder: BiLSTM (2 layers)
Decoder: LSTM (2 layers)
```

**Pros**: Faster inference
**Cons**: Weaker long-range modeling

### Option 3: TCN (Temporal Convolutional Network)

```
Encoder: TCN (dilated convolutions)
Decoder: TCN
```

**Pros**: Parallel processing, fast
**Cons**: Fixed receptive field

**Current choice**: Transformer encoder + LSTM decoder (balance of accuracy and speed)

---

## References

**See Also**:
- [Data Generation](data_generation.md) - Training data
- [Training Pipeline](training_pipeline.md) - How to train
- [Results/Correction Performance](../results/correction_performance.md) - Model performance

**Related Documents**:
- [Previous]: [Data Generation](data_generation.md)
- [Next]: [Training Pipeline](training_pipeline.md)
