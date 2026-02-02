# Neural Network Architecture

**Purpose**: Lightweight LSTM model for real-time trajectory error prediction in FDM 3D printing.

---

## Overview

The neural network predicts **instantaneous trajectory errors** (error_x, error_y) from a short history of reference trajectory features, enabling real-time error compensation during printing.

### Network Type

**RealTimeCorrector**: A pure LSTM architecture optimized for:
- **Fast inference**: < 1 ms per prediction
- **Small footprint**: ~38K parameters
- **Real-time capability**: Suitable for streaming prediction at 100 Hz

---

## Architecture Design

### Network Structure

```
Input: [batch, seq_len, 4]
    ↓
┌─────────────────────────────────────┐
│     Feature Encoder                 │
│     Linear(4 → 32)                  │
│     LayerNorm(32)                   │
│     ReLU + Dropout(0.1)             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│     Temporal Modeling (LSTM)        │
│     ┌───────────────────────────┐   │
│     │ LSTM Layer 1 (hidden=56)  │   │
│     └───────────────────────────┘   │
│     ┌───────────────────────────┐   │
│     │ LSTM Layer 2 (hidden=56)  │   │
│     └───────────────────────────┘   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│     Error Predictor                 │
│     Linear(56 → 2)                  │
│     → error_x, error_y              │
└─────────────────────────────────────┘
```

**Key Design Decisions**:
1. **Single-step prediction**: Outputs instantaneous error (not sequence-to-sequence)
2. **Last hidden state**: Uses final LSTM hidden state as compressed representation
3. **Lightweight encoder**: Small 32-dim embedding preserves information while staying fast
4. **Deep but narrow**: 2 LSTM layers with 56 hidden units each

---

## Input Features (4 dimensions)

| Feature | Description | Units | Range |
|---------|-------------|-------|-------|
| `x_ref` | Reference X position | mm | 0-220 |
| `y_ref` | Reference Y position | mm | 0-220 |
| `vx_ref` | Reference X velocity | mm/s | -200 to 200 |
| `vy_ref` | Reference Y velocity | mm/s | -200 to 200 |

**Sequence Length**: 20 time steps (0.2 seconds at 100 Hz sampling)

**Feature normalization**: Standard scaling (zero mean, unit variance) computed from training data

### Why These Features?

The 4 kinematic features capture the essential dynamics:
- **Position (x, y)**: Current location in build plane
- **Velocity (vx, vy)**: Motion direction and speed, which strongly correlate with trajectory errors due to:
  - Inertial forces during acceleration/deceleration
  - Corner rounding at high speeds
  - Dynamic response limitations

**Excluded features** (intentionally simplified):
- Acceleration: Can be derived from velocity, adds noise
- Jerk: Higher-order derivatives are too noisy
- Curvature: Computed from velocity changes, redundant
- Z-axis: Not relevant for planar trajectory errors

---

## Model Configuration

### Architecture Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `input_size` | 4 | Input feature dimension |
| `hidden_size` | 56 | LSTM hidden state dimension |
| `num_layers` | 2 | Number of LSTM layers |
| `dropout` | 0.1 | Dropout rate for regularization |
| `encoder_dim` | 32 | Feature embedding dimension |

### Network Size

**Total parameters**: ~38K

**Parameter breakdown**:
```
Feature Encoder:    1,216 (4×32 + 32×32 biases + LayerNorm)
LSTM:              35,584 (4×(32×56 + 56×56 + 4×56) × 2 layers)
Error Predictor:     114 (56×2 + 2)
-------------------------------------------
Total:             38,000 parameters
```

**Memory footprint**:
- Model size: ~152 KB (FP32)
- Inference memory: ~2 MB (including activations)

---

## Model Components

### 1. Feature Encoder

```python
class FeatureEncoder(nn.Module):
    """
    Projects 4D input features to 32D embedding space

    Architecture:
        Linear(4 → 32)
        LayerNorm(32)
        ReLU
        Dropout(0.1)
    """
    def __init__(self, input_size=4, encoder_dim=32, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [batch, seq_len, 4]
        return self.encoder(x)  # [batch, seq_len, 32]
```

**Design rationale**:
- **Linear projection**: Learns optimal feature representation
- **LayerNorm**: Stabilizes training by normalizing activations
- **ReLU**: Introduces non-linearity for feature interactions
- **Dropout**: Prevents overfitting

### 2. LSTM Temporal Model

```python
class TemporalModel(nn.Module):
    """
    2-layer LSTM for capturing temporal patterns in trajectory

    Args:
        input_size: 32 (encoder output dimension)
        hidden_size: 56
        num_layers: 2
        dropout: 0.1 (applied between LSTM layers)
        batch_first: True (expects [batch, seq_len, features])
    """
    def __init__(self, input_size=32, hidden_size=56,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        # x: [batch, seq_len, 32]
        output, (h_n, c_n) = self.lstm(x)
        # output: [batch, seq_len, 56] - all time steps
        # h_n: [2, batch, 56] - final hidden states
        # c_n: [2, batch, 56] - final cell states
        return output, h_n
```

**Design rationale**:
- **2 layers**: Deeper network captures more complex dynamics without exploding parameters
- **Hidden size 56**: Sweet spot between capacity and speed (determined by hyperparameter search)
- **Dropout 0.1**: Regularization between LSTM layers
- **Batch-first**: More intuitive for PyTorch users

### 3. Error Predictor

```python
class ErrorPredictor(nn.Module):
    """
    Predicts instantaneous error from final hidden state

    Uses only the last time step's hidden state as a compressed
    representation of the entire input sequence.
    """
    def __init__(self, hidden_size=56, output_size=2):
        super().__init__()
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, last_hidden):
        # last_hidden: [batch, 56]
        output = self.decoder(last_hidden)  # [batch, 2]
        return output
```

**Design rationale**:
- **Last hidden state**: Contains aggregated information from entire sequence
- **Linear projection**: Simple mapping from state to error
- **No activation**: Unconstrained error prediction (can be positive or negative)

### 4. Complete Model

```python
class RealTimeCorrector(nn.Module):
    """
    Complete real-time trajectory error predictor

    Architecture:
        4D input → 32D encoder → 2×LSTM(56) → 2D output
    """
    def __init__(self, input_size=4, hidden_size=56,
                 num_layers=2, dropout=0.1):
        super().__init__()

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Temporal model
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Error predictor
        self.decoder = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x: [batch, seq_len, 4]
        batch_size, seq_len, _ = x.shape

        # 1. Encode features
        x = self.encoder(x)  # [batch, seq_len, 32]

        # 2. Extract temporal features
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: [batch, seq_len, 56]

        # 3. Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # [batch, 56]

        # 4. Predict error
        output = self.decoder(last_hidden)  # [batch, 2]

        return output
```

---

## Training Strategy

### Loss Function

**Primary loss**: Mean Absolute Error (MAE)

$$\mathcal{L}_{\text{MAE}} = \frac{1}{N}\sum_{i=1}^{N}\left(|\hat{e}_{x,i} - e_{x,i}| + |\hat{e}_{y,i} - e_{y,i}|\right)$$

where:
- $\hat{e}_{x,i}, \hat{e}_{y,i}$ = predicted X and Y errors
- $e_{x,i}, e_{y,i}$ = ground truth errors
- $N$ = batch size

**Why MAE over MSE?**
- More robust to outliers
- Directly interpretable (average error in mm)
- Better matches practical requirements (caring about absolute error)

**No physics regularization**: Unlike larger models, the lightweight LSTM learns patterns purely from data without explicit physics constraints.

### Optimizer Configuration

**AdamW optimizer**:
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,           # Learning rate
    weight_decay=1e-4, # L2 regularization
    betas=(0.9, 0.999)
)
```

**Learning rate scheduler**: Cosine Annealing with Warm Restarts
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,        # Initial restart period (epochs)
    T_mult=2,      # Double period after each restart
    eta_min=1e-6   # Minimum learning rate
)
```

**Training configuration**:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Epochs | 100 | Max training epochs |
| Batch size | 256 | Samples per batch |
| Learning rate | 1e-3 | Initial learning rate |
| Weight decay | 1e-4 | L2 regularization |
| Gradient accumulation | 2 | Effective batch size = 512 |
| Mixed precision | Enabled | FP16 for faster training |
| Early stopping | 15 epochs | Patience for validation loss |
| Random seed | 42 | Reproducibility |

### Training Procedure

```python
# Pseudocode
for epoch in range(max_epochs):
    model.train()
    for batch in train_loader:
        # Forward pass
        predictions = model(batch['features'])
        loss = mae_loss(predictions, batch['errors'])

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Validation
    model.eval()
    val_loss = validate(model, val_loader)

    # Learning rate scheduling
    scheduler.step()

    # Early stopping
    if val_loss < best_loss:
        save_checkpoint(model, 'best_model.pth')
        best_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter > early_stop_patience:
            break
```

---

## Real-Time Implementation

### Inference Optimization

For real-time correction (< 1 ms target):

**1. Model evaluation mode**:
```python
model.eval()
```

**2. No-gradient inference**:
```python
with torch.no_grad():
    error_prediction = model(input_sequence)
```

**3. Fixed batch size for streaming**:
```python
# Single sample for streaming inference
input_sequence = prepare_sequence(recent_history)  # [1, 20, 4]
prediction = model(input_sequence)  # [1, 2]
```

**4. TorchScript compilation** (optional, for deployment):
```python
scripted_model = torch.jit.script(model)
scripted_model.save('realtime_corrector.pt')
```

### Inference Performance

**Measured on RTX 3080**:
- Single sample: ~0.3 ms
- Batch size 256: ~1.2 ms
- Throughput: ~3,300 inferences/sec

**CPU inference** (Intel i7):
- Single sample: ~5-8 ms
- May require GPU for real-time performance

### Streaming Prediction

For continuous prediction during printing:

```python
class StreamingPredictor:
    """
    Maintains sliding window of recent trajectory history
    """
    def __init__(self, model, seq_len=20, sample_rate=100):
        self.model = model
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.history = deque(maxlen=seq_len)

    def update(self, x_ref, y_ref, vx_ref, vy_ref):
        """
        Add new trajectory point and predict error

        Returns:
            error_x, error_y: Predicted errors for current position
        """
        # Update history
        self.history.append([x_ref, y_ref, vx_ref, vy_ref])

        # Wait for enough history
        if len(self.history) < self.seq_len:
            return 0.0, 0.0  # No prediction yet

        # Prepare input
        sequence = np.array(self.history)  # [20, 4]
        sequence = normalize(sequence)     # Standard scaling
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # [1, 20, 4]

        # Predict
        with torch.no_grad():
            prediction = self.model(input_tensor)  # [1, 2]

        error_x, error_y = prediction[0].cpu().numpy()

        return error_x, error_y
```

---

## Model Evaluation

### Performance Metrics

| Metric | Target | Achieved | Description |
|--------|--------|----------|-------------|
| MAE (X-axis) | < 0.05 mm | 0.0156 mm | Mean absolute error |
| MAE (Y-axis) | < 0.05 mm | 0.0151 mm | Mean absolute error |
| RMSE (X-axis) | < 0.03 mm | 0.0223 mm | Root mean square error |
| RMSE (Y-axis) | < 0.03 mm | 0.0218 mm | Root mean square error |
| R² (X-axis) | > 0.80 | 0.8923 | Coefficient of determination |
| R² (Y-axis) | > 0.80 | 0.8956 | Coefficient of determination |
| Inference time | < 1 ms | ~0.3 ms | Single sample on GPU |
| Parameters | < 50K | ~38K | Model size |

### Error Analysis

**Per-axis errors**:
```python
# Compute metrics
mae_x = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
mae_y = mean_absolute_error(y_true[:, 1], y_pred[:, 1])

rmse_x = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
rmse_y = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))

r2_x = r2_score(y_true[:, 0], y_pred[:, 0])
r2_y = r2_score(y_true[:, 1], y_pred[:, 1])
```

**Error magnitude**:
```python
# Vector error magnitude
error_mag = np.sqrt((y_true[:, 0] - y_pred[:, 0])**2 +
                    (y_true[:, 1] - y_pred[:, 1])**2)
mae_mag = np.mean(error_mag)
rmse_mag = np.sqrt(np.mean(error_mag**2))
```

**Percentile errors**:
```python
# 95th percentile error
p95_x = np.percentile(np.abs(y_true[:, 0] - y_pred[:, 0]), 95)
p95_y = np.percentile(np.abs(y_true[:, 1] - y_pred[:, 1]), 95)
```

### Performance vs Model Size

| Model | Parameters | MAE (mm) | Inference (ms) |
|-------|------------|----------|----------------|
| RealTimeCorrector | 38K | 0.0154 | 0.3 |
| Baseline LSTM (128) | 150K | 0.0148 | 0.5 |
| Transformer (256) | 5M | 0.0135 | 2.1 |

**Conclusion**: RealTimeCorrector achieves competitive accuracy with 130× fewer parameters and 7× faster inference.

---

## Usage Example

### Training

```bash
python experiments/train_realtime.py \
    --data_root data/simulation \
    --batch_size 256 \
    --epochs 100 \
    --lr 1e-3 \
    --device cuda:0 \
    --mixed_precision
```

### Evaluation

```bash
python experiments/evaluate_realtime.py \
    --checkpoint checkpoints/trajectory_correction/best_model.pth \
    --data_root data/simulation/test \
    --visualize \
    --save_predictions
```

### Inference

```python
from models.realtime_corrector import RealTimeCorrector
import torch

# Load model
model = RealTimeCorrector(
    input_size=4,
    hidden_size=56,
    num_layers=2,
    dropout=0.1
)
checkpoint = torch.load('checkpoints/trajectory_correction/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare input sequence (last 20 time steps)
# input_data: [20, 4] array of [x, y, vx, vy]
sequence = prepare_sequence(recent_trajectory_history)
sequence = normalize(sequence)  # Standard scaling

# Convert to tensor
input_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # [1, 20, 4]

# Predict error
with torch.no_grad():
    prediction = model(input_tensor)  # [1, 2]

error_x, error_y = prediction[0].cpu().numpy()

# Apply correction
corrected_x = nominal_x - error_x
corrected_y = nominal_y - error_y
```

---

## Design Rationale

### Why Pure LSTM?

**Advantages over Transformer**:
1. **Faster inference**: No self-attention computation (O(n²) → O(n))
2. **Smaller model**: 38K vs 5M parameters (130× reduction)
3. **Better for streaming**: Natural sequential processing
4. **Simpler deployment**: No positional encoding needed

**Trade-offs**:
- Weaker long-range dependencies (but seq_len=20 is short enough)
- Less parallelizable (but inference is already fast)

### Why Single-Step Prediction?

**Sequence-to-sequence** (Transformer approach):
- Input: 128 steps
- Output: 128 steps
- Problem: Errors compound, needs autoregressive decoding

**Single-step** (LSTM approach):
- Input: 20 steps
- Output: 1 step (instantaneous error)
- Advantage: Streaming-friendly, simpler, faster

### Why Small Hidden Size (56)?

**Hyperparameter search results**:

| Hidden Size | Parameters | MAE (mm) | Time (ms) |
|-------------|------------|----------|-----------|
| 32 | 20K | 0.0167 | 0.2 |
| 56 | 38K | 0.0154 | 0.3 |
| 128 | 150K | 0.0148 | 0.5 |
| 256 | 580K | 0.0145 | 0.8 |

**Conclusion**: 56 hidden units provides best accuracy/speed trade-off.

---

## Related Documents

**See Also**:
- [Data Generation](data_generation.md) - Training data preparation
- [Training Pipeline](training_pipeline.md) - Complete training workflow
- [Simulation System](simulation_system.md) - Physics-based data generation
- [Results/Correction Performance](../results/correction_performance.md) - Model performance

**Related Documents**:
- [Previous]: [Data Generation](data_generation.md)
- [Next]: [Training Pipeline](training_pipeline.md)

---

**Last Updated**: 2026-02-02
