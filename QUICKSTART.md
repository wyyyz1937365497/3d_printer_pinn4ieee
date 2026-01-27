# Quick Start Guide

Welcome to the 3D Printer PINN-Seq3D Framework! This guide will help you get started quickly.

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd 3d_printer_pinn4ieee
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📊 Generate Data

### Option 1: Generate Synthetic Data (Quick Start)

```bash
# Generate quality prediction data
python data/scripts/generate_physics_data.py --num_samples 10000

# Generate trajectory correction data
python data/scripts/generate_trajectory_data.py --num_sequences 5000
```

### Option 2: Use Your Own Data

Replace the synthetic data generation with your actual 3D printer data. The data format should be:

**Quality Prediction Data:**
- Features: `[batch, seq_len, num_features]`
  - Temperature (nozzle, bed)
  - Vibration (x, y, z)
  - Motor current (x, y, z)
  - Pressure
  - Position (x, y, z)

**Trajectory Correction Data:**
- Features: `[batch, seq_len, 4]`
  - Position (x, y, z)
  - Velocity

## 🎯 Train Models

### Quick Training (Synthetic Data)

```bash
# Train unified model with default settings
python experiments/train_unified_model.py
```

### Training with Custom Configuration

```python
from config import get_config
from models import UnifiedPINNSeq3D
from training import Trainer

# Create custom configuration
config = get_config(
    preset='unified',
    experiment_name='my_experiment',
    d_model=256,
    num_heads=8,
    learning_rate=1e-4,
    batch_size=64,
    num_epochs=50,
)

# Create model
model = UnifiedPINNSeq3D(config)

# Create trainer (you need to provide data loaders)
trainer = Trainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
)

# Train
trainer.train()
```

## 🔮 Use Trained Model

### Load Model and Make Predictions

```python
from inference import UnifiedPredictor
import numpy as np

# Load trained model
predictor = UnifiedPredictor.load_from_checkpoint(
    'checkpoints/unified_model/best_model.pth',
    device='cpu'  # or 'cuda'
)

# Prepare sensor data
sensor_data = np.random.randn(200, 12)  # [seq_len, num_features]

# Make predictions
results = predictor.predict(sensor_data)

# Access results
print(f"RUL: {results['quality']['rul'][0][0]:.2f} seconds")
print(f"Quality Score: {results['quality']['quality_score'][0][0]:.3f}")
print(f"Fault Class: {results['fault']['predicted_class'][0]}")
print(f"Trajectory Correction (dx): {results['trajectory']['dx'][0][0]:.4f} mm")
```

### Quality-Based Early Stopping

```python
# Monitor quality during printing
sensor_data = get_realtime_sensor_data()  # Your data

# Predict quality
quality = predictor.predict_quality_only(sensor_data)
quality_score = quality['quality_score'][0][0]

# Make decision
if quality_score < 0.5:
    print("Low quality detected! Stopping print.")
    stop_printing()
```

### Trajectory Correction

```python
# Current position
current_pos = np.array([100.0, 100.0, 0.2])  # x, y, z

# Get sensor data
sensor_data = get_recent_sensor_data()  # Your data

# Apply correction
corrected_pos = predictor.get_trajectory_correction(sensor_data, current_pos)

# Move to corrected position
move_to(corrected_pos)
```

## 📁 Project Structure

```
3d_printer_pinn4ieee/
├── config/              # Configuration files
├── data/                # Data directory
│   ├── scripts/        # Data generation scripts
│   └── processed/      # Processed data
├── models/              # Model definitions
│   ├── encoders/       # Encoder modules
│   ├── decoders/       # Decoder modules
│   └── physics/        # Physics constraints
├── training/            # Training utilities
├── inference/           # Inference utilities
├── utils/              # Helper functions
├── experiments/         # Experiment scripts
├── examples/           # Usage examples
└── checkpoints/        # Saved models
```

## 🎓 Examples

Check out `examples/usage_examples.py` for detailed examples:

```bash
python examples/usage_examples.py
```

Examples include:
1. Creating a model
2. Training a model
3. Making predictions
4. Quality-based early stopping
5. Trajectory correction
6. Custom configuration

## ⚙️ Configuration Presets

The framework includes several configuration presets:

- **`unified`**: Full model with all tasks (default)
- **`quality`**: Quality prediction only
- **`trajectory`**: Trajectory correction only
- **`fast`**: Lightweight model for fast inference
- **`research`**: Heavy model for research

```python
from config import get_config

# Use preset
config = get_config(preset='quality')

# Or customize
config = get_config(preset='unified', learning_rate=5e-4)
```

## 🔧 Common Tasks

### Change Model Size

```python
config = get_config(preset='unified')
config.model.d_model = 128  # Smaller model
config.model.num_layers = 4  # Fewer layers
```

### Adjust Loss Weights

```python
config = get_config(preset='unified')
config.lambda_quality = 2.0  # Emphasize quality
config.lambda_trajectory = 0.5  # De-emphasize trajectory
config.lambda_physics = 0.2  # Increase physics constraint
```

### Use Different Data

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self):
        # Load your data
        pass

    def __getitem__(self, idx):
        # Return dictionary with 'features' and targets
        return {
            'features': self.features[idx],
            'rul': self.targets['rul'][idx],
            'temperature': self.targets['temperature'][idx],
            # ... other targets
        }

# Create data loader
my_dataset = MyDataset()
train_loader = DataLoader(my_dataset, batch_size=64)
```

## 📈 Monitor Training

Training logs are saved to `logs/` directory. You can visualize them with TensorBoard:

```bash
tensorboard --logdir logs
```

## 🐛 Troubleshooting

### Out of Memory Error

```python
# Reduce batch size
config.training.batch_size = 32

# Or use gradient accumulation
config.training.accumulation_steps = 4

# Or use smaller model
config = get_config(preset='fast')
```

### Slow Training

```python
# Use mixed precision training
config.training.mixed_precision = True

# Increase batch size if you have enough memory
config.training.batch_size = 128

# Use fewer workers for data loading
config.num_workers = 2
```

### Poor Performance

```python
# Adjust loss weights
config.lambda_quality = 1.0
config.lambda_fault = 1.0
config.lambda_trajectory = 1.0
config.lambda_physics = 0.1

# Increase training epochs
config.training.num_epochs = 100

# Try different learning rate
config.training.learning_rate = 5e-4
```

## 📚 Next Steps

1. **Read the documentation**: Check `PROJECT_STRUCTURE.md` for detailed architecture
2. **Run examples**: Explore `examples/usage_examples.py`
3. **Customize**: Modify configuration and model architecture for your needs
4. **Train on real data**: Replace synthetic data with your actual 3D printer data

## 🤝 Need Help?

- Check `README.md` for overview
- Check `PROJECT_STRUCTURE.md` for architecture details
- Open an issue on GitHub
- Check existing issues and discussions

---

Happy printing! 🖨️✨
