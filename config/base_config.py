"""
Base configuration for training and evaluation
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    """Data configuration"""
    # Data paths
    data_dir: str = "data/processed"
    train_file: str = "train_data.pkl"
    val_file: str = "val_data.pkl"
    test_file: str = "test_data.pkl"

    # Data parameters
    seq_len: int = 200  # Sequence length (timesteps)
    pred_len: int = 50  # Prediction length (timesteps)
    sampling_rate: int = 1000  # Hz

    # Feature dimensions
    num_features: int = 12  # Number of input features (observable sensor data)
    num_quality_outputs: int = 5  # Implicit quality parameters: adhesion_strength, internal_stress, porosity, dimensional_accuracy, quality_score
    num_fault_classes: int = 4  # Normal, Nozzle Clog, Mechanical Loose, Motor Fault
    num_trajectory_outputs: int = 3  # dx, dy, dz

    # Preprocessing
    normalize: bool = True
    train_val_test_split: tuple = (0.7, 0.15, 0.15)


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic parameters
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Optimizer
    optimizer: str = "adamw"
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Training tricks
    gradient_clip: float = 1.0
    accumulation_steps: int = 2
    mixed_precision: bool = True
    early_stopping_patience: int = 10

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5  # Save every N epochs
    save_best_only: bool = True

    # Logging
    log_dir: str = "logs"
    log_interval: int = 10  # Log every N batches
    use_tensorboard: bool = True
    use_wandb: bool = False


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Encoder (Transformer)
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"

    # PINN parameters
    use_pinn: bool = True
    physics_loss_weight: float = 0.1

    # Quality prediction head
    quality_hidden_dims: list = field(default_factory=lambda: [256, 128])
    quality_dropout: float = 0.2

    # Fault classification head
    fault_hidden_dims: list = field(default_factory=lambda: [128])
    fault_dropout: float = 0.3

    # Trajectory correction head
    trajectory_use_lstm: bool = True
    trajectory_lstm_hidden: int = 128
    trajectory_lstm_layers: int = 2
    trajectory_bidirectional: bool = True
    trajectory_attention: bool = True


@dataclass
class PhysicsConfig:
    """Physics constraint configuration"""
    # Thermal physics
    thermal_diffusivity: float = 1.0
    heat_source_strength: float = 1.0

    # Vibration physics
    mass: float = 1.0
    damping: float = 0.5
    stiffness: float = 10.0

    # Energy conservation
    energy_loss_weight: float = 1.0

    # Motor coupling
    motor_coupling_weight: float = 1.0


@dataclass
class BaseConfig:
    """Main configuration class"""
    # Random seed
    seed: int = 42

    # Device
    device: str = "cuda"  # or "cpu"
    num_workers: int = 4

    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)

    # Experiment name
    experiment_name: str = "unified_pinn_seq3d"

    # Loss weights for multi-task learning
    lambda_quality: float = 1.0
    lambda_fault: float = 1.0
    lambda_trajectory: float = 1.0
    lambda_physics: float = 0.1

    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)
        os.makedirs(self.training.log_dir, exist_ok=True)
        os.makedirs(self.data.data_dir, exist_ok=True)

    def update(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self.data, key):
                setattr(self.data, key, value)
            elif hasattr(self.training, key):
                setattr(self.training, key, value)
            elif hasattr(self.model, key):
                setattr(self.model, key, value)
            elif hasattr(self.physics, key):
                setattr(self.physics, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")
