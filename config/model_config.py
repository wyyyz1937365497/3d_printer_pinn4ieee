"""
Model-specific configurations and presets
"""

from .base_config import BaseConfig, ModelConfig, TrainingConfig, DataConfig


def get_quality_prediction_config() -> BaseConfig:
    """Configuration for quality prediction model only"""
    config = BaseConfig()
    config.experiment_name = "quality_prediction_only"
    config.lambda_fault = 0.0
    config.lambda_trajectory = 0.0
    config.lambda_physics = 0.2
    config.model.use_pinn = True
    return config


def get_trajectory_correction_config() -> BaseConfig:
    """Configuration for trajectory correction model only"""
    config = BaseConfig()
    config.experiment_name = "trajectory_correction_only"
    config.lambda_quality = 0.0
    config.lambda_fault = 0.0
    config.lambda_trajectory = 1.0
    config.lambda_physics = 0.05
    config.model.trajectory_use_lstm = True
    config.model.trajectory_bidirectional = True
    config.data.seq_len = 10  # Shorter sequence for trajectory
    config.data.pred_len = 1
    return config


def get_unified_model_config() -> BaseConfig:
    """Configuration for unified model (all tasks)"""
    config = BaseConfig()
    config.experiment_name = "unified_model_all_tasks"
    config.lambda_quality = 1.0
    config.lambda_fault = 1.0
    config.lambda_trajectory = 1.0
    config.lambda_physics = 0.1
    return config


def get_fast_inference_config() -> BaseConfig:
    """Configuration for fast inference (lighter model)"""
    config = BaseConfig()
    config.model.d_model = 128
    config.model.num_heads = 4
    config.model.num_layers = 3
    config.model.dim_feedforward = 512
    config.training.mixed_precision = True
    config.training.batch_size = 128
    return config


def get_research_config() -> BaseConfig:
    """Configuration for research/development (heavier model)"""
    config = BaseConfig()
    config.model.d_model = 512
    config.model.num_heads = 16
    config.model.num_layers = 8
    config.model.dim_feedforward = 2048
    config.training.num_epochs = 200
    config.training.early_stopping_patience = 20
    return config


# Preset configurations dictionary
MODEL_PRESETS = {
    "quality": get_quality_prediction_config,
    "trajectory": get_trajectory_correction_config,
    "unified": get_unified_model_config,
    "fast": get_fast_inference_config,
    "research": get_research_config,
}


def get_config(preset: str = "unified", **kwargs) -> BaseConfig:
    """
    Get configuration with optional preset and custom overrides

    Args:
        preset: Name of the preset configuration
        **kwargs: Custom configuration overrides

    Returns:
        BaseConfig object with applied overrides
    """
    if preset in MODEL_PRESETS:
        config = MODEL_PRESETS[preset]()
    else:
        config = BaseConfig()

    if kwargs:
        config.update(**kwargs)

    return config
