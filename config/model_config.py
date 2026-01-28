"""
模型特定配置和预设
"""

from .base_config import BaseConfig, ModelConfig, TrainingConfig, DataConfig


def get_quality_prediction_config() -> BaseConfig:
    """仅用于质量预测模型的配置"""
    config = BaseConfig()
    config.experiment_name = "quality_prediction_only"
    config.lambda_fault = 0.0
    config.lambda_trajectory = 0.0
    config.lambda_physics = 0.2
    config.model.use_pinn = True
    return config


def get_trajectory_correction_config() -> BaseConfig:
    """仅用于轨迹校正模型的配置"""
    config = BaseConfig()
    config.experiment_name = "trajectory_correction_only"
    config.lambda_quality = 0.0
    config.lambda_fault = 0.0
    config.lambda_trajectory = 1.0
    config.lambda_physics = 0.05
    config.model.trajectory_use_lstm = True
    config.model.trajectory_bidirectional = True
    config.data.seq_len = 10  # 轨迹的较短序列
    config.data.pred_len = 1
    return config


def get_unified_model_config() -> BaseConfig:
    """统一模型（所有任务）的配置"""
    config = BaseConfig()
    config.experiment_name = "unified_model_all_tasks"
    config.lambda_quality = 1.0
    config.lambda_fault = 1.0
    config.lambda_trajectory = 1.0
    config.lambda_physics = 0.1
    return config


def get_fast_inference_config() -> BaseConfig:
    """快速推理（轻量级模型）的配置"""
    config = BaseConfig()
    config.model.d_model = 128
    config.model.num_heads = 4
    config.model.num_layers = 3
    config.model.dim_feedforward = 512
    config.training.mixed_precision = True
    config.training.batch_size = 128
    return config


def get_research_config() -> BaseConfig:
    """研究/开发（重量级模型）的配置"""
    config = BaseConfig()
    config.model.d_model = 512
    config.model.num_heads = 16
    config.model.num_layers = 8
    config.model.dim_feedforward = 2048
    config.training.num_epochs = 200
    config.training.early_stopping_patience = 20
    return config


# 预设配置字典
MODEL_PRESETS = {
    "quality": get_quality_prediction_config,
    "trajectory": get_trajectory_correction_config,
    "unified": get_unified_model_config,
    "fast": get_fast_inference_config,
    "research": get_research_config,
}


def get_config(preset: str = "unified", **kwargs) -> BaseConfig:
    """
    获取带有可选预设和自定义覆盖的配置

    参数:
        preset: 预设配置的名称
        **kwargs: 自定义配置覆盖

    返回:
        应用覆盖的BaseConfig对象
    """
    if preset in MODEL_PRESETS:
        config = MODEL_PRESETS[preset]()
    else:
        config = BaseConfig()

    if kwargs:
        config.update(**kwargs)

    return config