"""
用于训练和评估的基础配置
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径
    data_dir: str = "data/processed"
    train_dir: str = "data_simulation_layer25"  # MATLAB仿真数据
    val_dir: str = "validation_layer25"
    test_dir: str = "data_test"

    # 数据参数
    seq_len: int = 200  # 序列长度（时间步）
    pred_len: int = 50  # 预测长度（时间步）
    stride: int = 10  # 序列之间的步长
    sampling_rate: int = 100  # 采样频率（Hz，MATLAB使用100Hz）

    # 输入特征（共12个）- 明确定义
    input_features: list = field(default_factory=lambda: [
        # 理想轨迹（6个）- 理想轨迹
        'x_ref', 'y_ref', 'z_ref',           # 参考位置
        'vx_ref', 'vy_ref', 'vz_ref',        # 参考速度

        # 可观测测量值（6个）- 显式测量量
        'T_nozzle', 'T_interface',            # 温度
        'F_inertia_x', 'F_inertia_y',        # 惯性力
        'cooling_rate', 'layer_num'          # 冷却速率 + 层号
    ])

    # 输出标签 - 明确定义
    output_trajectory: list = field(default_factory=lambda: [
        'error_x', 'error_y'                  # 误差向量 (2D)
    ])

    output_quality: list = field(default_factory=lambda: [
        'adhesion_ratio',                    # 粘结强度比
        'internal_stress',                   # 内应力 (MPa)
        'porosity',                          # 孔隙率 (%)
        'dimensional_accuracy',              # 尺寸精度
        'quality_score'                      # 综合质量评分
    ])

    # 特征维度（向后兼容）
    num_features: int = 12  # 输入特征数量
    num_quality_outputs: int = 5  # 质量输出数量
    num_fault_classes: int = 4  # 正常、喷嘴堵塞、机械松动、电机故障
    num_trajectory_outputs: int = 2  # error_x, error_y

    # 预处理
    normalize: bool = True
    train_val_test_split: tuple = (0.7, 0.15, 0.15)


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基本参数
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # 优化器
    optimizer: str = "adamw"
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

    # 调度器
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # 训练技巧
    gradient_clip: float = 1.0
    accumulation_steps: int = 2
    mixed_precision: bool = True
    early_stopping_patience: int = 10

    # 检查点保存
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5  # 每N个epoch保存一次
    save_best_only: bool = True

    # 日志记录
    log_dir: str = "logs"
    log_interval: int = 10  # 每N个批次记录一次日志
    use_tensorboard: bool = True
    use_wandb: bool = False


@dataclass
class ModelConfig:
    """模型架构配置"""
    # 编码器（Transformer）
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"

    # PINN参数
    use_pinn: bool = True
    physics_loss_weight: float = 0.1

    # 质量预测头
    quality_hidden_dims: list = field(default_factory=lambda: [256, 128])
    quality_dropout: float = 0.2

    # 故障分类头
    fault_hidden_dims: list = field(default_factory=lambda: [128])
    fault_dropout: float = 0.3

    # 轨迹校正头
    trajectory_use_lstm: bool = True
    trajectory_lstm_hidden: int = 128
    trajectory_lstm_layers: int = 2
    trajectory_bidirectional: bool = True
    trajectory_attention: bool = True


@dataclass
class PhysicsConfig:
    """物理约束配置"""
    # 热物理学
    thermal_diffusivity: float = 1.0
    heat_source_strength: float = 1.0

    # 振动物理学
    mass: float = 1.0
    damping: float = 0.5
    stiffness: float = 10.0

    # 能量守恒
    energy_loss_weight: float = 1.0

    # 电机耦合
    motor_coupling_weight: float = 1.0


@dataclass
class BaseConfig:
    """主配置类"""
    # 随机种子
    seed: int = 42

    # 设备
    device: str = "cuda"  # 或 "cpu"
    num_workers: int = 4

    # 子配置
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)

    # 实验名称
    experiment_name: str = "unified_pinn_seq3d"

    # 多任务学习的损失权重
    lambda_quality: float = 1.0
    lambda_fault: float = 1.0
    lambda_trajectory: float = 1.0
    lambda_physics: float = 0.1

    def __post_init__(self):
        """创建必要的目录"""
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)
        os.makedirs(self.training.log_dir, exist_ok=True)
        os.makedirs(self.data.data_dir, exist_ok=True)

    def update(self, **kwargs):
        """使用新值更新配置"""
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
                raise ValueError(f"未知配置键: {key}")