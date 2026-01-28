"""
统一PINN-Seq3D模型

在单一模型中结合质量预测和轨迹校正
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .base_model import BaseModel
from .encoders import PINNTransformerEncoder
from .decoders.quality_decoder import QualityPredictionHead, FaultClassificationHead
from .decoders.trajectory_decoder import TrajectoryCorrectionHead


class UnifiedPINNSeq3D(BaseModel):
    """
    用于3D打印机质量预测和轨迹校正的统一模型

    架构:
        输入（传感器数据）-> 共享PINN引导的Transformer编码器
                                 -> 质量预测头（RUL、温度、振动、质量分数）
                                 -> 故障分类头（4种故障类型）
                                 -> 轨迹校正头（dx, dy, dz）
    """

    def __init__(self, config):
        """
        初始化统一模型

        参数:
            config: 包含模型参数的配置对象
        """
        super().__init__(config)

        # 共享编码器
        self.encoder = PINNTransformerEncoder(
            num_features=config.data.num_features,
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            d_ff=config.model.dim_feedforward,
            dropout=config.model.dropout,
            activation=config.model.activation,
        )

        # 质量预测头
        self.quality_head = QualityPredictionHead(
            d_model=config.model.d_model,
            hidden_dims=config.model.quality_hidden_dims,
            dropout=config.model.quality_dropout,
            num_outputs=config.data.num_quality_outputs,
        )

        # 故障分类头
        self.fault_head = FaultClassificationHead(
            d_model=config.model.d_model,
            hidden_dims=config.model.fault_hidden_dims,
            dropout=config.model.fault_dropout,
            num_classes=config.data.num_fault_classes,
        )

        # 轨迹校正头
        self.trajectory_head = TrajectoryCorrectionHead(
            d_model=config.model.d_model,
            lstm_hidden=config.model.trajectory_lstm_hidden,
            lstm_layers=config.model.trajectory_lstm_layers,
            bidirectional=config.model.trajectory_bidirectional,
            use_attention=config.model.trajectory_attention,
            num_outputs=config.data.num_trajectory_outputs,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            x: 输入张量 [batch, seq_len, num_features]
            mask: 可选的注意力掩码 [batch, seq_len, seq_len]

        返回:
            包含所有模型输出的字典
        """
        # 共享编码
        encoded = self.encoder(x, mask)  # [batch, seq_len, d_model]

        # 质量预测
        quality_outputs = self.quality_head(encoded)

        # 故障分类
        fault_outputs = self.fault_head(encoded)

        # 轨迹校正
        trajectory_outputs = self.trajectory_head(encoded, mask)

        # 合并所有输出
        outputs = {
            'encoded': encoded,
            **quality_outputs,
            **fault_outputs,
            **trajectory_outputs,
        }

        return outputs

    def predict_quality(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        仅预测质量指标

        参数:
            x: 输入张量
            mask: 可选的注意力掩码

        返回:
            质量预测输出
        """
        encoded = self.encoder(x, mask)
        return self.quality_head(encoded)

    def predict_fault(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        仅预测故障分类

        参数:
            x: 输入张量
            mask: 可选的注意力掩码

        返回:
            故障分类输出
        """
        encoded = self.encoder(x, mask)
        return self.fault_head(encoded)

    def predict_trajectory(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        仅预测轨迹校正

        参数:
            x: 输入张量
            mask: 可选的注意力掩码

        返回:
            轨迹校正输出
        """
        encoded = self.encoder(x, mask)
        return self.trajectory_head(encoded, mask)

    def get_model_info(self) -> Dict[str, any]:
        """
        获取模型信息

        返回:
            包含模型详细信息的字典
        """
        return {
            'model_type': 'UnifiedPINNSeq3D',
            'num_parameters': self.get_num_params(),
            'num_trainable_parameters': self.get_num_trainable_params(),
            'encoder_output_dim': self.encoder.get_output_dim(),
            'quality_outputs': self.quality_head.get_output_dim(),
            'fault_classes': self.fault_head.get_output_dim(),
            'trajectory_outputs': self.trajectory_head.get_output_dim(),
        }


class QualityPredictionOnlyModel(BaseModel):
    """
    仅质量预测模型（无轨迹校正）
    """

    def __init__(self, config):
        """
        初始化质量预测模型

        参数:
            config: 配置对象
        """
        super().__init__(config)

        # 编码器
        self.encoder = PINNTransformerEncoder(
            num_features=config.data.num_features,
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            d_ff=config.model.dim_feedforward,
            dropout=config.model.dropout,
            activation=config.model.activation,
        )

        # 质量预测头
        self.quality_head = QualityPredictionHead(
            d_model=config.model.d_model,
            hidden_dims=config.model.quality_hidden_dims,
            dropout=config.model.quality_dropout,
            num_outputs=config.data.num_quality_outputs,
        )

        # 故障分类头
        self.fault_head = FaultClassificationHead(
            d_model=config.model.d_model,
            hidden_dims=config.model.fault_hidden_dims,
            dropout=config.model.fault_dropout,
            num_classes=config.data.num_fault_classes,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            x: 输入张量
            mask: 可选的注意力掩码

        返回:
            质量和故障输出
        """
        encoded = self.encoder(x, mask)

        quality_outputs = self.quality_head(encoded)
        fault_outputs = self.fault_head(encoded)

        return {
            'encoded': encoded,
            **quality_outputs,
            **fault_outputs,
        }


class TrajectoryCorrectionOnlyModel(BaseModel):
    """
    仅轨迹校正模型（无质量预测）
    """

    def __init__(self, config):
        """
        初始化轨迹校正模型

        参数:
            config: 配置对象
        """
        super().__init__(config)

        # 编码器
        self.encoder = PINNTransformerEncoder(
            num_features=config.data.num_features,
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            d_ff=config.model.dim_feedforward,
            dropout=config.model.dropout,
            activation=config.model.activation,
        )

        # 轨迹校正头
        self.trajectory_head = TrajectoryCorrectionHead(
            d_model=config.model.d_model,
            lstm_hidden=config.model.trajectory_lstm_hidden,
            lstm_layers=config.model.trajectory_lstm_layers,
            bidirectional=config.model.trajectory_bidirectional,
            use_attention=config.model.trajectory_attention,
            num_outputs=config.data.num_trajectory_outputs,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            x: 输入张量
            mask: 可选的注意力掩码

        返回:
            轨迹校正输出
        """
        encoded = self.encoder(x, mask)
        trajectory_outputs = self.trajectory_head(encoded, mask)

        return {
            'encoded': encoded,
            **trajectory_outputs,
        }