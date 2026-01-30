"""
框架中所有模型的基类
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseModel(nn.Module, ABC):
    """
    所有模型的基类

    提供通用接口和工具
    """

    def __init__(self, config: Any):
        """
        初始化基模型

        参数:
            config: 配置对象
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        前向传播 - 必须由子类实现

        返回:
            模型输出
        """
        pass

    def get_num_params(self) -> int:
        """
        获取参数总数

        返回:
            参数数量
        """
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """
        获取可训练参数的数量

        返回:
            可训练参数数量
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self):
        """冻结所有模型参数"""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """解冻所有模型参数"""
        for param in self.parameters():
            param.requires_grad = True

    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None,
                       epoch: Optional[int] = None, loss: Optional[float] = None):
        """
        保存模型检查点

        参数:
            path: 保存检查点的路径
            optimizer: 优化器状态（可选）
            epoch: 当前周期（可选）
            loss: 当前损失（可选）
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config,
            'num_params': self.get_num_params(),
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None,
                       device: str = 'cpu') -> Dict[str, Any]:
        """
        加载模型检查点

        参数:
            path: 检查点路径
            optimizer: 要加载状态的优化器（可选）
            device: 加载模型的设备

        返回:
            包含附加信息的检查点字典
        """
        checkpoint = torch.load(path, map_location=device)

        self.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint

    @classmethod
    def from_checkpoint(cls, path: str, config: Any, device: str = 'cpu'):
        """
        从检查点创建模型

        参数:
            path: 检查点路径
            config: 配置对象
            device: 加载模型的设备

        返回:
            加载的模型实例
        """
        model = cls(config)
        model.load_checkpoint(path, device=device)
        return model

    def to_device(self, device: str):
        """
        将模型移动到设备

        参数:
            device: 要移动到的设备（'cpu' 或 'cuda'）

        返回:
            自身（用于方法链）
        """
        self.to(device)
        return self