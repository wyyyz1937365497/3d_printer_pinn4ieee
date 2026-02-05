"""
物理信息神经网络(PINN)轨迹误差预测模型

核心思想：
- 结合数据驱动学习和物理约束
- 物理约束基于质量-弹簧-阻尼系统
- 损失函数 = λ_data * MSE(预测误差, 真实误差) + λ_physics * MSE(预测误差, 理论误差)

理论依据：
  m·x'' + c·x' + k·x = F(t)
  稳态误差: error ≈ -(m/k)·a_ref
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict


class TrajectoryPINN(nn.Module):
    """
    轨迹误差预测的物理信息神经网络

    输入特征: [x, y, vx, vy, ax, ay, curvature] (7维)
    输出: [error_x, error_y] (2维)
    """

    def __init__(self,
                 hidden_sizes: List[int] = [128, 128, 64, 64],
                 activation: str = 'tanh',
                 mass: float = 0.35,          # kg
                 stiffness: float = 8000.0,   # N/m
                 damping: float = 15.0,       # Ns/m
                 lambda_data: float = 1.0,
                 lambda_physics: float = 0.1):
        """
        Args:
            hidden_sizes: 隐藏层大小列表
            activation: 激活函数 ('tanh', 'relu', 'selu')
            mass: 系统质量 (kg)
            stiffness: 刚度系数 (N/m)
            damping: 阻尼系数 (Ns/m)
            lambda_data: 数据损失权重
            lambda_physics: 物理损失权重
        """
        super(TrajectoryPINN, self).__init__()

        # 物理参数
        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping

        # 可学习的损失权重
        self.lambda_data = nn.Parameter(torch.tensor(lambda_data))
        self.lambda_physics = nn.Parameter(torch.tensor(lambda_physics))

        # 构建网络
        layers = []
        input_size = 7  # [x, y, vx, vy, ax, ay, curvature]
        output_size = 2  # [error_x, error_y]

        # 输入层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))

        # 隐藏层
        for i in range(len(hidden_sizes) - 1):
            layers.append(self._get_activation(activation))
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

        # 输出层
        layers.append(self._get_activation(activation))
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.network = nn.Sequential(*layers)

        # 初始化权重
        self._initialize_weights()

    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'selu': nn.SELU(),
            'leaky_relu': nn.LeakyReLU(0.1)
        }
        return activations.get(activation, nn.Tanh())

    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            inputs: (batch_size, 7) 特征张量
                    [x, y, vx, vy, ax, ay, curvature]

        Returns:
            errors: (batch_size, 2) 误差预测
                    [error_x, error_y]
        """
        return self.network(inputs)

    def compute_physics_loss(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        计算物理约束损失

        基于稳态误差理论: error ≈ -(m/k)·a_ref

        Args:
            inputs: (batch_size, 7) 特征张量

        Returns:
            physics_loss: 物理约束损失
        """
        # 提取加速度 (inputs[:, 4]=ax, inputs[:, 5]=ay)
        ax_ref = inputs[:, 4]
        ay_ref = inputs[:, 5]

        # 计算理论稳态误差
        ideal_error_x = -self.mass / self.stiffness * ax_ref
        ideal_error_y = -self.mass / self.stiffness * ay_ref

        # 模型预测误差
        pred_errors = self.forward(inputs)

        # 计算MSE损失
        physics_loss_x = nn.MSELoss()(pred_errors[:, 0], ideal_error_x)
        physics_loss_y = nn.MSELoss()(pred_errors[:, 1], ideal_error_y)

        return physics_loss_x + physics_loss_y

    def compute_total_loss(self,
                          inputs_data: torch.Tensor,
                          targets_data: torch.Tensor,
                          inputs_physics: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        """
        计算总损失（数据损失 + 物理损失）

        Args:
            inputs_data: 数据驱动输入 (batch_size, 7)
            targets_data: 数据驱动目标 (batch_size, 2) - 真实误差
            inputs_physics: 物理约束输入 (batch_size, 7) - 可选

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 数据驱动损失
        pred_errors = self.forward(inputs_data)
        data_loss = nn.MSELoss()(pred_errors, targets_data)

        # 物理约束损失
        if inputs_physics is None:
            inputs_physics = inputs_data

        physics_loss = self.compute_physics_loss(inputs_physics)

        # 可学习的权重
        lambda_data = torch.abs(self.lambda_data)
        lambda_physics = torch.abs(self.lambda_physics)

        # 总损失
        total_loss = lambda_data * data_loss + lambda_physics * physics_loss

        loss_dict = {
            'total': total_loss.item(),
            'data': data_loss.item(),
            'physics': physics_loss.item(),
            'lambda_data': lambda_data.item(),
            'lambda_physics': lambda_physics.item()
        }

        return total_loss, loss_dict

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        预测轨迹误差

        Args:
            inputs: (batch_size, 7) 特征张量

        Returns:
            errors: (batch_size, 2) 误差预测
        """
        self.eval()
        with torch.no_grad():
            return self.forward(inputs)

    def get_physics_parameters(self) -> Dict[str, float]:
        """获取物理参数"""
        return {
            'mass': self.mass,
            'stiffness': self.stiffness,
            'damping': self.damping,
            'natural_frequency': (self.stiffness / self.mass) ** 0.5,  # ωn = √(k/m)
            'damping_ratio': self.damping / (2 * (self.stiffness * self.mass) ** 0.5)  # ζ = c/(2√(km))
        }


class SequencePINN(nn.Module):
    """
    基于序列的PINN模型（使用LSTM/GRU）

    适用于处理时序轨迹数据
    """

    def __init__(self,
                 input_size: int = 7,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 2,
                 rnn_type: str = 'lstm',
                 mass: float = 0.35,
                 stiffness: float = 8000.0,
                 dropout: float = 0.1):
        """
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            num_layers: RNN层数
            output_size: 输出维度
            rnn_type: 'lstm' 或 'gru'
            mass: 系统质量
            stiffness: 刚度系数
            dropout: Dropout比例
        """
        super(SequencePINN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # 物理参数
        self.mass = mass
        self.stiffness = stiffness

        # RNN层
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
        else:
            self.rnn = nn.GRU(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )

        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            inputs: (batch_size, seq_len, input_size)

        Returns:
            outputs: (batch_size, seq_len, output_size)
        """
        # RNN前向传播
        rnn_out, _ = self.rnn(inputs)

        # 全连接层
        outputs = self.fc(rnn_out)

        return outputs

    def compute_physics_loss(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        计算物理约束损失（仅对最后一个时间步）

        Args:
            inputs: (batch_size, seq_len, input_size)

        Returns:
            physics_loss: 物理约束损失
        """
        # 提取最后一个时间步的加速度
        ax_ref = inputs[:, -1, 4]
        ay_ref = inputs[:, -1, 5]

        # 计算理论误差
        ideal_error_x = -self.mass / self.stiffness * ax_ref
        ideal_error_y = -self.mass / self.stiffness * ay_ref

        # 模型预测
        pred_errors = self.forward(inputs)[:, -1, :]

        # MSE损失
        physics_loss = nn.MSELoss()(pred_errors[:, 0], ideal_error_x) + \
                       nn.MSELoss()(pred_errors[:, 1], ideal_error_y)

        return physics_loss


def create_model(model_type: str = 'mlp',
                 **kwargs) -> nn.Module:
    """
    工厂函数：创建PINN模型

    Args:
        model_type: 'mlp' 或 'lstm'
        **kwargs: 模型参数

    Returns:
        model: PINN模型实例
    """
    if model_type.lower() == 'mlp':
        return TrajectoryPINN(**kwargs)
    elif model_type.lower() == 'lstm':
        return SequencePINN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
