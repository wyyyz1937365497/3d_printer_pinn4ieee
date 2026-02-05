"""
Physics-Informed Neural Network (PINN) for 3D Printing Trajectory Error Prediction

核心思想：
1. 使用神经网络拟合误差函数：error = f(state, acceleration)
2. 物理约束：2阶动力学方程 m·x'' + c·x' + k·x = F(t)
3. 数据约束：在测量点上，预测误差≈实测误差

优势：
- 需要的标注数据少（物理约束指导学习）
- 泛化能力强（符合物理定律）
- 结合仿真（物理loss）和实测（数据loss）
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List


class TrajectoryPINN(nn.Module):
    """
    轨迹误差PINN模型

    网络结构：
        Input: [x, y, vx, vy, ax, ay, curvature] (7维)
        Hidden: Multiple FC layers with activation
        Output: [error_x, error_y] (2维)

    物理约束：
        对于欠阻尼2阶系统：
        m·error'' + c·error' + k·error ≈ -m·a_ref (稳态)

    损失函数：
        Loss = λ_data·Loss_data + λ_physics·Loss_physics
    """

    def __init__(self,
                 hidden_sizes: List[int] = [128, 128, 64, 64],
                 activation: str = 'tanh',
                 # 物理参数（Ender 3 V2）
                 mass: float = 0.35,           # kg
                 stiffness: float = 8000.0,    # N/m
                 damping: float = 15.0,        # N·s/m
                 # 损失权重
                 lambda_data: float = 1.0,
                 lambda_physics: float = 0.1):
        super().__init__()

        # 物理参数
        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping

        # 计算导出参数
        self.omega_n = np.sqrt(stiffness / mass)  # 自然频率
        self.zeta = damping / (2 * np.sqrt(stiffness * mass))  # 阻尼比

        # 损失权重（可学习）
        self.lambda_data = nn.Parameter(torch.tensor(lambda_data))
        self.lambda_physics = nn.Parameter(torch.tensor(lambda_physics))

        # 构建网络
        layers = []
        input_size = 7  # x, y, vx, vy, ax, ay, curvature

        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_sizes[i-1], hidden_size))

            # 激活函数
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'swish':
                layers.append(nn.SiLU())
            else:
                layers.append(nn.Tanh())

            # BatchNorm（可选）
            # layers.append(nn.BatchNorm1d(hidden_size))

        # 输出层
        layers.append(nn.Linear(hidden_sizes[-1], 2))

        self.network = nn.Sequential(*layers)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            inputs: [batch, 7] 或 [seq_len, batch, 7]
                    [x, y, vx, vy, ax, ay, curvature]

        Returns:
            errors: [batch, 2] 或 [seq_len, batch, 2]
                    [error_x, error_y]
        """
        return self.network(inputs)

    def compute_physics_loss(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        物理约束损失

        简化的稳态约束：
            k·error ≈ -m·a_ref
            => error ≈ -(m/k)·a_ref

        对于欠阻尼系统，这是合理的近似

        Args:
            inputs: [batch, 7] - [x, y, vx, vy, ax, ay, curvature]

        Returns:
            physics_loss: scalar
        """
        batch_size = inputs.shape[0]

        # 提取加速度 (第4、5列)
        ax_ref = inputs[:, 4]
        ay_ref = inputs[:, 5]

        # 预测误差
        pred_errors = self.forward(inputs)
        pred_error_x = pred_errors[:, 0]
        pred_error_y = pred_errors[:, 1]

        # 计算物理约束的理想误差
        ideal_error_x = -self.mass / self.stiffness * ax_ref
        ideal_error_y = -self.mass / self.stiffness * ay_ref

        # MSE损失
        physics_loss_x = nn.MSELoss()(pred_error_x, ideal_error_x)
        physics_loss_y = nn.MSELoss()(pred_error_y, ideal_error_y)

        return physics_loss_x + physics_loss_y

    def compute_data_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        数据损失（在测量点上）

        Args:
            inputs: [batch, 7] - 输入特征
            targets: [batch, 2] - 真实误差 [error_x, error_y]

        Returns:
            data_loss: scalar
        """
        predictions = self.forward(inputs)
        return nn.MSELoss()(predictions, targets)

    def compute_total_loss(self,
                          inputs_data: torch.Tensor,
                          targets_data: torch.Tensor,
                          inputs_physics: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        计算总损失

        Loss = λ_data·Loss_data + λ_physics·Loss_physics

        Args:
            inputs_data: 用于数据loss的输入（实测数据）
            targets_data: 用于数据loss的目标（实测误差）
            inputs_physics: 用于物理loss的输入（大量采样点）

        Returns:
            total_loss, loss_dict
        """
        # 数据损失
        loss_data = self.compute_data_loss(inputs_data, targets_data)

        # 物理损失
        loss_physics = self.compute_physics_loss(inputs_physics)

        # 获取权重（确保非负）
        w_data = torch.relu(self.lambda_data)
        w_physics = torch.relu(self.lambda_physics)

        # 总损失
        loss_total = w_data * loss_data + w_physics * loss_physics

        loss_dict = {
            'total': loss_total.item(),
            'data': loss_data.item(),
            'physics': loss_physics.item(),
            'lambda_data': w_data.item(),
            'lambda_physics': w_physics.item()
        }

        return loss_total, loss_dict

    def predict(self,
                x: np.ndarray,
                y: np.ndarray,
                vx: np.ndarray,
                vy: np.ndarray,
                ax: np.ndarray,
                ay: np.ndarray,
                curvature: np.ndarray) -> np.ndarray:
        """
        预测轨迹误差

        Args:
            x, y: 位置 (mm)
            vx, vy: 速度 (mm/s)
            ax, ay: 加速度 (mm/s²)
            curvature: 路径曲率 (1/mm)

        Returns:
            errors: [n, 2] - [error_x, error_y] (mm)
            注意：修正量 = -误差
        """
        self.eval()

        with torch.no_grad():
            # 准备输入
            inputs = np.stack([x, y, vx, vy, ax, ay, curvature], axis=1)
            inputs_tensor = torch.FloatTensor(inputs)

            if torch.cuda.is_available():
                inputs_tensor = inputs_tensor.cuda()

            # 预测
            errors = self.forward(inputs_tensor).cpu().numpy()

        return errors

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_type': 'PINN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'mass': self.mass,
            'stiffness': self.stiffness,
            'damping': self.damping,
            'natural_frequency': self.omega_n,
            'damping_ratio': self.zeta
        }


class SequencePINN(TrajectoryPINN):
    """
    序列版本PINN - 考虑时间历史

    使用LSTM/GRU处理序列，但保留物理约束
    """

    def __init__(self,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 rnn_type: str = 'lstm',
                 **kwargs):
        """
        Args:
            hidden_size: RNN隐藏层大小
            num_layers: RNN层数
            rnn_type: 'lstm' or 'gru'
            **kwargs: 传递给TrajectoryPINN的其他参数
        """
        # 不调用父类的__init__，因为我们完全重写网络
        nn.Module.__init__(self)

        # 保存物理参数
        self.mass = kwargs.get('mass', 0.35)
        self.stiffness = kwargs.get('stiffness', 8000.0)
        self.damping = kwargs.get('damping', 15.0)
        self.omega_n = np.sqrt(self.stiffness / self.mass)
        self.zeta = self.damping / (2 * np.sqrt(self.stiffness * self.mass))

        self.lambda_data = nn.Parameter(torch.tensor(kwargs.get('lambda_data', 1.0)))
        self.lambda_physics = nn.Parameter(torch.tensor(kwargs.get('lambda_physics', 0.1)))

        # RNN层
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(7, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(7, hidden_size, num_layers, batch_first=True)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播（序列版本）

        Args:
            inputs: [batch, seq_len, 7]

        Returns:
            errors: [batch, seq_len, 2]
        """
        # RNN
        rnn_out, _ = self.rnn(inputs)

        # FC
        errors = self.fc(rnn_out)

        return errors


if __name__ == '__main__':
    # 测试代码
    print("Testing PINN models...")

    # 测试TrajectoryPINN
    model = TrajectoryPINN()
    info = model.get_model_info()
    print(f"\nTrajectoryPINN:")
    print(f"  Parameters: {info['total_parameters']:,}")
    print(f"  Natural freq: {info['natural_frequency']:.2f} rad/s")
    print(f"  Damping ratio: {info['damping_ratio']:.4f}")

    # 测试前向传播
    batch_size = 32
    inputs = torch.randn(batch_size, 7)
    outputs = model(inputs)
    print(f"  Input shape: {inputs.shape}")
    print(f"  Output shape: {outputs.shape}")

    # 测试损失计算
    targets = torch.randn(batch_size, 2)
    loss_total, loss_dict = model.compute_total_loss(inputs, targets, inputs)
    print(f"  Total loss: {loss_dict['total']:.6f}")
    print(f"  Data loss: {loss_dict['data']:.6f}")
    print(f"  Physics loss: {loss_dict['physics']:.6f}")

    # 测试SequencePINN
    print(f"\nSequencePINN:")
    seq_model = SequencePINN()
    seq_inputs = torch.randn(batch_size, 50, 7)  # seq_len=50
    seq_outputs = seq_model(seq_inputs)
    print(f"  Input shape: {seq_inputs.shape}")
    print(f"  Output shape: {seq_outputs.shape}")
    print(f"  Parameters: {sum(p.numel() for p in seq_model.parameters()):,}")
