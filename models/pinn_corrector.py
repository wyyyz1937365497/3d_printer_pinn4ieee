"""
Physics-Informed Neural Network (PINN) for 3D Printing Trajectory Correction

核心思想：
1. 使用神经网络拟合误差函数：error = f(x, y, vx, vy, ax, ay)
2. 物理约束：m·error'' + c·error' + k·error = -m·a_ref
3. 数据约束：在测量点上，预测误差≈实测误差

优势：
- 需要的标注数据少（物理约束指导学习）
- 泛化能力强（符合物理定律）
- 结合仿真（物理loss）和实测（数据loss）
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class TrajectoryErrorPINN(nn.Module):
    """
    轨迹误差PINN模型

    输入：[x, y, vx, vy, ax, ay] - 位置、速度、加速度
    输出：[error_x, error_y] - 轨迹误差

    物理约束：
    m·error'' + c·error' + k·error = -m·a_ref
    """

    def __init__(self,
                 hidden_sizes: list = [64, 64, 64],
                 mass: float = 0.35,
                 stiffness: float = 8000.0,
                 damping: float = 15.0):
        super().__init__()

        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping

        # 构建网络
        layers = []
        input_size = 6  # x, y, vx, vy, ax, ay

        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_sizes[i-1], hidden_size))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_sizes[-1], 2))  # 输出：error_x, error_y

        self.network = nn.Sequential(*layers)

        # 可学习权重
        self.data_loss_weight = nn.Parameter(torch.tensor(1.0))
        self.physics_loss_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            inputs: [batch, 6] - [x, y, vx, vy, ax, ay]

        Returns:
            errors: [batch, 2] - [error_x, error_y]
        """
        return self.network(inputs)

    def compute_derivatives(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算误差关于时间的导数（自动微分）

        Returns:
            error: [batch, 2] - 误差
            error_vel: [batch, 2] - 误差速度
            error_acc: [batch, 2] - 误差加速度
        """
        inputs.requires_grad_(True)

        error = self.forward(inputs)

        # 计算一阶导数（关于时间）
        # 这里简化：假设输入已经是按时间排序的序列
        # 实际应该使用更复杂的方法计算时间导数

        grad_outputs = torch.ones_like(error)
        error_vel = torch.autograd.grad(
            outputs=error,
            inputs=inputs,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]

        # 只取位置和速度的导数（前4个）
        error_vel_x = error_vel[:, 0]  # de_x/dx * dx/dt + de_x/dy * dy/dt ...
        error_vel_y = error_vel[:, 1]

        # 二阶导数（简化处理）
        # 实际应用中需要更精确的计算
        error_acc_x = torch.zeros_like(error_vel_x)
        error_acc_y = torch.zeros_like(error_vel_y)

        return error, (error_vel_x, error_vel_y), (error_acc_x, error_acc_y)

    def physics_loss(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        物理约束损失

        方程：m·error'' + c·error' + k·error = -m·a_ref

        简化版本（因为时间导数计算困难）：
        使用当前误差和参考加速度的关系
        """
        batch_size = inputs.shape[0]

        # 提取输入
        # inputs[:, 0] = x
        # inputs[:, 1] = y
        # inputs[:, 4] = ax (参考加速度)
        # inputs[:, 5] = ay

        ax_ref = inputs[:, 4]
        ay_ref = inputs[:, 5]

        # 预测误差
        error = self.forward(inputs)
        error_x = error[:, 0]
        error_y = error[:, 1]

        # 简化的物理约束（稳态响应）
        # k·error ≈ -m·a_ref
        # 理想情况下：error ≈ -m/k * a_ref

        error_x_ideal = -self.mass / self.stiffness * ax_ref
        error_y_ideal = -self.mass / self.stiffness * ay_ref

        physics_loss_x = nn.MSELoss()(error_x, error_x_ideal)
        physics_loss_y = nn.MSELoss()(error_y, error_y_ideal)

        return physics_loss_x + physics_loss_y

    def data_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        数据损失（在测量点上）

        Args:
            inputs: [batch, 6] - [x, y, vx, vy, ax, ay]
            targets: [batch, 2] - 实测误差 [error_x, error_y]
        """
        predictions = self.forward(inputs)
        return nn.MSELoss()(predictions, targets)

    def total_loss(self,
                   inputs_data: torch.Tensor,
                   targets_data: torch.Tensor,
                   inputs_physics: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        总损失 = 数据损失 + 物理损失

        Args:
            inputs_data: 测量点的输入（用于数据loss）
            targets_data: 测量点的目标（用于数据loss）
            inputs_physics: 大量采样点（用于物理loss）

        Returns:
            total_loss, loss_dict
        """
        # 数据损失
        loss_data = self.data_loss(inputs_data, targets_data)

        # 物理损失（在大量采样点上）
        loss_physics = self.physics_loss(inputs_physics)

        # 总损失
        w_data = torch.relu(self.data_loss_weight)
        w_physics = torch.relu(self.physics_loss_weight)

        loss_total = w_data * loss_data + w_physics * loss_physics

        loss_dict = {
            'total': loss_total.item(),
            'data': loss_data.item(),
            'physics': loss_physics.item(),
            'w_data': w_data.item(),
            'w_physics': w_physics.item()
        }

        return loss_total, loss_dict

    def predict_correction(self, x_ref: np.ndarray, y_ref: np.ndarray,
                          vx_ref: np.ndarray, vy_ref: np.ndarray,
                          ax_ref: np.ndarray, ay_ref: np.ndarray) -> np.ndarray:
        """
        预测修正量

        Args:
            参考轨迹（位置、速度、加速度）

        Returns:
            corrections: [n, 2] - [correction_x, correction_y]
            注意：修正量 = -预测误差
        """
        self.eval()

        with torch.no_grad():
            # 准备输入
            inputs = np.stack([x_ref, y_ref, vx_ref, vy_ref, ax_ref, ay_ref], axis=1)
            inputs_tensor = torch.FloatTensor(inputs)

            if torch.cuda.is_available():
                inputs_tensor = inputs_tensor.cuda()

            # 预测误差
            errors = self.forward(inputs_tensor).cpu().numpy()

            # 修正量 = -误差
            corrections = -errors

        return corrections


class PINNDataset(torch.utils.data.Dataset):
    """
    PINN数据集

    包含两部分：
    1. 标注数据（实测）：用于数据loss
    2. 无标注数据（仿真/采样）：用于物理loss
    """

    def __init__(self, labeled_file: str, unlabeled_files: list = None):
        """
        Args:
            labeled_file: 有标注的数据文件（实测）
            unlabeled_files: 无标注数据文件（仿真生成）
        """
        # 加载标注数据
        import h5py
        with h5py.File(labeled_file, 'r') as f:
            data = f['simulation_data']
            self.x_ref = data['x_ref'][:].flatten()
            self.y_ref = data['y_ref'][:].flatten()
            self.error_x = data['error_x'][:].flatten()
            self.error_y = data['error_y'][:].flatten()

            # 计算速度和加速度（数值微分）
            self.vx_ref = np.gradient(self.x_ref)
            self.vy_ref = np.gradient(self.y_ref)
            self.ax_ref = np.gradient(self.vx_ref)
            self.ay_ref = np.gradient(self.vy_ref)

        self.n_labeled = len(self.x_ref)

        # 加载无标注数据（如果有）
        self.unlabeled_data = []
        if unlabeled_files:
            for file in unlabeled_files:
                # 加载并处理
                pass

    def __len__(self):
        return self.n_labeled

    def __getitem__(self, idx):
        # 返回标注数据
        inputs = np.array([
            self.x_ref[idx],
            self.y_ref[idx],
            self.vx_ref[idx],
            self.vy_ref[idx],
            self.ax_ref[idx],
            self.ay_ref[idx]
        ])

        targets = np.array([self.error_x[idx], self.error_y[idx]])

        return {
            'inputs': torch.FloatTensor(inputs),
            'targets': torch.FloatTensor(targets)
        }


def train_pinn(labeled_data: str,
               unlabeled_data_files: list = None,
               epochs: int = 1000,
               lr: float = 1e-3):
    """
    训练PINN模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    model = TrajectoryErrorPINN().to(device)

    # 创建数据集
    dataset = PINNDataset(labeled_data, unlabeled_data_files)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        epoch_losses = {'total': 0, 'data': 0, 'physics': 0}

        for batch in dataloader:
            inputs_data = batch['inputs'].to(device)
            targets_data = batch['targets'].to(device)

            # 生成物理loss用的采样点（可以用仿真数据）
            inputs_physics = inputs_data  # 简化：用相同数据

            # 计算损失
            loss, loss_dict = model.total_loss(inputs_data, targets_data, inputs_physics)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]

        # 学习率调度
        scheduler.step()

        # 打印
        if (epoch + 1) % 100 == 0:
            n_batches = len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Total Loss: {epoch_losses['total']/n_batches:.6f}")
            print(f"  Data Loss: {epoch_losses['data']/n_batches:.6f}")
            print(f"  Physics Loss: {epoch_losses['physics']/n_batches:.6f}")
            print(f"  Weights: data={loss_dict['w_data']:.3f}, physics={loss_dict['w_physics']:.3f}")

    return model


if __name__ == '__main__':
    # 测试
    print("Testing PINN model...")

    model = TrajectoryErrorPINN()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # 测试前向传播
    batch_size = 10
    inputs = torch.randn(batch_size, 6)
    outputs = model(inputs)
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
