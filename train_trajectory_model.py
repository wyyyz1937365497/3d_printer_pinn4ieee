"""
训练脚本：使用MATLAB仿真数据训练轨迹误差预测模型

数据格式：
- 输入：29维轨迹特征（位置、速度、加速度、jerk、曲率等）
- 输出：2维误差向量（error_x, error_y）

Author: 3D Printer PINN Project
Date: 2026-01-29
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TrajectoryDataset(Dataset):
    """
    轨迹数据集：从HDF5文件加载序列数据
    """

    def __init__(self, h5_file, sequence_length=20, split='train', train_ratio=0.8):
        """
        Args:
            h5_file: HDF5文件路径
            sequence_length: 序列长度
            split: 'train' 或 'test'
            train_ratio: 训练集比例
        """
        self.sequence_length = sequence_length

        with h5py.File(h5_file, 'r') as f:
            features = f['features'][:]
            targets = f['targets'][:]

        # 计算分割点
        total_samples = len(features)
        # 由于我们需要连续的序列，分割时按样本数分割
        train_size = int(total_samples * train_ratio)
        # 确保train_size是sequence_length的整数倍
        train_size = (train_size // sequence_length) * sequence_length

        if split == 'train':
            self.features = features[:train_size]
            self.targets = targets[:train_size]
        else:
            self.features = features[train_size:]
            self.targets = targets[train_size:]

        # 计算可以生成多少个序列
        self.n_sequences = len(self.features) - sequence_length

        print(f"{split}集: {self.n_sequences} 个序列, {len(self.features)} 个样本")

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        """
        返回一个序列和对应的目标（序列中间点的误差）
        """
        # 提取序列
        seq_features = self.features[idx:idx + self.sequence_length]

        # 目标：预测序列中间点的误差
        target_idx = idx + self.sequence_length // 2
        target = self.targets[target_idx]

        # 从特征中分离出：
        # 1. 基础特征（除了速度方向和下一点位置）
        # 2. 速度方向（vx_norm, vy_norm）- 索引15-16
        # 3. 下一点位置（dx_next, dy_next）- 索引19-20

        # 基础特征：移除索引15-16（速度方向）和19-20（下一点位置）
        base_features = np.delete(seq_features, [15, 16, 19, 20], axis=1)

        # 速度方向
        velocities = seq_features[:, 15:17]

        # 下一点位置
        next_positions = seq_features[:, 19:21]

        return (
            torch.FloatTensor(base_features),
            torch.FloatTensor(velocities),
            torch.FloatTensor(next_positions),
            torch.FloatTensor(target)
        )


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SimplifiedTrajectoryPredictor(nn.Module):
    """
    简化的轨迹预测模型：直接使用29维特征，不需要分离输入
    """

    def __init__(self, input_size=29, d_model=128, nhead=8, num_layers=2,
                 output_size=2, sequence_length=20, dropout=0.1):
        super(SimplifiedTrajectoryPredictor, self).__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.sequence_length = sequence_length

        # 输入投影
        self.input_projection = nn.Linear(input_size, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, sequence_length)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: (batch_size, sequence_length, input_size)
        Returns:
            output: (batch_size, output_size) - 预测序列中心点的误差
        """
        # 输入投影
        projected = self.input_projection(x) * (self.d_model ** 0.5)

        # 添加位置编码
        encoded = self.pos_encoder(projected)

        # Transformer编码
        transformer_output = self.transformer_encoder(encoded)

        # LSTM处理
        lstm_output, _ = self.lstm(transformer_output)

        # 取序列中心点的特征
        center_idx = self.sequence_length // 2
        center_features = lstm_output[:, center_idx, :]

        # 输出投影
        output = self.output_projection(center_features)

        return output


def train_model(model, train_loader, val_loader, device, num_epochs=100,
                lr=0.001, patience=15):
    """
    训练模型
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for batch_base, batch_vel, batch_next, batch_target in train_loader:
            # 合并所有输入
            batch_x = torch.cat([batch_base, batch_vel, batch_next], dim=-1).to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_target)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_base, batch_vel, batch_next, batch_target in val_loader:
                batch_x = torch.cat([batch_base, batch_vel, batch_next], dim=-1).to(device)
                batch_target = batch_target.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_target)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # 学习率调度
        scheduler.step(avg_val_loss)

        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'models/best_trajectory_model.pth')
        else:
            epochs_no_improve += 1

        # 打印进度
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}, '
                  f'LR: {current_lr:.6f}')

        # 早停
        if epochs_no_improve >= patience:
            print(f'\n早停：验证损失在{patience}个epoch内没有改善')
            break

    return train_losses, val_losses


def evaluate_model(model, test_loader, device):
    """
    评估模型性能
    """
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_base, batch_vel, batch_next, batch_target in test_loader:
            batch_x = torch.cat([batch_base, batch_vel, batch_next], dim=-1).to(device)
            batch_target = batch_target.to(device)

            outputs = model(batch_x)

            predictions.append(outputs.cpu().numpy())
            targets.append(batch_target.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    # 计算误差统计
    error_x = predictions[:, 0] - targets[:, 0]
    error_y = predictions[:, 1] - targets[:, 1]
    error_mag = np.sqrt(error_x**2 + error_y**2)

    print('\n模型评估结果：')
    print(f'X方向误差: MAE={np.mean(np.abs(error_x)):.6f}, RMSE={np.sqrt(np.mean(error_x**2)):.6f}')
    print(f'Y方向误差: MAE={np.mean(np.abs(error_y)):.6f}, RMSE={np.sqrt(np.mean(error_y**2)):.6f}')
    print(f'总体误差: MAE={np.mean(error_mag):.6f}, RMSE={np.sqrt(np.mean(error_mag**2)):.6f}')

    return predictions, targets


def plot_results(train_losses, val_losses, predictions, targets):
    """
    可视化训练结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 训练损失曲线
    ax = axes[0, 0]
    ax.plot(train_losses, label='训练损失')
    ax.plot(val_losses, label='验证损失')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('损失 (MSE)')
    ax.set_title('训练过程')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 预测vs真实 - X方向
    ax = axes[0, 1]
    ax.scatter(targets[:, 0], predictions[:, 0], alpha=0.3, s=1)
    ax.plot([targets[:, 0].min(), targets[:, 0].max()],
            [targets[:, 0].min(), targets[:, 0].max()], 'r--', lw=2)
    ax.set_xlabel('真实值 (mm)')
    ax.set_ylabel('预测值 (mm)')
    ax.set_title('X方向误差预测')
    ax.grid(True, alpha=0.3)

    # 预测vs真实 - Y方向
    ax = axes[1, 0]
    ax.scatter(targets[:, 1], predictions[:, 1], alpha=0.3, s=1)
    ax.plot([targets[:, 1].min(), targets[:, 1].max()],
            [targets[:, 1].min(), targets[:, 1].max()], 'r--', lw=2)
    ax.set_xlabel('真实值 (mm)')
    ax.set_ylabel('预测值 (mm)')
    ax.set_title('Y方向误差预测')
    ax.grid(True, alpha=0.3)

    # 误差分布
    ax = axes[1, 1]
    errors = predictions - targets
    error_mag = np.sqrt(errors[:, 0]**2 + errors[:, 1]**2)
    ax.hist(error_mag, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(error_mag), color='r', linestyle='--',
               label=f'均值: {np.mean(error_mag):.6f}')
    ax.set_xlabel('误差大小 (mm)')
    ax.set_ylabel('频数')
    ax.set_title('预测误差分布')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    Path('models').mkdir(exist_ok=True)
    plt.savefig('models/training_results.png', dpi=300, bbox_inches='tight')
    print('\n✓ 训练结果已保存至: models/training_results.png')


def main():
    """
    主训练流程
    """
    print('='*60)
    print('轨迹误差预测模型训练')
    print('='*60)
    print()

    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    print()

    # 参数配置
    sequence_length = 20
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001

    # 加载数据
    print('加载数据...')
    h5_file = 'trajectory_data.h5'

    if not Path(h5_file).exists():
        print(f'错误：找不到数据文件 {h5_file}')
        print('请先运行数据转换：')
        print('  python matlab_simulation/convert_to_trajectory_features.py \\')
        print('      data_simulation_* -o trajectory_data.h5')
        return

    train_dataset = TrajectoryDataset(h5_file, sequence_length=sequence_length, split='train')
    val_dataset = TrajectoryDataset(h5_file, sequence_length=sequence_length, split='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f'训练批次数: {len(train_loader)}')
    print(f'验证批次数: {len(val_loader)}')
    print()

    # 创建模型
    print('创建模型...')
    model = SimplifiedTrajectoryPredictor(
        input_size=29,  # 所有特征
        d_model=128,
        nhead=8,
        num_layers=2,
        output_size=2,
        sequence_length=sequence_length,
        dropout=0.1
    ).to(device)

    # 计算模型参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数量: {n_params:,}')
    print()

    # 训练模型
    print('开始训练...')
    print('-'*60)
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, device,
        num_epochs=num_epochs,
        lr=learning_rate,
        patience=15
    )
    print('-'*60)
    print()

    # 评估模型
    print('评估模型...')
    print('-'*60)
    predictions, targets = evaluate_model(model, val_loader, device)
    print('-'*60)
    print()

    # 可视化结果
    print('生成可视化...')
    plot_results(train_losses, val_losses, predictions, targets)

    # 保存最终模型
    Path('models').mkdir(exist_ok=True)
    torch.save(model.state_dict(), 'models/trajectory_model_final.pth')
    print('✓ 最终模型已保存至: models/trajectory_model_final.pth')
    print()
    print('训练完成！')


if __name__ == '__main__':
    main()
