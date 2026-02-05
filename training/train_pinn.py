"""
PINN模型训练脚本

支持三种训练模式：
1. pretrain: 仅使用仿真数据预训练
2. finetune: 仅使用真实数据微调
3. hybrid: 混合真实数据和仿真数据（推荐）

数据格式：
- 真实数据: NPZ格式（来自视觉测量）
- 仿真数据: HDF5/MAT格式（来自MATLAB物理引擎）
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.pinn_trajectory_model import TrajectoryPINN, SequencePINN, create_model


class HybridDataset(Dataset):
    """
    混合数据集（真实测量数据 + 仿真数据）

    真实数据来自视觉系统采集的照片
    仿真数据来自物理引擎生成
    """

    def __init__(self,
                 real_data_files: list = None,
                 sim_data_files: list = None,
                 seq_len: int = 50,
                 mode: str = 'train'):
        """
        Args:
            real_data_files: 真实数据NPZ文件列表
            sim_data_files: 仿真数据文件列表
            seq_len: 序列长度（用于LSTM模型）
            mode: 'train' 或 'val'
        """
        self.seq_len = seq_len
        self.mode = mode
        self.samples = []
        self.scaler = None

        # 加载真实数据
        if real_data_files:
            self._load_real_data(real_data_files)

        # 加载仿真数据
        if sim_data_files:
            self._load_sim_data(sim_data_files)

        print(f"{mode}数据集加载完成:")
        print(f"  真实样本: {len([s for s in self.samples if s['source'] == 'real'])}")
        print(f"  仿真样本: {len([s for s in self.samples if s['source'] == 'sim'])}")
        print(f"  总样本数: {len(self.samples)}")

    def _load_real_data(self, files: list):
        """加载真实测量数据"""
        for file in files:
            try:
                data = np.load(file)

                # 提取轮廓和误差
                contours = data['contours']
                errors = data['errors']

                for contour, error in zip(contours, errors):
                    # 计算特征（简化版）
                    features = self._compute_features_from_contour(contour, error)

                    self.samples.append({
                        'inputs': features,
                        'targets': error,
                        'source': 'real'
                    })

            except Exception as e:
                print(f"警告: 加载真实数据失败 {file}: {e}")

    def _load_sim_data(self, files: list):
        """加载仿真数据"""
        for file in files:
            try:
                # 尝试使用h5py或scipy读取
                try:
                    import h5py
                    with h5py.File(file, 'r') as f:
                        trajectory_ref = f['trajectory_ref'][:]
                        trajectory_actual = f['trajectory_actual'][:]
                except:
                    from scipy.io import loadmat
                    mat = loadmat(file)
                    trajectory_ref = mat['trajectory_ref']
                    trajectory_actual = mat['trajectory_actual']

                # 计算误差和特征
                errors = trajectory_actual - trajectory_ref

                for i in range(len(trajectory_ref) - self.seq_len):
                    # 提取序列
                    seq_ref = trajectory_ref[i:i+self.seq_len]
                    seq_error = errors[i:i+self.seq_len]

                    # 计算特征
                    features = self._compute_features_from_trajectory(seq_ref)

                    self.samples.append({
                        'inputs': features,
                        'targets': seq_error[-1],  # 预测最后一个时间步的误差
                        'source': 'sim'
                    })

            except Exception as e:
                print(f"警告: 加载仿真数据失败 {file}: {e}")

    def _compute_features_from_contour(self, contour, error):
        """从轮廓计算特征"""
        # 简化：仅使用位置信息
        # 实际应该计算速度、加速度等

        x = contour[:, 0]
        y = contour[:, 1]

        # 计算一阶导数（速度）
        vx = np.gradient(x)
        vy = np.gradient(y)

        # 计算二阶导数（加速度）
        ax = np.gradient(vx)
        ay = np.gradient(vy)

        # 计算曲率
        curvature = self._compute_curvature(x, y)

        # 堆叠特征 (n_points, 7)
        features = np.stack([x, y, vx, vy, ax, ay, curvature], axis=1)

        return features

    def _compute_features_from_trajectory(self, trajectory):
        """从轨迹计算特征"""
        # trajectory形状: (seq_len, 2) 或 (seq_len, 4) [x, y, vx, vy]

        if trajectory.shape[1] == 2:
            x = trajectory[:, 0]
            y = trajectory[:, 1]
            vx = np.gradient(x)
            vy = np.gradient(y)
        else:
            x = trajectory[:, 0]
            y = trajectory[:, 1]
            vx = trajectory[:, 2]
            vy = trajectory[:, 3]

        ax = np.gradient(vx)
        ay = np.gradient(vy)
        curvature = self._compute_curvature(x, y)

        features = np.stack([x, y, vx, vy, ax, ay, curvature], axis=1)

        return features

    def _compute_curvature(self, x, y):
        """计算曲率"""
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-8)**1.5

        return curvature

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        inputs = torch.FloatTensor(sample['inputs'])
        targets = torch.FloatTensor(sample['targets'])

        return {
            'inputs': inputs,
            'targets': targets,
            'source': sample['source']
        }


def train_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()

    total_loss = 0
    total_data_loss = 0
    total_physics_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        inputs = batch['inputs'].to(device)
        targets = batch['targets'].to(device)

        # 确保targets形状正确
        if targets.dim() == 3:
            targets = targets.squeeze(1)

        # 计算损失
        loss, loss_dict = model.compute_total_loss(
            inputs_data=inputs,
            targets_data=targets
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_data_loss += loss_dict['data']
        total_physics_loss += loss_dict['physics']

        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                  f"Loss: {loss.item():.6f} "
                  f"(Data: {loss_dict['data']:.6f}, "
                  f"Physics: {loss_dict['physics']:.6f})")

    avg_loss = total_loss / len(dataloader)
    avg_data_loss = total_data_loss / len(dataloader)
    avg_physics_loss = total_physics_loss / len(dataloader)

    return avg_loss, avg_data_loss, avg_physics_loss


def validate(model, dataloader, device):
    """验证模型"""
    model.eval()

    total_loss = 0
    predictions = []
    targets_list = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)

            if targets.dim() == 3:
                targets = targets.squeeze(1)

            loss, loss_dict = model.compute_total_loss(
                inputs_data=inputs,
                targets_data=targets
            )

            total_loss += loss.item()

            # 保存预测结果
            preds = model.predict(inputs)
            predictions.append(preds.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # 计算R²
    all_preds = np.vstack(predictions)
    all_targets = np.vstack(targets_list)

    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets, axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return avg_loss, r2


def plot_training_history(history, output_dir):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # R²曲线
    axes[1].plot(history['val_r2'], label='Validation R²', linewidth=2, color='green')
    axes[1].axhline(y=0.9, color='r', linestyle='--', label='R² = 0.9')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('R² Score', fontsize=12)
    axes[1].set_title('Validation R² Score', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"训练曲线已保存: {output_dir}/training_curves.png")


def save_checkpoint(model, optimizer, epoch, history, save_path):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'model_info': {
            'type': 'TrajectoryPINN',
            'mass': model.mass,
            'stiffness': model.stiffness,
            'damping': model.damping
        }
    }

    torch.save(checkpoint, save_path)
    print(f"检查点已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='PINN模型训练')

    # 数据参数
    parser.add_argument('--real_data', type=str, nargs='*', default=[],
                       help='真实数据NPZ文件路径')
    parser.add_argument('--sim_data', type=str, nargs='*', default=[],
                       help='仿真数据文件路径')
    parser.add_argument('--mode', type=str, default='hybrid',
                       choices=['pretrain', 'finetune', 'hybrid'],
                       help='训练模式')

    # 模型参数
    parser.add_argument('--model_type', type=str, default='mlp',
                       choices=['mlp', 'lstm'],
                       help='模型类型')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[128, 128, 64, 64],
                       help='隐藏层大小')
    parser.add_argument('--mass', type=float, default=0.35,
                       help='系统质量(kg)')
    parser.add_argument('--stiffness', type=float, default=8000.0,
                       help='刚度系数(N/m)')
    parser.add_argument('--damping', type=float, default=15.0,
                       help='阻尼系数(Ns/m)')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--seq_len', type=int, default=50,
                       help='序列长度（LSTM模型）')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='checkpoints/pinn',
                       help='输出目录')
    parser.add_argument('--save_every', type=int, default=10,
                       help='每N个epoch保存一次')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 准备数据
    print("\n" + "="*70)
    print("准备数据集")
    print("="*70)

    if args.mode == 'pretrain' and len(args.sim_data) == 0:
        print("错误: pretrain模式需要提供仿真数据 (--sim_data)")
        return

    if args.mode == 'finetune' and len(args.real_data) == 0:
        print("错误: finetune模式需要提供真实数据 (--real_data)")
        return

    train_dataset = HybridDataset(
        real_data_files=args.real_data,
        sim_data_files=args.sim_data,
        seq_len=args.seq_len,
        mode='train'
    )

    # 划分训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)

    # 创建模型
    print("\n" + "="*70)
    print("创建模型")
    print("="*70)

    if args.model_type == 'mlp':
        model = TrajectoryPINN(
            hidden_sizes=args.hidden_sizes,
            mass=args.mass,
            stiffness=args.stiffness,
            damping=args.damping
        ).to(device)
    else:
        model = SequencePINN(
            input_size=7,
            hidden_size=args.hidden_sizes[0],
            num_layers=len(args.hidden_sizes),
            mass=args.mass,
            stiffness=args.stiffness
        ).to(device)

    print(f"模型类型: {args.model_type}")
    print(f"物理参数:")
    print(f"  质量: {model.mass} kg")
    print(f"  刚度: {model.stiffness} N/m")
    print(f"  阻尼: {model.damping} Ns/m")

    # 打印模型信息
    if args.model_type == 'mlp':
        total_params = sum(p.numel() for p in model.parameters())
        print(f"总参数量: {total_params:,}")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # 训练循环
    print("\n" + "="*70)
    print("开始训练")
    print("="*70)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2': []
    }

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # 训练
        train_loss, train_data_loss, train_physics_loss = train_epoch(
            model, train_loader, optimizer, device, epoch
        )

        # 验证
        val_loss, val_r2 = validate(model, val_loader, device)

        # 更新学习率
        scheduler.step(val_loss)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)

        print(f"\n训练损失: {train_loss:.6f}")
        print(f"  数据损失: {train_data_loss:.6f}")
        print(f"  物理损失: {train_physics_loss:.6f}")
        print(f"验证损失: {val_loss:.6f}")
        print(f"验证 R²: {val_r2:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, history,
                output_dir / 'best_model.pth'
            )
            print(f"  ✨ 最佳模型已保存")

        # 定期保存
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, history,
                output_dir / f'checkpoint_epoch{epoch}.pth'
            )

    # 绘制训练曲线
    print("\n绘制训练曲线...")
    plot_training_history(history, output_dir)

    # 保存最终模型
    save_checkpoint(
        model, optimizer, args.epochs, history,
        output_dir / 'final_model.pth'
    )

    print("\n" + "="*70)
    print("训练完成")
    print("="*70)
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"模型保存位置: {output_dir}")


if __name__ == '__main__':
    main()
