"""
PINN训练脚本

训练策略：
1. 混合数据：真实测量数据（少量）+ 仿真数据（大量）
2. 损失函数：Loss = λ_data·Loss_data + λ_physics·Loss_physics
3. 自适应权重：λ_data和λ_physics可学习

使用方法：
    # 使用仿真数据预训练
    python train_pinn.py --mode pretrain --data_dir data_simulation

    # 使用真实数据微调
    python train_pinn.py --mode finetune --real_data print_errors.npz

    # 混合训练
    python train_pinn.py --mode hybrid --real_data print_errors.npz --sim_data data_simulation
"""

import os
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

from models.pinn_trajectory_model import TrajectoryPINN, SequencePINN
from utils.vision_processor import VisionProcessor


class HybridDataset(Dataset):
    """
    混合数据集：真实数据 + 仿真数据

    格式：
    - inputs: [x, y, vx, vy, ax, ay, curvature]
    - targets: [error_x, error_y]
    """

    def __init__(self,
                 real_data_files: list = None,
                 sim_data_files: list = None,
                 seq_len: int = 1):  # seq_len=1 for point-wise, >1 for sequence
        """
        Args:
            real_data_files: 真实测量数据文件列表
            sim_data_files: 仿真数据文件列表
            seq_len: 序列长度（1=点对点，>1=序列）
        """
        self.seq_len = seq_len
        self.data_samples = []

        # 加载真实数据
        if real_data_files:
            print("加载真实数据...")
            for file in real_data_files:
                self._load_real_data(file)

        # 加载仿真数据
        if sim_data_files:
            print("加载仿真数据...")
            for file in sim_data_files:
                self._load_sim_data(file)

        print(f"总样本数: {len(self.data_samples)}")

    def _load_real_data(self, file_path: str):
        """加载真实测量数据"""
        # 支持多种格式
        if file_path.endswith('.npz'):
            data = np.load(file_path, allow_pickle=True)
            # 处理数据...
            # TODO: 根据实际数据格式调整
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
            # 处理数据...

    def _load_sim_data(self, file_path: str):
        """加载仿真数据"""
        import h5py
        with h5py.File(file_path, 'r') as f:
            sim_data = f['simulation_data']

            x_ref = sim_data['x_ref'][:].flatten()
            y_ref = sim_data['y_ref'][:].flatten()
            error_x = sim_data['error_x'][:].flatten()
            error_y = sim_data['error_y'][:].flatten()

            # 计算速度、加速度
            vx = np.gradient(x_ref)
            vy = np.gradient(y_ref)
            ax = np.gradient(vx)
            ay = np.gradient(vy)

            # 计算曲率
            curvature = self._compute_curvature(x_ref, y_ref)

            # 组合特征
            n_points = len(x_ref)
            for i in range(n_points):
                self.data_samples.append({
                    'inputs': np.array([
                        x_ref[i], y_ref[i],
                        vx[i], vy[i],
                        ax[i], ay[i],
                        curvature[i]
                    ]),
                    'targets': np.array([error_x[i], error_y[i]]),
                    'source': 'simulation'
                })

    def _compute_curvature(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """计算路径曲率"""
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2)**1.5

        # 避免除零
        curvature = np.divide(numerator, denominator,
                             out=np.zeros_like(numerator),
                             where=denominator!=0)

        return curvature

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]

        return {
            'inputs': torch.FloatTensor(sample['inputs']),
            'targets': torch.FloatTensor(sample['targets']),
            'source': sample['source']
        }


def train_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    epoch_losses = {'total': 0, 'data': 0, 'physics': 0}

    pbar = tqdm(dataloader, desc=f"[Train Epoch {epoch}]", ncols=120)

    for batch in pbar:
        inputs_data = batch['inputs'].to(device)
        targets_data = batch['targets'].to(device)

        # 生成物理loss用的采样点
        # 可以用inputs_data，或者生成更多采样点
        inputs_physics = inputs_data

        # 计算损失
        loss, loss_dict = model.compute_total_loss(
            inputs_data, targets_data, inputs_physics
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 记录
        for key in epoch_losses:
            epoch_losses[key] += loss_dict[key]

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss_dict["total"]:.6f}',
            'data': f'{loss_dict["data"]:.6f}',
            'phys': f'{loss_dict["physics"]:.6f}'
        })

    # 平均
    for key in epoch_losses:
        epoch_losses[key] /= len(dataloader)

    return epoch_losses


def validate(model, dataloader, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['inputs'].to(device)
            target = batch['targets'].to(device)

            # 预测
            pred = model(inputs)

            # 数据损失
            loss = nn.MSELoss()(pred, target)
            total_loss += loss.item()

            predictions.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())

    # 合并
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    # 计算指标
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets)**2))

    # R²
    ss_res = np.sum((targets - predictions)**2)
    ss_tot = np.sum((targets - np.mean(targets))**2)
    r2 = 1 - (ss_res / ss_tot)

    return {
        'loss': total_loss / len(dataloader),
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def main():
    parser = argparse.ArgumentParser(description='Train PINN for trajectory correction')
    parser.add_argument('--mode', type=str, default='hybrid',
                       choices=['pretrain', 'finetune', 'hybrid'],
                       help='Training mode')
    parser.add_argument('--real_data', type=str, default=None,
                       help='Real measurement data file')
    parser.add_argument('--sim_data', type=str, default='data_simulation_*',
                       help='Simulation data directory pattern')
    parser.add_argument('--model_type', type=str, default='pinn',
                       choices=['pinn', 'seq_pinn'],
                       help='Model type')
    parser.add_argument('--seq_len', type=int, default=1,
                       help='Sequence length (for seq_pinn)')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[128, 128, 64, 64],
                       help='Hidden layer sizes')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--lambda_data', type=float, default=1.0,
                       help='Data loss weight')
    parser.add_argument('--lambda_physics', type=float, default=0.1,
                       help='Physics loss weight')
    parser.add_argument('--output_dir', type=str, default='checkpoints/pinn',
                       help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    args = parser.parse_args()

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    if args.model_type == 'pinn':
        model = TrajectoryPINN(
            hidden_sizes=args.hidden_sizes,
            lambda_data=args.lambda_data,
            lambda_physics=args.lambda_physics
        ).to(device)
    else:
        model = SequencePINN(
            hidden_size=args.hidden_sizes[0],
            lambda_data=args.lambda_data,
            lambda_physics=args.lambda_physics
        ).to(device)

    info = model.get_model_info()
    print(f"\n模型信息:")
    print(f"  类型: {info['model_type']}")
    print(f"  参数量: {info['total_parameters']:,}")
    print(f"  自然频率: {info['natural_frequency']:.2f} rad/s")
    print(f"  阻尼比: {info['damping_ratio']:.4f}")

    # 准备数据
    real_files = [args.real_data] if args.real_data else []
    sim_files = [] if args.mode == 'finetune' else []

    if args.mode != 'finetune':
        import glob
        sim_files = glob.glob(args.sim_data + "/*.mat")

    dataset = HybridDataset(
        real_data_files=real_files,
        sim_data_files=sim_files,
        seq_len=args.seq_len
    )

    # 划分训练/验证
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"从epoch {start_epoch}恢复训练")

    # 训练循环
    print("\n" + "="*80)
    print("开始训练")
    print("="*80)

    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-"*80)

        # 训练
        train_losses = train_epoch(model, train_loader, optimizer, device, epoch)

        # 验证
        val_metrics = validate(model, val_loader, device)

        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 记录
        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_metrics'].append(val_metrics)

        # 打印
        print(f"训练损失: {train_losses['total']:.6f} (data: {train_losses['data']:.6f}, physics: {train_losses['physics']:.6f})")
        print(f"验证损失: {val_metrics['loss']:.6f}")
        print(f"验证指标: MAE={val_metrics['mae']:.6f}, RMSE={val_metrics['rmse']:.6f}, R²={val_metrics['r2']:.4f}")
        print(f"学习率: {current_lr:.2e}")
        print(f"权重: λ_data={model.lambda_data.item():.3f}, λ_physics={model.lambda_physics.item():.3f}")

        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']

            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_metrics': val_metrics,
                'history': history,
                'model_info': info
            }, checkpoint_path)

            print(f"[+] 保存最佳模型 (val_loss: {best_val_loss:.6f})")

    print("\n" + "="*80)
    print("训练完成")
    print("="*80)

    # 保存训练历史
    history_path = Path(args.output_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"训练历史已保存: {history_path}")


if __name__ == '__main__':
    main()
