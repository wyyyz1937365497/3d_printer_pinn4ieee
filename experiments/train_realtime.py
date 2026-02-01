"""
训练实时轨迹修正器

关键优化:
1. seq_len=20 (0.2秒历史)
2. pred_len=1 (单步预测)
3. 轻量级模型 (~46K参数)
4. MAE损失
5. 混合精度 + 梯度累积
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from data.realtime_dataset import RealTimeTrajectoryDataset
from models.realtime_corrector import RealTimeCorrector


def build_loaders(data_pattern, seq_len=20, batch_size=256, num_workers=2):
    """构建数据加载器"""
    import glob
    import random

    # 查找所有.mat文件
    all_dirs = glob.glob(data_pattern)
    mat_files = []
    for d in all_dirs:
        mat_files.extend(glob.glob(os.path.join(d, "*.mat")))

    if not mat_files:
        raise ValueError(f"未找到.mat文件: {data_pattern}")

    # 随机打乱
    random.shuffle(mat_files)

    # 划分数据集
    n_train = int(0.7 * len(mat_files))
    n_val = int(0.15 * len(mat_files))

    train_files = mat_files[:n_train]
    val_files = mat_files[n_train:n_train + n_val]
    test_files = mat_files[n_train + n_val:]

    print(f"\n数据集划分:")
    print(f"  总文件数: {len(mat_files)}")
    print(f"  训练集: {len(train_files)} 个文件")
    print(f"  验证集: {len(val_files)} 个文件")
    print(f"  测试集: {len(test_files)} 个文件")

    # 创建数据集
    train_dataset = RealTimeTrajectoryDataset(
        train_files,
        seq_len=seq_len,
        pred_len=1,
        scaler=None,
        mode='train'
    )

    val_dataset = RealTimeTrajectoryDataset(
        val_files,
        seq_len=seq_len,
        pred_len=1,
        scaler=train_dataset.scaler,
        mode='val'
    )

    print(f"\n序列配置:")
    print(f"  seq_len: {seq_len} ({seq_len/100:.2f}秒 @ 100Hz)")
    print(f"  pred_len: 1 (0.01秒提前)")
    print(f"  训练样本: {len(train_dataset):,}")
    print(f"  验证样本: {len(val_dataset):,}")

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, accumulation_steps=2):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="[Train]", ncols=120, leave=False)

    for step, batch in enumerate(pbar):
        # 数据移到设备
        inputs = batch['input'].to(device)
        targets = batch['target'].squeeze(1).to(device)  # [batch, 2]

        # 混合精度训练
        if scaler:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets) / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets) / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.6f}'})

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[Val]", ncols=120, leave=False)

        for batch in pbar:
            inputs = batch['input'].to(device)
            targets = batch['target'].squeeze(1).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()

            # 计算MAE
            mae = torch.abs(outputs - targets).mean()
            total_mae += mae.item()

            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    avg_loss = total_loss / len(val_loader)
    avg_mae = total_mae / len(val_loader)

    return avg_loss, avg_mae


def main():
    parser = argparse.ArgumentParser(description='Train real-time trajectory corrector')
    parser.add_argument('--data_dir', type=str, default="data_simulation_*",
                       help='Data directory pattern')
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--accumulation_steps', type=int, default=2,
                       help='Gradient accumulation steps')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 数据加载器
    train_loader, val_loader = build_loaders(
        args.data_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 创建模型
    model = RealTimeCorrector(
        input_size=4,
        hidden_size=56,
        num_layers=2,
        dropout=0.1
    ).to(device)

    # 打印模型信息
    info = model.get_model_info()
    print(f"\n模型配置:")
    print(f"  参数量: {info['total_parameters']:,}")
    print(f"  可训练参数: {info['trainable_parameters']:,}")

    # 损失函数 (MAE)
    criterion = nn.L1Loss()

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    # 混合精度
    scaler = GradScaler() if args.mixed_precision and device.type == 'cuda' else None

    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15

    print(f"\n开始训练...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Base LR: {args.lr}")
    print(f"  Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"  混合精度: {scaler is not None}")
    print("="*80)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'lr': []
    }

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-"*80)

        # 训练
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.accumulation_steps
        )

        # 验证
        val_loss, val_mae = validate(model, val_loader, criterion, device)

        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['lr'].append(current_lr)

        # 打印
        print(f"  训练损失: {train_loss:.6f}")
        print(f"  验证损失: {val_loss:.6f}")
        print(f"  验证MAE: {val_mae:.6f} mm")
        print(f"  学习率: {current_lr:.2e}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint_dir = Path('checkpoints/realtime_corrector')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'history': history,
                'model_info': info,
            }, checkpoint_dir / 'best_model.pth')

            print(f"  ✓ 保存最佳模型 (val_loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  ✗ 无改进 ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"\n早停: {patience}个epochs无改进")
                break

    print("\n" + "="*80)
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")

    # 保存训练历史
    history_path = Path('checkpoints/realtime_corrector/training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"训练历史已保存: {history_path}")


if __name__ == '__main__':
    main()
