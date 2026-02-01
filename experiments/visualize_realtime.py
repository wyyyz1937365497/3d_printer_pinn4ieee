"""
可视化实时轨迹修正器结果

生成6种可视化:
1. 训练曲线 (loss, lr)
2. 预测vs真实散点图
3. 误差时间序列对比
4. 误差分布直方图
5. 2D轨迹误差热图
6. 推理性能分析
"""

import os
import sys
import argparse
from pathlib import Path
import json
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

from data.realtime_dataset import RealTimeTrajectoryDataset
from models.realtime_corrector import RealTimeCorrector


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    model = RealTimeCorrector(
        input_size=4,
        hidden_size=56,
        num_layers=2,
        dropout=0.1
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def build_test_loader(data_pattern, seq_len=20, batch_size=256, num_workers=2):
    """构建测试数据加载器"""
    import glob
    import random

    all_dirs = glob.glob(data_pattern)
    mat_files = []
    for d in all_dirs:
        mat_files.extend(glob.glob(os.path.join(d, "*.mat")))

    random.shuffle(mat_files)

    n_train = int(0.7 * len(mat_files))
    test_files = mat_files[n_train + int(0.15 * len(mat_files)):]

    train_files = mat_files[:n_train]
    temp_train_dataset = RealTimeTrajectoryDataset(
        train_files, seq_len=seq_len, pred_len=1, scaler=None, mode='train'
    )

    test_dataset = RealTimeTrajectoryDataset(
        test_files, seq_len=seq_len, pred_len=1,
        scaler=temp_train_dataset.scaler, mode='test'
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return test_loader


def collect_predictions(model, test_loader, device):
    """收集所有预测结果"""
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].squeeze(1)

            outputs = model(inputs)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return preds, targets


def visualize_training(history, save_path):
    """1. 绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss曲线
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MAE)', fontsize=12)
    axes[0].set_title('Training Progress', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # 学习率曲线
    axes[1].plot(history['lr'], color='green', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Learning Rate', fontsize=12)
    axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ {save_path.name}")


def visualize_predictions(preds, targets, save_path):
    """2. 绘制预测vs真实散点图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # X轴
    r2_x = r2_score(targets[:, 0], preds[:, 0])
    axes[0].scatter(targets[:, 0], preds[:, 0], alpha=0.3, s=1, color='blue')
    min_val, max_val = targets[:, 0].min(), targets[:, 0].max()
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    axes[0].set_xlabel('True Error X (mm)', fontsize=12)
    axes[0].set_ylabel('Predicted Error X (mm)', fontsize=12)
    axes[0].set_title(f'X-Axis (R²={r2_x:.4f})', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Y轴
    r2_y = r2_score(targets[:, 1], preds[:, 1])
    axes[1].scatter(targets[:, 1], preds[:, 1], alpha=0.3, s=1, color='green')
    min_val, max_val = targets[:, 1].min(), targets[:, 1].max()
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    axes[1].set_xlabel('True Error Y (mm)', fontsize=12)
    axes[1].set_ylabel('Predicted Error Y (mm)', fontsize=12)
    axes[1].set_title(f'Y-Axis (R²={r2_y:.4f})', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ {save_path.name}")


def visualize_error_time_series(preds, targets, save_path, num_samples=1000):
    """3. 绘制误差时间序列"""
    preds_subset = preds[:num_samples]
    targets_subset = targets[:num_samples]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # X轴误差
    axes[0].plot(targets_subset[:, 0], label='True', alpha=0.7, linewidth=1)
    axes[0].plot(preds_subset[:, 0], label='Predicted', alpha=0.7, linewidth=1)
    axes[0].set_ylabel('Error X (mm)', fontsize=12)
    axes[0].set_title('X-Axis Error Over Time', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Y轴误差
    axes[1].plot(targets_subset[:, 1], label='True', alpha=0.7, linewidth=1)
    axes[1].plot(preds_subset[:, 1], label='Predicted', alpha=0.7, linewidth=1)
    axes[1].set_ylabel('Error Y (mm)', fontsize=12)
    axes[1].set_xlabel('Time Step', fontsize=12)
    axes[1].set_title('Y-Axis Error Over Time', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ {save_path.name}")


def visualize_error_distribution(preds, targets, save_path):
    """4. 绘制误差分布直方图"""
    errors = preds - targets

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # X轴误差分布
    axes[0].hist(errors[:, 0], bins=50, alpha=0.7, edgecolor='black', color='blue')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].set_xlabel('Prediction Error X (mm)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'X-Axis Error Distribution\n(Mean={errors[:, 0].mean():.4f}, Std={errors[:, 0].std():.4f})',
                     fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Y轴误差分布
    axes[1].hist(errors[:, 1], bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1].set_xlabel('Prediction Error Y (mm)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'Y-Axis Error Distribution\n(Mean={errors[:, 1].mean():.4f}, Std={errors[:, 1].std():.4f})',
                     fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ {save_path.name}")


def visualize_2d_error_heatmap(preds, targets, save_path):
    """5. 绘制2D误差热图"""
    errors = preds - targets
    error_magnitude = np.sqrt(errors[:, 0]**2 + errors[:, 1]**2)

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(targets[:, 0], targets[:, 1],
                        c=error_magnitude, cmap='hot', s=1, alpha=0.5)
    ax.set_xlabel('True Error X (mm)', fontsize=12)
    ax.set_ylabel('True Error Y (mm)', fontsize=12)
    ax.set_title('2D Error Magnitude Heatmap', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Error Magnitude (mm)', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ {save_path.name}")


def visualize_inference_performance(model, device, save_path):
    """6. 绘制推理性能分析"""
    model.eval()
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    avg_times = []
    throughputs = []

    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, 20, 4).to(device)

        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # 计时
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = model(dummy_input)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        end = time.perf_counter()

        avg_time = (end - start) / 100
        throughput = batch_size / avg_time

        avg_times.append(avg_time * 1000)  # ms
        throughputs.append(throughput)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 平均时间
    axes[0].plot(batch_sizes, avg_times, marker='o', linewidth=2, markersize=6)
    axes[0].axhline(1.0, color='r', linestyle='--', linewidth=2, label='1ms Threshold')
    axes[0].set_xlabel('Batch Size', fontsize=12)
    axes[0].set_ylabel('Average Inference Time (ms)', fontsize=12)
    axes[0].set_title('Inference Time vs Batch Size', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')

    # 吞吐量
    axes[1].plot(batch_sizes, throughputs, marker='o', color='green',
                linewidth=2, markersize=6)
    axes[1].set_xlabel('Batch Size', fontsize=12)
    axes[1].set_ylabel('Throughput (inferences/sec)', fontsize=12)
    axes[1].set_title('Throughput vs Batch Size', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ {save_path.name}")


def main():
    parser = argparse.ArgumentParser(description='Visualize real-time trajectory corrector results')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data_simulation_*',
                       help='Data directory pattern')
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--output_dir', type=str, default='results/realtime_visualization',
                       help='Output directory for visualizations')

    args = parser.parse_args()

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 加载模型
    print(f"\n加载模型: {args.checkpoint}")
    model, checkpoint = load_model(args.checkpoint, device)

    # 加载测试数据
    print("\n加载测试数据...")
    test_loader = build_test_loader(
        args.data_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 收集预测
    print("\n生成预测...")
    preds, targets = collect_predictions(model, test_loader, device)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n生成可视化到: {output_dir}")
    print("-" * 80)

    # 1. 训练曲线
    if 'history' in checkpoint:
        visualize_training(checkpoint['history'], output_dir / '01_training_curves.png')

    # 2. 预测vs真实
    visualize_predictions(preds, targets, output_dir / '02_predictions.png')

    # 3. 误差时间序列
    visualize_error_time_series(preds, targets, output_dir / '03_error_timeseries.png')

    # 4. 误差分布
    visualize_error_distribution(preds, targets, output_dir / '04_error_distribution.png')

    # 5. 2D热图
    visualize_2d_error_heatmap(preds, targets, output_dir / '05_error_heatmap.png')

    # 6. 推理性能
    visualize_inference_performance(model, device, output_dir / '06_inference_performance.png')

    print("-" * 80)
    print(f"\n✓ 所有可视化已生成到: {output_dir}")


if __name__ == '__main__':
    main()
