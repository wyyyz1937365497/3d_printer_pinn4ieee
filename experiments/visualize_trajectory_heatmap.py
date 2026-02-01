"""
可视化完整轨迹误差热图对比 (修正前 vs 修正后)

参照 main 分支的 visualize_trajectory_correction.py 实现
适配到新的实时修正系统
"""

import os
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import matplotlib.pyplot as plt
import glob

from data.realtime_dataset import RealTimeTrajectoryDataset, load_mat_file
from models.realtime_corrector import RealTimeCorrector


def main():
    parser = argparse.ArgumentParser(description='Visualize trajectory error heatmap comparison')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/simulation/*',
                       help='Data directory pattern')
    parser.add_argument('--file_index', type=int, default=0,
                       help='Which file to visualize (default: 0)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Stride for prediction (default: 1)')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 加载模型
    print(f"\n加载模型: {args.checkpoint}")
    model = RealTimeCorrector(
        input_size=4,
        hidden_size=56,
        num_layers=2,
        dropout=0.1
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 查找数据文件
    all_dirs = glob.glob(args.data_dir)
    mat_files = []
    for d in all_dirs:
        mat_files.extend(glob.glob(os.path.join(d, "*.mat")))

    if not mat_files:
        raise ValueError(f'未找到.mat文件: {args.data_dir}')

    mat_files = sorted(mat_files)
    file_index = max(0, min(args.file_index, len(mat_files) - 1))
    selected_file = mat_files[file_index]

    print(f"使用文件: {Path(selected_file).name}")

    # 加载单个文件的数据
    features, labels = load_mat_file(selected_file)
    # features: [N, 4] = [x_ref, y_ref, vx_ref, vy_ref]
    # labels: [N, 2] = [error_x, error_y]

    n = min(features.shape[0], labels.shape[0])
    x_ref = features[:n, 0]  # X位置
    y_ref = features[:n, 1]  # Y位置
    err_x = labels[:n, 0]    # X误差
    err_y = labels[:n, 1]    # Y误差

    # 需要归一化输入特征（使用与训练时相同的scaler）
    # 先创建一个临时数据集来拟合scaler
    temp_dataset = RealTimeTrajectoryDataset(
        mat_files[:10],  # 用部分文件拟合scaler
        seq_len=20,
        pred_len=1,
        scaler=None,
        mode='train'
    )

    # 归一化特征
    input_features = temp_dataset.scaler.transform(features[:n])

    # 预测整个轨迹的误差
    seq_len = 20
    pred_len = 1
    stride = args.stride

    print(f"\n预测轨迹误差...")
    print(f"  序列长度: {seq_len}")
    print(f"  预测步长: {pred_len}")
    print(f"  步幅: {stride}")

    pred_sum = np.zeros((n, 2), dtype=np.float32)
    pred_count = np.zeros((n, 1), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n - seq_len - pred_len + 1, stride):
            # 准备输入序列
            seq = input_features[i:i + seq_len]  # [seq_len, 4]
            inp = torch.FloatTensor(seq).unsqueeze(0).to(device)  # [1, seq_len, 4]

            # 预测
            pred = model(inp)  # [1, 2]
            pred_err = pred.cpu().numpy()[0]  # [2]

            # 累积预测结果
            start = i + seq_len
            end = start + pred_len
            pred_sum[start:end] += pred_err
            pred_count[start:end] += 1

    # 平均重复预测的位置
    pred_count[pred_count == 0] = 1
    pred_err = pred_sum / pred_count  # [n, 2]

    # 计算修正后的误差
    corr_err_x = err_x - pred_err[:, 0]  # 真实误差 - 预测误差 = 修正后剩余误差
    corr_err_y = err_y - pred_err[:, 1]

    # 计算误差幅度
    err_mag = np.sqrt(err_x ** 2 + err_y ** 2)
    corr_err_mag = np.sqrt(corr_err_x ** 2 + corr_err_y ** 2)

    # 统计
    print(f"\n误差统计:")
    print(f"  修正前 - 平均误差: {err_mag.mean():.4f} mm, 最大误差: {err_mag.max():.4f} mm")
    print(f"  修正后 - 平均误差: {corr_err_mag.mean():.4f} mm, 最大误差: {corr_err_mag.max():.4f} mm")
    print(f"  改善率: {(1 - corr_err_mag.mean() / err_mag.mean()) * 100:.1f}%")

    # 绘制热图
    results_dir = Path('results')
    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 使用统一的颜色范围（取两个误差的最大值）
    vmin = 0
    vmax = max(err_mag.max(), corr_err_mag.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 左图：修正前的误差热图
    sc1 = axes[0].scatter(x_ref, y_ref, c=err_mag, cmap='hot', s=2, alpha=0.7,
                         vmin=vmin, vmax=vmax)
    axes[0].set_title('Simulated Error Heatmap (Before Correction)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('X Position (mm)', fontsize=12)
    axes[0].set_ylabel('Y Position (mm)', fontsize=12)
    axes[0].axis('equal')
    cbar1 = plt.colorbar(sc1, ax=axes[0])
    cbar1.set_label('Error Magnitude (mm)', fontsize=11)

    # 右图：修正后的误差热图
    sc2 = axes[1].scatter(x_ref, y_ref, c=corr_err_mag, cmap='hot', s=2, alpha=0.7,
                         vmin=vmin, vmax=vmax)
    axes[1].set_title('Corrected Error Heatmap (After Correction)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('X Position (mm)', fontsize=12)
    axes[1].set_ylabel('Y Position (mm)', fontsize=12)
    axes[1].axis('equal')
    cbar2 = plt.colorbar(sc2, ax=axes[1])
    cbar2.set_label('Error Magnitude (mm)', fontsize=11)

    plt.tight_layout()

    # 保存
    out_path = figures_dir / 'trajectory_error_heatmap_compare.png'
    plt.savefig(out_path, dpi=200)
    print(f"\n✓ 保存图片: {out_path}")

    # 同时保存高分辨率版本
    out_path_hd = figures_dir / 'trajectory_error_heatmap_compare_hd.png'
    plt.savefig(out_path_hd, dpi=300)
    print(f"✓ 保存高清图片: {out_path_hd}")

    plt.close()

    # 额外生成：误差改善统计图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 误差幅度分布对比
    axes[0, 0].hist(err_mag, bins=50, alpha=0.6, label='Before', color='red')
    axes[0, 0].hist(corr_err_mag, bins=50, alpha=0.6, label='After', color='green')
    axes[0, 0].set_xlabel('Error Magnitude (mm)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Error Distribution Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # X轴误差对比
    axes[0, 1].scatter(err_x, corr_err_x, alpha=0.3, s=1)
    min_val = min(err_x.min(), corr_err_x.min())
    max_val = max(err_x.max(), corr_err_x.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='No improvement')
    axes[0, 1].set_xlabel('Error X Before (mm)', fontsize=12)
    axes[0, 1].set_ylabel('Error X After (mm)', fontsize=12)
    axes[0, 1].set_title('X-Axis Error Before vs After', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')

    # Y轴误差对比
    axes[1, 0].scatter(err_y, corr_err_y, alpha=0.3, s=1)
    min_val = min(err_y.min(), corr_err_y.min())
    max_val = max(err_y.max(), corr_err_y.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='No improvement')
    axes[1, 0].set_xlabel('Error Y Before (mm)', fontsize=12)
    axes[1, 0].set_ylabel('Error Y After (mm)', fontsize=12)
    axes[1, 0].set_title('Y-Axis Error Before vs After', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axis('equal')

    # 误差改善率统计
    improvement_x = (1 - np.abs(corr_err_x) / np.abs(err_x)) * 100
    improvement_y = (1 - np.abs(corr_err_y) / np.abs(err_y)) * 100

    # 过滤无穷值
    improvement_x = improvement_x[np.isfinite(improvement_x)]
    improvement_y = improvement_y[np.isfinite(improvement_y)]

    axes[1, 1].hist(improvement_x, bins=50, alpha=0.6, label='X-axis', color='blue')
    axes[1, 1].hist(improvement_y, bins=50, alpha=0.6, label='Y-axis', color='orange')
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    axes[1, 1].set_xlabel('Improvement Rate (%)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Error Improvement Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    out_path_stats = figures_dir / 'trajectory_error_improvement_stats.png'
    plt.savefig(out_path_stats, dpi=200)
    print(f"✓ 保存统计图: {out_path_stats}")

    plt.close()


if __name__ == '__main__':
    main()
