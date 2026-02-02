"""
可视化实时轨迹修正效果 - 从已有数据生成

功能：
1. 从已有.mat文件加载仿真数据（未修正）
2. 使用训练好的模型进行逐点预测和修正
3. 生成修正前后的对比可视化

输出：
- heatmap_comparison_hd.png - 修正前后热图对比
- correction_metrics.json - 修正效果统计

使用方法：
    python visualize_correction_from_data.py \
        --checkpoint checkpoints/realtime_corrector/best_model.pth \
        --data_dir "data_simulation_3DBenchy*" \
        --layer 25 \
        --seq_len 50
"""

import os
import sys
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.realtime_dataset import RealTimeTrajectoryDataset
from models.realtime_corrector import RealTimeCorrector


def load_model(checkpoint_path, device):
    """加载训练好的模型（自动从checkpoint读取配置）"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_info' in checkpoint:
        model_info = checkpoint['model_info']
        hidden_size = model_info.get('hidden_size', 56)
        num_layers = model_info.get('num_layers', 2)
        dropout = model_info.get('dropout', 0.1)
    else:
        hidden_size = 56
        num_layers = 2
        dropout = 0.1

    model = RealTimeCorrector(
        input_size=4,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def load_simulation_data(data_dir, target_layer):
    """加载指定层的仿真数据"""
    import h5py
    import glob

    # 查找目标层的文件
    all_files = glob.glob(os.path.join(data_dir, f"layer{target_layer:02d}_*.mat"))

    if not all_files:
        raise ValueError(f"未找到layer {target_layer}的数据文件")

    # 使用第一个找到的文件
    filepath = all_files[0]
    print(f"加载数据: {filepath}")

    with h5py.File(filepath, 'r') as f:
        sim_data = f['simulation_data']

        # 提取数据
        data = {
            'time': sim_data['time'][:],
            'x_ref': sim_data['x_ref'][:],
            'y_ref': sim_data['y_ref'][:],
            'vx_ref': sim_data['vx_ref'][:],
            'vy_ref': sim_data['vy_ref'][:],
            'error_x': sim_data['error_x'][:],
            'error_y': sim_data['error_y'][:],
            'error_magnitude': sim_data['error_magnitude'][:] if 'error_magnitude' in sim_data else None,
        }

    return data


def apply_correction(model, data, scaler, seq_len, device):
    """应用实时修正"""
    print("\n应用实时修正...")

    # 准备输入特征
    x_ref = data['x_ref']
    y_ref = data['y_ref']
    vx_ref = data['vx_ref']
    vy_ref = data['vy_ref']

    # 组合特征 [N, 4]
    features = np.stack([x_ref, y_ref, vx_ref, vy_ref], axis=1)

    # 归一化
    features_normalized = scaler.transform(features)

    # 创建序列
    n_points = len(features_normalized)
    corrected_x = x_ref.copy()
    corrected_y = y_ref.copy()

    # 对于前seq_len个点，无法预测，保持原样
    # 从seq_len开始，每个点都基于前seq_len个点预测误差并修正

    model.eval()
    with torch.no_grad():
        for i in range(seq_len, n_points):
            # 获取历史序列 [seq_len, 4]
            history = features_normalized[i-seq_len:i]

            # 转为tensor
            input_tensor = torch.FloatTensor(history).unsqueeze(0).to(device)  # [1, seq_len, 4]

            # 预测误差
            pred_error = model(input_tensor).cpu().numpy()[0]  # [2]

            # 应用修正：修正位置 = 参考位置 - 预测误差
            corrected_x[i] = x_ref[i] - pred_error[0]
            corrected_y[i] = y_ref[i] - pred_error[1]

    print(f"  ✓ 已修正 {n_points} 个点")

    return corrected_x, corrected_y


def compute_corrected_error(x_corrected, y_corrected, x_ref, y_ref):
    """计算修正后的误差（假设修正后位置就是实际位置）"""
    # 修正后误差 = 参考位置 - 修正后位置
    error_x_corr = x_ref - x_corrected
    error_y_corr = y_ref - y_corrected
    error_mag_corr = np.sqrt(error_x_corr**2 + error_y_corr**2)

    return error_x_corr, error_y_corr, error_mag_corr


def plot_heatmap_comparison_hd(x_ref, y_ref, err_uncorr, err_corr, output_dir):
    """生成高分辨率热图对比"""
    print("\n生成热图对比图...")

    # 转换为微米
    err_uncorr_um = err_uncorr * 1000
    err_corr_um = err_corr * 1000

    # 计算统计
    rms_uncorr = np.sqrt(np.mean(err_uncorr_um**2))
    rms_corr = np.sqrt(np.mean(err_corr_um**2))
    mean_uncorr = np.mean(err_uncorr_um)
    mean_corr = np.mean(err_corr_um)
    max_uncorr = np.max(err_uncorr_um)
    max_corr = np.max(err_corr_um)

    improvement = (rms_uncorr - rms_corr) / rms_uncorr * 100

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # 统一颜色范围
    vmax = max(max_uncorr, max_corr)

    # 未修正热图
    sc1 = axes[0].scatter(x_ref, y_ref, c=err_uncorr_um,
                          cmap='hot', s=1, alpha=0.6,
                          vmin=0, vmax=vmax)
    axes[0].set_title('Uncorrected Trajectory\n(Original Simulation)',
                      fontsize=16, fontweight='bold', pad=15)
    axes[0].set_xlabel('X Position (mm)', fontsize=13)
    axes[0].set_ylabel('Y Position (mm)', fontsize=13)
    axes[0].axis('equal')
    cbar1 = plt.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Error (μm)', fontsize=12)

    # 添加统计文本
    stats1 = f'RMS: {rms_uncorr:.1f} μm\nMean: {mean_uncorr:.1f} μm\nMax: {max_uncorr:.1f} μm'
    axes[0].text(0.02, 0.98, stats1, transform=axes[0].transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 修正后热图
    sc2 = axes[1].scatter(x_ref, y_ref, c=err_corr_um,
                          cmap='hot', s=1, alpha=0.6,
                          vmin=0, vmax=vmax)
    axes[1].set_title('Corrected Trajectory\n(After Real-Time Correction)',
                      fontsize=16, fontweight='bold', pad=15)
    axes[1].set_xlabel('X Position (mm)', fontsize=13)
    axes[1].set_ylabel('Y Position (mm)', fontsize=13)
    axes[1].axis('equal')
    cbar2 = plt.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Error (μm)', fontsize=12)

    # 添加统计文本
    stats2 = f'RMS: {rms_corr:.1f} μm\nMean: {mean_corr:.1f} μm\nMax: {max_corr:.1f} μm'
    axes[1].text(0.02, 0.98, stats2, transform=axes[1].transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # 总标题
    plt.suptitle(f'Real-Time Trajectory Error Correction\nImprovement: {improvement:.1f}% RMS Error Reduction',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存
    output_path = os.path.join(output_dir, 'heatmap_comparison_hd.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 保存: {output_path}")

    return {
        'rms_uncorr': rms_uncorr,
        'rms_corr': rms_corr,
        'mean_uncorr': mean_uncorr,
        'mean_corr': mean_corr,
        'max_uncorr': max_uncorr,
        'max_corr': max_corr,
        'improvement': improvement
    }


def main():
    parser = argparse.ArgumentParser(description='Visualize real-time correction from existing data')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Data directory pattern (e.g., "data_simulation_3DBenchy*")')
    parser.add_argument('--layer', type=int, default=25,
                       help='Layer number to visualize')
    parser.add_argument('--seq_len', type=int, default=50,
                       help='Sequence length (must match training)')
    parser.add_argument('--output_dir', type=str,
                       default='results/realtime_correction',
                       help='Output directory')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("实时轨迹修正可视化 - 从已有数据")
    print("="*70)
    print(f"\n配置:")
    print(f"  模型: {args.checkpoint}")
    print(f"  数据目录: {args.data_dir}")
    print(f"  目标层: {args.layer}")
    print(f"  序列长度: {args.seq_len}")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")

    # 1. 加载模型
    print(f"\n加载模型...")
    model, checkpoint = load_model(args.checkpoint, device)
    print(f"  ✓ 模型加载完成")

    # 2. 加载数据
    data = load_simulation_data(args.data_dir, args.layer)
    print(f"  ✓ 数据加载完成: {len(data['x_ref'])} 个点")

    # 3. 准备scaler（用训练数据拟合）
    print(f"\n准备scaler...")
    import glob
    all_dirs = glob.glob(args.data_dir)
    all_files = []
    for d in all_dirs:
        all_files.extend(glob.glob(os.path.join(d, "*.mat")))

    # 使用一部分文件拟合scaler（避免太慢）
    import random
    random.seed(42)
    random.shuffle(all_files)
    train_files = all_files[:50]  # 用50个文件拟合

    temp_dataset = RealTimeTrajectoryDataset(
        train_files, seq_len=args.seq_len, pred_len=1,
        scaler=None, mode='train'
    )
    scaler = temp_dataset.scaler
    print(f"  ✓ Scaler拟合完成 (使用{len(train_files)}个文件)")

    # 4. 应用修正
    x_corrected, y_corrected = apply_correction(
        model, data, scaler, args.seq_len, device
    )

    # 5. 计算修正后误差
    err_x_corr, err_y_corr, err_mag_corr = compute_corrected_error(
        x_corrected, y_corrected,
        data['x_ref'], data['y_ref']
    )

    # 6. 生成可视化
    metrics = plot_heatmap_comparison_hd(
        data['x_ref'], data['y_ref'],
        data['error_magnitude'] if data['error_magnitude'] is not None
                                  else np.sqrt(data['error_x']**2 + data['error_y']**2),
        err_mag_corr,
        args.output_dir
    )

    # 7. 保存统计报告
    report = {
        'layer': args.layer,
        'model_checkpoint': args.checkpoint,
        'sequence_length': args.seq_len,
        'num_points': len(data['x_ref']),
        'metrics': metrics
    }

    report_path = os.path.join(args.output_dir, 'correction_metrics.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print("修正效果统计")
    print(f"{'='*70}")
    print(f"\n未修正轨迹:")
    print(f"  RMS误差: {metrics['rms_uncorr']:.2f} μm")
    print(f"  平均误差: {metrics['mean_uncorr']:.2f} μm")
    print(f"  最大误差: {metrics['max_uncorr']:.2f} μm")

    print(f"\n修正后轨迹:")
    print(f"  RMS误差: {metrics['rms_corr']:.2f} μm")
    print(f"  平均误差: {metrics['mean_corr']:.2f} μm")
    print(f"  最大误差: {metrics['max_corr']:.2f} μm")

    print(f"\n改进:")
    print(f"  RMS误差减少: {metrics['improvement']:.1f}%")

    print(f"\n✓ 完成！")
    print(f"\n生成的文件:")
    print(f"  - {args.output_dir}/heatmap_comparison_hd.png")
    print(f"  - {args.output_dir}/correction_metrics.json")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
