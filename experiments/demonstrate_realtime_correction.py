"""
简化版实时轨迹修正演示

使用现有数据快速演示修正效果，无需MATLAB Engine
适用于：
- 快速验证模型效果
- 演示实时预测过程
- 生成可视化结果
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.realtime_corrector import RealTimeCorrector
from data.realtime_dataset import RealTimeTrajectoryDataset
import glob


def demonstrate_correction(checkpoint_path, data_dir, output_dir, device='cuda'):
    """
    演示实时轨迹修正效果

    Args:
        checkpoint_path: 模型检查点路径
        data_dir: 数据目录
        output_dir: 输出目录
        device: 计算设备
    """
    print("="*70)
    print("实时轨迹修正演示（简化版）")
    print("="*70)

    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载模型
    print(f"\n[1/4] 加载模型: {checkpoint_path}")
    model = RealTimeCorrector(
        input_size=4,
        hidden_size=56,
        num_layers=2,
        dropout=0.1
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("  ✓ 模型加载成功")

    # 2. 加载数据
    print(f"\n[2/4] 加载数据: {data_dir}")
    all_dirs = glob.glob(data_dir)
    mat_files = []
    for d in all_dirs:
        mat_files.extend(glob.glob(os.path.join(d, "*.mat")))

    if not mat_files:
        raise ValueError(f'未找到数据文件: {data_dir}')

    # 使用第一个文件
    mat_file = mat_files[0]
    print(f"  使用文件: {Path(mat_file).name}")

    # 创建数据集以获取scaler
    dataset = RealTimeTrajectoryDataset(
        mat_files[:min(5, len(mat_files))],
        seq_len=20,
        pred_len=1,
        scaler=None,
        mode='train'
    )

    # 加载单个文件的数据
    features, labels = dataset.load_single_file(mat_file)

    print(f"  样本数: {features.shape[0]}")
    print(f"  ✓ 数据加载成功")

    # 3. 实时预测并修正
    print(f"\n[3/4] 实时预测并修正轨迹")

    seq_len = 20
    history = deque(maxlen=seq_len)

    x_ref = features[:, 0]
    y_ref = features[:, 1]
    vx_ref = features[:, 2]
    vy_ref = features[:, 3]

    err_x_orig = labels[:, 0]
    err_y_orig = labels[:, 1]

    # 修正后的轨迹
    x_corrected = x_ref.copy()
    y_corrected = y_ref.copy()

    # 预测的误差
    pred_errors = []

    print(f"  序列长度: {seq_len}")
    print(f"  推理设备: {device}")

    with torch.no_grad():
        for i in range(len(x_ref)):
            # 当前特征
            feat = np.array([x_ref[i], y_ref[i], vx_ref[i], vy_ref[i]])
            history.append(feat)

            if len(history) < seq_len:
                # 历史不足，零误差
                pred_err = np.array([0.0, 0.0])
            else:
                # 准备输入
                seq = np.array(history)  # [seq_len, 4]
                seq_norm = dataset.scaler.transform(seq)
                inp = torch.FloatTensor(seq_norm).unsqueeze(0).to(device)

                # 预测
                pred = model(inp)
                pred_err = pred.cpu().numpy()[0]

            pred_errors.append(pred_err)

            # 应用修正
            x_corrected[i] -= pred_err[0]
            y_corrected[i] -= pred_err[1]

            # 每1000点打印进度
            if (i + 1) % 1000 == 0:
                sys.stdout.write(f"\r  进度: {i+1}/{len(x_ref)} ({(i+1)/len(x_ref)*100:.1f}%)")
                sys.stdout.flush()

    pred_errors = np.array(pred_errors)
    print(f"\n  ✓ 预测完成")

    # 4. 计算修正后的误差（简化假设）
    # 注意：这里假设修正后的误差 = 原始误差 - 预测误差
    # 在真实MATLAB仿真中，这会通过重新仿真得到
    err_x_corr = err_x_orig - pred_errors[:, 0]
    err_y_corr = err_y_orig - pred_errors[:, 1]

    err_mag_orig = np.sqrt(err_x_orig**2 + err_y_orig**2)
    err_mag_corr = np.sqrt(err_x_corr**2 + err_y_corr**2)

    print(f"\n[4/4] 生成可视化")

    # 统计
    improvement = (1 - err_mag_corr / err_mag_orig) * 100
    improvement = improvement[np.isfinite(improvement)]

    print(f"\n{'='*70}")
    print(f"修正效果统计")
    print(f"{'='*70}")
    print(f"\n原始误差（未修正）:")
    print(f"  平均: {err_mag_orig.mean():.4f} mm")
    print(f"  最大: {err_mag_orig.max():.4f} mm")
    print(f"  标准差: {err_mag_orig.std():.4f} mm")

    print(f"\n修正后误差:")
    print(f"  平均: {err_mag_corr.mean():.4f} mm")
    print(f"  最大: {err_mag_corr.max():.4f} mm")
    print(f"  标准差: {err_mag_corr.std():.4f} mm")

    print(f"\n改善率:")
    print(f"  平均改善: {improvement.mean():.1f}%")
    print(f"  中位改善: {np.median(improvement):.1f}%")
    print(f"  误差降低: {(1 - err_mag_corr.mean()/err_mag_orig.mean())*100:.1f}%")
    print(f"{'='*70}\n")

    # 创建可视化
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. 误差热图对比
    ax1 = fig.add_subplot(gs[0, :])
    vmin = 0
    vmax = max(err_mag_orig.max(), err_mag_corr.max())

    sc1 = ax1.scatter(x_ref, y_ref, c=err_mag_orig, cmap='hot',
                     s=2, alpha=0.7, vmin=vmin, vmax=vmax, label='Original')
    ax1.set_title('Original Error (Before Correction)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Position (mm)', fontsize=11)
    ax1.set_ylabel('Y Position (mm)', fontsize=11)
    ax1.axis('equal')
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label('Error (mm)', fontsize=10)

    # 2. 误差分布
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(err_mag_orig, bins=50, alpha=0.6, label='Original',
             color='red', density=True)
    ax2.hist(err_mag_corr, bins=50, alpha=0.6, label='Corrected',
             color='green', density=True)
    ax2.set_xlabel('Error Magnitude (mm)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 误差相关性
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(err_mag_orig, err_mag_corr, alpha=0.3, s=1, c='steelblue')
    limit = max(err_mag_orig.max(), err_mag_corr.max())
    ax3.plot([0, limit], [0, limit], 'r--', linewidth=2, label='Perfect')
    ax3.set_xlabel('Original Error (mm)', fontsize=11)
    ax3.set_ylabel('Corrected Error (mm)', fontsize=11)
    ax3.set_title('Error Correlation', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    # 4. 改善率分布
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(improvement, bins=50, color='steelblue', alpha=0.7,
             edgecolor='black')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    ax4.axvline(improvement.mean(), color='green', linestyle='-',
                linewidth=2, label=f'Mean: {improvement.mean():.1f}%')
    ax4.set_xlabel('Improvement (%)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Improvement Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. 时间序列对比（前1000个点）
    ax5 = fig.add_subplot(gs[2, 1])
    n_show = min(1000, len(err_mag_orig))
    t = np.arange(n_show) / 100  # 假设100Hz采样
    ax5.plot(t, err_mag_orig[:n_show], alpha=0.7, label='Original', linewidth=1)
    ax5.plot(t, err_mag_corr[:n_show], alpha=0.7, label='Corrected', linewidth=1)
    ax5.set_xlabel('Time (s)', fontsize=11)
    ax5.set_ylabel('Error (mm)', fontsize=11)
    ax5.set_title('Error Time Series (First 1000 points)',
                  fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    plt.suptitle('Real-Time Trajectory Error Correction Demo',
                 fontsize=16, fontweight='bold', y=0.995)

    # 保存
    output_path = os.path.join(output_dir, 'correction_demo.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 保存图片: {output_path}")

    # 高分辨率版本
    output_path_hd = os.path.join(output_dir, 'correction_demo_hd.png')
    plt.savefig(output_path_hd, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存高清图: {output_path_hd}")

    plt.close()

    print(f"\n✓ 演示完成！")
    print(f"  结果保存在: {output_dir}")
    print(f"\n提示：这是简化演示，使用模拟数据验证修正效果。")
    print(f"      完整演示需要MATLAB Engine，详见使用指南。")


def main():
    parser = argparse.ArgumentParser(
        description='Simple real-time correction demo (no MATLAB required)'
    )
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/realtime_corrector/best_model.pth',
                       help='Model checkpoint path')
    parser.add_argument('--data_dir', type=str,
                       default='data/simulation/*',
                       help='Data directory pattern')
    parser.add_argument('--output_dir', type=str,
                       default='results/correction_demo',
                       help='Output directory')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')

    args = parser.parse_args()

    try:
        demonstrate_correction(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device
        )
    except FileNotFoundError as e:
        print(f"\n✗ 错误: {e}")
        print(f"\n请确保：")
        print(f"  1. 模型文件存在: {args.checkpoint}")
        print(f"  2. 数据文件存在: {args.data_dir}")
        print(f"  3. 已运行数据生成脚本")
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
