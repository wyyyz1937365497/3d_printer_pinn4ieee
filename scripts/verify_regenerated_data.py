#!/usr/bin/env python3
"""
验证新生成的仿真数据的误差范围

该脚本检查使用更新后参数生成的.mat文件,
验证误差是否在目标范围内 (±50-100 μm)

使用方法:
    python verify_regenerated_data.py <data_directory>

示例:
    python verify_regenerated_data.py ../test_output_new_params
    python verify_regenerated_data.py ../data_simulation_3DBenchy_PLA_1h28m_sampled_48layers
"""

import sys
import h5py
import numpy as np
import glob
from pathlib import Path


def analyze_mat_file(mat_path):
    """分析单个.mat文件的误差范围"""
    try:
        with h5py.File(mat_path, 'r') as f:
            # 查找误差数据
            if 'error_x' in f:
                error_x = f['error_x'][:]
                error_y = f['error_y'][:]
            elif 'trajectory_error' in f:
                traj_error = f['trajectory_error']
                if 'error_x' in traj_error:
                    error_x = traj_error['error_x'][:]
                    error_y = traj_error['error_y'][:]
                else:
                    return None
            else:
                return None

            # 转换为微米
            error_x_um = error_x * 1000
            error_y_um = error_y * 1000

            # 统计分析
            stats = {
                'x_range_um': (float(np.min(error_x_um)), float(np.max(error_x_um))),
                'y_range_um': (float(np.min(error_y_um)), float(np.max(error_y_um))),
                'x_std_um': float(np.std(error_x_um)),
                'y_std_um': float(np.std(error_y_um)),
                'x_rms_um': float(np.sqrt(np.mean(error_x_um**2))),
                'y_rms_um': float(np.sqrt(np.mean(error_y_um**2))),
                'x_max_abs_um': float(np.max(np.abs(error_x_um))),
                'y_max_abs_um': float(np.max(np.abs(error_y_um))),
                'num_samples': len(error_x_um)
            }

            return stats
    except Exception as e:
        print(f"  错误: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("使用方法: python verify_regenerated_data.py <data_directory>")
        print("\n示例:")
        print("  python verify_regenerated_data.py ../test_output_new_params")
        print("  python verify_regenerated_data.py ../data_simulation_*")
        sys.exit(1)

    data_dir = sys.argv[1]

    # 查找所有.mat文件
    pattern = f"{data_dir}/*.mat"
    mat_files = sorted(glob.glob(pattern))

    if not mat_files:
        print(f"错误: 在 {data_dir} 中未找到.mat文件")
        sys.exit(1)

    print("=" * 80)
    print("验证新生成的仿真数据误差范围")
    print("=" * 80)
    print(f"\n数据目录: {data_dir}")
    print(f"找到 {len(mat_files)} 个.mat文件\n")

    # 分析每个文件
    all_stats = []
    for i, mat_file in enumerate(mat_files, 1):
        print(f"[{i}/{len(mat_files)}] 分析 {Path(mat_file).name}")

        stats = analyze_mat_file(mat_file)
        if stats:
            all_stats.append(stats)
            print(f"  X轴误差: [{stats['x_range_um'][0]:.2f}, {stats['x_range_um'][1]:.2f}] μm")
            print(f"  Y轴误差: [{stats['y_range_um'][0]:.2f}, {stats['y_range_um'][1]:.2f}] μm")
            print(f"  X轴最大: ±{stats['x_max_abs_um']:.2f} μm")
            print(f"  Y轴最大: ±{stats['y_max_abs_um']:.2f} μm")
            print(f"  样本数: {stats['num_samples']:,}")
        else:
            print(f"  跳过 (无误差数据)")

    if not all_stats:
        print("\n错误: 没有找到有效的误差数据")
        sys.exit(1)

    # 汇总统计
    print("\n" + "=" * 80)
    print("汇总统计 (所有文件)")
    print("=" * 80)

    # 计算平均统计
    avg_x_max = np.mean([s['x_max_abs_um'] for s in all_stats])
    avg_y_max = np.mean([s['y_max_abs_um'] for s in all_stats])
    avg_x_rms = np.mean([s['x_rms_um'] for s in all_stats])
    avg_y_rms = np.mean([s['y_rms_um'] for s in all_stats])

    print(f"\n平均最大绝对误差:")
    print(f"  X轴: ±{avg_x_max:.2f} μm")
    print(f"  Y轴: ±{avg_y_max:.2f} μm")

    print(f"\n平均RMS误差:")
    print(f"  X轴: {avg_x_rms:.2f} μm")
    print(f"  Y轴: {avg_y_rms:.2f} μm")

    # 验证是否在目标范围内
    target_min = 50  # μm
    target_max = 100  # μm

    print("\n" + "=" * 80)
    print("目标验证 (目标: ±50-100 μm)")
    print("=" * 80)

    x_in_range = target_min <= avg_x_max <= target_max
    y_in_range = target_min <= avg_y_max <= target_max

    if x_in_range:
        print(f"✓ X轴误差: ±{avg_x_max:.2f} μm - 在目标范围内!")
    elif avg_x_max < target_min:
        print(f"✗ X轴误差: ±{avg_x_max:.2f} μm - 太小 (目标: >=50 μm)")
        print("  建议: 进一步降低刚度或质量")
    else:
        print(f"✗ X轴误差: ±{avg_x_max:.2f} μm - 太大 (目标: <=100 μm)")
        print("  建议: 增加刚度或质量")

    if y_in_range:
        print(f"✓ Y轴误差: ±{avg_y_max:.2f} μm - 在目标范围内!")
    elif avg_y_max < target_min:
        print(f"✗ Y轴误差: ±{avg_y_max:.2f} μm - 太小 (目标: >=50 μm)")
        print("  建议: 进一步降低刚度或质量")
    else:
        print(f"✗ Y轴误差: ±{avg_y_max:.2f} μm - 太大 (目标: <=100 μm)")
        print("  建议: 增加刚度或质量")

    print("\n" + "=" * 80)

    if x_in_range and y_in_range:
        print("✓ 参数验证成功!")
        print("  误差范围符合Ender-3实际精度")
        print("  可以使用这些数据训练模型")
        return 0
    else:
        print("✗ 参数需要调整")
        print("  请修改 matlab_simulation/physics_parameters.m")
        return 1


if __name__ == '__main__':
    sys.exit(main())
