"""
检查训练数据 - 快速查看生成的数据统计

使用方法:
    python check_training_data.py --data_dir "data/simulation/*.mat"
"""

import argparse
import glob
import h5py
import numpy as np
from pathlib import Path
import sys

# 设置Windows控制台编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def check_mat_file(filepath):
    """检查单个.mat文件"""
    try:
        with h5py.File(filepath, 'r') as f:
            if 'simulation_data' not in f:
                return None, "无simulation_data字段"

            sim_data = f['simulation_data']

            # 提取基本信息
            if 'time' in sim_data:
                time = sim_data['time'][:]
                n_points = len(time)
            else:
                return None, "无time字段"

            # 提取轨迹和误差
            has_trajectory = all(k in sim_data for k in ['x_ref', 'y_ref', 'vx_ref', 'vy_ref'])
            has_error = all(k in sim_data for k in ['error_x', 'error_y'])

            info = {
                'n_points': n_points,
                'has_trajectory': has_trajectory,
                'has_error': has_error,
            }

            # 如果有误差数据，计算统计
            if has_error:
                error_x = sim_data['error_x'][:]
                error_y = sim_data['error_y'][:]
                error_mag = np.sqrt(error_x**2 + error_y**2)

                info['error_mean'] = np.mean(error_mag) * 1000  # μm
                info['error_std'] = np.std(error_mag) * 1000
                info['error_max'] = np.max(error_mag) * 1000
                info['error_min'] = np.min(error_mag) * 1000

            return info, None

    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(description='检查训练数据统计')
    parser.add_argument('--data_dir', type=str, default='data/simulation/*.mat',
                       help='数据文件路径模式')
    args = parser.parse_args()

    # 查找文件
    mat_files = glob.glob(args.data_dir)

    if not mat_files:
        print(f"❌ 未找到.mat文件: {args.data_dir}")
        return

    print(f"找到 {len(mat_files)} 个.mat文件")
    print("=" * 80)

    # 统计信息
    valid_files = []
    invalid_files = []

    total_points = 0
    error_means = []
    error_maxs = []

    for filepath in sorted(mat_files):
        filename = Path(filepath).name
        info, error = check_mat_file(filepath)

        if info is not None:
            valid_files.append((filepath, info))
            total_points += info['n_points']

            if 'error_mean' in info:
                error_means.append(info['error_mean'])
                error_maxs.append(info['error_max'])

            print(f"✓ {filename:40s} | {info['n_points']:6d} 点", end='')
            if 'error_mean' in info:
                print(f" | 误差: {info['error_mean']:6.2f} ± {info['error_std']:5.2f} μm (max: {info['error_max']:6.2f} μm)")
            else:
                print()
        else:
            invalid_files.append((filepath, error))
            print(f"✗ {filename:40s} | 错误: {error}")

    # 打印汇总
    print("\n" + "=" * 80)
    print("汇总统计")
    print("=" * 80)

    print(f"\n文件统计:")
    print(f"  总文件数: {len(mat_files)}")
    print(f"  有效文件: {len(valid_files)}")
    print(f"  无效文件: {len(invalid_files)}")

    if valid_files:
        print(f"\n数据点统计:")
        print(f"  总数据点: {total_points:,}")
        print(f"  平均每文件: {total_points // len(valid_files):,} 点")

        if error_means:
            error_means = np.array(error_means)
            error_maxs = np.array(error_maxs)

            print(f"\n误差统计 (跨所有文件):")
            print(f"  平均误差: {np.mean(error_means):.2f} μm")
            print(f"  误差标准差: {np.std(error_means):.2f} μm")
            print(f"  最小平均误差: {np.min(error_means):.2f} μm")
            print(f"  最大平均误差: {np.max(error_means):.2f} μm")
            print(f"  平均最大误差: {np.mean(error_maxs):.2f} μm")

        print(f"\n数据质量:")
        if np.mean(error_means) > 20 and np.mean(error_means) < 200:
            print(f"  ✓ 误差范围合理（{np.mean(error_means):.1f} μm）")
        else:
            print(f"  ⚠ 误差范围可能需要调整（当前: {np.mean(error_means):.1f} μm）")

    if invalid_files:
        print(f"\n无效文件详情:")
        for filepath, error in invalid_files:
            print(f"  - {Path(filepath).name}: {error}")

    print("\n" + "=" * 80)
    print("下一步:")
    print("=" * 80)
    print(f"\n1. 开始训练:")
    print(f"   python experiments/train_realtime.py --data_dir \"{args.data_dir}\"")
    print(f"\n2. 如果数据不足，可以运行:")
    print(f"   matlab -batch " + "\"collect_training_data\"")


if __name__ == '__main__':
    main()
