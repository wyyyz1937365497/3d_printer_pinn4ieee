#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集统计脚本
统计当前已收集的仿真数据量和训练样本数
"""

import os
import glob
import numpy as np
from pathlib import Path

def count_mat_files():
    """统计.mat文件"""
    print("=" * 80)
    print("数据文件统计")
    print("=" * 80)
    print()

    # 查找所有数据目录
    data_dirs = glob.glob("data_simulation_*")
    data_dirs.sort()

    if not data_dirs:
        print("[!] 未找到任何数据目录")
        return

    total_files = 0
    total_size_mb = 0

    print(f"{'目录':<60} {'文件数':>10} {'大小(MB)':>12}")
    print("-" * 85)

    for data_dir in data_dirs:
        mat_files = glob.glob(os.path.join(data_dir, "*.mat"))
        n_files = len(mat_files)

        # 计算目录大小
        dir_size = 0
        for mat_file in mat_files:
            if os.path.exists(mat_file):
                dir_size += os.path.getsize(mat_file) / (1024 * 1024)

        total_files += n_files
        total_size_mb += dir_size

        dir_name = os.path.basename(data_dir)
        print(f"{dir_name:<60} {n_files:>10} {dir_size:>11.1f}")

    print("-" * 85)
    print(f"{'总计':<60} {total_files:>10} {total_size_mb:>11.1f}")
    print()

    return total_files, data_dirs

def estimate_training_samples(total_files):
    """估算训练样本数"""
    print("=" * 80)
    print("训练样本估算")
    print("=" * 80)
    print()

    # 假设每个文件平均点数（根据不同gcode类型）
    avg_points_per_layer = {
        '3DBenchy': 1200,
        'bearing': 1000,
        'Nautilus': 2000,
        'boat': 1200,
    }

    # 根据文件名判断类型
    def get_avg_points(dirname):
        for key, val in avg_points_per_layer.items():
            if key in dirname:
                return val
        return 1200  # 默认值

    # 计算总数据点
    data_dirs = glob.glob("data_simulation_*")
    total_points = 0
    for data_dir in data_dirs:
        mat_files = glob.glob(os.path.join(data_dir, "*.mat"))
        n_files = len(mat_files)
        avg_points = get_avg_points(os.path.basename(data_dir))
        total_points += n_files * avg_points

    print(f"配置参数:")
    print(f"  序列长度 (seq_len): 200")
    print(f"  预测长度 (pred_len): 50")
    print(f"  采样间隔 (stride): 5")
    print()

    # 估算训练样本数
    # 每个序列需要 seq_len + pred_len 个点，序列之间stride间隔
    stride = 5
    seq_len = 200
    pred_len = 50

    # 粗略估算：每个原始点约生成 1/stride 个样本
    estimated_samples = total_points // stride

    print(f"原始数据点: ~{total_points:,}")
    print(f"训练样本数 (stride={stride}): ~{estimated_samples:,}")
    print()

    # 模型参数统计
    model_params = 896030
    ratio = model_params / estimated_samples if estimated_samples > 0 else float('inf')

    print(f"模型参数: {model_params:,}")
    print(f"参数/样本比: {ratio:.1f}:1", end="")

    if ratio < 10:
        print(" ✅✅ (优秀！)")
    elif ratio < 20:
        print(" ✅ (良好)")
    elif ratio < 50:
        print(" ⚠️ (可接受)")
    else:
        print(" ❌ (不足)")

    print()

    # 目标样本数（按照论文级别 20:1 比例）
    target_ratio = 20
    target_samples = model_params / target_ratio

    print(f"目标样本数 (20:1比例): ~{target_samples:,.0f}")
    if estimated_samples < target_samples:
        shortage = target_samples - estimated_samples
        shortage_pct = (shortage / target_samples) * 100
        print(f"还需收集: ~{shortage:,.0f} 样本 (短缺 {shortage_pct:.1f}%)")
    else:
        print(f"✅ 已达到目标！")

    print()

    return estimated_samples

def check_per_directory():
    """详细统计每个目录"""
    print("=" * 80)
    print("各目录详细统计")
    print("=" * 80)
    print()

    data_dirs = glob.glob("data_simulation_*")
    data_dirs.sort()

    print(f"{'目录':<55} {'层数':>8} {'预期点数':>12} {'预期样本':>12}")
    print("-" * 90)

    total_expected_points = 0
    total_expected_samples = 0

    for data_dir in data_dirs:
        mat_files = glob.glob(os.path.join(data_dir, "*.mat"))
        n_layers = len(mat_files)

        # 根据目录名估算点数/层
        dir_name = os.path.basename(data_dir)
        if 'Benchy' in dir_name or 'boat' in dir_name:
            points_per_layer = 1200
        elif 'bearing' in dir_name:
            points_per_layer = 1000
        elif 'Nautilus' in dir_name:
            points_per_layer = 2000
        else:
            points_per_layer = 1200

        expected_points = n_layers * points_per_layer
        expected_samples = expected_points // 5

        total_expected_points += expected_points
        total_expected_samples += expected_samples

        # 简化目录名显示
        if len(dir_name) > 53:
            dir_display = "..." + dir_name[-50:]
        else:
            dir_display = dir_name

        print(f"{dir_display:<55} {n_layers:>8} {expected_points:>12,} {expected_samples:>12,}")

    print("-" * 90)
    print(f"{'总计':<55} {'':>8} {total_expected_points:>12,} {total_expected_samples:>12,}")
    print()

def show_progress():
    """显示收集进度"""
    print("=" * 80)
    print("数据收集进度")
    print("=" * 80)
    print()

    # 目标配置（基于collect_data_single_param.m）
    targets = {
        '3DBenchy': 48,
        'bearing': 75,
        'Nautilus': 56,
        'boat': 74,
    }

    data_dirs = glob.glob("data_simulation_*")
    total_completed = 0
    total_target = 0

    print(f"{'文件':<20} {'目标':>10} {'已完成':>10} {'进度':>10} {'状态':>10}")
    print("-" * 65)

    for name, target in targets.items():
        # 查找对应目录
        matching_dirs = [d for d in data_dirs if name in d]

        if matching_dirs:
            data_dir = matching_dirs[0]
            mat_files = glob.glob(os.path.join(data_dir, "*.mat"))
            completed = len(mat_files)
        else:
            completed = 0

        total_completed += completed
        total_target += target

        progress = (completed / target * 100) if target > 0 else 0

        if progress >= 100:
            status = "✅"
        elif progress > 0:
            status = "🔄"
        else:
            status = "⏳"

        print(f"{name:<20} {target:>10} {completed:>10} {progress:>9.1f}% {status:>10}")

    print("-" * 65)
    overall_progress = (total_completed / total_target * 100) if total_target > 0 else 0
    print(f"{'总计':<20} {total_target:>10} {total_completed:>10} {overall_progress:>9.1f}%")
    print()

    return total_completed, total_target

def main():
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "3D打印PINN数据集统计工具" + " " * 32 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # 1. 统计.mat文件
    total_files, data_dirs = count_mat_files()

    if total_files == 0:
        print("[!] 未找到任何数据文件")
        return

    # 2. 显示收集进度
    show_progress()

    # 3. 估算训练样本
    estimate_training_samples(total_files)

    # 4. 详细目录统计
    check_per_directory()

    print("=" * 80)
    print("下一步建议:")
    print("=" * 80)
    print()
    print("1. 验证数据加载:")
    print('   python -c "from data.simulation import PrinterSimulationDataset; import glob;')
    print('                files = glob.glob(\'data_simulation_*/*.mat\');')
    print('                print(f\'找到 {len(files)} 个.mat文件\');')
    print('                ds = PrinterSimulationDataset(files, seq_len=200, pred_len=50,')
    print('                                          stride=5, mode=\'train\', fit_scaler=True);')
    print('                print(f\'训练样本: {len(ds)}\')"')
    print()
    print("2. 开始训练模型:")
    print('   python experiments/train_implicit_state_tcn_optimized.py \\')
    print('       --data_dir "data_simulation_*" \\')
    print('       --epochs 100 \\')
    print('       --batch_size 256 \\')
    print('       --lr 1e-3 \\')
    print('       --lambda_physics 0.05 \\')
    print('       --num_workers 8')
    print()
    print("=" * 80)
    print()

if __name__ == "__main__":
    main()
