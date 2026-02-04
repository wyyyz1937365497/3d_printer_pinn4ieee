"""
分析G-code文件中的实际打印速度

用途：
1. 统计G-code中所有F（速度）指令
2. 分析实际打印速度范围
3. 验证与MATLAB参数的匹配度
4. 为physics_parameters.m提供实际速度参考

使用方法：
    python test_gcode_files/analyze_gcode_speed.py \
        --gcode test_gcode_files/Tremendous_Hillar_PLA_10m22s.gcode
"""

import argparse
import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def analyze_gcode_speed(gcode_file, output_dir=None):
    """分析G-code中的速度分布"""

    print(f"分析G-code文件: {gcode_file}\n")

    # 读取G-code
    with open(gcode_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # 提取速度（F指令）
    speeds = []  # mm/s
    travel_speeds = []  # 空移速度
    print_speeds = []  # 打印速度
    line_numbers = []

    for i, line in enumerate(lines, 1):
        # 只处理G1/G0移动指令
        if not (line.strip().startswith('G1') or line.strip().startswith('G0')):
            continue

        # 提取F参数（速度）
        match = re.search(r'F(\d+\.?\d*)', line)
        if match:
            speed_mm_min = float(match.group(1))
            speed_mm_s = speed_mm_min / 60.0  # 转换为mm/s
            speeds.append(speed_mm_s)
            line_numbers.append(i)

            # 区分空移和打印（有E值变化为打印）
            e_match = re.search(r'E(-?\d+\.?\d*)', line)
            if e_match:
                e_value = float(e_match.group(1))
                # E值变化大于0.01认为是打印
                if abs(e_value) > 0.01:
                    print_speeds.append(speed_mm_s)
            else:
                travel_speeds.append(speed_mm_s)

    # 转换为numpy数组
    speeds = np.array(speeds)
    print_speeds = np.array(print_speeds)
    travel_speeds = np.array(travel_speeds)

    # 统计分析
    print("="*70)
    print("速度统计结果")
    print("="*70)

    print(f"\n总移动指令数: {len(speeds)}")
    print(f"  打印移动: {len(print_speeds)}")
    print(f"  空移移动: {len(travel_speeds)}")

    if len(speeds) > 0:
        print(f"\n所有移动 (包括打印和空移):")
        print(f"  最小速度: {np.min(speeds):.1f} mm/s")
        print(f"  最大速度: {np.max(speeds):.1f} mm/s")
        print(f"  平均速度: {np.mean(speeds):.1f} mm/s")
        print(f"  中位速度: {np.median(speeds):.1f} mm/s")
        print(f"  标准差: {np.std(speeds):.1f} mm/s")

        # 速度分位数
        print(f"\n速度分位数:")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"  {p}%: {np.percentile(speeds, p):.1f} mm/s")

    if len(print_speeds) > 0:
        print(f"\n仅打印移动 (有挤出):")
        print(f"  最小速度: {np.min(print_speeds):.1f} mm/s")
        print(f"  最大速度: {np.max(print_speeds):.1f} mm/s")
        print(f"  平均速度: {np.mean(print_speeds):.1f} mm/s")
        print(f"  中位速度: {np.median(print_speeds):.1f} mm/s")
        print(f"  标准差: {np.std(print_speeds):.1f} mm/s")

    if len(travel_speeds) > 0:
        print(f"\n仅空移移动 (无挤出):")
        print(f"  最小速度: {np.min(travel_speeds):.1f} mm/s")
        print(f"  最大速度: {np.max(travel_speeds):.1f} mm/s")
        print(f"  平均速度: {np.mean(travel_speeds):.1f} mm/s")
        print(f"  中位速度: {np.median(travel_speeds):.1f} mm/s")

    # 速度区间分布
    print(f"\n速度区间分布:")
    bins = [0, 20, 40, 60, 80, 100, 150, 200, 300, 500, 10000]
    for i in range(len(bins)-1):
        count = np.sum((speeds >= bins[i]) & (speeds < bins[i+1]))
        pct = count / len(speeds) * 100
        bar = '█' * int(pct / 2)
        print(f"  {bins[i]:4d}-{bins[i+1]:4d} mm/s: {pct:5.1f}% {bar}")

    # 与MATLAB参数对比
    print(f"\n与MATLAB参数对比:")
    print(f"  MATLAB最大速度: 500 mm/s")
    print(f"  G-code最大速度: {np.max(speeds):.1f} mm/s")
    if np.max(speeds) <= 500:
        print(f"  匹配度: 匹配")
    else:
        print(f"  匹配度: G-code超过MATLAB设定")

    # 推荐的测试件速度
    print(f"\n推荐测试件速度:")
    print(f"  基于中位打印速度: {np.median(print_speeds) if len(print_speeds) > 0 else 50:.1f} mm/s")
    print(f"  建议使用: 50 mm/s (保守值，符合实际打印)")

    # 生成可视化
    if output_dir:
        plot_speed_distribution(speeds, print_speeds, travel_speeds, output_dir, gcode_file)

    # 生成JSON报告
    if output_dir:
        speed_report = {
            'gcode_file': str(gcode_file),
            'total_moves': int(len(speeds)),
            'print_moves': int(len(print_speeds)),
            'travel_moves': int(len(travel_speeds)),
            'all_moves': {
                'min_mm_s': float(np.min(speeds)) if len(speeds) > 0 else None,
                'max_mm_s': float(np.max(speeds)) if len(speeds) > 0 else None,
                'mean_mm_s': float(np.mean(speeds)) if len(speeds) > 0 else None,
                'median_mm_s': float(np.median(speeds)) if len(speeds) > 0 else None,
                'std_mm_s': float(np.std(speeds)) if len(speeds) > 0 else None,
            },
            'print_moves_only': {
                'min_mm_s': float(np.min(print_speeds)) if len(print_speeds) > 0 else None,
                'max_mm_s': float(np.max(print_speeds)) if len(print_speeds) > 0 else None,
                'mean_mm_s': float(np.mean(print_speeds)) if len(print_speeds) > 0 else None,
                'median_mm_s': float(np.median(print_speeds)) if len(print_speeds) > 0 else None,
                'std_mm_s': float(np.std(print_speeds)) if len(print_speeds) > 0 else None,
            },
            'travel_moves_only': {
                'min_mm_s': float(np.min(travel_speeds)) if len(travel_speeds) > 0 else None,
                'max_mm_s': float(np.max(travel_speeds)) if len(travel_speeds) > 0 else None,
                'mean_mm_s': float(np.mean(travel_speeds)) if len(travel_speeds) > 0 else None,
                'median_mm_s': float(np.median(travel_speeds)) if len(travel_speeds) > 0 else None,
                'std_mm_s': float(np.std(travel_speeds)) if len(travel_speeds) > 0 else None,
            },
            'recommended_test_speed': 50.0,
            'matlab_max_speed': 500.0,
        }

        report_file = Path(output_dir) / 'speed_analysis_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(speed_report, f, indent=2)

        print(f"\n[OK] Report saved: {report_file}")

    return speed_report if output_dir else None


def plot_speed_distribution(speeds, print_speeds, travel_speeds, output_dir, gcode_file):
    """绘制速度分布图"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 速度直方图
    ax = axes[0, 0]
    ax.hist(speeds, bins=50, alpha=0.7, edgecolor='black', label='所有移动')
    ax.axvline(np.mean(speeds), color='r', linestyle='--', linewidth=2, label=f'平均: {np.mean(speeds):.1f} mm/s')
    ax.axvline(np.median(speeds), color='g', linestyle='--', linewidth=2, label=f'中位: {np.median(speeds):.1f} mm/s')
    ax.set_xlabel('速度 (mm/s)', fontsize=12)
    ax.set_ylabel('频次', fontsize=12)
    ax.set_title('G-code速度分布', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. 打印vs空移速度对比
    ax = axes[0, 1]
    if len(print_speeds) > 0 and len(travel_speeds) > 0:
        ax.hist(print_speeds, bins=30, alpha=0.7, label='打印移动', edgecolor='black')
        ax.hist(travel_speeds, bins=30, alpha=0.7, label='空移移动', edgecolor='black')
        ax.set_xlabel('速度 (mm/s)', fontsize=12)
        ax.set_ylabel('频次', fontsize=12)
        ax.set_title('打印 vs 空移速度对比', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # 3. 速度累积分布
    ax = axes[1, 0]
    sorted_speeds = np.sort(speeds)
    cumulative = np.arange(1, len(speeds) + 1) / len(speeds) * 100
    ax.plot(sorted_speeds, cumulative, linewidth=2)
    ax.axvline(np.percentile(speeds, 50), color='r', linestyle='--', alpha=0.7, label='中位数')
    ax.axvline(np.percentile(speeds, 95), color='orange', linestyle='--', alpha=0.7, label='95%分位')
    ax.set_xlabel('速度 (mm/s)', fontsize=12)
    ax.set_ylabel('累积百分比 (%)', fontsize=12)
    ax.set_title('速度累积分布', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. 速度箱型图
    ax = axes[1, 1]
    box_data = [speeds, print_speeds if len(print_speeds) > 0 else speeds, travel_speeds if len(travel_speeds) > 0 else speeds]
    bp = ax.boxplot(box_data, labels=['所有移动', '打印移动', '空移移动'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightyellow']):
        patch.set_facecolor(color)
    ax.set_ylabel('速度 (mm/s)', fontsize=12)
    ax.set_title('速度箱型图对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存图片
    output_path = Path(output_dir) / 'speed_distribution_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[OK] Chart saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='分析G-code中的打印速度')
    parser.add_argument('--gcode', type=str, required=True,
                       help='G-code文件路径')
    parser.add_argument('--output_dir', type=str,
                       default='test_gcode_files/analysis',
                       help='输出目录（用于保存图表和报告）')

    args = parser.parse_args()

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 分析速度
    print("="*70)
    print("G-code速度分析工具")
    print("="*70)
    print(f"\n输入: {args.gcode}")
    print(f"输出: {args.output_dir}\n")

    analyze_gcode_speed(args.gcode, args.output_dir)

    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)
    print(f"\n建议的后续操作:")
    print(f"  1. 查看速度分布图表: {args.output_dir}/speed_distribution_analysis.png")
    print(f"  2. 阅读详细报告: {args.output_dir}/speed_analysis_report.json")
    print(f"  3. 根据实际速度更新MATLAB参数")
    print(f"  4. 使用推荐速度(50 mm/s)生成测试件G-code")


if __name__ == '__main__':
    main()
