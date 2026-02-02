"""
实时轨迹修正可视化 - 调用MATLAB完整仿真

功能：
1. 调用MATLAB进行逐点实时修正仿真
2. MATLAB内部：每一点都预测误差→修正→物理仿真
3. 加载仿真结果生成可视化
4. 完全模拟真实3D打印过程

使用方法：
    python experiments/visualize_realtime_correction.py \
        --checkpoint checkpoints/realtime_corrector/best_model.pth \
        --gcode test_gcode_files/3DBenchy_PLA_1h28m.gcode \
        --layer 25

作者: 3D Printer PINN Project
日期: 2026-02-02
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
import json
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_matlab_simulation(gcode_file, layer_num, checkpoint_path, output_dir):
    """
    调用MATLAB运行实时修正仿真

    Args:
        gcode_file: G-code文件路径
        layer_num: 层编号
        checkpoint_path: 模型检查点路径
        output_dir: 输出目录

    Returns:
        results: MATLAB仿真结果字典
    """
    print("="*70)
    print("实时轨迹修正仿真 - 调用MATLAB")
    print("="*70)
    print(f"\n配置:")
    print(f"  G-code: {gcode_file}")
    print(f"  层编号: {layer_num}")
    print(f"  模型: {checkpoint_path}")
    print(f"  输出目录: {output_dir}")

    # 启动MATLAB引擎
    print(f"\n启动MATLAB引擎...")
    matlab = matlab.engine.start_matlab()
    print("  ✓ MATLAB启动成功")

    # 添加仿真路径
    sim_path = str(project_root / 'simulation')
    matlab.addpath(sim_path, nargout=0)
    print(f"  ✓ 添加路径: {sim_path}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 调用MATLAB仿真函数
    print(f"\n运行MATLAB仿真...")
    print(f"  (每个点: 预测误差 → 修正轨迹 → 物理仿真)\n")

    try:
        # 调用仿真函数
        matlab.simulate_realtime_correction(
            gcode_file,
            float(layer_num),
            checkpoint_path,
            output_dir,
            nargout=0  # 不返回输出，结果保存在文件中
        )

        print(f"\n  ✓ MATLAB仿真完成")

    except Exception as e:
        print(f"\n  ✗ MATLAB仿真失败: {e}")
        print(f"\n可能的原因:")
        print(f"  1. G-code文件不存在: {gcode_file}")
        print(f"  2. 模型文件不存在: {checkpoint_path}")
        print(f"  3. MATLAB函数路径错误")
        print(f"  4. Python环境未正确配置")
        raise
    finally:
        # 关闭MATLAB
        matlab.quit()
        print(f"\n  ✓ MATLAB已关闭")

    # 加载仿真结果
    result_file = os.path.join(output_dir, f'realtime_correction_layer_{layer_num}.mat')
    print(f"\n加载仿真结果: {result_file}")

    # 使用MATLAB加载（因为保存的是v7.3格式）
    matlab = matlab.engine.start_matlab()
    mat_data = matlab.load(result_file)
    matlab.quit()

    # 转换为Python字典
    results = convert_matlab_results(mat_data)

    return results


def convert_matlab_results(mat_data):
    """将MATLAB结果转换为Python字典"""
    results = {}

    # 时间
    results['time'] = np.array(mat_data['time'][0]).flatten()

    # 参考轨迹
    traj = mat_data['trajectory'][0]
    results['x_ref'] = np.array(traj['x_ref'][0]).flatten()
    results['y_ref'] = np.array(traj['y_ref'][0]).flatten()
    results['z_ref'] = np.array(traj['z_ref'][0]).flatten()
    results['vx'] = np.array(traj['vx'][0]).flatten()
    results['vy'] = np.array(traj['vy'][0]).flatten()

    # 修正后轨迹
    traj_corr = mat_data['trajectory_corrected'][0]
    results['x_corrected'] = np.array(traj_corr['x'][0]).flatten()
    results['y_corrected'] = np.array(traj_corr['y'][0]).flatten()

    # 实际轨迹
    traj_act = mat_data['trajectory_actual'][0]
    results['x_actual'] = np.array(traj_act['x'][0]).flatten()
    results['y_actual'] = np.array(traj_act['y'][0]).flatten()

    # 未修正误差
    err_uncorr = mat_data['error_uncorrected'][0]
    results['error_x_uncorrected'] = np.array(err_uncorr['x'][0]).flatten()
    results['error_y_uncorrected'] = np.array(err_uncorr['y'][0]).flatten()
    results['error_mag_uncorrected'] = np.array(err_uncorr['mag'][0]).flatten()

    # 修正后误差
    err_corr = mat_data['error_corrected'][0]
    results['error_x_corrected'] = np.array(err_corr['x'][0]).flatten()
    results['error_y_corrected'] = np.array(err_corr['y'][0]).flatten()
    results['error_mag_corrected'] = np.array(err_corr['mag'][0]).flatten()

    # 预测误差
    pred_err = mat_data['predicted_error'][0]
    results['pred_error_x'] = np.array(pred_err['x'][0]).flatten()
    results['pred_error_y'] = np.array(pred_err['y'][0]).flatten()

    # 性能统计
    perf = mat_data['performance'][0]
    results['mean_inference_time_ms'] = float(perf['mean_inference_time_ms'][0])
    results['max_inference_time_ms'] = float(perf['max_inference_time_ms'][0])
    results['throughput_pred_per_sec'] = float(perf['throughput_pred_per_sec'][0])

    return results


def visualize_results(results, output_dir):
    """生成完整的对比可视化"""
    print(f"\n生成可视化...")

    os.makedirs(output_dir, exist_ok=True)

    # 提取数据
    x_ref = results['x_ref']
    y_ref = results['y_ref']
    x_corr = results['x_corrected']
    y_corr = results['y_corrected']
    x_act = results['x_actual']
    y_act = results['y_actual']

    err_mag_uncorr = results['error_mag_uncorrected']
    err_mag_corr = results['error_mag_corrected']

    # 1. 主要对比图：误差热图
    _plot_heatmap_comparison(x_ref, y_ref, err_mag_uncorr,
                           x_corr, y_corr, err_mag_corr, output_dir)

    # 2. 三轨迹对比图
    _plot_three_trajectories(x_ref, y_ref, x_corr, y_corr, x_act, y_act, output_dir)

    # 3. 详细统计图
    _plot_statistics(results, output_dir)

    # 4. 时间序列图
    _plot_time_series(results, output_dir)

    # 5. 保存统计报告
    _save_statistics_report(results, output_dir)

    print(f"\n✓ 所有可视化已保存到: {output_dir}")

    # 打印总结
    _print_summary(results)


def _plot_heatmap_comparison(x_ref, y_ref, err_mag_uncorr,
                           x_corr, y_corr, err_mag_corr, output_dir):
    """绘制误差热图对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    vmin = 0
    vmax = max(err_mag_uncorr.max(), err_mag_corr.max())

    # 未修正
    sc1 = axes[0].scatter(x_ref, y_ref, c=err_mag_uncorr * 1000, cmap='hot',
                         s=2, alpha=0.7, vmin=vmin*1000, vmax=vmax*1000)
    axes[0].set_title('Uncorrected Simulation\n(Before Real-Time Correction)',
                     fontsize=14, fontweight='bold')
    axes[0].set_xlabel('X Position (mm)', fontsize=12)
    axes[0].set_ylabel('Y Position (mm)', fontsize=12)
    axes[0].axis('equal')
    cbar1 = plt.colorbar(sc1, ax=axes[0])
    cbar1.set_label('Error (μm)', fontsize=11)

    # 修正后
    sc2 = axes[1].scatter(x_corr, y_corr, c=err_mag_corr * 1000, cmap='hot',
                         s=2, alpha=0.7, vmin=vmin*1000, vmax=vmax*1000)
    axes[1].set_title('Corrected Simulation\n(After Real-Time Correction)',
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('X Position (mm)', fontsize=12)
    axes[1].set_ylabel('Y Position (mm)', fontsize=12)
    axes[1].axis('equal')
    cbar2 = plt.colorbar(sc2, ax=axes[1])
    cbar2.set_label('Error (μm)', fontsize=11)

    plt.suptitle('Real-Time Trajectory Error Correction',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'heatmap_comparison.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'heatmap_comparison_hd.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 热图对比")


def _plot_three_trajectories(x_ref, y_ref, x_corr, y_corr, x_act, y_act, output_dir):
    """绘制三条轨迹对比：参考、修正指令、实际位置"""
    fig, ax = plt.subplots(figsize=(10, 10))

    # 只显示部分点以保持清晰
    n_show = min(5000, len(x_ref))
    step = max(1, n_show // 5000)

    # 参考轨迹（黑色虚线）
    ax.plot(x_ref[::step], y_ref[::step], 'k--', linewidth=0.5,
           alpha=0.5, label='Reference (G-code)')

    # 修正后的轨迹（发送给电机的指令，蓝色）
    ax.plot(x_corr[::step], y_corr[::step], 'b-', linewidth=0.8,
           alpha=0.6, label='Corrected Command (to Motor)')

    # 实际轨迹（执行后的位置，绿色）
    ax.plot(x_act[::step], y_act[::step], 'g-', linewidth=0.8,
           alpha=0.6, label='Actual Position (After Correction)')

    ax.set_xlabel('X Position (mm)', fontsize=12)
    ax.set_ylabel('Y Position (mm)', fontsize=12)
    ax.set_title('Three-Way Trajectory Comparison',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'three_trajectories.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 三轨迹对比")


def _plot_statistics(results, output_dir):
    """绘制详细统计图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    err_mag_uncorr = results['error_mag_uncorrected']
    err_mag_corr = results['error_mag_corrected']
    err_x_uncorr = results['error_x_uncorrected']
    err_y_uncorr = results['error_y_uncorrected']
    err_x_corr = results['error_x_corrected']
    err_y_corr = results['error_y_corrected']

    # 计算改善率
    improvement = (1 - err_mag_corr / err_mag_uncorr) * 100
    improvement = improvement[np.isfinite(improvement)]

    # 误差分布
    axes[0, 0].hist(err_mag_uncorr * 1000, bins=50, alpha=0.6,
                   label='Uncorrected', color='red', density=True)
    axes[0, 0].hist(err_mag_corr * 1000, bins=50, alpha=0.6,
                   label='Corrected', color='green', density=True)
    axes[0, 0].set_xlabel('Error Magnitude (μm)', fontsize=12)
    axes[0, 0].set_ylabel('Density', fontsize=12)
    axes[0, 0].set_title('Error Distribution', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # X轴误差相关性
    axes[0, 1].scatter(err_x_uncorr * 1000, err_x_corr * 1000,
                      alpha=0.3, s=1, c='steelblue')
    limit = max(abs(err_x_uncorr).max(), abs(err_x_corr).max()) * 1000
    axes[0, 1].plot([-limit, limit], [-limit, limit], 'r--',
                  linewidth=2, label='Perfect correlation')
    axes[0, 1].set_xlabel('Uncorrected Error X (μm)', fontsize=12)
    axes[0, 1].set_ylabel('Corrected Error X (μm)', fontsize=12)
    axes[0, 1].set_title('X-Axis Error Correlation', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')

    # 改善率分布
    axes[1, 0].hist(improvement, bins=50, color='steelblue', alpha=0.7,
                   edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    axes[1, 0].axvline(improvement.mean(), color='green', linestyle='-',
                      linewidth=2, label=f'Mean: {improvement.mean():.1f}%')
    axes[1, 0].set_xlabel('Improvement (%)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Improvement Distribution', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 推理时间统计
    # 由于没有保存每个点的推理时间，显示平均值
    mean_time = results['mean_inference_time_ms']
    max_time = results['max_inference_time_ms']
    throughput = results['throughput_pred_per_sec']

    axes[1, 1].barh(['Mean', 'Max'], [mean_time, max_time],
                    color=['green', 'orange'], alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(1.0, color='red', linestyle='--', linewidth=2,
                       label='Real-time limit (1ms)')
    axes[1, 1].set_xlabel('Inference Time (ms)', fontsize=12)
    axes[1, 1].set_title(f'Real-Time Performance (Throughput: {throughput:.0f} pred/s)',
                         fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_statistics.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 详细统计")


def _plot_time_series(results, output_dir):
    """绘制时间序列对比"""
    n_show = min(2000, len(results['time']))
    t = results['time'][:n_show]

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(t, results['error_mag_uncorrected'][:n_show] * 1000,
           alpha=0.7, label='Uncorrected', linewidth=1, color='red')
    ax.plot(t, results['error_mag_corrected'][:n_show] * 1000,
           alpha=0.7, label='Corrected', linewidth=1, color='green')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Error Magnitude (μm)', fontsize=12)
    ax.set_title(f'Error Time Series (First {n_show} points)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_series.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 时间序列")


def _save_statistics_report(results, output_dir):
    """保存统计报告"""
    err_mag_uncorr = results['error_mag_uncorrected']
    err_mag_corr = results['error_mag_corrected']

    improvement = (1 - err_mag_corr / err_mag_uncorr) * 100
    improvement = improvement[np.isfinite(improvement)]

    report = {
        'timestamp': datetime.now().isoformat(),
        'uncorrected': {
            'mean_error_um': float(err_mag_uncorr.mean() * 1000),
            'max_error_um': float(err_mag_uncorr.max() * 1000),
            'std_error_um': float(err_mag_uncorr.std() * 1000),
        },
        'corrected': {
            'mean_error_um': float(err_mag_corr.mean() * 1000),
            'max_error_um': float(err_mag_corr.max() * 1000),
            'std_error_um': float(err_mag_corr.std() * 1000),
        },
        'improvement': {
            'mean_improvement_percent': float(improvement.mean()),
            'median_improvement_percent': float(np.median(improvement)),
            'mean_error_reduction_percent': float(
                (1 - err_mag_corr.mean() / err_mag_uncorr.mean()) * 100
            ),
        },
        'performance': {
            'mean_inference_time_ms': results['mean_inference_time_ms'],
            'max_inference_time_ms': results['max_inference_time_ms'],
            'throughput_predictions_per_sec': results['throughput_pred_per_sec'],
            'realtime_capable': bool(results['mean_inference_time_ms'] < 1.0),
        }
    }

    report_path = os.path.join(output_dir, 'correction_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"  ✓ 统计报告")


def _print_summary(results):
    """打印结果总结"""
    err_mag_uncorr = results['error_mag_uncorrected']
    err_mag_corr = results['error_mag_corrected']

    improvement = (1 - err_mag_corr.mean() / err_mag_uncorr.mean()) * 100

    print(f"\n{'='*70}")
    print(f"实时轨迹修正仿真 - 结果总结")
    print(f"{'='*70}\n")

    print(f"未修正误差（参考）:")
    print(f"  平均: {err_mag_uncorr.mean()*1000:.1f} μm")
    print(f"  最大: {err_mag_uncorr.max()*1000:.1f} μm")
    print(f"  标准差: {err_mag_uncorr.std()*1000:.1f} μm")

    print(f"\n修正后误差:")
    print(f"  平均: {err_mag_corr.mean()*1000:.1f} μm")
    print(f"  最大: {err_mag_corr.max()*1000:.1f} μm")
    print(f"  标准差: {err_mag_corr.std()*1000:.1f} μm")

    print(f"\n改善效果:")
    print(f"  平均误差降低: {improvement:.1f}%")

    print(f"\n实时性能:")
    print(f"  平均推理时间: {results['mean_inference_time_ms']:.3f} ms")
    print(f"  最大推理时间: {results['max_inference_time_ms']:.3f} ms")
    print(f"  吞吐量: {results['throughput_pred_per_sec']:.0f} predictions/s")
    print(f"  实时性: {'✓ 满足' if results['mean_inference_time_ms'] < 1.0 else '✗ 不满足'} (< 1ms)")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize real-time trajectory correction with MATLAB simulation'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--gcode', type=str,
                       default='test_gcode_files/3DBenchy_PLA_1h28m.gcode',
                       help='G-code file to simulate')
    parser.add_argument('--layer', type=int, default=25,
                       help='Layer number to simulate')
    parser.add_argument('--output_dir', type=str,
                       default='results/realtime_correction',
                       help='Output directory')

    args = parser.parse_args()

    try:
        # 运行MATLAB仿真
        results = run_matlab_simulation(
            gcode_file=args.gcode,
            layer_num=args.layer,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir
        )

        # 生成可视化
        visualize_results(results, args.output_dir)

        print(f"\n✓ 完成！")
        print(f"\n生成的文件:")
        print(f"  - heatmap_comparison.png / heatmap_comparison_hd.png")
        print(f"  - three_trajectories.png")
        print(f"  - detailed_statistics.png")
        print(f"  - time_series.png")
        print(f"  - correction_report.json")
        print(f"  - realtime_correction_layer_{args.layer}.mat")

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
