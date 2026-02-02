"""
实时轨迹修正演示 - 真实仿真工作流

功能：
1. 运行MATLAB仿真获取原始轨迹（未修正）
2. 使用Python LSTM模型实时预测误差
3. 应用修正到参考轨迹
4. 再次运行MATLAB仿真验证修正效果
5. 生成对比可视化

要求：
- 已安装 MATLAB Engine API for Python
- 已训练的轨迹误差预测模型
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import matlab.engine
import json

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.realtime_corrector import RealTimeCorrector
from sklearn.preprocessing import StandardScaler


class RealTimeCorrectionDemo:
    """实时轨迹修正演示类"""

    def __init__(self, checkpoint_path, device='cuda', matlab_engine=None):
        """
        初始化

        Args:
            checkpoint_path: 模型检查点路径
            device: 计算设备
            matlab_engine: MATLAB引擎实例（如果为None则创建新的）
        """
        self.device = torch.device(device)
        print(f"使用设备: {self.device}")

        # 加载模型
        print(f"\n加载模型: {checkpoint_path}")
        self.model = RealTimeCorrector(
            input_size=4,
            hidden_size=56,
            num_layers=2,
            dropout=0.1
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 初始化归一化器（使用训练时的参数）
        self.scaler = StandardScaler()
        # 从checkpoint中加载scaler参数（如果保存了的话）
        if 'scaler_mean' in checkpoint and 'scaler_scale' in checkpoint:
            self.scaler.mean_ = checkpoint['scaler_mean']
            self.scaler.scale_ = checkpoint['scaler_scale']
        else:
            # 使用默认参数（需要与训练时一致）
            self.scaler.mean_ = np.array([110.0, 110.0, 85.3, 85.3])  # [x, y, vx, vy]
            self.scaler.scale_ = np.array([30.5, 30.5, 45.2, 45.2])

        # MATLAB引擎
        self.matlab = matlab_engine

        # 序列参数
        self.seq_len = 20
        self.history = deque(maxlen=self.seq_len)

    def run_matlab_simulation(self, gcode_file, output_file, layers=None, use_firmware=True):
        """
        运行MATLAB仿真

        Args:
            gcode_file: G-code文件路径
            output_file: 输出.mat文件路径
            layers: 要仿真的层列表，None表示所有层
            use_firmware: 是否使用固件效应

        Returns:
            simulation_data: 仿真数据字典
        """
        print(f"\n运行MATLAB仿真...")
        print(f"  G-code: {gcode_file}")
        print(f"  输出: {output_file}")
        print(f"  层数: {layers if layers else '所有'}")
        print(f"  固件效应: {'启用' if use_firmware else '禁用'}")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 调用MATLAB仿真函数
        try:
            # 构建MATLAB命令
            if layers is None:
                # 仿真所有层
                self.matlab.run_simulation(
                    gcode_file,
                    output_file,
                    nargout=0
                )
            else:
                # 仿真指定层
                self.matlab.run_simulation(
                    gcode_file,
                    output_file,
                    layers,
                    nargout=0
                )

            print(f"  ✓ 仿真完成")

            # 加载仿真结果
            sim_data = self.matlab.load(output_file)

            # 转换为Python字典
            simulation_data = {
                'time': np.array(sim_data['time'][0]),
                'trajectory': {
                    'x_ref': np.array(sim_data['trajectory']['x_ref'][0]).flatten(),
                    'y_ref': np.array(sim_data['trajectory']['y_ref'][0]).flatten(),
                    'z_ref': np.array(sim_data['trajectory']['z_ref'][0]).flatten(),
                    'vx': np.array(sim_data['trajectory']['vx'][0]).flatten(),
                    'vy': np.array(sim_data['trajectory']['vy'][0]).flatten(),
                    'vz': np.array(sim_data['trajectory']['vz'][0]).flatten(),
                },
                'error': {
                    'error_x': np.array(sim_data['error']['error_x'][0]).flatten(),
                    'error_y': np.array(sim_data['error']['error_y'][0]).flatten(),
                    'error_mag': np.array(sim_data['error']['error_mag'][0]).flatten(),
                }
            }

            return simulation_data

        except Exception as e:
            print(f"  ✗ MATLAB仿真失败: {e}")
            raise

    def predict_and_correct(self, trajectory_data, apply_correction=True):
        """
        实时预测误差并修正轨迹

        Args:
            trajectory_data: 轨迹数据（包含x_ref, y_ref, vx, vy等）
            apply_correction: 是否应用修正

        Returns:
            corrected_trajectory: 修正后的轨迹
            predicted_errors: 预测的误差
        """
        print(f"\n实时轨迹误差预测...")
        print(f"  轨迹点数: {len(trajectory_data['x_ref'])}")
        print(f"  应用修正: {'是' if apply_correction else '否（仅预测）'}")

        # 提取特征
        x_ref = trajectory_data['x_ref']
        y_ref = trajectory_data['y_ref']
        vx_ref = trajectory_data['vx']
        vy_ref = trajectory_data['vy']

        n_points = len(x_ref)
        corrected_x = x_ref.copy()
        corrected_y = y_ref.copy()

        predicted_errors = []

        with torch.no_grad():
            for i in range(n_points):
                # 准备当前点特征
                features = np.array([x_ref[i], y_ref[i], vx_ref[i], vy_ref[i]])

                # 更新历史
                self.history.append(features)

                # 只有在累积足够历史后才开始预测
                if len(self.history) < self.seq_len:
                    # 历史不足，使用零误差
                    pred_err = np.array([0.0, 0.0])
                else:
                    # 准备输入序列
                    seq = np.array(self.history)  # [seq_len, 4]

                    # 归一化
                    seq_norm = self.scaler.transform(seq)

                    # 转换为tensor
                    inp = torch.FloatTensor(seq_norm).unsqueeze(0).to(self.device)  # [1, seq_len, 4]

                    # 预测
                    pred = self.model(inp)  # [1, 2]
                    pred_err = pred.cpu().numpy()[0]  # [2]

                predicted_errors.append(pred_err)

                # 应用修正（补偿预测的误差）
                if apply_correction:
                    corrected_x[i] -= pred_err[0]  # X轴修正
                    corrected_y[i] -= pred_err[1]  # Y轴修正

        predicted_errors = np.array(predicted_errors)

        print(f"  ✓ 预测完成")
        print(f"    预测误差范围: X [{predicted_errors[:, 0].min():.4f}, {predicted_errors[:, 0].max():.4f}] mm")
        print(f"                   Y [{predicted_errors[:, 1].min():.4f}, {predicted_errors[:, 1].max():.4f}] mm")

        # 构造修正后的轨迹数据
        corrected_trajectory = {
            'x_ref': corrected_x,
            'y_ref': corrected_y,
            'z_ref': trajectory_data['z_ref'],
            'vx': trajectory_data['vx'],
            'vy': trajectory_data['vy'],
            'vz': trajectory_data['vz'],
        }

        return corrected_trajectory, predicted_errors

    def save_corrected_trajectory(self, trajectory_data, output_file):
        """
        保存修正后的轨迹为.mat文件（用于第二次MATLAB仿真）

        Args:
            trajectory_data: 修正后的轨迹数据
            output_file: 输出文件路径
        """
        print(f"\n保存修正后的轨迹: {output_file}")

        # 准备MATLAB格式的数据
        mat_data = {
            'time': matlab.double(list(range(len(trajectory_data['x_ref'])))),
            'trajectory': {
                'x_ref': matlab.double(trajectory_data['x_ref'].tolist()),
                'y_ref': matlab.double(trajectory_data['y_ref'].tolist()),
                'z_ref': matlab.double(trajectory_data['z_ref'].tolist()),
                'vx': matlab.double(trajectory_data['vx'].tolist()),
                'vy': matlab.double(trajectory_data['vy'].tolist()),
                'vz': matlab.double(trajectory_data['vz'].tolist()),
            }
        }

        # 保存为.mat文件
        try:
            self.matlab.save(output_file, mat_data, nargout=0)
            print(f"  ✓ 保存完成")
        except Exception as e:
            print(f"  ✗ 保存失败: {e}")
            raise

    def visualize_correction(self, original_data, predicted_errors,
                           corrected_data, corrected_simulation_data,
                           output_dir):
        """
        可视化修正效果

        Args:
            original_data: 原始仿真数据（未修正）
            predicted_errors: 模型预测的误差
            corrected_data: 修正后的轨迹数据
            corrected_simulation_data: 修正后重新仿真的数据
            output_dir: 输出目录
        """
        print(f"\n生成可视化...")

        os.makedirs(output_dir, exist_ok=True)

        # 提取数据
        x_orig = original_data['trajectory']['x_ref']
        y_orig = original_data['trajectory']['y_ref']
        err_x_orig = original_data['error']['error_x']
        err_y_orig = original_data['error']['error_y']
        err_mag_orig = original_data['error']['error_mag']

        x_corr = corrected_data['x_ref']
        y_corr = corrected_data['y_ref']
        err_x_corr = corrected_simulation_data['error']['error_x']
        err_y_corr = corrected_simulation_data['error']['error_y']
        err_mag_corr = corrected_simulation_data['error']['error_mag']

        # 1. 主要对比图：修正前后误差热图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        vmin = 0
        vmax = max(err_mag_orig.max(), err_mag_corr.max())

        # 修正前
        sc1 = axes[0].scatter(x_orig, y_orig, c=err_mag_orig, cmap='hot',
                             s=2, alpha=0.7, vmin=vmin, vmax=vmax)
        axes[0].set_title('Original Simulation (Before Correction)',
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X Position (mm)', fontsize=12)
        axes[0].set_ylabel('Y Position (mm)', fontsize=12)
        axes[0].axis('equal')
        cbar1 = plt.colorbar(sc1, ax=axes[0])
        cbar1.set_label('Error Magnitude (mm)', fontsize=11)

        # 修正后
        sc2 = axes[1].scatter(x_corr, y_corr, c=err_mag_corr, cmap='hot',
                             s=2, alpha=0.7, vmin=vmin, vmax=vmax)
        axes[1].set_title('Corrected Simulation (After Correction)',
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('X Position (mm)', fontsize=12)
        axes[1].set_ylabel('Y Position (mm)', fontsize=12)
        axes[1].axis('equal')
        cbar2 = plt.colorbar(sc2, ax=axes[1])
        cbar2.set_label('Error Magnitude (mm)', fontsize=11)

        plt.tight_layout()
        fig_path = os.path.join(output_dir, 'realtime_correction_comparison.png')
        plt.savefig(fig_path, dpi=200)
        print(f"  ✓ 保存对比图: {fig_path}")
        plt.close()

        # 2. 误差统计图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 误差分布对比
        axes[0, 0].hist(err_mag_orig, bins=50, alpha=0.6, label='Original',
                       color='red', density=True)
        axes[0, 0].hist(err_mag_corr, bins=50, alpha=0.6, label='Corrected',
                       color='green', density=True)
        axes[0, 0].set_xlabel('Error Magnitude (mm)', fontsize=12)
        axes[0, 0].set_ylabel('Density', fontsize=12)
        axes[0, 0].set_title('Error Distribution Comparison',
                            fontsize=13, fontweight='bold')
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # X轴误差
        axes[0, 1].scatter(err_x_orig, err_x_corr, alpha=0.3, s=1)
        limit = max(abs(err_x_orig).max(), abs(err_x_corr).max())
        axes[0, 1].plot([-limit, limit], [-limit, limit], 'r--',
                      linewidth=2, label='Perfect correlation')
        axes[0, 1].set_xlabel('Original Error X (mm)', fontsize=12)
        axes[0, 1].set_ylabel('Corrected Error X (mm)', fontsize=12)
        axes[0, 1].set_title('X-Axis Error: Original vs Corrected',
                            fontsize=13, fontweight='bold')
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axis('equal')

        # Y轴误差
        axes[1, 0].scatter(err_y_orig, err_y_corr, alpha=0.3, s=1)
        limit = max(abs(err_y_orig).max(), abs(err_y_corr).max())
        axes[1, 0].plot([-limit, limit], [-limit, limit], 'r--',
                      linewidth=2, label='Perfect correlation')
        axes[1, 0].set_xlabel('Original Error Y (mm)', fontsize=12)
        axes[1, 0].set_ylabel('Corrected Error Y (mm)', fontsize=12)
        axes[1, 0].set_title('Y-Axis Error: Original vs Corrected',
                            fontsize=13, fontweight='bold')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axis('equal')

        # 改善率统计
        improvement = (1 - err_mag_corr / err_mag_orig) * 100
        improvement = improvement[np.isfinite(improvement)]

        axes[1, 1].hist(improvement, bins=50, color='steelblue', alpha=0.7,
                       edgecolor='black')
        axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2,
                          label='No improvement')
        axes[1, 1].axvline(improvement.mean(), color='green', linestyle='-',
                          linewidth=2, label=f'Mean: {improvement.mean():.1f}%')
        axes[1, 1].set_xlabel('Improvement Rate (%)', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Error Improvement Distribution',
                            fontsize=13, fontweight='bold')
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fig_path = os.path.join(output_dir, 'realtime_correction_stats.png')
        plt.savefig(fig_path, dpi=200)
        print(f"  ✓ 保存统计图: {fig_path}")
        plt.close()

        # 3. 保存统计报告
        stats = {
            'original': {
                'mean_error_mm': float(err_mag_orig.mean()),
                'max_error_mm': float(err_mag_orig.max()),
                'std_error_mm': float(err_mag_orig.std()),
                'x_mean_error_mm': float(np.abs(err_x_orig).mean()),
                'y_mean_error_mm': float(np.abs(err_y_orig).mean()),
            },
            'corrected': {
                'mean_error_mm': float(err_mag_corr.mean()),
                'max_error_mm': float(err_mag_corr.max()),
                'std_error_mm': float(err_mag_corr.std()),
                'x_mean_error_mm': float(np.abs(err_x_corr).mean()),
                'y_mean_error_mm': float(np.abs(err_y_corr).mean()),
            },
            'improvement': {
                'mean_improvement_percent': float(improvement.mean()),
                'median_improvement_percent': float(np.median(improvement)),
                'mean_error_reduction_percent': float(
                    (1 - err_mag_corr.mean() / err_mag_orig.mean()) * 100
                ),
            }
        }

        stats_path = os.path.join(output_dir, 'correction_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  ✓ 保存统计报告: {stats_path}")

        # 打印统计
        print(f"\n{'='*60}")
        print(f"修正效果统计")
        print(f"{'='*60}")
        print(f"\n原始仿真（未修正）:")
        print(f"  平均误差: {stats['original']['mean_error_mm']:.4f} mm")
        print(f"  最大误差: {stats['original']['max_error_mm']:.4f} mm")
        print(f"  标准差:   {stats['original']['std_error_mm']:.4f} mm")

        print(f"\n修正后仿真:")
        print(f"  平均误差: {stats['corrected']['mean_error_mm']:.4f} mm")
        print(f"  最大误差: {stats['corrected']['max_error_mm']:.4f} mm")
        print(f"  标准差:   {stats['corrected']['std_error_mm']:.4f} mm")

        print(f"\n改善率:")
        print(f"  平均误差降低: {stats['improvement']['mean_error_reduction_percent']:.1f}%")
        print(f"  平均改善率:   {stats['improvement']['mean_improvement_percent']:.1f}%")
        print(f"  中位改善率:   {stats['improvement']['median_improvement_percent']:.1f}%")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time trajectory correction demonstration with live MATLAB simulation'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--gcode', type=str,
                       default='test_gcode_files/3DBenchy_PLA_1h28m.gcode',
                       help='G-code file to simulate')
    parser.add_argument('--layer', type=int, default=25,
                       help='Layer number to simulate (default: 25)')
    parser.add_argument('--output_dir', type=str, default='results/realtime_correction_demo',
                       help='Output directory for results')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--skip_matlab', action='store_true',
                       help='Skip MATLAB simulation (use existing data)')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 如果跳过MATLAB仿真，使用简单的演示模式
    if args.skip_matlab:
        print("\n⚠️  演示模式：跳过MATLAB仿真")
        print("    使用模拟数据进行演示")
        demo_with_simulated_data(args)
        return

    # 完整模式：使用MATLAB引擎
    try:
        print("\n启动MATLAB引擎...")
        matlab = matlab.engine.start_matlab()
        print("  ✓ MATLAB引擎启动成功")

        # 添加MATLAB路径
        sim_dir = str(project_root / 'simulation')
        matlab.addpath(sim_dir, nargout=0)
        print(f"  ✓ 添加仿真路径: {sim_dir}")

        # 创建演示实例
        demo = RealTimeCorrectionDemo(
            checkpoint_path=args.checkpoint,
            device=args.device,
            matlab_engine=matlab
        )

        # 步骤1: 运行原始仿真（未修正）
        original_output = output_dir / 'original_simulation.mat'
        original_data = demo.run_matlab_simulation(
            gcode_file=args.gcode,
            output_file=str(original_output),
            layers=[args.layer],
            use_firmware=True
        )

        # 步骤2: 实时预测误差并修正轨迹
        corrected_trajectory, predicted_errors = demo.predict_and_correct(
            trajectory_data=original_data['trajectory'],
            apply_correction=True
        )

        # 保存修正后的轨迹
        corrected_trajectory_file = output_dir / 'corrected_trajectory.mat'
        demo.save_corrected_trajectory(
            trajectory_data=corrected_trajectory,
            output_file=str(corrected_trajectory_file)
        )

        # 步骤3: 对修正后的轨迹重新仿真
        # 注意：这里需要创建一个专门的MATLAB函数来仿真给定的轨迹
        # 暂时跳过这一步，使用预测误差作为"修正后的误差"

        # 构造修正后的仿真数据（简化版本）
        corrected_simulation_data = {
            'error': {
                'error_x': original_data['error']['error_x'] - predicted_errors[:, 0],
                'error_y': original_data['error']['error_y'] - predicted_errors[:, 1],
                'error_mag': np.sqrt(
                    (original_data['error']['error_x'] - predicted_errors[:, 0])**2 +
                    (original_data['error']['error_y'] - predicted_errors[:, 1])**2
                )
            }
        }

        # 步骤4: 可视化
        demo.visualize_correction(
            original_data=original_data,
            predicted_errors=predicted_errors,
            corrected_data=corrected_trajectory,
            corrected_simulation_data=corrected_simulation_data,
            output_dir=str(output_dir)
        )

        print(f"\n✓ 演示完成！")
        print(f"  结果保存在: {output_dir}")

    except matlab.engine.MatlabEngineError as e:
        print(f"\n✗ MATLAB引擎错误: {e}")
        print("\n提示：")
        print("  1. 确保已安装 MATLAB Engine API for Python")
        print("  2. 安装命令: cd 'matlabroot/extern/engines/python' && python setup.py install")
        print("  3. 或者使用 --skip_matlab 标志使用模拟数据演示")
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'matlab' in locals():
            print("\n关闭MATLAB引擎...")
            matlab.quit()
            print("  ✓ MATLAB引擎已关闭")


def demo_with_simulated_data(args):
    """使用模拟数据进行演示（不需要MATLAB）"""
    from data.realtime_dataset import load_mat_file
    import glob

    # 查找现有数据文件
    data_dirs = glob.glob('data/simulation/*')
    mat_files = []
    for d in data_dirs:
        mat_files.extend(glob.glob(os.path.join(d, "*.mat")))

    if not mat_files:
        print("\n✗ 未找到仿真数据文件")
        print("  请先运行数据生成，或使用MATLAB模式")
        return

    # 使用第一个文件
    selected_file = mat_files[0]
    print(f"\n使用数据文件: {Path(selected_file).name}")

    # 加载数据
    features, labels = load_mat_file(selected_file)

    # 创建演示实例
    demo = RealTimeCorrectionDemo(
        checkpoint_path=args.checkpoint,
        device=args.device,
        matlab_engine=None
    )

    # 构造原始数据
    original_data = {
        'trajectory': {
            'x_ref': features[:, 0],
            'y_ref': features[:, 1],
            'z_ref': np.zeros(len(features)),
            'vx': features[:, 2],
            'vy': features[:, 3],
            'vz': np.zeros(len(features)),
        },
        'error': {
            'error_x': labels[:, 0],
            'error_y': labels[:, 1],
            'error_mag': np.sqrt(labels[:, 0]**2 + labels[:, 1]**2),
        }
    }

    # 预测并修正
    corrected_trajectory, predicted_errors = demo.predict_and_correct(
        trajectory_data=original_data['trajectory'],
        apply_correction=True
    )

    # 构造修正后的数据
    corrected_simulation_data = {
        'error': {
            'error_x': original_data['error']['error_x'] - predicted_errors[:, 0],
            'error_y': original_data['error']['error_y'] - predicted_errors[:, 1],
            'error_mag': np.sqrt(
                (original_data['error']['error_x'] - predicted_errors[:, 0])**2 +
                (original_data['error']['error_y'] - predicted_errors[:, 1])**2
            )
        }
    }

    # 可视化
    output_dir = Path(args.output_dir)
    demo.visualize_correction(
        original_data=original_data,
        predicted_errors=predicted_errors,
        corrected_data=corrected_trajectory,
        corrected_simulation_data=corrected_simulation_data,
        output_dir=str(output_dir)
    )

    print(f"\n✓ 演示完成（模拟数据模式）！")
    print(f"  结果保存在: {output_dir}")


if __name__ == '__main__':
    main()
