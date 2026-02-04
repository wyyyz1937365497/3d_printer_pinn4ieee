"""
修正版闭环G-code优化系统

**关键修正**：
1. 优化目标：让实际打印轨迹最接近原始理想轨迹
2. 使用增强版Python解析器（支持速度变化）
3. 调用改进的MATLAB仿真函数
4. 正确计算误差：error = x_act - original_trajectory

**流程**：
1. 解析原始G-code → ideal_traj（目标形状）
2. 应用LSTM修正 → corrected_traj（发送给打印机）
3. MATLAB仿真：corrected_traj → actual_traj（打印结果）
4. 误差：error = actual_traj - ideal_traj
5. 基于误差调整修正量
6. 迭代优化

作者：基于原始closedloop_gcode_optimizer.py修正
日期：2026-02-04
"""

import os
import sys
import numpy as np
import matlab.engine
import json
from datetime import datetime
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.realtime_corrector import RealTimeCorrector
from data.realtime_dataset import RealTimeTrajectoryDataset
from data.gcode_parser_enhanced import EnhancedGCodeParser
from sklearn.preprocessing import StandardScaler
import torch
import glob
import random


class FixedClosedLoopOptimizer:
    """修正的闭环G-code优化器"""

    def __init__(self, checkpoint_path, device='cuda'):
        print("="*70)
        print("初始化闭环优化系统")
        print("="*70)

        # 加载物理参数
        print("\n加载物理参数...")
        self.params = self._load_matlab_params()

        # 启动MATLAB引擎
        print("\n启动MATLAB引擎...")
        self.matlab = matlab.engine.start_matlab()
        self.matlab.cd(str(project_root / 'simulation'))

        # 添加物理参数到MATLAB工作空间
        self.matlab.workspace['params'] = self._params_to_matlab_struct(self.params)
        print("  [OK] MATLAB引擎启动完成")

        # 加载LSTM模型
        print("\n加载LSTM模型...")
        self._load_model(checkpoint_path, device)
        self.device = device
        self.seq_len = 50

        print("\n[OK] 闭环优化系统初始化完成\n")

    def _load_matlab_params(self):
        """加载MATLAB物理参数"""
        # 简化版本：直接定义参数
        # 实际可以从physics_parameters.m读取
        params = {
            'dynamics': {
                'x': {
                    'mass': 0.35,      # kg
                    'stiffness': 8000,  # N/m
                    'damping': 15.0     # N·s/m
                },
                'y': {
                    'mass': 0.45,      # kg
                    'stiffness': 8000,  # N/m
                    'damping': 15.0     # N·s/m
                }
            },
            'printing': {
                'nozzle_temp': 220,
                'bed_temp': 60
            }
        }

        return params

    def _params_to_matlab_struct(self, params):
        """将Python参数转换为MATLAB结构体"""
        struct = self.matlab.struct()

        # Dynamics
        dynamics = self.matlab.struct()

        x_dyn = self.matlab.struct()
        x_dyn['mass'] = params['dynamics']['x']['mass']
        x_dyn['stiffness'] = params['dynamics']['x']['stiffness']
        x_dyn['damping'] = params['dynamics']['x']['damping']
        dynamics['x'] = x_dyn

        y_dyn = self.matlab.struct()
        y_dyn['mass'] = params['dynamics']['y']['mass']
        y_dyn['stiffness'] = params['dynamics']['y']['stiffness']
        y_dyn['damping'] = params['dynamics']['y']['damping']
        dynamics['y'] = y_dyn

        struct['dynamics'] = dynamics

        # Printing
        printing = self.matlab.struct()
        printing['nozzle_temp'] = params['printing']['nozzle_temp']
        printing['bed_temp'] = params['printing']['bed_temp']
        struct['printing'] = printing

        return struct

    def _load_model(self, checkpoint_path, device):
        """加载LSTM模型和Scaler"""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 模型配置
        if 'model_info' in checkpoint:
            model_info = checkpoint['model_info']
            hidden_size = model_info.get('hidden_size', 56)
            num_layers = model_info.get('num_layers', 2)
            dropout = model_info.get('dropout', 0.1)
        else:
            hidden_size = 56
            num_layers = 2
            dropout = 0.1

        self.model = RealTimeCorrector(
            input_size=4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 加载Scaler
        print("加载数据标准化器...")
        all_files = glob.glob("data_simulation_*/*.mat")
        random.seed(42)
        random.shuffle(all_files)
        train_files = all_files[:50]

        temp_dataset = RealTimeTrajectoryDataset(
            train_files, seq_len=self.seq_len, pred_len=1,
            scaler=None, mode='train'
        )
        self.scaler = temp_dataset.scaler

        print("  [OK] 模型加载完成")

    def parse_gcode(self, gcode_file):
        """使用增强解析器解析G-code"""
        parser = EnhancedGCodeParser(gcode_file)
        trajectory = parser.parse(keep_type_annotations=True)

        # 分离轮廓和其他移动
        outline_indices = [i for i, t in enumerate(trajectory['move_type'])
                          if t in ['Outer wall', 'Inner wall']]

        print(f"\n轮廓移动: {len(outline_indices)}/{len(trajectory['move_type'])}")

        return trajectory, outline_indices

    def predict_initial_correction(self, trajectory):
        """预测初始修正量"""
        print("\n应用LSTM预测初始修正...")

        n = len(trajectory['x'])
        corrections = np.zeros((n, 2))

        # 准备特征
        features = np.stack([
            trajectory['x'],
            trajectory['y'],
            trajectory['vx'],
            trajectory['vy']
        ], axis=1)

        features_norm = self.scaler.transform(features)

        # 应用LSTM预测
        with torch.no_grad():
            for i in range(self.seq_len, n):
                history = features_norm[i-self.seq_len:i]
                input_tensor = torch.FloatTensor(history).unsqueeze(0).to(self.device)
                pred_error = self.model(input_tensor).cpu().numpy()[0]

                # 反向修正
                corrections[i] = -pred_error

        # 统计
        correction_mag = np.linalg.norm(corrections, axis=1)
        print(f"  平均修正量: {np.mean(correction_mag[self.seq_len:])*1000:.2f} um")
        print(f"  最大修正量: {np.max(correction_mag)*1000:.2f} um")

        return corrections

    def apply_correction(self, trajectory, corrections, outline_indices):
        """应用修正（只修正轮廓）"""
        corrected = trajectory.copy()

        for key in ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']:
            corrected[key] = trajectory[key].copy()

        # 只对轮廓应用修正
        for i in outline_indices:
            corrected['x'][i] += corrections[i, 0]
            corrected['y'][i] += corrections[i, 1]

        # 重新计算速度和加速度
        n = len(outline_indices)
        if n > 1:
            sorted_indices = sorted(outline_indices)

            for idx in range(1, len(sorted_indices)):
                i = sorted_indices[idx]
                i_prev = sorted_indices[idx-1]

                dt = corrected['time'][i] - corrected['time'][i_prev]
                if dt > 0:
                    corrected['vx'][i] = (corrected['x'][i] - corrected['x'][i_prev]) / dt
                    corrected['vy'][i] = (corrected['y'][i] - corrected['y'][i_prev]) / dt

            for idx in range(2, len(sorted_indices)):
                i = sorted_indices[idx]
                i_prev = sorted_indices[idx-1]

                dt = corrected['time'][i] - corrected['time'][i_prev]
                if dt > 0:
                    corrected['ax'][i] = (corrected['vx'][i] - corrected['vx'][i_prev]) / dt
                    corrected['ay'][i] = (corrected['vy'][i] - corrected['vy'][i_prev]) / dt

        return corrected

    def matlab_simulate(self, input_traj, ideal_traj):
        """
        调用改进的MATLAB仿真

        关键：传入两个轨迹
        - input_traj: 输入轨迹（将发送给打印机）
        - ideal_traj: 理想轨迹（目标形状）

        返回相对于理想轨迹的误差
        """
        # 创建MATLAB结构体
        input_struct = self.matlab.struct()
        input_struct['time'] = matlab.double(input_traj['time'].tolist())
        input_struct['x'] = matlab.double(input_traj['x'].tolist())
        input_struct['y'] = matlab.double(input_traj['y'].tolist())
        input_struct['z'] = matlab.double(input_traj['z'].tolist())
        input_struct['vx'] = matlab.double(input_traj['vx'].tolist())
        input_struct['vy'] = matlab.double(input_traj['vy'].tolist())
        input_struct['ax'] = matlab.double(input_traj['ax'].tolist())
        input_struct['ay'] = matlab.double(input_traj['ay'].tolist())

        ideal_struct = self.matlab.struct()
        ideal_struct['time'] = matlab.double(ideal_traj['time'].tolist())
        ideal_struct['x'] = matlab.double(ideal_traj['x'].tolist())
        ideal_struct['y'] = matlab.double(ideal_traj['y'].tolist())
        ideal_struct['z'] = matlab.double(ideal_traj['z'].tolist())

        # 调用改进的MATLAB函数
        self.matlab.workspace['input_traj'] = input_struct
        self.matlab.workspace['ideal_traj'] = ideal_struct

        results = self.matlab.eval(
            'simulate_trajectory_error_from_python(input_traj, ideal_traj, params)',
            nargout=1
        )

        # 提取结果
        simulated_data = {
            'x_ideal': np.array(results['x_ideal']).flatten(),
            'y_ideal': np.array(results['y_ideal']).flatten(),
            'x_input': np.array(results['x_input']).flatten(),
            'y_input': np.array(results['y_input']).flatten(),
            'x_act': np.array(results['x_act']).flatten(),
            'y_act': np.array(results['y_act']).flatten(),
            'error_x': np.array(results['error_x']).flatten(),  # 相对于理想轨迹的误差
            'error_y': np.array(results['error_y']).flatten(),
            'error_magnitude': np.array(results['error_magnitude']).flatten()
        }

        return simulated_data

    def optimize(self, gcode_file, max_iterations=5, tolerance=20e-3):
        """执行闭环优化"""

        print("="*70)
        print("闭环G-code优化（修正版）")
        print("="*70)
        print(f"\n关键修正：")
        print(f"  - 优化目标：实际轨迹 ≈ 理想轨迹")
        print(f"  - 误差计算：error = x_act - x_ideal")
        print(f"  - 使用增强解析器：支持速度变化\n")

        # 解析G-code
        print("步骤1: 解析原始G-code")
        ideal_traj, outline_indices = self.parse_gcode(gcode_file)
        print(f"  [OK] 理想轨迹已提取\n")

        # 初始轨迹
        current_traj = ideal_traj.copy()

        # 优化历史
        optimization_history = []
        learning_rate = 0.3
        max_correction = 0.5  # mm

        # 追踪最佳
        best_traj = None
        best_rms = float('inf')
        best_iteration = 0

        # === 优化循环 ===
        for iteration in range(max_iterations):
            print(f"\n{'='*70}")
            print(f"迭代 {iteration + 1}/{max_iterations}")
            print(f"{'='*70}")

            # 步骤1：预测/应用修正
            if iteration == 0:
                print("\n步骤1: LSTM预测初始修正")
                correction = self.predict_initial_correction(current_traj)
                print(f"  [测试] 翻转LSTM符号")
                correction_xy = -correction
            else:
                print("\n步骤1: 基于反馈调整修正")
                # 使用相对于理想轨迹的误差
                correction_xy = np.zeros((len(current_traj['x']), 2))
                correction_xy[outline_indices, 0] = last_sim_data['error_x'][outline_indices] * learning_rate
                correction_xy[outline_indices, 1] = last_sim_data['error_y'][outline_indices] * learning_rate

            # 限制修正幅度
            correction_xy = np.clip(correction_xy, -max_correction, max_correction)
            print(f"  [限制后] 最大修正: {np.max(np.linalg.norm(correction_xy, axis=1))*1000:.2f} um")

            # 应用修正
            corrected_traj = self.apply_correction(current_traj, correction_xy, outline_indices)

            # 调试信息
            if iteration == 0:
                print(f"\n[调试] 修正应用检查:")
                print(f"  理想轨迹X范围: [{np.min(ideal_traj['x']):.2f}, {np.max(ideal_traj['x']):.2f}] mm")
                print(f"  修正后X范围: [{np.min(corrected_traj['x']):.2f}, {np.max(corrected_traj['x']):.2f}] mm")

            # 步骤2：MATLAB仿真
            print("\n步骤2: MATLAB仿真修正后轨迹...")
            sim_data = self.matlab_simulate(corrected_traj, ideal_traj)

            # 步骤3：计算相对于理想轨迹的误差
            error_mag = sim_data['error_magnitude']
            rms_error = np.sqrt(np.mean(error_mag**2))
            max_error = np.max(error_mag)
            mean_error = np.mean(error_mag)

            print(f"\n误差统计（相对于理想轨迹）:")
            print(f"  RMS误差: {rms_error*1000:.2f} um")
            print(f"  平均误差: {mean_error*1000:.2f} um")
            print(f"  最大误差: {max_error*1000:.2f} um")

            # 记录历史
            iteration_record = {
                'iteration': iteration + 1,
                'rms_error_um': rms_error * 1000,
                'mean_error_um': mean_error * 1000,
                'max_error_um': max_error * 1000
            }
            optimization_history.append(iteration_record)

            # 更新最佳
            if rms_error < best_rms:
                best_rms = rms_error
                best_traj = corrected_traj.copy()
                best_iteration = iteration + 1
                print(f"\n[更新] 找到更好的轨迹 (迭代 {best_iteration}, RMS: {best_rms*1000:.2f} um)")

            # 检查收敛
            if rms_error < tolerance:
                print(f"\n[OK] 收敛！RMS误差 < {tolerance*1000:.0f} um")
                break

            # 调整学习率
            if iteration > 0:
                improvement = optimization_history[-2]['rms_error_um'] - optimization_history[-1]['rms_error_um']

                if improvement > 5:
                    learning_rate = min(0.3, learning_rate * 1.1)
                    print(f"\n改进显著 ({improvement:.2f} um)，增大学习率: {learning_rate:.3f}")
                elif improvement < -5:
                    learning_rate *= 0.5
                    print(f"\n误差增加 ({improvement:.2f} um)，减小学习率: {learning_rate:.3f}")
                else:
                    print(f"\n改进: {improvement:.2f} um，保持学习率: {learning_rate:.3f}")
            else:
                print(f"\n初始学习率: {learning_rate}")

            # 更新当前轨迹
            current_traj = corrected_traj
            last_sim_data = sim_data

        # 返回最佳轨迹
        if best_traj is not None:
            print(f"\n[总结] 使用迭代 {best_iteration} 的轨迹作为最终结果")
            print(f"  最佳RMS误差: {best_rms*1000:.2f} um")
            return best_traj, ideal_traj, optimization_history
        else:
            return corrected_traj, ideal_traj, optimization_history

    def __del__(self):
        if hasattr(self, 'matlab'):
            try:
                self.matlab.quit()
                print("\n[MATLAB] 已关闭")
            except:
                pass


def generate_optimization_report(history, sim_data, output_dir):
    """生成优化报告"""
    report_path = os.path.join(output_dir, 'optimization_report_fixed.json')

    report = {
        'timestamp': datetime.now().isoformat(),
        'version': 'fixed - optimizes for ideal trajectory',
        'total_iterations': len(history),
        'final_rms_error_um': history[-1]['rms_error_um'],
        'final_mean_error_um': history[-1]['mean_error_um'],
        'final_max_error_um': history[-1]['max_error_um'],
        'iterations': history
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] 报告已保存: {report_path}")


def generate_corrected_gcode(original_gcode, ideal_traj, optimized_traj, output_file):
    """生成修正后的G-code"""
    print(f"\n生成修正后的G-code...")

    with open(original_gcode, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    corrected_lines = []
    traj_idx = 0

    for line in lines:
        line_stripped = line.strip()

        # 保留注释和非移动指令
        if line_stripped.startswith(';') or not (line_stripped.startswith('G1') or line_stripped.startswith('G0')):
            corrected_lines.append(line)
            continue

        # 修改移动指令
        if traj_idx < len(optimized_traj['x']):
            import re
            new_line = line_stripped

            # 替换X和Y
            if 'X' in new_line:
                new_line = re.sub(r'X-?\d+\.?\d*',
                                f'X{optimized_traj["x"][traj_idx]:.4f}',
                                new_line)
            if 'Y' in new_line:
                new_line = re.sub(r'Y-?\d+\.?\d*',
                                f'Y{optimized_traj["y"][traj_idx]:.4f}',
                                new_line)

            corrected_lines.append(new_line + '\n')
            traj_idx += 1
        else:
            corrected_lines.append(line)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(corrected_lines)

    print(f"  [OK] 保存到: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='修正版闭环G-code优化')
    parser.add_argument('--gcode', type=str, required=True,
                       help='输入G-code文件')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='LSTM模型checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/closedloop_fixed',
                       help='输出目录')
    parser.add_argument('--max_iterations', type=int, default=5,
                       help='最大迭代次数')
    parser.add_argument('--tolerance', type=float, default=20.0,
                       help='收敛阈值(um)')

    args = parser.parse_args()

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("修正版闭环G-code优化系统")
    print("="*70)
    print(f"\n配置:")
    print(f"  输入G-code: {args.gcode}")
    print(f"  LSTM模型: {args.checkpoint}")
    print(f"  最大迭代: {args.max_iterations}")
    print(f"  收敛阈值: {args.tolerance} um")
    print(f"  输出目录: {args.output_dir}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}\n")

    # 创建优化器
    optimizer = FixedClosedLoopOptimizer(args.checkpoint, device)

    # 执行优化
    optimized_traj, ideal_traj, history = optimizer.optimize(
        args.gcode,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance * 1e-3
    )

    # 生成报告
    generate_optimization_report(history, None, args.output_dir)

    # 生成修正后的G-code
    input_path = Path(args.gcode)
    output_file = Path(args.output_dir) / f"{input_path.stem}_corrected.gcode"
    generate_corrected_gcode(args.gcode, ideal_traj, optimized_traj, str(output_file))

    print("\n" + "="*70)
    print("优化完成！")
    print("="*70)
    print(f"\n生成的文件:")
    print(f"  - {output_file}")
    print(f"  - {args.output_dir}/optimization_report_fixed.json")


if __name__ == '__main__':
    main()
