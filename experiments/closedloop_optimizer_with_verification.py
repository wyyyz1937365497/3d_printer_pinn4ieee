"""
闭环G-code优化器 - 带仿真验证

核心逻辑：
1. 使用LSTM预测误差
2. 修正G-code：corrected = ideal - predicted_error (反向补偿)
3. 仿真验证：simulated_error_after_correction
4. 迭代优化直到收敛

使用方法：
    python closedloop_optimizer_with_verification.py \
        --gcode test.gcode \
        --checkpoint checkpoints/realtime_corrector/best_model.pth \
        --max_iterations 10
"""

import argparse
import re
import numpy as np
import torch
from pathlib import Path
from collections import deque
import json
import time

project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

from models.realtime_corrector import RealTimeCorrector
from data.realtime_dataset import RealTimeTrajectoryDataset
from data.gcode_physics_simulator_enhanced import (
    PrinterPhysicsSimulator,
    EnhancedGCodeParser
)


class ClosedLoopOptimizer:
    """
    闭环G-code优化器

    流程：
    1. 解析原始G-code
    2. LSTM预测误差
    3. 修正：corrected = ideal - predicted (关键：反向补偿！)
    4. 仿真验证
    5. 判断是否收敛
    6. 如未收敛，迭代优化
    """

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        # 加载模型
        print("加载LSTM模型...")
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

        self.model = RealTimeCorrector(
            input_size=4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.device = device
        self.seq_len = 50

        # 加载scaler
        print("加载数据标准化器...")
        import glob
        import random
        all_files = glob.glob("data_simulation_*/*.mat")
        if len(all_files) > 0:
            random.seed(42)
            random.shuffle(all_files)
            train_files = all_files[:min(50, len(all_files))]

            temp_dataset = RealTimeTrajectoryDataset(
                train_files, seq_len=self.seq_len, pred_len=1,
                scaler=None, mode='train'
            )
            self.scaler = temp_dataset.scaler
        else:
            print("  [WARNING] 未找到训练数据，使用默认scaler")
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            # 拟合一些默认数据
            dummy_data = np.random.randn(100, 4)
            self.scaler.fit(dummy_data)

        # 创建物理仿真器
        print("创建物理仿真器...")
        self.simulator = PrinterPhysicsSimulator()

        print("  [OK] 初始化完成")

    def parse_gcode(self, gcode_file: str) -> list:
        """解析G-code文件"""
        print(f"\n解析G-code: {gcode_file}")
        parser = EnhancedGCodeParser(gcode_file)
        moves = parser.parse()

        # 只保留打印移动
        printing_moves = [m for m in moves if m['type'] == 'printing']
        print(f"  保留 {len(printing_moves)} 个打印移动")

        return printing_moves

    def predict_corrections(self, moves: list) -> dict:
        """使用LSTM预测修正量"""
        print(f"\n应用LSTM模型预测修正量...")

        n_points = len(moves)

        # 提取特征
        x_ref = np.array([m['x'] for m in moves])
        y_ref = np.array([m['y'] for m in moves])
        vx_ref = np.array([m['vx'] for m in moves])
        vy_ref = np.array([m['vy'] for m in moves])

        features = np.stack([x_ref, y_ref, vx_ref, vy_ref], axis=1)
        features_norm = self.scaler.transform(features)

        # 预测修正量
        corrections = {}
        correction_list = []

        with torch.no_grad():
            for i in range(self.seq_len, n_points):
                history = features_norm[i-self.seq_len:i]
                input_tensor = torch.FloatTensor(history).unsqueeze(0).to(self.device)
                pred_error = self.model(input_tensor).cpu().numpy()[0]

                # 记录修正量（这是预测的误差）
                corrections[i] = {
                    'x': pred_error[0],
                    'y': pred_error[1]
                }
                correction_list.append([pred_error[0], pred_error[1]])

        correction_array = np.array(correction_list)

        # 统计
        print(f"  预测了 {len(corrections)} 个点的修正量")
        print(f"  修正量统计:")
        print(f"    X轴: Mean={np.mean(correction_array[:,0])*1000:.2f} um, "
              f"RMS={np.sqrt(np.mean(correction_array[:,0]**2))*1000:.2f} um")
        print(f"    Y轴: Mean={np.mean(correction_array[:,1])*1000:.2f} um, "
              f"RMS={np.sqrt(np.mean(correction_array[:,1]**2))*1000:.2f} um")

        return corrections

    def apply_corrections(self, gcode_file: str, moves: list, corrections: dict,
                         output_file: str) -> str:
        """
        应用修正到G-code

        关键：反向补偿
        corrected_position = ideal_position - predicted_error

        这样：
        actual = corrected + actual_error
              = (ideal - predicted) + actual
              ≈ ideal (如果predicted ≈ actual)
        """
        print(f"\n应用修正到G-code...")
        print(f"  策略：反向补偿 (corrected = ideal - predicted_error)")

        with open(gcode_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        corrected_lines = []
        corrections_applied = 0

        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()

            # 保留注释和非移动指令
            if not line_stripped.startswith('G1') and not line_stripped.startswith('G0'):
                corrected_lines.append(line)
                continue

            # 检查是否需要修正
            # 找到对应的move索引
            move_idx = None
            if len([m for m in moves if m['line_index'] == i]) > 0:
                # 找到这个line对应的move索引
                for idx, m in enumerate(moves):
                    if m['line_index'] == i and idx in corrections:
                        move_idx = idx
                        break

            if move_idx is not None:
                # 提取原始坐标
                x_match = re.search(r'X(-?\d+\.?\d*)', line_stripped)
                y_match = re.search(r'Y(-?\d+\.?\d*)', line_stripped)

                orig_x = float(x_match.group(1)) if x_match else None
                orig_y = float(y_match.group(1)) if y_match else None

                # 获取预测误差
                correction_x = corrections[move_idx]['x']
                correction_y = corrections[move_idx]['y']

                # 应用修正：原始坐标 - 预测误差（反向补偿！）
                new_x = orig_x - correction_x if orig_x is not None else None
                new_y = orig_y - correction_y if orig_y is not None else None

                # 生成新行
                new_line = line_stripped

                if new_x is not None and 'X' in new_line:
                    new_line = re.sub(r'X-?\d+\.?\d*', f'X{new_x:.4f}', new_line)
                if new_y is not None and 'Y' in new_line:
                    new_line = re.sub(r'Y-?\d+\.?\d*', f'Y{new_y:.4f}', new_line)

                corrected_lines.append(new_line + '\n')
                corrections_applied += 1

                # 打印前几个修正
                if corrections_applied <= 3:
                    print(f"  修正 #{corrections_applied}:")
                    if orig_x is not None:
                        print(f"    X: {orig_x:.4f} - ({correction_x*1000:.2f} um) = {new_x:.4f}")
                    if orig_y is not None:
                        print(f"    Y: {orig_y:.4f} - ({correction_y*1000:.2f} um) = {new_y:.4f}")
            else:
                # 非打印移动，保持不变
                corrected_lines.append(line)

        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(corrected_lines)

        print(f"  [OK] 应用了 {corrections_applied} 处修正")
        print(f"  [OK] 保存到: {output_file}")

        return output_file

    def verify_with_simulation(self, gcode_file: str) -> dict:
        """使用物理仿真验证修正效果"""
        print(f"\n仿真验证: {gcode_file}")

        try:
            result = self.simulator.simulate_gcode(gcode_file, filter_printing=True)

            error_x = result['x']['error']
            error_y = result['y']['error']
            error_mag = np.sqrt(error_x**2 + error_y**2)

            metrics = {
                'rms_error_um': np.sqrt(np.mean(error_mag**2)) * 1000,
                'mean_error_um': np.mean(error_mag) * 1000,
                'max_error_um': np.max(error_mag) * 1000,
                'rms_x_um': np.sqrt(np.mean(error_x**2)) * 1000,
                'rms_y_um': np.sqrt(np.mean(error_y**2)) * 1000,
                'n_points': len(error_mag)
            }

            print(f"  仿真结果:")
            print(f"    RMS误差: {metrics['rms_error_um']:.2f} um")
            print(f"    平均误差: {metrics['mean_error_um']:.2f} um")
            print(f"    最大误差: {metrics['max_error_um']:.2f} um")

            return metrics

        except Exception as e:
            print(f"  [ERROR] 仿真失败: {e}")
            return None

    def optimize(self, gcode_file: str, output_dir: str,
                 max_iterations: int = 10, tolerance: float = 5.0) -> dict:
        """
        执行闭环优化

        Args:
            gcode_file: 原始G-code文件
            output_dir: 输出目录
            max_iterations: 最大迭代次数
            tolerance: 收敛容差（um），如果误差改进小于此值则停止

        Returns:
            优化历史
        """
        print("\n" + "="*80)
        print("闭环G-code优化")
        print("="*80)
        print(f"\n配置:")
        print(f"  输入: {gcode_file}")
        print(f"  输出: {output_dir}")
        print(f"  最大迭代: {max_iterations}")
        print(f"  收敛容差: {tolerance} um")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        history = []
        current_gcode = gcode_file

        # 迭代0：仿真原始G-code
        print("\n" + "-"*80)
        print(f"Iteration 0: 仿真原始G-code")
        print("-"*80)

        metrics_orig = self.verify_with_simulation(current_gcode)

        if metrics_orig is None:
            print("[ERROR] 无法仿真原始G-code")
            return None

        history.append({
            'iteration': 0,
            'gcode_file': current_gcode,
            'metrics': metrics_orig
        })

        best_rms = metrics_orig['rms_error_um']
        best_iteration = 0

        # 迭代优化
        for iteration in range(1, max_iterations + 1):
            print("\n" + "-"*80)
            print(f"Iteration {iteration}: 修正+验证")
            print("-"*80)

            # 1. 解析当前G-code
            moves = self.parse_gcode(current_gcode)

            # 2. LSTM预测修正量
            corrections = self.predict_corrections(moves)

            # 3. 应用修正
            input_path = Path(current_gcode)
            output_file = Path(output_dir) / f"{input_path.stem}_iter{iteration}.gcode"

            current_gcode = str(self.apply_corrections(
                current_gcode, moves, corrections, str(output_file)
            ))

            # 4. 仿真验证
            metrics_new = self.verify_with_simulation(current_gcode)

            if metrics_new is None:
                print(f"  [WARNING] 仿真失败，停止迭代")
                break

            history.append({
                'iteration': iteration,
                'gcode_file': current_gcode,
                'metrics': metrics_new
            })

            # 5. 判断是否收敛
            improvement = best_rms - metrics_new['rms_error_um']

            print(f"\n  进度:")
            print(f"    当前RMS: {metrics_new['rms_error_um']:.2f} um")
            print(f"    最佳RMS: {best_rms:.2f} um (iter {best_iteration})")
            print(f"    改进: {improvement:.2f} um ({improvement/best_rms*100:.1f}%)")

            if metrics_new['rms_error_um'] < best_rms:
                best_rms = metrics_new['rms_error_um']
                best_iteration = iteration
                print(f"    ✓ 新的最佳结果！")

                # 检查是否收敛
                if improvement < tolerance:
                    print(f"\n  收敛！改进小于容差 ({tolerance} um)")
                    break
            else:
                print(f"    ✗ 误差增加，停止迭代")
                break

        # 总结
        print("\n" + "="*80)
        print("优化完成")
        print("="*80)

        print(f"\n迭代历史:")
        for h in history:
            iter_num = h['iteration']
            rms = h['metrics']['rms_error_um']
            marker = " ★" if iter_num == best_iteration else ""
            print(f"  Iter {iter_num}: RMS={rms:.2f} um{marker}")

        total_improvement = history[0]['metrics']['rms_error_um'] - best_rms
        improvement_pct = total_improvement / history[0]['metrics']['rms_error_um'] * 100

        print(f"\n总改进: {total_improvement:.2f} um ({improvement_pct:.1f}%)")
        print(f"最佳文件: {history[best_iteration]['gcode_file']}")

        # 保存历史
        history_file = Path(output_dir) / 'optimization_history.json'
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\n优化历史已保存: {history_file}")

        return {
            'history': history,
            'best_iteration': best_iteration,
            'best_gcode': history[best_iteration]['gcode_file'],
            'total_improvement_um': total_improvement,
            'improvement_pct': improvement_pct
        }


def main():
    parser = argparse.ArgumentParser(description='闭环G-code优化器')
    parser.add_argument('--gcode', type=str, required=True,
                       help='原始G-code文件')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='LSTM模型checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/closedloop_optimization',
                       help='输出目录')
    parser.add_argument('--max_iterations', type=int, default=10,
                       help='最大迭代次数')
    parser.add_argument('--tolerance', type=float, default=5.0,
                       help='收敛容差（um）')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（cuda或cpu）')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    # 创建优化器
    optimizer = ClosedLoopOptimizer(args.checkpoint, device)

    # 执行优化
    result = optimizer.optimize(
        args.gcode,
        args.output_dir,
        args.max_iterations,
        args.tolerance
    )

    if result:
        print(f"\n✅ 优化成功！")
        print(f"最佳G-code: {result['best_gcode']}")


if __name__ == '__main__':
    main()
