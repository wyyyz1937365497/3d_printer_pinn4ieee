"""
G-code轮廓修正脚本 - 只修正外壁和内壁

问题：
    原始脚本对所有G1/G0移动都应用修正，导致内部填充也被修改，
    破坏了打印结构。

解决方案：
    识别G-code中的TYPE注释，只对轮廓（Outer wall, Inner wall）应用修正，
    保持填充（infill）、支撑（support）等不变。

G-code类型说明：
    ;TYPE:Outer wall - 外壁（需要修正）
    ;TYPE:Inner wall - 内壁（需要修正）
    ;TYPE:Internal solid infill - 内部实心填充（不修正）
    ;TYPE:Gap infill - 间隙填充（不修正）
    ;TYPE:Skirt - 裙边（不修正）
    ;TYPE:Support - 支撑（不修正）

使用方法：
    python experiments/gcode_outline_correction.py \
        --gcode test_gcode_files/3DBenchy_PLA_1h28m.gcode \
        --checkpoint checkpoints/realtime_corrector/best_model.pth \
        --output results/outline_correction
"""

import argparse
import re
import numpy as np
import torch
from pathlib import Path
from collections import deque

project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

from models.realtime_corrector import RealTimeCorrector
from data.realtime_dataset import RealTimeTrajectoryDataset


class OutlineAwareGCodeParser:
    """G-code解析器 - 区分轮廓和填充"""

    def __init__(self, gcode_file):
        self.gcode_file = gcode_file
        self.moves = []  # 所有移动指令
        self.outline_only_moves = []  # 只包含轮廓的移动

    def parse(self):
        """解析G-code，区分轮廓和填充"""
        print(f"解析G-code文件: {self.gcode_file}")

        with open(self.gcode_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        current_type = None
        line_number = 0

        for line in lines:
            line_number += 1
            line_stripped = line.strip()

            # 检测TYPE注释
            if line_stripped.startswith(';TYPE:'):
                current_type = line_stripped[6:].strip()
                continue

            # 解析G1/G0移动指令
            if line_stripped.startswith('G1') or line_stripped.startswith('G0'):
                move = self._parse_move(line_stripped, line_number, current_type)
                if move and ('x' in move or 'y' in move):
                    self.moves.append(move)

                    # 只记录轮廓类型的移动
                    if current_type in ['Outer wall', 'Inner wall']:
                        self.outline_only_moves.append(move)

        print(f"  总移动指令数: {len(self.moves)}")
        print(f"  轮廓移动数: {len(self.outline_only_moves)}")
        print(f"  填充等其他移动数: {len(self.moves) - len(self.outline_only_moves)}")

        return self.moves

    def _parse_move(self, line, line_number, move_type):
        """解析单行移动指令"""
        params = {
            'raw': line,
            'line_number': line_number,
            'type': move_type  # 移动类型
        }

        # 提取X, Y, Z, E, F参数
        for param in ['X', 'Y', 'Z', 'E', 'F']:
            match = re.search(f'{param}(-?\\d+\\.?\\d*)', line)
            if match:
                params[param.lower()] = float(match.group(1))

        return params


class OutlineOnlyCorrector:
    """只修正轮廓的修正器"""

    def __init__(self, checkpoint_path, device='cuda'):
        # 加载模型
        print("加载模型...")
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
        random.seed(42)
        random.shuffle(all_files)
        train_files = all_files[:50]

        temp_dataset = RealTimeTrajectoryDataset(
            train_files, seq_len=self.seq_len, pred_len=1,
            scaler=None, mode='train'
        )
        self.scaler = temp_dataset.scaler

        print("  [OK] 模型和Scaler加载完成")

    def correct_outline_only(self, moves):
        """
        只对轮廓移动应用修正

        Args:
            moves: 所有移动指令列表

        Returns:
            corrected_positions: {line_number: (x_corrected, y_corrected)}
        """
        print(f"\n应用模型修正（仅轮廓）...")

        # 提取轮廓移动
        outline_moves = [m for m in moves if m['type'] in ['Outer wall', 'Inner wall']]

        if len(outline_moves) == 0:
            print("  [WARNING] 未找到轮廓移动")
            return {}

        print(f"  轮廓移动数: {len(outline_moves)}")

        # 准备特征
        x_ref = np.array([m.get('x', 0) for m in outline_moves])
        y_ref = np.array([m.get('y', 0) for m in outline_moves])

        # 估算速度（简化）
        vx_ref = np.zeros(len(outline_moves))
        vy_ref = np.zeros(len(outline_moves))

        for i in range(1, len(outline_moves)):
            dx = x_ref[i] - x_ref[i-1]
            dy = y_ref[i] - y_ref[i-1]
            dist = np.sqrt(dx**2 + dy**2)

            if dist > 0:
                speed = 30.0  # mm/s (默认)
                if 'f' in outline_moves[i]:
                    speed = outline_moves[i]['f'] / 60.0

                vx_ref[i] = (dx / dist) * speed
                vy_ref[i] = (dy / dist) * speed

        features = np.stack([x_ref, y_ref, vx_ref, vy_ref], axis=1)
        features_norm = self.scaler.transform(features)

        # 应用修正
        corrected_positions = {}

        with torch.no_grad():
            for i in range(self.seq_len, len(outline_moves)):
                history = features_norm[i-self.seq_len:i]
                input_tensor = torch.FloatTensor(history).unsqueeze(0).to(self.device)
                pred_error = self.model(input_tensor).cpu().numpy()[0]

                # 修正公式
                x_corrected = x_ref[i] + pred_error[0]
                y_corrected = y_ref[i] + pred_error[1]

                # 记录修正后的位置（使用原始行号）
                line_number = outline_moves[i]['line_number']
                corrected_positions[line_number] = (x_corrected, y_corrected)

        print(f"  [OK] 修正了 {len(corrected_positions)} 个轮廓点")

        # 统计修正量
        corrections = []
        for i, line_num in enumerate(corrected_positions.keys()):
            if i < len(outline_moves) - self.seq_len:
                orig_x = outline_moves[i + self.seq_len].get('x', 0)
                orig_y = outline_moves[i + self.seq_len].get('y', 0)
                corr_x, corr_y = corrected_positions[line_num]
                correction_mag = np.sqrt((corr_x - orig_x)**2 + (corr_y - orig_y)**2)
                corrections.append(correction_mag)

        if corrections:
            print(f"  修正量统计:")
            print(f"    Mean: {np.mean(corrections)*1000:.2f} um")
            print(f"    Max:  {np.max(corrections)*1000:.2f} um")
            print(f"    RMS:  {np.sqrt(np.mean(np.array(corrections)**2))*1000:.2f} um")

        return corrected_positions


def generate_corrected_gcode(input_gcode, corrected_positions, output_file):
    """生成修正后的G-code文件

    只修正轮廓移动，保持其他移动不变
    """
    print(f"\n生成修正后的G-code...")

    with open(input_gcode, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    corrected_lines = []
    corrections_applied = 0

    for i, line in enumerate(lines, 1):
        line_stripped = line.strip()

        # 保留注释和非移动指令
        if not line_stripped.startswith('G1') and not line_stripped.startswith('G0'):
            corrected_lines.append(line)
            continue

        # 检查是否是需要修正的行
        if i in corrected_positions:
            x_corrected, y_corrected = corrected_positions[i]

            # 修改X和Y坐标
            new_line = line_stripped

            # X坐标
            if 'X' in new_line:
                new_line = re.sub(r'X-?\\d+\\.?\\d*', f'X{x_corrected:.4f}', new_line)

            # Y坐标
            if 'Y' in new_line:
                new_line = re.sub(r'Y-?\\d+\\.?\\d*', f'Y{y_corrected:.4f}', new_line)

            corrected_lines.append(new_line + '\n')
            corrections_applied += 1
        else:
            # 非轮廓移动，保持不变
            corrected_lines.append(line)

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(corrected_lines)

    print(f"  [OK] 保存到: {output_file}")
    print(f"  应用了 {corrections_applied} 处修正")


def main():
    parser = argparse.ArgumentParser(description='G-code轮廓修正（仅修正外壁和内壁）')
    parser.add_argument('--gcode', type=str, required=True,
                       help='原始G-code文件')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--output_dir', type=str, default='results/outline_correction',
                       help='输出目录')

    args = parser.parse_args()

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("G-code轮廓修正系统（仅修正外壁和内壁）")
    print("="*70)
    print(f"\n配置:")
    print(f"  输入G-code: {args.gcode}")
    print(f"  模型: {args.checkpoint}")
    print(f"  输出目录: {args.output_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}\n")

    # 1. 解析G-code
    print("\n" + "="*70)
    print("步骤 1: 解析G-code")
    print("="*70)
    parser = OutlineAwareGCodeParser(args.gcode)
    moves = parser.parse()

    # 2. 加载模型并修正
    print("\n" + "="*70)
    print("步骤 2: 应用模型修正（仅轮廓）")
    print("="*70)
    corrector = OutlineOnlyCorrector(args.checkpoint, device)
    corrected_positions = corrector.correct_outline_only(moves)

    # 3. 生成修正后的G-code
    print("\n" + "="*70)
    print("步骤 3: 生成修正后G-code")
    print("="*70)

    input_path = Path(args.gcode)
    output_file = Path(args.output_dir) / f"{input_path.stem}_outline_corrected.gcode"

    generate_corrected_gcode(args.gcode, corrected_positions, str(output_file))

    print("\n" + "="*70)
    print("完成！")
    print("="*70)
    print(f"\n生成文件: {output_file}")
    print("\n修正说明:")
    print("  - ✅ 外壁（Outer wall）已修正")
    print("  - ✅ 内壁（Inner wall）已修正")
    print("  - ❌ 填充（Infill）保持不变")
    print("  - ❌ 支撑（Support）保持不变")
    print("  - ❌ 裙边（Skirt）保持不变")


if __name__ == '__main__':
    main()
