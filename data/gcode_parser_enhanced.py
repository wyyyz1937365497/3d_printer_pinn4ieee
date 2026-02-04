"""
增强版G-code解析器

**关键改进**：
1. 正确处理速度变化（F指令）
2. 计算真实的加速度（而不是假设恒定速度）
3. 保留完整的时间信息
4. 与MATLAB无缝对接

用途：
- 为MATLAB仿真提供准确的轨迹数据
- 支持闭环优化
- 重新生成训练数据
"""

import numpy as np
import re
from collections import defaultdict
from pathlib import Path


class EnhancedGCodeParser:
    """增强版G-code解析器，支持速度变化"""

    def __init__(self, gcode_file):
        self.gcode_file = gcode_file
        self.moves = []  # 存储所有移动指令

    def parse(self, keep_type_annotations=True):
        """
        解析G-code文件

        参数：
            keep_type_annotations: 是否保留;TYPE:注释

        返回：
            trajectory字典，包含：
                - time: 时间数组（秒）
                - x, y, z: 位置数组（mm）
                - vx, vy, vz: 速度数组（mm/s）
                - ax, ay, az: 加速度数组（mm/s²）
                - feedrate: 进给速度数组（mm/min）
                - is_extruding: 是否在挤出
                - move_type: 移动类型（outline/infill/travel等）
        """
        print(f"解析G-code文件: {self.gcode_file}")

        with open(self.gcode_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # 解析状态
        current_pos = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'e': 0.0}
        current_feedrate = 1800.0  # mm/min (默认)
        current_type = 'unknown'

        # 存储原始移动数据
        raw_moves = []

        # 第一遍：提取所有移动指令
        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # 检测类型注释
            if keep_type_annotations and line.startswith(';TYPE:'):
                current_type = line[6:].strip()
                continue

            # 只处理G1/G0移动指令
            if not (line.startswith('G1') or line.startswith('G0')):
                continue

            # 解析移动参数
            move = {
                'line_num': line_num,
                'raw': line,
                'type': current_type,
                'pos': current_pos.copy(),
                'feedrate': current_feedrate,
                'has_move': False,
                'extruding': False
            }

            # 提取X, Y, Z, E, F参数
            for param in ['X', 'Y', 'Z', 'E', 'F']:
                match = re.search(f'{param}(-?\\d+\\.?\\d*)', line)
                if match:
                    value = float(match.group(1))

                    if param == 'F':
                        move['feedrate'] = value
                        current_feedrate = value
                    else:
                        move['pos'][param.lower()] = value
                        if param in ['X', 'Y', 'Z']:
                            move['has_move'] = True
                        elif param == 'E':
                            # 检测是否有明显的挤出
                            e_diff = value - current_pos['e']
                            move['extruding'] = abs(e_diff) > 0.01

            # 更新当前位置（为下一次移动做准备）
            for axis in ['x', 'y', 'z', 'e']:
                if axis in move['pos']:
                    current_pos[axis] = move['pos'][axis]

            # 只保留有实际位置移动的指令
            if move['has_move']:
                raw_moves.append(move)

        print(f"  提取了 {len(raw_moves)} 个移动指令")

        # 第二遍：计算轨迹（考虑实际速度）
        trajectory = self._compute_trajectory(raw_moves)

        return trajectory

    def _compute_trajectory(self, raw_moves):
        """
        从原始移动指令计算完整轨迹

        关键：正确处理速度变化
        """
        print("\n计算轨迹（考虑速度变化）...")

        n = len(raw_moves)

        # 初始化数组
        time = np.zeros(n)
        x = np.zeros(n)
        y = np.zeros(n)
        z = np.zeros(n)
        vx = np.zeros(n)
        vy = np.zeros(n)
        vz = np.zeros(n)
        ax = np.zeros(n)
        ay = np.zeros(n)
        az = np.zeros(n)
        feedrate = np.zeros(n)
        is_extruding = np.zeros(n, dtype=bool)
        move_type = []

        current_time = 0.0

        for i, move in enumerate(raw_moves):
            # 位置
            x[i] = move['pos']['x']
            y[i] = move['pos']['y']
            z[i] = move['pos']['z']

            # 进给速度（mm/min → mm/s）
            fr_mm_min = move['feedrate']
            fr_mm_s = fr_mm_min / 60.0
            feedrate[i] = fr_mm_s

            # 是否挤出
            is_extruding[i] = move['extruding']
            move_type.append(move['type'])

            # 计算速度和移动时间
            if i > 0:
                dx = x[i] - x[i-1]
                dy = y[i] - y[i-1]
                dz = z[i] - z[i-1]

                distance = np.sqrt(dx**2 + dy**2 + dz**2)

                if distance > 1e-6:  # 有实际移动
                    # 使用实际进给速度计算时间
                    move_time = distance / fr_mm_s

                    # 速度分量
                    vx[i] = dx / move_time
                    vy[i] = dy / move_time
                    vz[i] = dz / move_time

                    # 累积时间
                    current_time += move_time
                else:
                    # 没有移动
                    vx[i] = 0
                    vy[i] = 0
                    vz[i] = 0
                    move_time = 0

                time[i] = current_time
            else:
                # 第一个点
                time[i] = 0
                vx[i] = 0
                vy[i] = 0
                vz[i] = 0

        # 计算加速度（中心差分）
        for i in range(2, n):
            dt = time[i] - time[i-1]

            if dt > 1e-6:
                ax[i] = (vx[i] - vx[i-1]) / dt
                ay[i] = (vy[i] - vy[i-1]) / dt
                az[i] = (vz[i] - vz[i-1]) / dt

        # 统计信息
        print(f"  轨迹点数: {n}")
        print(f"  总时间: {time[-1]:.2f} s")
        print(f"  X范围: [{np.min(x):.2f}, {np.max(x):.2f}] mm")
        print(f"  Y范围: [{np.min(y):.2f}, {np.max(y):.2f}] mm")
        print(f"  Z范围: [{np.min(z):.2f}, {np.max(z):.2f}] mm")
        print(f"  速度范围: [{np.min(feedrate):.1f}, {np.max(feedrate):.1f}] mm/s")
        print(f"  最大加速度: X={np.max(np.abs(ax)):.1f}, Y={np.max(np.abs(ay)):.1f} mm/s²")

        return {
            'time': time,
            'x': x,
            'y': y,
            'z': z,
            'vx': vx,
            'vy': vy,
            'vz': vz,
            'ax': ax,
            'ay': ay,
            'az': az,
            'feedrate': feedrate,
            'is_extruding': is_extruding,
            'move_type': move_type
        }


def parse_for_training(gcode_file, params=None):
    """
    为训练数据生成解析G-code

    返回格式与MATLAB的parse_gcode.m兼容，但使用Python解析器

    参数：
        gcode_file: G-code文件路径
        params: 物理参数（可选，用于验证）

    返回：
        trajectory_data: 包含所有轨迹信息的字典
    """
    parser = EnhancedGCodeParser(gcode_file)
    trajectory = parser.parse(keep_type_annotations=False)

    # 转换为MATLAB兼容格式
    trajectory_data = {
        'time': trajectory['time'],
        'x': trajectory['x'],
        'y': trajectory['y'],
        'z': trajectory['z'],
        'vx': trajectory['vx'],
        'vy': trajectory['vy'],
        'vz': trajectory['vz'],
        'ax': trajectory['ax'],
        'ay': trajectory['ay'],
        'az': trajectory['az'],
        'feedrate': trajectory['feedrate'],
        'is_extruding': trajectory['is_extruding']
    }

    return trajectory_data


# 测试代码
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("用法: python gcode_parser_enhanced.py <gcode_file>")
        sys.exit(1)

    gcode_file = sys.argv[1]

    parser = EnhancedGCodeParser(gcode_file)
    trajectory = parser.parse()

    print("\n解析完成！")
    print(f"轨迹点数: {len(trajectory['time'])}")
    print(f"总时间: {trajectory['time'][-1]:.2f} s")
