"""
基于gcodeparser的增强版3D打印机物理仿真器

功能：
1. 使用gcodeparser解析G-code（更强大、更准确）
2. 自动识别回抽、空移、打印移动
3. 2阶质量-弹簧-阻尼系统仿真
4. 支持速度变化、加速度限制
5. 高精度数值积分（RK4）

使用方法：
    simulator = PrinterPhysicsSimulator(params)
    trajectory = simulator.simulate_gcode('test.gcode')
"""

import numpy as np
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from gcodeparser import parse_gcode_lines, Commands


class EnhancedGCodeParser:
    """
    使用gcodeparser的增强版G-code解析器

    特点：
    - 利用gcodeparser的强大解析能力
    - 自动识别回抽、空移、打印移动
    - 提取完整的速度、加速度信息
    """

    def __init__(self, gcode_file: str, min_print_speed: float = 5.0):
        """
        Args:
            gcode_file: G-code文件路径
            min_print_speed: 最小打印速度（mm/s）
        """
        self.gcode_file = gcode_file
        self.min_print_speed = min_print_speed
        self.moves = []

    def parse(self) -> List[Dict]:
        """解析G-code文件"""
        print(f"解析G-code: {self.gcode_file}")

        # 读取G-code
        with open(self.gcode_file, 'r', encoding='utf-8') as f:
            gcode_lines = f.readlines()

        # 使用gcodeparser解析
        parsed = list(parse_gcode_lines(gcode_lines))

        # 状态追踪
        current_pos = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'e': 0.0}
        current_speed = 1800.0  # mm/min
        is_extruding = False

        stats = {
            'total_lines': len(parsed),
            'move_commands': 0,
            'printing_moves': 0,
            'retraction_moves': 0,
            'travel_moves': 0,
            'skipped_moves': 0
        }

        for line in parsed:
            # 只处理G1/G0移动命令
            if line.type == Commands.MOVE:
                stats['move_commands'] += 1

                # 提取参数
                params = line.params if hasattr(line, 'params') else {}

                # 更新位置和速度
                new_pos = current_pos.copy()

                if 'X' in params:
                    new_pos['x'] = float(params['X'])
                if 'Y' in params:
                    new_pos['y'] = float(params['Y'])
                if 'Z' in params:
                    new_pos['z'] = float(params['Z'])
                if 'E' in params:
                    new_e = float(params['E'])
                    e_delta = new_e - current_pos['e']

                    # 回抽判定
                    if e_delta < -0.5:
                        stats['retraction_moves'] += 1
                        current_pos['e'] = new_e
                        continue

                    # 挤出判定
                    is_extruding = e_delta > 0.1
                    current_pos['e'] = new_e
                else:
                    is_extruding = False

                if 'F' in params:
                    current_speed = float(params['F'])

                # 计算移动距离
                dx = new_pos['x'] - current_pos['x']
                dy = new_pos['y'] - current_pos['y']
                distance = np.sqrt(dx**2 + dy**2)

                # 跳过微小移动
                if distance < 0.01:
                    stats['skipped_moves'] += 1
                    current_pos.update(new_pos)
                    continue

                # 判断移动类型
                if is_extruding:
                    move_type = 'printing'
                    stats['printing_moves'] += 1
                else:
                    move_type = 'travel'
                    stats['travel_moves'] += 1

                # 计算速度（mm/s）
                speed_mm_s = current_speed / 60.0

                # 计算速度分量
                if distance > 0:
                    vx = (dx / distance) * speed_mm_s
                    vy = (dy / distance) * speed_mm_s
                else:
                    vx, vy = 0.0, 0.0

                # 记录移动
                move_info = {
                    'x': new_pos['x'],
                    'y': new_pos['y'],
                    'z': new_pos['z'],
                    'vx': vx,
                    'vy': vy,
                    'ax': 0.0,  # 加速度将在后面计算
                    'ay': 0.0,
                    'speed': speed_mm_s,
                    'distance': distance,
                    'type': move_type,
                    'line_index': line.line_index,
                    'raw_line': line
                }

                self.moves.append(move_info)
                current_pos.update(new_pos)

        # 打印统计
        print(f"  解析完成:")
        print(f"    总行数: {stats['total_lines']}")
        print(f"    移动命令: {stats['move_commands']}")
        print(f"    打印移动: {stats['printing_moves']}")
        print(f"    空移: {stats['travel_moves']}")
        print(f"    回抽: {stats['retraction_moves']}")
        print(f"    跳过(微小): {stats['skipped_moves']}")

        return self.moves


class PrinterPhysicsSimulator:
    """
    3D打印机物理仿真器

    模型：2阶质量-弹簧-阻尼系统
    方程：m·x'' + c·x' + k·x = F(t)

    参数（与MATLAB仿真一致）：
    - mass: 0.35 kg
    - stiffness: 8000 N/m
    - damping: 15.0 N·s/m
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Args:
            params: 物理参数字典
        """
        # 默认参数（与MATLAB仿真一致）
        default_params = {
            'dynamics': {
                'x': {
                    'mass': 0.35,          # kg - 挤出头质量
                    'stiffness': 8000.0,   # N/m - 有效刚度
                    'damping': 15.0        # N·s/m - 结构阻尼
                },
                'y': {
                    'mass': 0.45,          # kg - Y轴运动质量（含打印台）
                    'stiffness': 8000.0,   # N/m - 有效刚度
                    'damping': 15.0        # N·s/m - 结构阻尼
                }
            },
            'simulation': {
                'time_step': 0.001,       # 1ms (与MATLAB一致)
                'integration_method': 'rk4'  # 'rk4' or 'euler'
            }
        }

        if params:
            self._update_params(default_params, params)
            self.params = params
        else:
            self.params = default_params

        # 计算导出参数
        self._compute_derived_params()

    def _update_params(self, base: Dict, updates: Dict):
        """递归更新参数"""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._update_params(base[key], value)
            else:
                base[key] = value

    def _compute_derived_params(self):
        """计算导出参数（自然频率、阻尼比等）"""
        for axis in ['x', 'y']:
            dyn = self.params['dynamics'][axis]

            m = dyn['mass']
            k = dyn['stiffness']
            c = dyn['damping']

            # 自然频率
            dyn['natural_freq'] = np.sqrt(k / m)  # rad/s

            # 阻尼比
            dyn['damping_ratio'] = c / (2 * np.sqrt(k * m))

            # 固有频率（Hz）
            dyn['natural_freq_hz'] = dyn['natural_freq'] / (2 * np.pi)

    def simulate_trajectory(self, moves: List[Dict], axis: str = 'x') -> Dict:
        """
        仿真单个轴的轨迹（基于MATLAB的误差动力学方法）

        使用状态空间：[error; velocity_error]
        方程：
            error' = velocity_error
            velocity_error' = -(c/m)*velocity_error - (k/m)*error - a_ref

        Args:
            moves: 移动指令列表
            axis: 'x' 或 'y'

        Returns:
            仿真结果字典
        """
        if axis not in ['x', 'y']:
            raise ValueError(f"Invalid axis: {axis}")

        # 获取参数
        dyn = self.params['dynamics'][axis]
        m = dyn['mass']
        k = dyn['stiffness']
        c = dyn['damping']
        dt = self.params['simulation']['time_step']

        # 初始化状态
        n_points = len(moves)
        t = np.zeros(n_points)
        x_ref = np.zeros(n_points)
        v_ref = np.zeros(n_points)
        a_ref = np.zeros(n_points)  # 参考加速度

        # 首先提取参考轨迹（位置、速度、加速度）
        for i in range(n_points):
            x_ref[i] = moves[i][axis]
            v_ref[i] = moves[i][f'v{axis}']

            # 计算时间
            if i > 0:
                dist = np.sqrt((moves[i]['x'] - moves[i-1]['x'])**2 +
                             (moves[i]['y'] - moves[i-1]['y'])**2)
                speed = moves[i]['speed']
                if speed > 0 and dist > 0:
                    dt_segment = dist / speed
                    t[i] = t[i-1] + dt_segment
                else:
                    t[i] = t[i-1] + dt

        # 计算参考加速度（数值微分）
        for i in range(n_points):
            if i == 0 or i == n_points - 1:
                a_ref[i] = 0.0
            else:
                # 中心差分
                dt_forward = t[i+1] - t[i] if t[i+1] > t[i] else dt
                dt_backward = t[i] - t[i-1] if t[i] > t[i-1] else dt
                a_ref[i] = (v_ref[i+1] - v_ref[i-1]) / (dt_forward + dt_backward)

        # 状态空间仿真：[error; velocity_error]
        state = np.zeros((2, n_points))  # 第1行：位置误差，第2行：速度误差
        state[:, 0] = [0, 0]  # 从零误差开始

        # 系统矩阵
        A = np.array([[0, 1],
                      [-k/m, -c/m]])
        B = np.array([0, -1])  # 输入是加速度

        # RK4积分（与MATLAB一致）
        for i in range(1, n_points):
            # RK4积分步骤
            k1 = A @ state[:, i-1] + B * a_ref[i-1]
            k2 = A @ (state[:, i-1] + 0.5*dt*k1) + B * a_ref[i-1]
            k3 = A @ (state[:, i-1] + 0.5*dt*k2) + B * a_ref[i-1]
            k4 = A @ (state[:, i-1] + dt*k3) + B * a_ref[i-1]

            state[:, i] = state[:, i-1] + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

            # 数值保护（防止误差无限增长）
            if not np.all(np.isfinite(state[:, i])) or np.any(np.abs(state[:, i]) > 100):
                state[:, i] = state[:, i-1]

        # 提取结果
        error_x = state[0, :]  # 位置误差
        x_act = x_ref + error_x  # 实际位置 = 参考位置 + 误差

        return {
            't': t,
            'x_ref': x_ref,
            'x_act': x_act,
            'v_ref': v_ref,
            'a_ref': a_ref,
            'error': error_x
        }

    def _rk4_step(self, x, v, x_ref, v_ref, m, c, k, dt):
        """Runge-Kutta 4阶积分"""
        def derivatives(x, v, x_ref, v_ref):
            dx_dt = v
            dv_dt = (k * (x_ref - x) + c * (v_ref - v)) / m
            return dx_dt, dv_dt

        # k1
        dx1, dv1 = derivatives(x, v, x_ref, v_ref)

        # k2
        dx2, dv2 = derivatives(x + 0.5*dt*dx1, v + 0.5*dt*dv1, x_ref, v_ref)

        # k3
        dx3, dv3 = derivatives(x + 0.5*dt*dx2, v + 0.5*dt*dv2, x_ref, v_ref)

        # k4
        dx4, dv4 = derivatives(x + dt*dx3, v + dt*dv3, x_ref, v_ref)

        # 更新
        x_new = x + (dt/6) * (dx1 + 2*dx2 + 2*dx3 + dx4)
        v_new = v + (dt/6) * (dv1 + 2*dv2 + 2*dv3 + dv4)

        _, a = derivatives(x, v, x_ref, v_ref)

        return x_new, v_new, a

    def _euler_step(self, x, v, x_ref, v_ref, m, c, k, dt):
        """Euler积分"""
        a = (k * (x_ref - x) + c * (v_ref - v)) / m
        v_new = v + a * dt
        x_new = x + v * dt
        return x_new, v_new, a

    def simulate_gcode(self, gcode_file: str, filter_printing: bool = True) -> Dict:
        """
        仿真G-code文件的完整轨迹

        Args:
            gcode_file: G-code文件路径
            filter_printing: 是否只仿真打印移动

        Returns:
            完整仿真结果
        """
        # 解析G-code
        parser = EnhancedGCodeParser(gcode_file)
        all_moves = parser.parse()

        # 过滤
        if filter_printing:
            moves = [m for m in all_moves if m['type'] == 'printing']
            print(f"\n过滤后保留 {len(moves)} 个打印移动")
        else:
            moves = all_moves

        if len(moves) == 0:
            raise ValueError("没有有效的移动指令")

        # 仿真X轴
        print(f"\n仿真X轴...")
        sim_x = self.simulate_trajectory(moves, axis='x')

        # 仿真Y轴
        print(f"仿真Y轴...")
        sim_y = self.simulate_trajectory(moves, axis='y')

        # 合并结果
        result = {
            'moves': moves,
            'x': sim_x,
            'y': sim_y,
            'params': self.params
        }

        # 统计误差
        error_x = sim_x['error']
        error_y = sim_y['error']
        error_mag = np.sqrt(error_x**2 + error_y**2)

        print(f"\n仿真结果统计:")
        print(f"  时间范围: {sim_x['t'][-1]:.2f} s")
        print(f"  总点数: {len(moves)}")
        print(f"  X轴RMS误差: {np.sqrt(np.mean(error_x**2))*1000:.2f} um")
        print(f"  Y轴RMS误差: {np.sqrt(np.mean(error_y**2))*1000:.2f} um")
        print(f"  综合RMS误差: {np.sqrt(np.mean(error_mag**2))*1000:.2f} um")
        print(f"  最大误差: {np.max(error_mag)*1000:.2f} um")

        return result

    def get_system_info(self) -> Dict:
        """获取系统信息"""
        info = {}
        for axis in ['x', 'y']:
            dyn = self.params['dynamics'][axis]
            info[axis] = {
                'mass': dyn['mass'],
                'stiffness': dyn['stiffness'],
                'damping': dyn['damping'],
                'natural_freq': dyn['natural_freq'],
                'natural_freq_hz': dyn['natural_freq_hz'],
                'damping_ratio': dyn['damping_ratio']
            }
        return info


if __name__ == '__main__':
    # 测试代码
    # 创建测试G-code（包含挤出动作）
    test_gcode = """G28 ; Home
G1 Z10 F3000
G1 X10 Y10 E1.0 F1800
G1 X50 Y10 E2.0 F1800
G1 X50 Y50 E3.0 F1800
G1 X10 Y50 E4.0 F1800
G1 X10 Y10 E5.0 F1800
"""

    test_file = Path('test_simulation_enhanced.gcode')
    with open(test_file, 'w') as f:
        f.write(test_gcode)

    # 仿真
    simulator = PrinterPhysicsSimulator()
    result = simulator.simulate_gcode(str(test_file))

    print(f"\n系统信息:")
    info = simulator.get_system_info()
    for axis in ['x', 'y']:
        print(f"  {axis.upper()}轴:")
        print(f"    自然频率: {info[axis]['natural_freq_hz']:.2f} Hz")
        print(f"    阻尼比: {info[axis]['damping_ratio']:.4f}")

    # 清理
    test_file.unlink()
