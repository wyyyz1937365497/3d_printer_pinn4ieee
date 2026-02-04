"""
使用增强解析器重新生成训练数据

**关键改进**：
1. 使用Python增强解析器（支持速度变化）
2. 调用改进的MATLAB仿真函数
3. 误差基于理想轨迹计算
4. Python-MATLAB互操作

**用途**：
- 为LSTM模型重新生成训练数据
- 基于改进的误差模型
- 包含真实的速度变化

**流程**：
1. Python: 解析G-code → ideal_traj
2. MATLAB: simulate(ideal_traj) → actual_traj, error
3. Python: 保存训练数据（ideal + error）

作者：Claude
日期：2026-02-04
"""

import os
import sys
import numpy as np
import matlab.engine
import h5py
from pathlib import Path
from tqdm import tqdm
import json

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.gcode_parser_enhanced import EnhancedGCodeParser


class TrainingDataRegenerator:
    """训练数据重新生成器"""

    def __init__(self):
        print("="*70)
        print("训练数据重新生成系统（增强版）")
        print("="*70)

        # 启动MATLAB
        print("\n启动MATLAB引擎...")
        self.matlab = matlab.engine.start_matlab()
        self.matlab.cd(str(project_root / 'simulation'))

        # 加载物理参数
        print("加载物理参数...")
        self._load_params()

        print("  [OK] 初始化完成\n")

    def _load_params(self):
        """加载物理参数到MATLAB"""
        params_struct = self.matlab.struct()

        # Dynamics
        dynamics = self.matlab.struct()

        x_dyn = self.matlab.struct()
        x_dyn['mass'] = 0.35
        x_dyn['stiffness'] = 8000
        x_dyn['damping'] = 15.0
        dynamics['x'] = x_dyn

        y_dyn = self.matlab.struct()
        y_dyn['mass'] = 0.45
        y_dyn['stiffness'] = 8000
        y_dyn['damping'] = 15.0
        dynamics['y'] = y_dyn

        params_struct['dynamics'] = dynamics

        # Printing
        printing = self.matlab.struct()
        printing['nozzle_temp'] = 220
        printing['bed_temp'] = 60
        params_struct['printing'] = printing

        self.matlab.workspace['params'] = params_struct

    def generate_from_gcode(self, gcode_file, output_file=None):
        """
        从G-code生成训练数据

        参数：
            gcode_file: G-code文件路径
            output_file: 输出HDF5文件路径（可选）

        返回：
            data_dict: 包含轨迹和误差的字典
        """
        print(f"\n{'='*70}")
        print(f"处理: {Path(gcode_file).name}")
        print(f"{'='*70}")

        # 步骤1：Python解析G-code
        print("\n[1/3] 使用Python增强解析器解析G-code...")
        parser = EnhancedGCodeParser(gcode_file)
        ideal_traj = parser.parse(keep_type_annotations=False)

        print(f"  轨迹点数: {len(ideal_traj['time'])}")
        print(f"  总时间: {ideal_traj['time'][-1]:.2f} s")

        # 步骤2：MATLAB仿真（ideal_traj既是输入也是理想）
        print("\n[2/3] MATLAB仿真轨迹误差...")

        # 创建MATLAB结构体
        input_struct = self._traj_to_matlab(ideal_traj)
        ideal_struct = self._traj_to_matlab(ideal_traj)

        self.matlab.workspace['input_traj'] = input_struct
        self.matlab.workspace['ideal_traj'] = ideal_struct

        # 调用改进的仿真函数
        results = self.matlab.eval(
            'simulate_trajectory_error_from_python(input_traj, ideal_traj, params)',
            nargout=1
        )

        # 提取结果
        error_x = np.array(results['error_x']).flatten()
        error_y = np.array(results['error_y']).flatten()
        error_mag = np.array(results['error_magnitude']).flatten()

        print(f"  RMS误差: {np.sqrt(np.mean(error_mag**2))*1000:.2f} um")
        print(f"  最大误差: {np.max(error_mag)*1000:.2f} um")

        # 步骤3：保存数据
        print("\n[3/3] 保存训练数据...")

        data_dict = {
            'x_ref': ideal_traj['x'],
            'y_ref': ideal_traj['y'],
            'z_ref': ideal_traj['z'],
            'vx_ref': ideal_traj['vx'],
            'vy_ref': ideal_traj['vy'],
            'ax_ref': ideal_traj['ax'],
            'ay_ref': ideal_traj['ay'],
            'time': ideal_traj['time'],
            'error_x': error_x,
            'error_y': error_y,
            'error_magnitude': error_mag,
            'feedrate': ideal_traj['feedrate'],
            'is_extruding': ideal_traj['is_extruding'].astype(np.uint8)
        }

        if output_file:
            self._save_hdf5(data_dict, output_file)
            print(f"  [OK] 保存: {output_file}")

        return data_dict

    def _traj_to_matlab(self, traj):
        """转换Python轨迹字典为MATLAB结构体"""
        struct = self.matlab.struct()
        struct['time'] = matlab.double(traj['time'].tolist())
        struct['x'] = matlab.double(traj['x'].tolist())
        struct['y'] = matlab.double(traj['y'].tolist())
        struct['z'] = matlab.double(traj['z'].tolist())
        struct['vx'] = matlab.double(traj['vx'].tolist())
        struct['vy'] = matlab.double(traj['vy'].tolist())
        struct['ax'] = matlab.double(traj['ax'].tolist())
        struct['ay'] = matlab.double(traj['ay'].tolist())
        return struct

    def _save_hdf5(self, data_dict, filename):
        """保存为HDF5格式（与原始格式兼容）"""
        with h5py.File(filename, 'w') as f:
            for key, value in data_dict.items():
                f.create_dataset(key, data=value)

    def batch_generate(self, gcode_files, output_dir):
        """批量生成训练数据"""
        print(f"\n批量处理 {len(gcode_files)} 个文件...")
        print(f"输出目录: {output_dir}\n")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        summary = []

        for gcode_file in tqdm(gcode_files, desc="生成数据"):
            try:
                output_name = Path(gcode_file).stem + '.mat'
                output_file = Path(output_dir) / output_name

                data = self.generate_from_gcode(gcode_file, str(output_file))

                summary.append({
                    'gcode': gcode_file,
                    'output': str(output_file),
                    'points': len(data['time']),
                    'rms_error_um': np.sqrt(np.mean(data['error_magnitude']**2)) * 1000
                })

            except Exception as e:
                print(f"\n[ERROR] 处理失败: {gcode_file}")
                print(f"  {e}")
                summary.append({
                    'gcode': gcode_file,
                    'error': str(e)
                })

        # 保存摘要
        summary_file = Path(output_dir) / 'generation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n[OK] 批量生成完成")
        print(f"  成功: {sum(1 for s in summary if 'error' not in s)}/{len(gcode_files)}")
        print(f"  摘要: {summary_file}")

    def __del__(self):
        if hasattr(self, 'matlab'):
            try:
                self.matlab.quit()
                print("\n[MATLAB] 已关闭")
            except:
                pass


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='使用增强解析器重新生成训练数据'
    )
    parser.add_argument('--gcode', type=str, nargs='+',
                       help='G-code文件或目录')
    parser.add_argument('--output_dir', type=str,
                       default='data_python_parser_enhanced',
                       help='输出目录')
    parser.add_argument('--single', action='store_true',
                       help='单个文件模式（不是批量）')

    args = parser.parse_args()

    print("="*70)
    print("训练数据重新生成（增强版）")
    print("="*70)
    print(f"\n关键改进：")
    print(f"  - Python解析器：正确处理速度变化")
    print(f"  - MATLAB仿真：基于理想轨迹计算误差")
    print(f"  - 误差定义：error = actual - ideal\n")

    # 收集G-code文件
    gcode_files = []
    for path in args.gcode:
        p = Path(path)
        if p.is_file() and p.suffix in ['.gcode', '.gco']:
            gcode_files.append(str(p))
        elif p.is_dir():
            gcode_files.extend([str(f) for f in p.glob('*.gcode')])
            gcode_files.extend([str(f) for f in p.glob('*.gco')])

    print(f"找到 {len(gcode_files)} 个G-code文件\n")

    # 创建生成器
    generator = TrainingDataRegenerator()

    # 生成数据
    if args.single or len(gcode_files) == 1:
        # 单文件模式
        for gcode_file in gcode_files:
            output_name = Path(gcode_file).stem + '.mat'
            output_file = Path(args.output_dir) / output_name

            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

            data = generator.generate_from_gcode(gcode_file, str(output_file))

            print(f"\n{'='*70}")
            print(f"完成！")
            print(f"  输出: {output_file}")
            print(f"  点数: {len(data['time'])}")
            print(f"  RMS误差: {np.sqrt(np.mean(data['error_magnitude']**2))*1000:.2f} um")
    else:
        # 批量模式
        generator.batch_generate(gcode_files, args.output_dir)

    print("\n" + "="*70)
    print("数据生成完成！")
    print("="*70)
    print(f"\n下一步：")
    print(f"  1. 检查生成的数据: {args.output_dir}")
    print(f"  2. 重新训练模型")
    print(f"  3. 运行闭环优化验证")


if __name__ == '__main__':
    main()
