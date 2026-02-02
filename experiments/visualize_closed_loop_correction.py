"""
闭环实时修正仿真可视化

完整流程：
1. 读取参考轨迹
2. 逐点应用模型预测误差 → 得到修正轨迹
3. 用修正轨迹运行MATLAB物理仿真
4. 比较修正前后与理想轨迹的误差
5. 生成热力图对比

使用方法：
    python visualize_closed_loop_correction.py \
        --checkpoint checkpoints/realtime_corrector/best_model.pth \
        --gcode test_gcode_files/3DBenchy_PLA_1h28m.gcode \
        --layer 25 \
        --seq_len 50
"""

import os
import sys
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
import torch

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.realtime_dataset import RealTimeTrajectoryDataset
from models.realtime_corrector import RealTimeCorrector

# MATLAB引擎（可选）
try:
    import matlab.engine
    HAS_MATLAB = True
except ImportError:
    HAS_MATLAB = False
    print("[WARNING] MATLAB Engine未安装，将使用简化的物理仿真")


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
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

    model = RealTimeCorrector(
        input_size=4, hidden_size=hidden_size,
        num_layers=num_layers, dropout=dropout
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def load_reference_trajectory_from_gcode(gcode_file, layer_num, physics_params=None):
    """从gcode文件提取参考轨迹（通过MATLAB）"""
    if not HAS_MATLAB:
        raise ImportError("需要MATLAB Engine来解析gcode文件")

    print(f"解析gcode文件: {gcode_file}")
    print(f"  目标层: {layer_num}")

    # 启动MATLAB
    eng = matlab.engine.start_matlab()
    sim_path = str(project_root / 'simulation')
    eng.addpath(sim_path, nargout=0)

    # 调用MATLAB函数提取轨迹
    try:
        # 先获取物理参数
        if physics_params is None:
            eng.addpath(sim_path, nargout=0)
            params = eng.physics_parameters()
        else:
            params = physics_params

        # 提取单层轨迹
        result = eng.extract_single_layer_trajectory(
            gcode_file,
            float(layer_num),
            params,
            nargout=1
        )

        # 转换为numpy数组
        trajectory = {
            'time': np.array(result['time']).flatten(),
            'x': np.array(result['x']).flatten(),
            'y': np.array(result['y']).flatten(),
            'vx': np.array(result['vx']).flatten(),
            'vy': np.array(result['vy']).flatten(),
        }

        print(f"  [OK] 提取了 {len(trajectory['time'])} 个点")

    finally:
        eng.quit()

    return trajectory


def load_reference_trajectory_from_mat(data_dir, layer_num):
    """从已有的.mat文件加载参考轨迹"""
    import h5py
    import glob

    # 查找文件
    pattern = os.path.join(data_dir, f"layer{layer_num:02d}_*.mat")
    files = glob.glob(pattern)

    if not files:
        raise ValueError(f"未找到layer {layer_num}的文件: {pattern}")

    filepath = files[0]
    print(f"加载数据: {filepath}")

    with h5py.File(filepath, 'r') as f:
        sim_data = f['simulation_data']

        trajectory = {
            'time': sim_data['time'][:].flatten(),
            'x': sim_data['x_ref'][:].flatten(),
            'y': sim_data['y_ref'][:].flatten(),
            'vx': sim_data['vx_ref'][:].flatten(),
            'vy': sim_data['vy_ref'][:].flatten(),
        }

    print(f"  [OK] 加载了 {len(trajectory['time'])} 个点")

    return trajectory


def apply_model_correction(model, trajectory, scaler, seq_len, device):
    """应用模型进行逐点修正"""
    print("\n应用模型进行实时修正...")

    x_ref = trajectory['x']
    y_ref = trajectory['y']
    vx_ref = trajectory['vx']
    vy_ref = trajectory['vy']

    n_points = len(x_ref)

    # 准备特征
    features = np.stack([x_ref, y_ref, vx_ref, vy_ref], axis=1)

    # 归一化
    features_norm = scaler.transform(features)

    # 修正后的轨迹 - 试试反向修正
    x_corrected = x_ref.copy()
    y_corrected = y_ref.copy()

    # 存储预测的误差
    predicted_errors_x = np.zeros(n_points)
    predicted_errors_y = np.zeros(n_points)

    model.eval()

    print(f"  处理 {n_points} 个点...")

    with torch.no_grad():
        for i in range(seq_len, n_points):
            # 获取历史 [seq_len, 4]
            history = features_norm[i-seq_len:i]

            # 预测误差
            input_tensor = torch.FloatTensor(history).unsqueeze(0).to(device)
            pred_error = model(input_tensor).cpu().numpy()[0]

            # 存储预测误差
            predicted_errors_x[i] = pred_error[0]
            predicted_errors_y[i] = pred_error[1]

            # 尝试反向修正: corrected = reference + pred_error
            # 因为: actual = reference + error
            # 如果我们提前补偿，应该让 command = reference + pred_error
            x_corrected[i] = x_ref[i] + pred_error[0]
            y_corrected[i] = y_ref[i] + pred_error[1]

    # 打印预测误差统计
    pred_err_mag = np.sqrt(predicted_errors_x[seq_len:]**2 + predicted_errors_y[seq_len:]**2)
    print(f"  预测误差统计:")
    print(f"    RMS: {np.sqrt(np.mean(pred_err_mag**2))*1000:.2f} um")
    print(f"    Mean: {np.mean(pred_err_mag)*1000:.2f} um")
    print(f"    Max: {np.max(pred_err_mag)*1000:.2f} um")

    print(f"  [OK] 完成，修正了 {n_points - seq_len} 个点")

    return x_corrected, y_corrected, predicted_errors_x, predicted_errors_y


def run_matlab_physics_simulation(x_corrected, y_corrected, vx_ref, vy_ref, time, ax_ref, ay_ref):
    """用MATLAB运行物理仿真（修正轨迹 → 实际轨迹）"""
    if not HAS_MATLAB:
        raise ImportError("需要MATLAB Engine来运行物理仿真")

    print("\n运行MATLAB物理仿真...")

    # 启动MATLAB
    eng = matlab.engine.start_matlab()
    sim_path = str(project_root / 'simulation')
    eng.addpath(sim_path, nargout=0)

    try:
        # 获取物理参数
        params = eng.physics_parameters()

        # 准备输入数据结构（完整格式）
        # MATLAB期望列向量 (N×1)，需要转置
        def to_col_vector(arr):
            """将numpy数组转换为MATLAB列向量 (N×1)"""
            return matlab.double([[x] for x in arr.flatten()])

        traj_data = eng.struct(
            'time', to_col_vector(time),
            'x', to_col_vector(x_corrected),
            'y', to_col_vector(y_corrected),
            'z', to_col_vector(np.zeros(len(time))),  # Z轴（单层仿真，设为0）
            'vx', to_col_vector(vx_ref),
            'vy', to_col_vector(vy_ref),
            'vz', to_col_vector(np.zeros(len(time))),  # Z轴速度
            'ax', to_col_vector(ax_ref),
            'ay', to_col_vector(ay_ref),
            'az', to_col_vector(np.zeros(len(time)))   # Z轴加速度
        )

        # 调用MATLAB仿真函数（带固件效应）
        result = eng.simulate_trajectory_error_with_firmware_effects(traj_data, params, nargout=1)

        # 提取结果
        error_x = np.array(result['error_x']).flatten()
        error_y = np.array(result['error_y']).flatten()

        # 计算实际位置：actual = reference - error
        x_actual = x_corrected - error_x
        y_actual = y_corrected - error_y

        print(f"  [OK] MATLAB仿真完成")
        print(f"    误差RMS: {np.sqrt(np.mean(error_x**2 + error_y**2))*1000:.1f} um")

    except Exception as e:
        print(f"  [ERROR] MATLAB仿真失败: {e}")
        raise

    finally:
        eng.quit()

    return x_actual, y_actual


def plot_correction_comparison(x_ref, y_ref, x_act_uncorr, y_act_uncorr,
                                x_act_corr, y_act_corr, output_dir):
    """生成修正前后的热力图对比"""
    print("\n生成热力图对比...")

    # 计算误差
    err_uncorr = np.sqrt((x_act_uncorr - x_ref)**2 + (y_act_uncorr - y_ref)**2)
    err_corr = np.sqrt((x_act_corr - x_ref)**2 + (y_act_corr - y_ref)**2)

    # 转换为微米
    err_uncorr_um = err_uncorr * 1000
    err_corr_um = err_corr * 1000

    # 统计
    rms_uncorr = np.sqrt(np.mean(err_uncorr_um**2))
    rms_corr = np.sqrt(np.mean(err_corr_um**2))
    mean_uncorr = np.mean(err_uncorr_um)
    mean_corr = np.mean(err_corr_um)
    max_uncorr = np.max(err_uncorr_um)
    max_corr = np.max(err_corr_um)

    improvement = (rms_uncorr - rms_corr) / rms_uncorr * 100

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    vmax = max(max_uncorr, max_corr)

    # 未修正热图
    sc1 = axes[0].scatter(x_ref, y_ref, c=err_uncorr_um,
                          cmap='hot', s=1, alpha=0.6,
                          vmin=0, vmax=vmax)
    axes[0].set_title('Uncorrected Simulation\n(Before Real-Time Correction)',
                      fontsize=16, fontweight='bold', pad=15)
    axes[0].set_xlabel('X Position (mm)', fontsize=13)
    axes[0].set_ylabel('Y Position (mm)', fontsize=13)
    axes[0].axis('equal')
    cbar1 = plt.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Error from Ideal (um)', fontsize=12)

    stats1 = f'RMS: {rms_uncorr:.1f} um\nMean: {mean_uncorr:.1f} um\nMax: {max_uncorr:.1f} um'
    axes[0].text(0.02, 0.98, stats1, transform=axes[0].transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 修正后热图
    sc2 = axes[1].scatter(x_ref, y_ref, c=err_corr_um,
                          cmap='hot', s=1, alpha=0.6,
                          vmin=0, vmax=vmax)
    axes[1].set_title('Corrected Simulation\n(After Real-Time Correction)',
                      fontsize=16, fontweight='bold', pad=15)
    axes[1].set_xlabel('X Position (mm)', fontsize=13)
    axes[1].set_ylabel('Y Position (mm)', fontsize=13)
    axes[1].axis('equal')
    cbar2 = plt.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Error from Ideal (um)', fontsize=12)

    stats2 = f'RMS: {rms_corr:.1f} um\nMean: {mean_corr:.1f} um\nMax: {max_corr:.1f} um'
    axes[1].text(0.02, 0.98, stats2, transform=axes[1].transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.suptitle(f'Real-Time Trajectory Error Correction (Closed-Loop)\n'
                 f'Improvement: {improvement:.1f}% RMS Error Reduction',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存
    output_path = os.path.join(output_dir, 'heatmap_comparison_hd.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] 保存: {output_path}")

    return {
        'rms_uncorr': rms_uncorr,
        'rms_corr': rms_corr,
        'mean_uncorr': mean_uncorr,
        'mean_corr': mean_corr,
        'max_uncorr': max_uncorr,
        'max_corr': max_corr,
        'improvement': improvement
    }


def main():
    parser = argparse.ArgumentParser(description='Closed-loop real-time correction visualization')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--gcode', type=str,
                       default='test_gcode_files/3DBenchy_PLA_1h28m.gcode',
                       help='G-code file')
    parser.add_argument('--layer', type=int, default=25,
                       help='Layer number to visualize')
    parser.add_argument('--data_dir', type=str,
                       default='data_simulation_3DBenchy_PLA_1h28m_sampled_240layers',
                       help='Alternative: load from existing .mat files')
    parser.add_argument('--seq_len', type=int, default=50,
                       help='Sequence length')
    parser.add_argument('--output_dir', type=str,
                       default='results/realtime_correction',
                       help='Output directory')
    parser.add_argument('--use_matlab_data', action='store_true',
                       help='Load from existing .mat instead of parsing gcode')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("闭环实时修正仿真")
    print("="*70)
    print(f"\n配置:")
    print(f"  模型: {args.checkpoint}")
    print(f"  序列长度: {args.seq_len}")
    print(f"  层编号: {args.layer}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")

    # 1. 加载模型
    print(f"\n{'='*70}")
    print("步骤 1: 加载模型")
    print(f"{'='*70}")
    model, checkpoint = load_model(args.checkpoint, device)
    print(f"  [OK] 模型加载完成")

    # 2. 准备scaler
    print(f"\n{'='*70}")
    print("步骤 2: 准备数据标准化")
    print(f"{'='*70}")
    import glob
    import random

    if args.use_matlab_data:
        # 从现有数据加载
        data_dirs = [args.data_dir]
    else:
        # 从所有数据加载
        data_dirs = glob.glob("data_simulation_*")

    all_files = []
    for d in data_dirs:
        all_files.extend(glob.glob(os.path.join(d, "*.mat")))

    random.seed(42)
    random.shuffle(all_files)
    train_files = all_files[:min(50, len(all_files))]

    temp_dataset = RealTimeTrajectoryDataset(
        train_files, seq_len=args.seq_len, pred_len=1,
        scaler=None, mode='train'
    )
    scaler = temp_dataset.scaler
    print(f"  [OK] Scaler准备完成 (使用{len(train_files)}个文件)")

    # 3. 加载参考轨迹
    print(f"\n{'='*70}")
    print("步骤 3: 加载参考轨迹")
    print(f"{'='*70}")

    if args.use_matlab_data:
        trajectory = load_reference_trajectory_from_mat(args.data_dir, args.layer)
    else:
        trajectory = load_reference_trajectory_from_gcode(args.gcode, args.layer)

    # 4. 应用模型修正
    print(f"\n{'='*70}")
    print("步骤 4: 应用模型进行实时修正")
    print(f"{'='*70}")
    x_corrected, y_corrected, pred_err_x, pred_err_y = apply_model_correction(
        model, trajectory, scaler, args.seq_len, device
    )

    # 5. 物理仿真
    print(f"\n{'='*70}")
    print("步骤 5: 物理仿真")
    print(f"{'='*70}")

    # 计算加速度（从速度求导）
    time = trajectory['time']
    vx = trajectory['vx']
    vy = trajectory['vy']
    ax = np.gradient(vx, time)
    ay = np.gradient(vy, time)

    # 未修正仿真（参考轨迹 → 实际轨迹）
    print("\n未修正仿真:")
    x_act_uncorr, y_act_uncorr = run_matlab_physics_simulation(
        trajectory['x'], trajectory['y'],
        vx, vy,
        time, ax, ay
    )

    # 修正后仿真（修正轨迹 → 实际轨迹）
    print("\n修正后仿真:")
    x_act_corr, y_act_corr = run_matlab_physics_simulation(
        x_corrected, y_corrected,
        vx, vy,
        time, ax, ay
    )

    # 6. 生成可视化
    print(f"\n{'='*70}")
    print("步骤 6: 生成可视化")
    print(f"{'='*70}")
    metrics = plot_correction_comparison(
        trajectory['x'], trajectory['y'],
        x_act_uncorr, y_act_uncorr,
        x_act_corr, y_act_corr,
        args.output_dir
    )

    # 7. 保存报告
    report = {
        'config': {
            'checkpoint': args.checkpoint,
            'layer': args.layer,
            'seq_len': args.seq_len,
        },
        'metrics': metrics
    }

    report_path = os.path.join(args.output_dir, 'correction_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print("修正效果统计")
    print(f"{'='*70}")
    print(f"\n未修正仿真:")
    print(f"  RMS误差: {metrics['rms_uncorr']:.2f} um")
    print(f"  平均误差: {metrics['mean_uncorr']:.2f} um")
    print(f"  最大误差: {metrics['max_uncorr']:.2f} um")

    print(f"\n修正后仿真:")
    print(f"  RMS误差: {metrics['rms_corr']:.2f} um")
    print(f"  平均误差: {metrics['mean_corr']:.2f} um")
    print(f"  最大误差: {metrics['max_corr']:.2f} um")

    print(f"\n改进:")
    print(f"  RMS误差减少: {metrics['improvement']:.1f}%")

    print(f"\n[OK] 完成！")
    print(f"\n生成的文件:")
    print(f"  - {args.output_dir}/heatmap_comparison_hd.png")
    print(f"  - {args.output_dir}/correction_report.json")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
