# MATLAB 3D Printer Simulation System

完整的FDM 3D打印机物理仿真系统，用于生成PINN训练数据。

## 系统概述

本系统实现了以下物理模型：

### 1. 轨迹误差模型（二阶系统动力学）
- **物理原理**：质量-弹簧-阻尼系统
  - 运动方程：`m·x'' + c·x' + k·x = F(t)`
  - m: 移动质量（Ender-3 V2喷头组件）
  - c: 阻尼系数（皮带和摩擦）
  - k: 刚度（GT2皮带弹性）
  - F: 惯性力（来自加速度变化）

- **输出状态量（向量形式！）**：
  - `error_x`, `error_y`: 位置误差的X、Y分量（mm）
  - `error_magnitude`: 位置误差幅值（mm）
  - `error_direction`: 位置误差方向（弧度）
  - `F_inertia_x`, `F_inertia_y`: 惯性力（N）
  - `F_elastic_x`, `F_elastic_y`: 皮带弹性力（N）
  - `belt_stretch_x`, `belt_stretch_y`: 皮带伸长量（mm）

### 2. 温度场模型（移动热源）
- **物理原理**：瞬态热传导方程
  - `∂T/∂t = α·∇²T + Q_source - Q_cooling`
  - α: 热扩散系数（PLA: ~8.7×10⁻⁸ m²/s）
  - Q_source: 挤出热输入
  - Q_cooling: 对流+辐射冷却

- **输出状态量**：
  - `T_nozzle`: 喷嘴位置温度（°C）
  - `T_interface`: 层间界面温度（°C）
  - `T_surface`: 表面温度（°C）
  - `cooling_rate`: 冷却速率（°C/s）
  - `temp_gradient_z`: 垂直温度梯度（°C/mm）
  - `interlayer_time`: 层间时间间隔（s）

### 3. 层间粘结强度模型
- **物理原理**：Wool-O'Connor聚合物愈合模型
  - 基于界面温度和扩散时间
  - 考虑冷却速率影响

- **输出状态量**：
  - `adhesion_ratio`: 粘结强度比（0-1，1为完全粘结）

### 4. G-code特征提取
- **输出状态量**：
  - `is_extruding`: 是否在挤出（布尔）
  - `is_travel`: 是否为空移（布尔）
  - `is_corner`: 是否为转角（布尔）
  - `corner_angle`: 转角角度（度）
  - `curvature`: 路径曲率（1/mm）
  - `layer_num`: 层号
  - `dist_from_last_corner`: 距上次转角的距离（mm）

## 物理参数来源

### Ender-3 V2机械参数
- **移动质量**：
  - X轴: 0.485 kg（挤出器组件+滑车）
  - Y轴: 0.650 kg（包括X轴组件）
- **GT2皮带参数**：
  - 刚度: ~150,000 N/m（含预紧力）
  - 阻尼: 25 N·s/m
- **步进电机（NEMA 17 42-34）**：
  - 保持扭矩: 0.40 N·m
  - 转子惯量: 54×10⁻⁶ kg·m²
- **运动限制**（来自Marlin固件）：
  - 最大加速度: 500 mm/s²
  - 最大速度: 500 mm/s
  - Jerk限制: 8-10 mm/s

### PLA材料参数
- **热学性质**：
  - 热导率: 0.13 W/(m·K)
  - 比热容: 1,200 J/(kg·K)
  - 热扩散率: 8.7×10⁻⁸ m²/s
  - 玻璃化温度: 60°C
  - 熔点: 171°C
- **力学性质**：
  - 密度: 1,240 kg/m³
  - 弹性模量: 3.5 GPa
  - 拉伸强度: 70 MPa

### 传热参数
- **对流传热系数**（来自文献）：
  - 自然对流（无风扇）: 10 W/(m²·K)
  - 强制对流（风扇开启）: 44 W/(m²·K)
  - 床接触: 150 W/(m²·K)
- **辐射**（线性化）: 10 W/(m²·K)

## 文件结构

```
matlab_simulation/
├── physics_parameters.m          # 物理参数配置
├── parse_gcode.m                 # G-code解析器
├── simulate_trajectory_error.m   # 轨迹误差仿真（二阶系统）
├── simulate_thermal_field.m      # 温度场仿真（移动热源）
├── run_full_simulation.m         # 主仿真脚本
├── convert_matlab_to_python.py   # MATLAB→Python转换器
└── README.md                      # 本文件
```

## 快速开始

### 1. MATLAB仿真

```matlab
% 在MATLAB中运行
% 添加到路径
addpath('matlab_simulation')

% 运行完整仿真
simulation_data = run_full_simulation('Tremendous Hillar_PLA_17m1s.gcode');

% 结果保存在: Tremendous Hillar_PLA_17m1s_simulation.mat
```

### 2. 转换为Python格式

```bash
# 安装依赖
pip install numpy scipy h5py pandas

# 转换为训练数据集（HDF5格式，推荐）
python matlab_simulation/convert_matlab_to_python.py \
    "Tremendous Hillar_PLA_17m1s_simulation.mat" \
    training \
    -o training_data

# 这会生成 training_data.h5 文件
```

### 3. 在Python中使用数据

```python
import h5py
import numpy as np

# 读取训练数据
with h5py.File('training_data.h5', 'r') as f:
    # 输入变量
    x_ref = f['inputs/x_ref'][:]
    y_ref = f['inputs/y_ref'][:]
    vx_ref = f['inputs/vx_ref'][:]
    ax_ref = f['inputs/ax_ref'][:]
    corner_angle = f['inputs/corner_angle'][:]

    # 系统状态
    T_interface = f['states/T_interface'][:]
    F_inertia_x = f['states/F_inertia_x'][:]
    belt_stretch_x = f['states/belt_stretch_x'][:]

    # 输出（目标）
    error_x = f['outputs/error_x'][:]      # X方向误差向量
    error_y = f['outputs/error_y'][:]      # Y方向误差向量
    error_magnitude = f['outputs/error_magnitude'][:]
    adhesion_ratio = f['outputs/adhesion_ratio'][:]

# 用于PINN训练
# inputs = [x_ref, y_ref, vx_ref, ax_ref, corner_angle, ...]
# targets = [error_x, error_y, adhesion_ratio, ...]
```

## 输出数据结构

### 完整变量列表（50+个状态量）

#### 时间
- `time`: 时间（s）

#### 参考轨迹（G-code）
- `x_ref`, `y_ref`, `z_ref`: 规划位置（mm）

#### 参考运动学
- `vx_ref`, `vy_ref`, `vz_ref`: 规划速度（mm/s）
- `ax_ref`, `ay_ref`, `az_ref`: 规划加速度（mm/s²）
- `v_mag_ref`: 速度幅值（mm/s）
- `a_mag_ref`: 加速度幅值（mm/s²）
- `jx_ref`, `jy_ref`, `jz_ref`: Jerk（mm/s³）
- `jerk_mag`: Jerk幅值（mm/s³）

#### 实际轨迹（含动力学）
- `x_act`, `y_act`, `z_act`: 实际位置（mm）
- `vx_act`, `vy_act`, `vz_act`: 实际速度（mm/s）
- `ax_act`, `ay_act`, `az_act`: 实际加速度（mm/s²）

#### 动力学量
- `F_inertia_x`, `F_inertia_y`: 惯性力（N）
- `F_elastic_x`, `F_elastic_y`: 弹性力（N）
- `belt_stretch_x`, `belt_stretch_y`: 皮带伸长（mm）

#### 轨迹误差（向量！）
- `error_x`: X方向误差分量（mm）
- `error_y`: Y方向误差分量（mm）
- `error_magnitude`: 误差幅值（mm）
- `error_direction`: 误差方向（弧度）

#### G-code特征
- `is_extruding`: 挤出标志（布尔）
- `is_travel`: 空移标志（布尔）
- `is_corner`: 转角标志（布尔）
- `corner_angle`: 转角角度（度）
- `curvature`: 曲率（1/mm）
- `layer_num`: 层号
- `dist_from_last_corner`: 距上次转角距离（mm）

#### 温度场
- `T_nozzle`: 喷嘴温度（°C）
- `T_interface`: 层间温度（°C）
- `T_surface`: 表面温度（°C）
- `cooling_rate`: 冷却速率（°C/s）
- `temp_gradient_z`: 垂直温度梯度（°C/mm）
- `interlayer_time`: 层间时间（s）

#### 粘结强度
- `adhesion_ratio`: 层间粘结强度比（0-1）

## 物理公式参考

### 轨迹误差模型

二阶系统传递函数：
```
H(s) = ωn² / (s² + 2ζωn·s + ωn²)
```

其中：
- `ωn = √(k/m)` - 固有频率（rad/s）
- `ζ = c/(2√(mk))` - 阻尼比

Ender-3 V2 X轴：
- ωn ≈ 555 rad/s（88.4 Hz）
- ζ ≈ 0.04（欠阻尼，会有震荡）

### 温度场模型

牛顿冷却定律：
```
dT/dt = -h·A/(m·cp) · (T - T_amb)
```

其中：
- h: 对流传热系数
- A: 表面积
- m: 质量
- cp: 比热容

### 层间粘结模型

Wool-O'Connor愈合模型（简化）：
```
σ_adhesion = σ_bulk · [1 - exp(-t/τ(T))]
τ(T) = τ₀ · exp(Ea/RT)
```

## 验证数据

### Ender-3 V2性能指标（来自文献）
- 典型转角误差: ~0.3 mm
- X轴共振频率: ~45 Hz
- Y轴共振频率: ~35 Hz

### 典型仿真结果
- 最大位置误差: 0.2-0.5 mm（取决于加速度）
- 层间温度范围: 80-180°C
- 层间时间: 5-30 s（取决于打印速度）
- 粘结强度比: 0.6-0.95

## 参数扫描

可以通过修改`physics_parameters.m`进行参数扫描：

```matlab
% 示例：扫描不同加速度
accelerations = [200, 300, 400, 500];  % mm/s²
for i = 1:length(accelerations)
    params = physics_parameters();
    params.motion.max_accel = accelerations(i);

    % 重新运行仿真
    % ...
end
```

## 与Python PINN集成

### 数据格式
推荐使用`training`格式，数据组织为：

```
training_data.h5
├── inputs/      # 输入特征（G-code、运动学）
├── states/      # 系统状态（力、温度）
├── outputs/     # 目标输出（误差、粘结）
└── descriptions # 变量描述
```

### 训练建议

1. **输入归一化**：
   - 位置: 减去均值，除以标准差
   - 速度: 归一化到[0, 1]
   - 角度: 使用sin/cos编码

2. **输出缩放**：
   - 误差: 可能需要对数变换（非均匀分布）
   - 粘结比: 已在[0, 1]

3. **采样策略**：
   - 对转角处过采样（关键位置）
   - 时间间隔均匀采样

## 故障排除

### MATLAB问题

**错误**: "Cannot open G-code file"
- 检查文件路径是否正确
- 确认G-code文件使用Unix换行符（LF）

**错误**: "Index exceeds array bounds"
- 可能是G-code解析问题
- 检查G-code是否为有效的RepRap/Marlin格式

**仿真很慢**
- 减小`sim.dt_thermal`（温度场时间步）
- 降低网格分辨率（dx, dy, dz）
- 关闭debug绘图

### Python转换问题

**错误**: "Module 'h5py' not found"
- 安装: `pip install h5py`

**错误**: "KeyError: 'simulation_data'"
- 确认MATLAB文件包含simulation_data变量
- 检查是否使用了`-v7.3`标志保存

## 引用和参考文献

### 物理参数来源
1. Ender-3 V2技术规格 - Creality官方文档
2. PLA热学性质 - [Simplify3D材料指南](https://www.simplify3d.com/resources/materials-guide/)
3. GT2皮带规格 - 机械工程手册
4. NEMA 17电机 - 步进电机数据手册

### 传热系数
- 对流系数（无风扇）: 10 W/(m²·K)
- 对流系数（有风扇）: 44 W/(m²·K)
- 来源: FDM热传递文献，2024-2025年研究

### 动力学建模
- 二阶系统控制理论
- 3D打印机轨迹误差研究（2024-2025）

## 贡献和反馈

如有问题或建议，请提交issue或pull request。

## 许可证

本项目采用MIT许可证。

---

**生成日期**: 2026-01-27
**版本**: 1.0.0
**作者**: 3D Printer PINN Project
