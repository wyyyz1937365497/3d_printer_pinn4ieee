# MATLAB仿真系统使用指南

**日期**: 2026-01-27
**版本**: 1.0
**适用于**: 3D Printer PINN Project

---

## 快速导航

- [快速开始](#快速开始) - 5分钟上手
- [系统概述](#系统概述) - 了解整个系统
- [优化策略](#优化策略) - 30-40倍效率提升的关键
- [GPU加速](#gpu加速) - 使用cuda1加速仿真
- [参数配置](#参数配置) - 自定义物理参数
- [数据转换](#数据转换) - MATLAB→Python
- [故障排除](#故障排除) - 常见问题解决

---

## 快速开始

### 方法1：优化模式（推荐）⭐

**适用场景**: 所有层形状相同，需要高效生成训练数据

```matlab
cd('F:\TJ\3d_print\3d_printer_pinn4ieee')
collect_data_optimized
```

**预期结果**:
- 100,000+ 训练样本（含数据增强）
- 仿真时间：1.5 小时
- GPU：cuda1（不影响cuda0上的训练）

### 方法2：单次仿真

**适用场景**: 测试、调试、快速验证

```matlab
addpath('matlab_simulation')

% 基础仿真（CPU）
data = run_full_simulation('Tremendous Hillar_PLA_17m1s.gcode');

% GPU加速仿真
data = run_full_simulation_gpu('Tremendous Hillar_PLA_17m1s.gcode', ...
                               'output.mat', [], 1);  % 使用GPU 1
```

### 方法3：参数扫描

**适用场景**: 探索参数空间影响

```matlab
% 扫描加速度
accelerations = [200, 300, 400, 500];
for i = 1:length(accelerations)
    params = physics_parameters();
    params.motion.max_accel = accelerations(i);

    data = run_full_simulation_gpu('Tremendous Hillar_PLA_17m1s.gcode', ...
                                   sprintf('output_accel%d.mat', i), ...
                                   [], 1);
end
```

---

## 系统概述

### 核心组件

```
matlab_simulation/
├── physics_parameters.m           # 物理参数（Ender-3 V2 + PLA）
├── parse_gcode.m                  # G-code解析器
├── parse_gcode_improved.m         # 改进版解析器（推荐）
├── simulate_trajectory_error.m    # 轨迹误差（二阶系统）
├── simulate_trajectory_error_gpu.m # GPU加速版
├── simulate_thermal_field.m       # 温度场（移动热源）
├── run_full_simulation.m          # 完整仿真流程（CPU）
├── run_full_simulation_gpu.m      # 完整仿真流程（GPU）
├── gpu_utils.m                    # GPU工具函数
└── convert_matlab_to_python.py    # 数据转换器

根目录/
├── collect_data_optimized.m       # ⭐ 优化数据收集（推荐）
├── colleat_data.m                 # 标准数据收集（CPU）
└── colleat_data_gpu.m             # GPU数据收集
```

### 物理模型

#### 1. 轨迹误差模型

**原理**: 质量弹簧阻尼系统（二阶动力学）

```
运动方程: m·x'' + c·x' + k·x = F_inertia

其中:
- m: 移动质量（X: 0.485kg, Y: 0.650kg）
- c: 阻尼系数（GT2皮带: 25 N·s/m）
- k: 刚度（GT2皮带: 150,000 N/m）
- F_inertia: 惯性力（来自加速度变化）

固有频率:
- X轴: ωn ≈ 555 rad/s (88.4 Hz)
- Y轴: ωn ≈ 478 rad/s (76.2 Hz)
阻尼比: ζ ≈ 0.04（欠阻尼）
```

**输出**:
```matlab
results.error_x           % X方向误差向量 (mm)
results.error_y           % Y方向误差向量 (mm)
results.error_magnitude   % 误差幅值 (mm)
results.error_direction   % 误差方向 (rad)
results.F_inertia_x       % X方向惯性力 (N)
results.F_elastic_x       % X方向弹性力 (N)
results.belt_stretch_x    % X方向皮带伸长 (mm)
```

#### 2. 温度场模型

**原理**: 移动热源瞬态热传导

```
∂T/∂t = α·∇²T + Q_source - Q_cooling

其中:
- α: 热扩散系数（PLA: 8.7×10⁻⁸ m²/s）
- Q_source: 挤出热输入（喷嘴温度）
- Q_cooling: 对流冷却（h=44 W/(m²·K) 风扇开启）
```

**输出**:
```matlab
results.T_nozzle         % 喷嘴温度 (°C)
results.T_interface      % 层间界面温度 (°C)
results.T_surface        % 表面温度 (°C)
results.cooling_rate     % 冷却速率 (°C/s)
results.temp_gradient_z  % 垂直温度梯度 (°C/mm)
results.interlayer_time  % 层间时间间隔 (s)
```

#### 3. 层间粘结强度模型

**原理**: Wool-O'Connor聚合物愈合模型

```matlab
adhesion_ratio = 1 - exp(-interlayer_time / tau_healing)

其中:
- interlayer_time: 层间时间间隔
- tau_healing: 温度依赖的特征时间
```

**输出**:
```matlab
results.adhesion_ratio   % 粘结强度比 (0-1, 1为完全粘结)
```

---

## 优化策略

### 关键洞察

**发现**: 你的打印模型每一层形状都相同

这意味着：
- ✅ 几何多样性：从单层提取（转角、曲率）
- ✅ 物理多样性：从参数变化获得（速度、加速度、温度）
- ❌ 不需要：扫描所有50层获取几何信息

### 优化前后对比

| 指标 | 原策略 | 优化策略 | 提升 |
|------|--------|----------|------|
| 仿真层数 | 15层 | 1+3层 | 73% ↓ |
| 参数组合 | 少 | 180种 | 多样性↑ |
| 仿真时间 | 2-3天 | 1.5小时 | **30-40倍** |
| 原始样本 | 45K点 | 309K点 | **6.9倍** |
| 最终样本 | 4.2K | 28.8K | **6.9倍** |

### 优化策略详解

**单层参数扫描**（主力数据）:
```matlab
目标层: 第25层（中间层，代表性强）
参数空间:
  - 加速度: 5个值（200-500 mm/s²）
  - 速度: 4个值（100-400 mm/s）
  - 风扇: 3个值（0, 128, 255）
  - 温度: 3个值（20, 25, 30°C）
总组合: 5×4×3×3 = 180种

样本生成:
  - 每次仿真: 3000点
  - 时间窗口滑动: 280样本/仿真
  - 100种配置: 28,000样本
  - 数据增强×3: 84,000样本
```

**三层验证**（层间效应）:
```matlab
验证层: [1, 25, 50]（首、中、末）
参数: 每层10种关键配置
样本: 3×10×280 = 8,400样本
```

**总数据量**:
```
原始样本: 28,000 + 8,400 = 36,400
增强后: 36,400 × 3 = 109,200 样本 ✅
满足模型需求: 80,000-100,000 样本
```

### 使用优化脚本

```matlab
% 查看脚本内容了解详细配置
edit collect_data_optimized.m

% 三种策略可选
strategy = 1;  % 快速验证（30分钟，~10K样本）
strategy = 2;  % 标准数据集（1.5小时，~100K样本）⭐推荐
strategy = 3;  % 完整数据集（3小时，~180K样本）

% 运行
collect_data_optimized
```

---

## GPU加速

### 为什么使用GPU

| 数据点数 | CPU时间 | GPU时间 | 加速比 |
|---------|---------|---------|--------|
| 1,000   | ~2s     | ~3s     | 0.67x  |
| 10,000  | ~20s    | ~5s     | **4x** |
| 100,000 | ~200s   | ~15s    | **13x** |

**建议**: 数据量>10,000点时使用GPU

### GPU配置

你有2个GPU：
- **cuda0 (GPU 0)**: 正在训练，占用中
- **cuda1 (GPU 1)**: 空闲，用于仿真

### 使用方法

#### 方法1：自动GPU（推荐）

```matlab
% 使用优化脚本（默认cuda1）
collect_data_optimized
```

#### 方法2：手动指定

```matlab
% 使用GPU 1
data = run_full_simulation_gpu('Tremendous Hillar_PLA_17m1s.gcode', ...
                               'output.mat', [], 1);

% 使用GPU 0（不推荐，会干扰训练）
data = run_full_simulation_gpu('Tremendous Hillar_PLA_17m1s.gcode', ...
                               'output.mat', [], 0);

% 自动选择（选择内存最多的GPU）
data = run_full_simulation_gpu('Tremendous Hillar_PLA_17m1s.gcode', ...
                               'output.mat', [], []);
```

#### 方法3：检查GPU状态

```matlab
% 检查GPU数量
gpuDeviceCount
% 预期输出: 2

% 查看cuda1信息
gpuDevice(1)
% 输出: GPU名称、内存、利用率等

% 检查cuda1可用内存
gpu = gpuDevice(1);
fprintf('Free memory: %.2f GB\n', gpu.FreeMemory / 1e9);
% 建议: 至少2GB可用
```

### GPU加速原理

**关键优化**:
1. 向量化矩阵运算
2. GPU并行RK4求解
3. 减少CPU-GPU传输

**代码示例**:
```matlab
% CPU版本
for i = 1:n_points
    x_state(i+1) = x_state(i) + dt * velocity;
end

% GPU版本
ax_gpu = gpuArray(ax_ref);
x_state_gpu = gpuArray(x_state);
k1_gpu = Ax_gpu * x_state_gpu(:, 1:n_steps) + Bx_gpu * ax_gpu(1:n_steps);
% ... 全部在GPU上并行计算
x_state = gather(x_state_gpu);  % 一次性传回CPU
```

---

## 参数配置

### 修改物理参数

所有参数在 `matlab_simulation/physics_parameters.m` 中定义。

#### 运动参数

```matlab
params = physics_parameters();

% 修改最大加速度
params.motion.max_accel = 500;  % mm/s²（默认500）

% 修改最大速度
params.motion.max_velocity = 400;  % mm/s（默认500）

% 修改Jerk限制
params.motion.jerk_limit = 10;  % mm/s（默认8）
```

#### 热学参数

```matlab
% 修改环境温度
params.environment.ambient_temp = 25;  % °C（默认20）

% 修改风扇对流系数
params.heat_transfer.h_convection_with_fan = 44;  % W/(m²·K)（默认44）
params.heat_transfer.h_convection_no_fan = 10;    % W/(m²·K)（默认10）

% 修改喷嘴温度
params.material.nozzle_temp = 210;  % °C（默认210）
```

#### 动力学参数

```matlab
% 修改X轴质量
params.dynamics.x.mass = 0.485;  % kg（默认0.485）

% 修改皮带刚度
params.dynamics.x.stiffness = 150000;  % N/m（默认150000）

% 修改阻尼系数
params.dynamics.x.damping = 25;  % N·s/m（默认25）
```

### 参数扫描示例

```matlab
% 多参数网格扫描
accel_grid = 200:100:500;           % 5个值
velocity_grid = 100:100:400;        % 4个值
fan_grid = [0, 128, 255];           % 3个值

[accel_vals, vel_vals, fan_vals] = ...
    ndgrid(accel_grid, velocity_grid, fan_grid);

n_combos = length(accel_vals);  % 5×4×3 = 60种组合

for i = 1:n_combos
    params = physics_parameters();
    params.motion.max_accel = accel_vals(i);
    params.motion.max_velocity = vel_vals(i);

    if fan_vals(i) == 0
        params.heat_transfer.h_convection_with_fan = 10;
    else
        params.heat_transfer.h_convection_with_fan = 44;
    end

    % 运行仿真
    data = run_full_simulation_gpu('Tremendous Hillar_PLA_17m1s.gcode', ...
                                   sprintf('combo_%03d.mat', i), ...
                                   [], 1);
end
```

---

## 数据转换

### MATLAB → Python

#### 安装依赖

```bash
pip install numpy scipy h5py pandas
```

#### 转换命令

```bash
# 基础转换
python matlab_simulation/convert_matlab_to_python.py \
    "Tremendous Hillar_PLA_17m1s_simulation.mat" \
    training \
    -o training_data

# 批量转换
python matlab_simulation/convert_matlab_to_python.py \
    "data_simulation_layer25/*.mat" \
    training \
    -o training_data_combined
```

#### 输出格式

生成的HDF5文件结构：
```
training_data.h5
├── inputs/          # 输入特征
│   ├── x_ref        # 参考X位置 (mm)
│   ├── y_ref        # 参考Y位置 (mm)
│   ├── vx_ref       # 参考X速度 (mm/s)
│   ├── vy_ref       # 参考Y速度 (mm/s)
│   ├── ax_ref       # 参考X加速度 (mm/s²)
│   ├── ay_ref       # 参考Y加速度 (mm/s²)
│   ├── corner_angle # 转角角度 (度)
│   ├── curvature    # 曲率 (1/mm)
│   └── layer_num    # 层号
├── states/          # 系统状态
│   ├── T_interface  # 层间温度 (°C)
│   ├── F_inertia_x  # X惯性力 (N)
│   ├── F_inertia_y  # Y惯性力 (N)
│   └── belt_stretch_x # X皮带伸长 (mm)
├── outputs/         # 目标输出
│   ├── error_x      # X误差向量 (mm)
│   ├── error_y      # Y误差向量 (mm)
│   ├── error_magnitude # 误差幅值 (mm)
│   └── adhesion_ratio # 粘结强度比 (0-1)
└── descriptions     # 变量描述
```

### Python中使用数据

```python
import h5py
import numpy as np

# 读取数据
with h5py.File('training_data.h5', 'r') as f:
    # 输入
    X_ref = f['inputs/x_ref'][:]
    Y_ref = f['inputs/y_ref'][:]
    VX_ref = f['inputs/vx_ref'][:]
    AY_ref = f['inputs/ay_ref'][:]

    # 输出
    error_x = f['outputs/error_x'][:]
    error_y = f['outputs/error_y'][:]
    adhesion = f['outputs/adhesion_ratio'][:]

# 用于训练
inputs = np.stack([X_ref, Y_ref, VX_ref, AY_ref], axis=-1)
targets = np.stack([error_x, error_y, adhesion], axis=-1)

print(f"样本数: {len(inputs)}")
print(f"输入形状: {inputs.shape}")
print(f"输出形状: {targets.shape}")
```

---

## 故障排除

### MATLAB问题

#### 问题1：找不到G-code文件

**错误**: `Cannot open file 'Tremendous Hillar_PLA_17m1s.gcode'`

**解决**:
```matlab
% 检查当前目录
pwd

% 检查文件是否存在
exist('Tremendous Hillar_PLA_17m1s.gcode', 'file')

% 使用绝对路径
gcode_file = 'F:\TJ\3d_print\3d_printer_pinn4ieee\Tremendous Hillar_PLA_17m1s.gcode';
```

#### 问题2：GPU不可用

**错误**: `Parallel Computing Toolbox not found`

**解决**:
```matlab
% 方案A：安装工具箱
% 在MATLAB中: Add-Ons → Get Add-Ons → 搜索 "Parallel Computing"

% 方案B：使用CPU版本
data = run_full_simulation('Tremendous Hillar_PLA_17m1s.gcode');
```

#### 问题3：GPU内存不足

**错误**: `Out of memory on device`

**解决**:
```matlab
% 方案A：减少数据量
options.layers = 'first';  % 只仿真第一层

% 方案B：使用CPU
gpu_id = [];
data = run_full_simulation_gpu(..., [], gpu_id);

% 方案C：清理GPU
gpu = gpuDevice(1);
reset(gpu);  % 重置GPU
```

#### 问题4：仿真速度慢

**诊断**:
```matlab
% 检查数据点数
data = load('output.mat');
fprintf('数据点数: %d\n', length(data.simulation_data.time));

% 如果>10000点，应该用GPU
```

**解决**:
```matlab
% 使用GPU版本
data = run_full_simulation_gpu(..., [], 1);

% 或减小时间步（降低精度但更快）
params = physics_parameters();
params.sim.dt_thermal = 0.1;  % 默认0.05，增大到0.1
```

### Python转换问题

#### 问题1：缺少模块

**错误**: `ModuleNotFoundError: No module named 'h5py'`

**解决**:
```bash
pip install numpy scipy h5py pandas
```

#### 问题2：找不到MATLAB变量

**错误**: `KeyError: 'simulation_data'`

**解决**:
```matlab
% 确保使用-v7.3标志保存
save('output.mat', 'simulation_data', '-v7.3');

% 或使用转换脚本自动处理
run_full_simulation(..., 'output.mat');  % 内部已处理
```

#### 问题3：数据维度问题

**错误**: `ValueError: all input arrays must have the same shape`

**原因**: MATLAB .mat文件包含不同长度的序列

**解决**:
```python
# 转换脚本已自动处理，使用padding
python matlab_simulation/convert_matlab_to_python.py \
    "output.mat" \
    training \
    -o training_data \
    --pad True
```

### 性能问题

#### 仿真太慢

**诊断步骤**:
1. 检查数据量
2. 检查是否使用GPU
3. 检查时间步设置

**优化方案**:
```matlab
% 1. 减少层数
options.layers = [1, 25, 50];  % 只仿真3层

% 2. 使用GPU
data = run_full_simulation_gpu(..., [], 1);

% 3. 增大时间步
params.sim.dt_thermal = 0.1;  % 降低精度

% 4. 关闭绘图
options.debug_plot = false;
```

#### GPU反而更慢

**原因**: 数据量太小，CPU-GPU传输开销 > 计算收益

**解决**:
```matlab
% 检查数据点数
if n_points < 1000
    % 使用CPU
    data = run_full_simulation(...);
else
    % 使用GPU
    data = run_full_simulation_gpu(..., [], 1);
end
```

---

## 最佳实践

### 数据生成工作流

```matlab
% 1. 快速测试（5分钟）
params = physics_parameters();
options.layers = 1;  % 只仿真第一层
data = run_full_simulation_gpu('Tremendous Hillar_PLA_17m1s.gcode', ...
                               'test.mat', options, 1);

% 2. 检查结果
load('test.mat');
fprintf('点数: %d\n', length(simulation_data.time));
fprintf('最大误差: %.3f mm\n', max(simulation_data.error_magnitude));

% 3. 生成完整数据集（1.5小时）
collect_data_optimized  % 使用优化脚本

% 4. 转换为Python
% 在命令行运行
% python matlab_simulation/convert_matlab_to_python.py ...
```

### 参数扫描建议

```matlab
% 智能参数选择（拉丁超立方采样，而非全网格）
n_configs = 50;  % 目标配置数

% 关键参数范围
accel_range = [200, 500];   % mm/s²
velocity_range = [100, 400]; % mm/s
fan_options = [0, 128, 255];

% 使用lhs设计（需要Statistics Toolbox）
if exist('lhsdesign', 'file')
    designs = lhsdesign(n_configs, 2);  % 2个连续参数

    accelerations = accel_range(1) + designs(:,1) * diff(accel_range);
    velocities = velocity_range(1) + designs(:,2) * diff(velocity_range);
    fan_speeds = fan_options(randi(3, n_configs, 1));
else
    % 随机采样
    accelerations = accel_range(1) + rand(n_configs, 1) * diff(accel_range);
    velocities = velocity_range(1) + rand(n_configs, 1) * diff(velocity_range);
    fan_speeds = fan_options(randi(3, n_configs, 1));
end

% 运行仿真
for i = 1:n_configs
    params = physics_parameters();
    params.motion.max_accel = accelerations(i);
    params.motion.max_velocity = velocities(i);
    % ...
end
```

### 数据质量检查

```matlab
% 检查仿真合理性
load('output.mat');

% 1. 检查误差范围
fprintf('误差统计:\n');
fprintf('  最大: %.3f mm\n', max(simulation_data.error_magnitude));
fprintf('  平均: %.3f mm\n', mean(simulation_data.error_magnitude));
fprintf('  合理范围: 0.1-0.5 mm\n');

% 2. 检查温度范围
fprintf('\n温度统计:\n');
fprintf('  层间温度: %.1f - %.1f °C\n', ...
        min(simulation_data.T_interface), ...
        max(simulation_data.T_interface));
fprintf('  合理范围: 80-180 °C\n');

% 3. 检查粘结强度
fprintf('\n粘结强度:\n');
fprintf('  平均: %.2f\n', mean(simulation_data.adhesion_ratio));
fprintf('  合理范围: 0.6-0.95\n');

% 4. 可视化检查
figure;
subplot(3,1,1);
plot(simulation_data.time, simulation_data.error_magnitude);
ylabel('误差 (mm)');
title('轨迹误差');

subplot(3,1,2);
plot(simulation_data.time, simulation_data.T_interface);
ylabel('温度 (°C)');
title('层间温度');

subplot(3,1,3);
plot(simulation_data.time, simulation_data.adhesion_ratio);
ylabel('粘结强度比');
xlabel('时间 (s)');
title('层间粘结');
```

---

## 附录

### A. 完整参数列表

#### Ender-3 V2参数

```matlab
动力学:
  X轴质量: 0.485 kg
  Y轴质量: 0.650 kg
  X轴刚度: 150,000 N/m
  Y轴刚度: 150,000 N/m
  X轴阻尼: 25 N·s/m
  Y轴阻尼: 25 N·s/m
  X轴固有频率: 88.4 Hz
  Y轴固有频率: 76.2 Hz

运动限制:
  最大加速度: 500 mm/s²
  最大速度: 500 mm/s
  Jerk限制: 8-10 mm/s
```

#### PLA参数

```matlab
热学:
  热导率: 0.13 W/(m·K)
  比热容: 1,200 J/(kg·K)
  热扩散率: 8.7×10⁻⁸ m²/s
  玻璃化温度: 60°C
  熔点: 171°C

力学:
  密度: 1,240 kg/m³
  弹性模量: 3.5 GPa
  拉伸强度: 70 MPa
```

#### 传热参数

```matlab
对流系数:
  无风扇: 10 W/(m²·K)
  有风扇: 44 W/(m²·K)
  床接触: 150 W/(m²·K)

辐射:
  线性化: 10 W/(m²·K)
```

### B. 物理公式汇总

#### 轨迹误差

```
传递函数: H(s) = ωn² / (s² + 2ζωn·s + ωn²)
固有频率: ωn = √(k/m)
阻尼比: ζ = c/(2√(mk))
```

#### 温度场

```
热传导: ∂T/∂t = α·∇²T + Q_source - Q_cooling
牛顿冷却: dT/dt = -h·A/(m·cp) · (T - T_amb)
```

#### 粘结强度

```
愈合模型: σ = σ_bulk · [1 - exp(-t/τ)]
特征时间: τ = τ₀ · exp(Ea/RT)
```

### C. 输出变量完整列表

```
时间:
  time - 时间 (s)

参考轨迹:
  x_ref, y_ref, z_ref - 规划位置 (mm)

参考运动学:
  vx_ref, vy_ref, vz_ref - 规划速度 (mm/s)
  ax_ref, ay_ref, az_ref - 规划加速度 (mm/s²)
  v_mag_ref - 速度幅值 (mm/s)
  a_mag_ref - 加速度幅值 (mm/s²)
  jx_ref, jy_ref, jz_ref - Jerk (mm/s³)
  jerk_mag - Jerk幅值 (mm/s³)

实际轨迹:
  x_act, y_act, z_act - 实际位置 (mm)
  vx_act, vy_act, vz_act - 实际速度 (mm/s)
  ax_act, ay_act, az_act - 实际加速度 (mm/s²)

动力学:
  F_inertia_x, F_inertia_y - 惯性力 (N)
  F_elastic_x, F_elastic_y - 弹性力 (N)
  belt_stretch_x, belt_stretch_y - 皮带伸长 (mm)

轨迹误差:
  error_x - X方向误差向量 (mm)
  error_y - Y方向误差向量 (mm)
  error_magnitude - 误差幅值 (mm)
  error_direction - 误差方向 (rad)

G-code特征:
  is_extruding - 挤出标志
  is_travel - 空移标志
  is_corner - 转角标志
  corner_angle - 转角角度 (度)
  curvature - 曲率 (1/mm)
  layer_num - 层号
  dist_from_last_corner - 距上次转角距离 (mm)

温度场:
  T_nozzle - 喷嘴温度 (°C)
  T_interface - 层间温度 (°C)
  T_surface - 表面温度 (°C)
  cooling_rate - 冷却速率 (°C/s)
  temp_gradient_z - 垂直温度梯度 (°C/mm)
  interlayer_time - 层间时间 (s)

粘结强度:
  adhesion_ratio - 粘结强度比 (0-1)
```

---

**文档版本**: 1.0
**最后更新**: 2026-01-27
**作者**: 3D Printer PINN Project
