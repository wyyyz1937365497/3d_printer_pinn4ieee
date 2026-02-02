# MATLAB仿真系统 - 统一入口

**版本**: 3.0
**更新日期**: 2026-02-01
**主要特性**: 单一入口点，参数化控制，自动GPU加速

---

## 快速开始

### 基本用法

```matlab
% 添加仿真路径
addpath('simulation')

% 最简单：仿真第一层
data = run_simulation('path/to/print.gcode');

% 查看结果
disp(data);
```

### 指定层数

```matlab
% 仿真单个特定层（例如第25层）
data = run_simulation('print.gcode', 'Layers', 25);

% 仿真多个层（例如第25、50、75层）
data = run_simulation('print.gcode', 'Layers', [25, 50, 75]);

% 仿真层范围（例如从第10到100层，每5层采样一次）
data = run_simulation('print.gcode', 'Layers', [10, 5, 100]);

% 仿真所有层
data = run_simulation('print.gcode', 'Layers', 'all');
```

### 控制输出

```matlab
% 指定输出文件路径
data = run_simulation('print.gcode', ...
    'OutputFile', 'results/my_simulation.mat');

% 高精度仿真（更小的时间步长）
data = run_simulation('print.gcode', ...
    'TimeStep', 0.005);  % 5ms采样（200Hz）

% 关闭详细输出
data = run_simulation('print.gcode', 'Verbose', false);
```

### GPU/CPU选择

```matlab
% 自动选择GPU（默认）
data = run_simulation('print.gcode', 'UseGPU', true);

% 强制使用CPU
data = run_simulation('print.gcode', 'UseGPU', false);

% 指定GPU设备ID（例如cuda1）
data = run_simulation('print.gcode', 'UseGPU', 1);
```

### 高级选项

```matlab
% 包含裙边/边缘
data = run_simulation('print.gcode', 'IncludeSkirt', true);

% 选择特定打印类型
data = run_simulation('print.gcode', ...
    'IncludeType', {'Outer wall', 'Inner wall', 'Infill'});

% 启用固件效应（转角偏差、步进共振等）
data = run_simulation('print.gcode', 'FirmwareEffects', true);
```

---

## 完整参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `'OutputFile'` | char | `''` | 输出.mat文件路径（默认自动生成） |
| `'Layers'` | char/numeric | `'first'` | 层选择：`'first'`, `'all'`, 整数，或`[start, step, end]` |
| `'UseGPU'` | logical/numeric | `true` | GPU选择：`true`, `false`, 或GPU ID |
| `'TimeStep'` | scalar | `0.01` | 仿真时间步长（秒） |
| `'IncludeSkirt'` | logical | `false` | 是否包含裙边/边缘 |
| `'IncludeType'` | cell | `{'Outer wall', 'Inner wall'}` | 包含的打印类型 |
| `'Verbose'` | logical | `true` | 是否显示进度信息 |
| `'FirmwareEffects'` | logical | `false` | 是否启用固件效应模拟 |

---

## 输出数据结构

```matlab
simulation_data =
    % 时间和轨迹
    time: [T×1 double]           % 时间 (s)
    x_ref, y_ref, z_ref: [T×1]  % 参考位置 (mm)
    x_act, y_act, z_act: [T×1]  % 实际位置 (mm)

    % 运动学
    vx_ref, vy_ref, vz_ref: [T×1]  % 参考速度 (mm/s)
    vx_act, vy_act, vz_act: [T×1]  % 实际速度 (mm/s)
    ax_ref, ay_ref, az_ref: [T×1]  % 加速度 (mm/s²)
    jx_ref, jy_ref, jz_ref: [T×1]  % Jerk (mm/s³)

    % 误差
    error_x, error_y: [T×1]           % X/Y误差 (mm)
    error_magnitude: [T×1]            % 误差幅值 (mm)
    error_direction: [T×1]            % 误差方向 (rad)

    % 动力学
    F_inertia_x, F_inertia_y: [T×1]  % 惯性力 (N)
    F_elastic_x, F_elastic_y: [T×1]  % 弹性力 (N)
    belt_stretch_x, belt_stretch_y: [T×1]  % 皮带伸长 (mm)

    % 热场
    T_nozzle: [T×1]           % 喷嘴温度 (°C)
    T_interface: [T×1]       % 层间温度 (°C)
    T_surface: [T×1]         % 表面温度 (°C)
    cooling_rate: [T×1]      % 冷却速率 (°C/s)
    temp_gradient_z: [T×1]  % 温度梯度 (°C/mm)

    % 质量指标
    internal_stress: [T×1]  % 内应力 (MPa)
    porosity: [T×1]        % 孔隙率 (0-100%)
    dimensional_accuracy: [T×1]  % 尺寸误差 (mm)
    quality_score: [T×1]   % 综合质量评分 (0-1)

    % G-code特征
    is_extruding: [T×1 logical]  % 挤出标志
    is_travel: [T×1 logical]     % 移动标志
    is_corner: [T×1 logical]     % 转角标志
    layer_num: [T×1 double]      % 层号
```

---

## 核心模块说明

### 1. 主要入口

**`run_simulation.m`** - 统一仿真入口
- 参数化控制（G-code文件、层数、时间步长等）
- 自动GPU/CPU选择
- 集成所有物理模型

### 2. 轨迹处理

**`parse_gcode_improved.m`** - G-code解析器
- 提取2D轨迹和层信息
- 支持层选择和打印类型过滤
- 计算速度、加速度、jerk

**`reconstruct_trajectory.m`** - 轨迹重建
- S曲线/梯形速度规划
- 时间插值（0.01s采样）
- 生成密集时间序列

### 3. 物理仿真

**`simulate_trajectory_error.m`** - CPU版轨迹误差
- 二阶质量-弹簧-阻尼系统
- RK4数值求解

**`simulate_trajectory_error_gpu.m`** - GPU加速版
- 向量化矩阵运算
- 4-13倍加速（数据量>10K点）

**`simulate_thermal_field.m`** - 热场仿真
- 移动热源模型
- 计算温度场、冷却速率、温度梯度

**`calculate_thermal_history.m`** - 多层热累积
- 三阶段物理模型：加热 → 冷却 → 热扩散
- 预测每层初始温度

**`calculate_quality_metrics.m`** - 质量指标
- 内应力、孔隙率、尺寸精度
- 综合质量评分

### 4. 支持模块

**`physics_parameters.m`** - 物理参数配置
- Ender-3 V2打印机参数
- PLA材料属性
- 传热系数

**`setup_gpu.m`** - GPU初始化
- 自动检测GPU
- 选择cuda1（不影响cuda0训练）
- CPU fallback机制

### 5. 固件效应包

**`+planner/junction_deviation.m`** - 转角偏差
**`+stepper/microstep_resonance.m`** - 微步共振
**`+stepper/timer_jitter.m`** - 定时器抖动

---

## 使用示例

### 示例1：快速验证

```matlab
% 仿真第一层，快速验证模型
addpath('simulation');
data = run_simulation('test.gcode', 'Verbose', true);
```

### 示例2：训练数据生成

```matlab
% 仿真多个层，用于模型训练
layers = [25, 25, 75];  % 第25、50、75层
data = run_simulation('train.gcode', ...
    'Layers', layers, ...
    'OutputFile', 'data/simulation/train_data.mat');
```

### 示例3：高精度全模型仿真

```matlab
% 仿真所有层，高精度
data = run_simulation('full.gcode', ...
    'Layers', 'all', ...
    'TimeStep', 0.005, ...  % 5ms采样
    'UseGPU', true);
```

### 示例4：固件效应研究

```matlab
% 启用固件效应
data = run_simulation('test.gcode', ...
    'FirmwareEffects', true, ...
    'Layers', 25);
```

---

## 工作流程

```
run_simulation.m (统一入口)
   ↓
1. G-code解析 (parse_gcode_improved.m)
   - 提取轨迹点
   - 过滤层和类型
   ↓
2. 热场仿真 (simulate_thermal_field.m)
   - 移动热源模型
   - 多层热累积
   ↓
3. 质量计算 (calculate_quality_metrics.m)
   - 基于参考轨迹+热场
   ↓
4. 轨迹误差仿真 (simulate_trajectory_error*.m)
   - GPU加速（可选）
   - 固件效应（可选）
   ↓
5. 数据融合与保存
   - 输出统一格式.mat文件
```

---

## 性能参考

| 数据量 | CPU | GPU | 加速比 |
|--------|-----|-----|--------|
| 1K点   | 2s  | 3s  | 0.67x  |
| 5K点   | 8s  | 5s  | 1.6x   |
| 10K点  | 15s | 6s  | 2.5x   |
| 50K点  | 75s | 15s | 5.0x   |
| 100K点 | 150s| 22s | 6.8x   |

**建议**: 数据量>500点时使用GPU

---

## 故障排除

### 问题1：找不到函数

```matlab
% 确保添加了路径
addpath('simulation')
```

### 问题2：GPU不可用

仿真会自动使用CPU版本，无需修改代码。

手动检查GPU：
```matlab
gpuDeviceCount  % 应该>0
```

### 问题3：G-code解析失败

确认G-code文件格式：
- 必须包含G1（移动）命令
- 建议使用切片软件导出的标准文件

---

## 迁移指南

### 从旧版本迁移

**旧代码**:
```matlab
% run_full_simulation_gpu.m
data = run_full_simulation_gpu('print.gcode', 'output.mat', opts, 1);
```

**新代码**:
```matlab
% run_simulation.m
data = run_simulation('print.gcode', ...
    'OutputFile', 'output.mat', ...
    'UseGPU', 1);
```

### 从批量脚本迁移

**旧代码**:
```matlab
% regenerate_all_datasets.m
regenerate_all_datasets();
```

**新代码**:
```matlab
% 使用循环调用run_simulation
gcode_files = {'file1.gcode', 'file2.gcode'};
for i = 1:length(gcode_files)
    run_simulation(gcode_files{i}, 'Layers', [25, 25, 75]);
end
```

---

## 更新日志

### v3.0 (2026-02-01)

**重大变更**:
- ✅ 统一入口点：`run_simulation.m`
- ✅ 参数化控制：G-code文件、层数、时间步长等
- ✅ 删除冗余脚本：6个旧入口文件
- ✅ 简化文档：专注于使用方法

**删除的文件**:
- `run_full_simulation.m`
- `run_full_simulation_gpu.m`
- `test_firmware_effects.m`
- `test_firmware_effects_simple.m`
- `regenerate_all_datasets.m`
- `simulate_trajectory_error_with_firmware_effects.m`（功能集成到主流程）

**保留的核心模块**:
- `parse_gcode_improved.m`
- `simulate_trajectory_error.m` / `.gpu.m`
- `simulate_thermal_field.m`
- `calculate_thermal_history.m`
- `calculate_quality_metrics.m`
- `physics_parameters.m`
- `setup_gpu.m`
- `+planner/` 和 `+stepper/` 包

---

**最后更新**: 2026-02-01
**维护者**: 3D Printer PINN Project Team
