# Improved G-code Parser Guide

## 概述

改进的G-code解析器（`parse_gcode_improved.m`）专为2D轨迹提取而设计，能够更好地处理实际打印过程中的各种情况。

## 主要改进

### 1. 智能层过滤
- 可以选择提取特定层（第一层、所有层或指定层号）
- 自动跳过准备动作（skirt、brim等）

### 2. 打印类型识别
识别并可选择性地包含以下类型的打印动作：
- `Inner wall`（内壁）
- `Outer wall`（外壁）
- `Sparse infill`（稀疏填充）
- `Internal solid infill`（内部实心填充）
- `Top surface`（顶面）
- `Bottom surface`（底面）
- `Skirt`（裙边，默认排除）

### 3. 纯挤出轨迹
只提取真正的挤出移动，自动过滤空移（travel moves）。

### 4. 2D聚焦
专注于X-Y平面轨迹，更适合2D仿真。

## 使用方法

### 基本用法

```matlab
% 加载参数
params = physics_parameters();

% 使用默认选项（第一层，排除skirt）
data = parse_gcode_improved('print.gcode', params);
```

### 高级选项

```matlab
% 提取所有层
opts = struct();
opts.layers = 'all';
data = parse_gcode_improved('print.gcode', params, opts);

% 提取特定层（1, 3, 5）
opts.layers = [1, 3, 5];
data = parse_gcode_improved('print.gcode', params, opts);

% 包含skirt
opts.layers = 'first';
opts.include_skirt = true;
data = parse_gcode_improved('print.gcode', params, opts);

% 只包含特定打印类型
opts.layers = 'all';
opts.include_type = {'Inner wall', 'Outer wall'};  % 只包含内外壁
data = parse_gcode_improved('print.gcode', params, opts);
```

## 与原始解析器的对比

| 特性 | 原始解析器 | 改进解析器 |
|------|-----------|-----------|
| 维度 | 3D (X, Y, Z) | 2D聚焦 (X, Y) |
| 层处理 | 全部解析 | 可选择层 |
| 准备动作 | 包含 | 可选排除 |
| 打印类型 | 无区分 | 按类型分类 |
| 空移过滤 | 基于E值 | 更精确 |
| 轨迹纯净度 | 包含非打印移动 | 仅挤出移动 |
| 适用场景 | 完整3D仿真 | 2D轨迹分析 |

## 输出数据结构

改进的解析器输出与原始解析器兼容，包含以下额外字段：

```matlab
trajectory_data
├── x, y, z              % 位置坐标
├── time                 % 时间
├── vx, vy, vz           % 速度
├── ax, ay, az           % 加速度
├── jx, jy, jz           % Jerk
├── is_extruding         % 是否挤出
├── is_corner            % 是否转角
├── corner_angle         % 转角角度
├── layer_num            % 层号
├── print_type           % 打印类型（新增）
└── segment_idx          % 线段索引（新增）
```

## 完整工作流示例

### 示例1：快速2D仿真

```matlab
% 添加路径
addpath('matlab_simulation');

% 加载参数
params = physics_parameters();

% 解析第一层
opts = struct('layers', 'first', 'include_skirt', false);
trajectory_data = parse_gcode_improved('print.gcode', params, opts);

% 仿真轨迹误差
results = simulate_trajectory_error(trajectory_data, params);

% 绘制轨迹
figure;
plot(trajectory_data.x, trajectory_data.y, 'b-');
axis equal;
title('2D Printing Trajectory (First Layer)');
```

### 示例2：多层分析

```matlab
% 解析所有层
opts = struct('layers', 'all');
trajectory_data = parse_gcode_improved('print.gcode', params, opts);

% 统计每层的转角数
unique_layers = unique(trajectory_data.layer_num);
for i = 1:length(unique_layers)
    layer_mask = trajectory_data.layer_num == unique_layers(i);
    n_corners = sum(trajectory_data.is_corner(layer_mask));
    fprintf('Layer %d: %d corners\n', unique_layers(i), n_corners);
end
```

### 示例3：按打印类型分析

```matlab
% 只分析内外壁
opts = struct();
opts.layers = 'first';
opts.include_type = {'Inner wall', 'Outer wall'};

trajectory_data = parse_gcode_improved('print.gcode', params, opts);

% 分别统计每种类型的特征
unique_types = unique(trajectory_data.print_type);
for i = 1:length(unique_types)
    type_mask = strcmp(trajectory_data.print_type, unique_types{i});
    mean_velocity = mean(trajectory_data.v_actual(type_mask));
    fprintf('%s: Mean velocity = %.2f mm/s\n', unique_types{i}, mean_velocity);
end
```

## 集成到完整仿真

### 方法1：直接使用

```matlab
% 使用改进解析器的完整仿真
opts = struct('use_improved', true, 'layers', 'first');
simulation_data = run_full_simulation('print.gcode', 'output.mat', opts);
```

### 方法2：单独使用

```matlab
% 步骤1：解析G-code
params = physics_parameters();
opts = struct('layers', 'first');
trajectory_data = parse_gcode_improved('print.gcode', params, opts);

% 步骤2：仿真
trajectory_results = simulate_trajectory_error(trajectory_data, params);
thermal_results = simulate_thermal_field(trajectory_data, params);

% 步骤3：保存
save('results.mat', 'trajectory_data', 'trajectory_results', 'thermal_results');
```

## 常见问题

### Q: 为什么选择第一层？
A: 第一层最关键，因为它直接影响打印平台的粘附。同时，第一层通常包含最完整的轨迹信息。

### Q: 何时应该使用原始解析器？
A: 当你需要完整的3D信息或需要分析Z轴运动时。

### Q: 如何判断是否应该包含skirt？
A: Skirt不属于实际打印件，但如果需要分析热历史或打印头移动路径，可以包含它。

### Q: 可以提取多个不连续的层吗？
A: 可以：`opts.layers = [1, 5, 10];`

## 性能考虑

- **第一层**: 最快（~1000-5000点）
- **所有层**: 较慢（取决于打印件复杂度）
- **特定层**: 中等

建议在开发阶段使用第一层，在生产阶段根据需要扩展。

## 与Python集成

改进解析器的输出完全兼容Python转换器：

```bash
# 生成仿真数据后
python matlab_simulation/convert_matlab_to_python.py \
    "simulation_results.mat" \
    training \
    -o training_data
```

## 更新日志

### v1.0 (2026-01-27)
- 初始版本
- 支持层过滤
- 支持打印类型识别
- 自动过滤空移
- 与原始解析器输出兼容
