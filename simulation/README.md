# 3D打印机仿真系统 (FDM 3D Printer Simulation)

**版本**: 4.0 (并行数据收集)
**更新日期**: 2026-02-02

---

## 📁 目录结构

### 核心仿真模块 (`simulation/`)

```
simulation/
├── parse_gcode_improved.m                    # G-code解析器
├── reconstruct_trajectory.m                  # 轨迹重构（S-curve速度规划）
├── physics_parameters.m                      # 物理参数定义
├── simulate_trajectory_error.m               # 轨迹误差仿真（CPU）
├── simulate_trajectory_error_gpu.m           # 轨迹误差仿真（GPU）
├── simulate_trajectory_error_with_firmware_effects.m  # 固件增强仿真
├── run_simulation.m                          # 统一仿真接口
├── setup_gpu.m                               # GPU配置
│
├── +planner/                                 # 运动规划包
│   └── junction_deviation.m                  # Junction Deviation算法
│
├── +stepper/                                 # 步进电机包
│   ├── microstep_resonance.m                 # 微步谐振模型
│   └── timer_jitter.m                        # 定时器抖动模型
│
└── archives/                                 # 归档文件（旧版本）
    ├── run_full_simulation.m                 # 旧版完整仿真
    ├── simulate_thermal_field.m              # 热场仿真（已移除）
    └── ...
```

### 数据收集脚本 (根目录)

```
根目录/
├── collect_data_parallel.m                   # ⭐ 并行数据收集（主入口）
├── collect_3dbenchy.m                        # 3DBenchy数据收集
├── collect_bearing5.m                        # bearing5数据收集
├── collect_boat.m                            # simple_boat5数据收集
├── collect_nautilus.m                        # Nautilus数据收集
└── collect_all.m                             # 批量收集所有文件
```

---

## 🚀 快速开始

### 1. 收集单个文件

```matlab
% 3DBenchy（默认采样48层）
collect_3dbenchy

% Bearing5（全量75层）
collect_bearing5

% Nautilus（全量56层）
collect_nautilus

% Boat（采样74层）
collect_boat
```

### 2. 批量收集所有文件

```matlab
% 使用默认配置
collect_all

% 收集所有文件的所有层
collect_all('all')

% 统一采样配置
collect_all('sampled:5')
```

### 3. 自定义收集

```matlab
% 指定层范围
collect_3dbenchy(1:50)

% 指定采样间隔
collect_3dbenchy('sampled:2')

% 使用并行版本（自定义worker数）
collect_data_parallel('test.gcode', 'all', 'NumWorkers', 16)
```

---

## 🔄 仿真流程

### 轨迹误差仿真（包含固件效应）

```
G-code文件
    ↓
[parse_gcode_improved.m] 提取轨迹点
    ↓
[reconstruct_trajectory.m] S-curve速度规划
    ↓
[simulate_trajectory_error_with_firmware_effects.m]
    ├→ 基础动力学（惯性+弹性）→ 50-80 μm
    ├→ Junction Deviation（转角圆化）→ 20-50 μm
    ├→ 微步谐振（高频振动）→ 10-30 μm
    └→ 定时器抖动（脉冲不规则）→ 5-15 μm
    ↓
误差向量 (error_x, error_y) → 总计 ~0.1 mm
```

### 并行数据收集流程

```
1. 检测文件层数
   ├→ 读取文件头 "; total layer number: XX" (最快)
   └→ Fallback: 扫描文件统计 LAYER 标记

2. 预提取轨迹（所有worker共享）
   ├→ 一次性解析所有层
   └→ 组织到 containers.Map 缓存

3. 并行仿真（parfor）
   ├→ Worker 1: 层 1, 16, 31, 46...
   ├→ Worker 2: 层 2, 17, 32, 47...
   ├→ ...
   └→ 每个worker: 从缓存获取轨迹 → 运行仿真 → 保存

4. 输出
   └── data_simulation_<gcode>_<config>/layer<NN>_ender3v2.mat
```

---

## ⚡ 性能优化

### 并行计算加速

| 任务 | 单线程 | 15核并行 | 加速比 |
|------|--------|----------|--------|
| 3DBenchy 48层 | 40-50分钟 | 5-8分钟 | **6-10x** |
| Bearing5 75层 | 60-75分钟 | 8-12分钟 | **6-10x** |
| Boat 74层 | 60-75分钟 | 8-12分钟 | **6-10x** |
| Nautilus 56层 | 45-55分钟 | 6-9分钟 | **6-10x** |

### 关键优化点

1. **共享轨迹缓存**: 所有worker共享预提取的轨迹，避免重复解析gcode
2. **静默模式**: 使用 `evalc` 抑制并行worker的详细输出
3. **CPU模式**: 并行环境使用CPU，避免GPU资源竞争
4. **断点续传**: 自动跳过已完成的层

---

## 📊 输出数据格式

每个 `.mat` 文件包含：

```matlab
simulation_data = struct(
    % 时间
    'time',              % 时间向量 (s)

    % 参考轨迹
    'x_ref', 'y_ref', 'z_ref',           % 参考位置 (mm)
    'vx_ref', 'vy_ref', 'vz_ref',         % 参考速度 (mm/s)
    'ax_ref', 'ay_ref', 'az_ref',         % 参考加速度 (mm/s²)
    'jx_ref', 'jy_ref', 'jz_ref',         % 参考加加速度 (mm/s³)

    % 误差向量
    'error_x', 'error_y',                 % X/Y误差 (mm)
    'error_magnitude',                    % 误差幅值 (mm)
    'error_direction',                    % 误差方向 (rad)

    % G-code特征
    'is_extruding',                       % 是否挤出
    'is_travel',                          % 是否移动
    'layer_num',                          % 层号

    % 系统信息
    'params'                              % 物理参数
);
```

---

## 🔧 配置参数

### 修改物理参数

```matlab
% 编辑 physics_parameters.m
params.motion.max_accel = 500;           % 最大加速度 (mm/s²)
params.motion.max_velocity = 300;        % 最大速度 (mm/s)
params.dynamics.x.mass = 0.35;           % X轴质量 (kg)
params.dynamics.x.stiffness = 15000;     % X轴刚度 (N/m)
```

### 修改采样配置

```matlab
% 编辑收集脚本中的参数
LAYER_START = 1;         % 起始层
LAYER_STEP = 2;          % 采样间隔（2 = 隔层采样）
MAX_LAYERS = 50;         % 每文件最多采集层数
```

---

## 📝 下一步操作

### 1. 检查数据质量

```bash
python check_training_data.py --data_dir "data_simulation_*/layer*.mat"
```

### 2. 训练模型

```bash
python experiments/train_realtime.py \
    --data_dir "data_simulation_*/layer*.mat" \
    --seq_len 20 \
    --batch_size 256 \
    --epochs 100
```

### 3. 可视化结果

```bash
python experiments/visualize_realtime_correction.py \
    --checkpoint checkpoints/realtime_corrector/best_model.pth \
    --gcode test_gcode_files/3DBenchy_PLA_1h28m.gcode \
    --layer 25
```

---

## 🔧 故障排除

### 问题1：找不到函数

**错误**: `Undefined function 'physics_parameters'`

**解决**: 确保添加了路径
```matlab
addpath('simulation')
```

### 问题2：并行池未启动

**错误**: `Parallel Computing Toolbox not found`

**解决**: 会自动使用单线程，或手动启动：
```matlab
parpool('local', 8)  % 启动8个worker
```

### 问题3：GPU不可用

**错误**: GPU相关错误

**解决**: 并行版本默认使用CPU，GPU不影响数据收集

---

## 📚 版本历史

### v4.0 (2026-02-02) - 并行数据收集系统

**新增**:
- ✅ `collect_data_parallel.m` - 并行数据收集（6-10倍加速）
- ✅ 共享轨迹缓存 - 避免重复解析gcode
- ✅ 固件效应增强 - Junction Deviation、微步谐振、定时器抖动
- ✅ 自适应层数检测 - 自动读取gcode文件头

**改进**:
- ✅ 所有收集脚本支持并行
- ✅ 简化仿真流程（移除热场和质量评估）
- ✅ 统一仿真接口 `run_simulation.m`

**移除**:
- ❌ 热场仿真（`simulate_thermal_field.m`）
- ❌ 质量评估（`calculate_quality_metrics.m`）
- ❌ 旧版单线程收集脚本

### v3.1 (2026-01-29) - 质量特征

详见 `archives/README_v3.md`

### v2.0 (2026-01-27) - 轨迹重建

详见 `archives/README_v2.md`

---

## 📧 联系

**项目**: 3D Printer PINN Project
**维护**: Project Team
**许可**: 详见项目根目录 LICENSE 文件
