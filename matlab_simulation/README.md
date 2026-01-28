# MATLAB仿真系统

**版本**: 2.0
**更新日期**: 2026-01-27

---

## 📁 核心文件

### 主入口（根目录）

**`collect_data.m`** - 数据收集主脚本
- 使用最新的轨迹重建和热累积模型
- 单层参数扫描 + 三层验证策略
- 30-40倍效率提升

**快速开始**:
```matlab
cd('F:\TJ\3d_print\3d_printer_pinn4ieee')
collect_data
```

### 核心模块（matlab_simulation/）

#### 1. 轨迹重建

**`reconstruct_trajectory.m`** - G-code轨迹重建
- 解析G-code关键点
- S曲线/梯形速度曲线规划
- 时间插值（0.01s采样）
- 输出密集时间序列：位置、速度、加速度、jerk

**关键创新**:
- 从33个关键点 → 2000-5000个密集点
- 考虑物理约束（v_max, a_max, j_max）
- 模拟Ender-3 V2运动规划

#### 2. 热累积模型

**`calculate_thermal_history.m`** - 多层热累积计算
- 三阶段物理模型：加热 → 冷却 → 热扩散
- 考虑喷嘴加热、层间冷却、下层热传导
- 预测每层初始温度

**关键创新**:
- 物理驱动（非简单线性模型）
- 第25层初始温度：60-70°C（符合文献）
- 考虑热输入衰减效应

#### 3. 动力学仿真

**`simulate_trajectory_error.m`** - CPU版轨迹误差
- 二阶质量-弹簧-阻尼系统
- RK4数值求解
- 作为GPU版本的fallback

**`simulate_trajectory_error_gpu.m`** - GPU加速版
- 向量化矩阵运算
- 4-13倍加速（数据量>10K点）

#### 4. 热场仿真

**`simulate_thermal_field.m`** - 热场演化
- 移动热源模型
- 集成热累积模型
- 计算温度场、冷却速率、温度梯度

#### 5. 粘结强度

**`calculate_adhesion_strength.m`** - 层间粘结预测
- Wool-O'Connor聚合物愈合模型
- 基于界面温度和时间
- 输出粘结强度比（0-1）

#### 6. 完整仿真

**`run_full_simulation_gpu.m`** - 完整仿真流程
- 集成所有模块
- 自动GPU/CPU选择
- 数据融合和保存

#### 7. 支持模块

**`physics_parameters.m`** - 物理参数配置
- Ender-3 V2参数（质量、刚度、阻尼）
- PLA材料参数（热学、力学）
- 传热系数（对流、辐射）
- 所有参数有文献来源

**`setup_gpu.m`** - GPU初始化
- 自动检测GPU
- 选择cuda1（不影响cuda0训练）
- CPU fallback机制

### 数据转换

**`convert_matlab_to_python.py`** - MATLAB → Python
- 转换.mat文件为HDF5格式
- 自动数据增强（时间窗口、噪声）
- 适配Python训练

---

## 🔄 工作流程

```
1. collect_data.m (主入口)
   ↓
2. reconstruct_trajectory.m → 密集时间序列（理想轨迹）
   ↓
3. calculate_thermal_history.m → 初始温度
   ↓
4. simulate_thermal_field.m → 热场演化
   ↓
5. calculate_quality_metrics.m → 质量特征 ✨NEW
   (基于理想轨迹+热场，不依赖误差)
   ↓
6. simulate_trajectory_error_gpu.m → 轨迹误差
   (动力学仿真，产生误差向量)
   ↓
7. run_full_simulation_gpu.m → 数据融合、保存
   ↓
8. convert_matlab_to_python.py → Python格式
```

**重要变更 (v3.0)**:
- 质量参数计算移至轨迹误差仿真**之前**
- 质量参数仅基于理想轨迹+热场计算
- 误差向量由动力学仿真独立产生

---

## 📊 输出数据

### .mat文件结构

每个仿真生成一个.mat文件，包含：

```matlab
simulation_data =
    time: [T×1 double]           % 时间 (s)

    % 参考轨迹
    x_ref, y_ref, z_ref: [T×1 double]  % 位置 (mm)
    vx_ref, vy_ref, vz_ref: [T×1 double] % 速度 (mm/s)
    ax_ref, ay_ref, az_ref: [T×1 double] % 加速度 (mm/s²)
    jx_ref, jy_ref, jz_ref: [T×1 double]  % Jerk (mm/s³)

    % 实际轨迹
    x_act, y_act, z_act: [T×1 double]
    vx_act, vy_act, vz_act: [T×1 double]
    ax_act, ay_act, az_act: [T×1 double]

    % 误差
    error_x, error_y: [T×1 double]         % X/Y误差 (mm)
    error_magnitude: [T×1 double]         % 误差幅值 (mm)
    error_direction: [T×1 double]         % 误差方向 (rad)

    % 动力学
    F_inertia_x, F_inertia_y: [T×1 double] % 惯性力 (N)
    F_elastic_x, F_elastic_y: [T×1 double] % 弹性力 (N)
    belt_stretch_x, belt_stretch_y: [T×1 double] % 皮带伸长 (mm)

    % 热场
    T_nozzle: [T×1 double]           % 喷嘴温度 (°C)
    T_interface: [T×1 double]       % 层间温度 (°C)
    T_surface: [T×1 double]         % 表面温度 (°C)
    cooling_rate: [T×1 double]      % 冷却速率 (°C/s)
    temp_gradient_z: [T×1 double]  % 温度梯度 (°C/mm)
    interlayer_time: [T×1 double]  % 层间时间 (s)

    % 粘结
    adhesion_ratio: [T×1 double]   % 粘结强度比 (0-1)

    % ✨ 质量特征 (Implicit Quality Parameters) - NEW
    internal_stress: [T×1 double]  % 内应力 (MPa)
    porosity: [T×1 double]         % 孔隙率 (0-100%)
    dimensional_accuracy: [T×1 double]  % 尺寸误差 (mm)
    quality_score: [T×1 double]    % 综合质量评分 (0-1)

    % G-code特征
    is_extruding: [T×1 logical]    % 挤出标志
    print_type: {T×1 cell}         % 打印类型
    layer_num: [T×1 double]        % 层号

    % 参数引用
    params: struct                 % 使用的物理参数
```

---

## 🚀 使用方法

### 方法1：标准数据生成（推荐）

```matlab
cd('F:\TJ\3d_print\3d_printer_pinn4ieee')
collect_data
```

**输出**:
- `data_simulation_layer25/` - 100个参数配置的仿真数据
- `validation_layer*/` - 三层验证数据
- 总计：~109,200 样本（含增强）
- 时间：~1.5 小时

### 方法2：单次测试

```matlab
addpath('matlab_simulation')

% 配置参数
params = physics_parameters();
params.debug.verbose = false;  % 关闭图表

% 配置选项
options = struct();
options.layers = 25;           % 第25层
options.time_step = 0.01;      % 10ms采样
options.include_type = {'Outer wall', 'Inner wall'};

% 运行仿真
data = run_full_simulation_gpu('Tremendous Hillar_PLA_17m1s.gcode', ...
                               'test_output.mat', options, params, 1);
```

### 方法3：转换为Python

```bash
python matlab_simulation/convert_matlab_to_python.py \
    "data_simulation_layer25/*.mat" \
    training \
    -o training_data
```

---

## 📐 参数配置

### 修改运动参数

```matlab
params = physics_parameters();
params.motion.max_accel = 400;      % mm/s²
params.motion.max_velocity = 300;   % mm/s
params.motion.jerk_limit = 10;      % mm/s³
```

### 修改热学参数

```matlab
params.environment.ambient_temp = 25;  % °C
params.heat_transfer.h_convection_with_fan = 44;  % W/(m²·K)
params.material.nozzle_temp = 210;  % °C
```

### 修改采样率

```matlab
options.time_step = 0.01;  % 10ms (100Hz)
% options.time_step = 0.005;  % 5ms (200Hz) - 更高质量
% options.time_step = 0.02;   % 20ms (50Hz) - 更快
```

---

## 🎯 性能指标

### 数据生成

| 指标 | 数值 |
|------|------|
| 原始G-code点数 | 33点/层 |
| 重建后点数 | 2000-5000点/层 |
| 点数提升 | 60-150倍 |
| 采样率 | 100 Hz |
| 生成速度 | ~30秒/仿真（GPU） |

### 数据质量

| 指标 | 数值 | 文献对比 |
|------|------|---------|
| 轨迹误差 | 0.3-0.5 mm | 0.3-0.5 mm [8] ✅ |
| 层间温度（L25） | 60-70°C | 65-75°C [5] ✅ |
| 粘结强度比 | 0.75-0.90 | 0.60-0.95 [9] ✅ |

---

## 🔧 故障排除

### 问题1：找不到函数

**错误**: `Undefined function 'setup_gpu'`

**解决**: 确保添加了路径
```matlab
addpath('matlab_simulation')
```

### 问题2：GPU不可用

**错误**: `Parallel Computing Toolbox not found`

**解决**: 会自动使用CPU版本，或检查：
```matlab
gpuDeviceCount  % 应该输出2
```

### 问题3：仿真太慢

**原因**: 数据量大，未使用GPU

**解决**: 检查GPU设置
```matlab
gpu_info = setup_gpu(1);  % 使用cuda1
fprintf('使用GPU: %d\n', gpu_info.use_gpu);
```

---

## 📚 相关文档

- **TECHNICAL_DOCUMENTATION.md** - 完整技术文档（公式、算法、推导）
- **THESIS_WRITING_QUICK_REF.md** - 论文写作速查表
- **THESIS_DOCUMENTATION.md** - 文献综述和理论基础
- **USER_GUIDE.md** - 使用指南
- **QUICK_START.md** - 快速开始

---

## 📝 更新日志

### v3.0 (2026-01-28) - 质量特征与数据加载

**新增**:
- ✅ `calculate_quality_metrics.m` - 计算隐式质量参数（内应力、孔隙率、尺寸精度、质量评分）
- ✅ `data/simulation/dataset.py` - Python数据集类，直接加载MATLAB .mat文件
- ✅ `data/scripts/prepare_training_data.py` - 数据预处理pipeline
- ✅ `experiments/quick_train_simulation.py` - 快速训练脚本
- ✅ `docs/SIMULATION_DATA_GUIDE.md` - 完整使用指南

**改进**:
- ✅ 配置文件明确定义12个输入特征和7个输出特征
- ✅ 数据转换脚本支持新的质量特征
- ✅ MATLAB和Python数据完全对齐

**修复**:
- ✅ 移除转角识别（is_corner）功能
- ✅ 修正字段名称（jerk_limit → max_jerk）
- ✅ 修复G-code解析（layer number, type parsing）

### v2.0 (2026-01-27)

**新增**:
- ✅ `reconstruct_trajectory.m` - 完整轨迹重建
- ✅ `calculate_thermal_history.m` - 物理驱动热累积模型
- ✅ `collect_data.m` - 新的主入口（原collect_data_optimized_v2.m）

**改进**:
- ✅ 采样点数提升60-150倍
- ✅ 物理一致性显著提升
- ✅ 热累积模型符合文献验证

**移除**:
- ❌ 旧的G-code解析器（parse_gcode.m, parse_gcode_improved.m）
- ❌ 旧的收集脚本（colleat_data.m, collect_data_optimized.m）
- ❌ CPU版完整仿真（run_full_simulation.m）

---

**最后更新**: 2026-01-28
**维护者**: 3D Printer PINN Project Team
