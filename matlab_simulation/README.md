# 3D打印机MATLAB仿真系统

用于生成PINN训练数据的完整MATLAB仿真系统。

## 📋 概述

本仿真系统通过物理建模生成3D打印过程的高质量数据，包括：

1. **轨迹误差模型**：基于二阶震荡系统（质量-弹簧-阻尼）
2. **温度场模型**：基于移动热源的热传导方程
3. **粘结力模型**：基于分子扩散理论

## 🎯 核心物理问题

### 1. 轨迹误差（速度突变影响）
- **物理模型**：`m·x'' + c·x' + k·x = F(t)`
- **关键因素**：
  - 惯性力：`F_inertia = m × a`
  - 皮带弹性：`ΔL = F / k`
  - 二阶系统响应：超调、震荡、稳态误差

### 2. 层间粘结力（温度历史影响）
- **物理模型**：分子扩散理论
- **关键因素**：
  - 层间温度：决定分子扩散系数
  - 冷却速率：影响分子扩散时间
  - 时间高于熔点：决定扩散深度
  - 环境温度：影响冷却速率 ✅

### 3. G-code信息源
- 转角检测：标记易产生误差的位置
- 曲率计算：路径弯曲程度
- 轨迹特征：长度、方向、距离等

## 📂 文件结构

```
matlab_simulation/
├── run_full_simulation.m          # 主控制脚本
├── generate_or_parse_gcode.m      # G-code生成/解析
├── simulate_trajectory_error.m    # 轨迹误差仿真
├── simulate_thermal_field.m       # 温度场仿真
├── calculate_adhesion_strength.m  # 粘结力计算
├── export_to_python.m             # Python格式转换
└── README.md                      # 本文档
```

## 🚀 使用方法

### 方法1：完整仿真（推荐）

在MATLAB中运行：

```matlab
% 导航到matlab_simulation目录
cd matlab_simulation

% 运行完整仿真
run_full_simulation
```

**参数调整**：
在 `run_full_simulation.m` 中修改 `params` 结构体：

```matlab
% 仿真数量
params.num_samples = 500;  % 样本数量

% 轨迹类型
params.trajectory_type = 'random_rectangles';  % 或 'sine_wave', 'spiral'

% 运动参数
params.print_speed = 50;      % 打印速度 (mm/s)
params.acceleration = 1500;   % 加速度 (mm/s^2)

% 传动系统（二阶系统参数）
params.mass_x = 0.5;          % X轴质量 (kg)
params.stiffness_x = 50000;   % X轴刚度 (N/m)
params.damping_x = 20;        % X轴阻尼 (N·s/m)

% 热学参数
params.T_nozzle = 220;        % 喷嘴温度 (°C)
params.T_bed = 60;            % 热床温度 (°C)
params.T_ambient = 25;        % 环境温度 (°C) ⭐ 重要！
```

### 方法2：单独运行模块

#### 仅生成轨迹
```matlab
params = struct();  % 设置参数
[gcode_data, trajectory] = generate_or_parse_gcode(params);
```

#### 仅仿真轨迹误差
```matlab
params = struct();  % 设置参数
trajectory = struct();  % 输入轨迹
trajectory_error = simulate_trajectory_error(trajectory, params);
```

#### 仅仿真温度场
```matlab
params = struct();  % 设置参数
trajectory = struct();  % 输入轨迹
thermal_field = simulate_thermal_field(trajectory, params);
```

### 方法3：转换已有数据

如果你已经有 `.mat` 数据：

```matlab
export_to_python('./output/your_data.mat', params)
```

## 📊 输出数据格式

### MATLAB格式

原始数据保存在 `./output/` 目录：

```
output/
├── 3d_print_simulation_v1_data.mat        # 完整数据（MATLAB格式）
├── 3d_print_simulation_v1_data_python.mat # Python兼容格式
├── 3d_print_simulation_v1_data_X.csv      # 特征矩阵（CSV）
├── 3d_print_simulation_v1_data_y.csv      # 目标矩阵（CSV）
└── *_loader.py                             # Python加载脚本
```

### Python格式

数据结构：

```python
import scipy.io as sio

# 加载数据
data = sio.loadmat('3d_print_simulation_v1_data_python.mat')

X = data['X']  # (num_samples, 50) - 特征矩阵
y = data['y']  # (num_samples, 4) - 目标矩阵

feature_names = data['feature_names']  # 特征名称
target_names = data['target_names']    # 目标名称
```

## 🔬 输出状态量清单

### 输入特征（50个）

#### A. 轨迹误差模块（20个）
| # | 特征名 | 单位 | 说明 |
|---|--------|------|------|
| 1 | mean_epsilon_x | mm | X方向平均位置误差 |
| 2 | mean_epsilon_y | mm | Y方向平均位置误差 |
| 3 | max_epsilon_r | mm | 最大位置误差幅值 |
| 4 | rms_error | mm | RMS位置误差 |
| 5-6 | mean_vx/vy_act | mm/s | 平均实际速度 |
| 7 | max_v_ref | mm/s | 最大参考速度 |
| 8-9 | mean_abs_ax/ay_ref | mm/s² | 平均加速度绝对值 |
| 10-11 | max_abs_jx/jy_ref | mm/s³ | 最大加加速度绝对值 |
| 12-13 | mean_abs_F_inertia_x/y | N | 平均惯性力绝对值 |
| 14-15 | max_abs_delta_L_x/y | mm | 最大皮带伸长量 |
| 16 | omega_n_x | rad/s | X轴固有频率 |
| 17 | zeta_x | - | X轴阻尼比 |
| 18-20 | print_speed, acceleration, jerk | mm/s, mm/s², mm/s³ | 运动参数设置 |

#### B. 温度场模块（18个）
| # | 特征名 | 单位 | 说明 |
|---|--------|------|------|
| 21-24 | T_path: mean/max/min/std | °C | 喷嘴路径温度统计 |
| 25 | mean_T_interface | °C | 平均层间温度 |
| 26-27 | cooling_rate: mean/max | °C/s | 冷却速率 |
| 28 | mean_time_above_melting | s | 平均时间高于熔点 |
| 29 | mean_gradient_z | °C/mm | Z方向温度梯度 |
| 30 | mean_gradient_xy | °C/mm | XY平面温度梯度 |
| 31 | mean_thermal_accumulation_time | s | 平均热累积时间 |
| 32-35 | T_nozzle, T_bed, T_ambient, fan_speed | °C, °C, °C, RPM | 温度和风扇设置 |
| 36-38 | mean_vx/vy_ref, layer_height | mm/s, mm/s, mm | 速度和层高 |

#### C. G-code特征模块（8个）
| # | 特征名 | 单位 | 说明 |
|---|--------|------|------|
| 39 | corner_density | - | 转角密度（转角数/总点数） |
| 40 | mean_corner_angle | ° | 平均转角角度 |
| 41-42 | curvature: max/mean | 1/mm | 曲率统计 |
| 43 | mean_d_last_corner | mm | 平均距离上次转角 |
| 44-45 | num_layers, num_corners | - | 层数和转角总数 |
| 46 | extrusion_width | mm | 挤出宽度 |

#### D. 其他参数（4个）
| # | 特征名 | 单位 | 说明 |
|---|--------|------|------|
| 47 | nozzle_diameter | mm | 喷嘴直径 |
| 48 | extrusion_multiplier | - | 挤出倍率 |
| 49-50 | mass_x, stiffness_x | kg, N/m | X轴质量和刚度 |

### 输出目标（4个）

| # | 目标名 | 单位 | 说明 |
|---|--------|------|------|
| 1 | max_trajectory_error | mm | 最大轨迹误差 ⭐ |
| 2 | mean_adhesion_strength | MPa | 平均层间粘结强度 ⭐ |
| 3 | weak_bond_ratio | - | 弱粘结区域比例 |
| 4 | quality_score | - | 综合质量评分 (0-1) |

## 🎨 因果关系链

### 轨迹误差因果链
```
G-code（转角、速度变化）
    ↓
参考加速度 a_ref(t)
    ↓
惯性力 F_inertia = m × a_ref
    ↓
二阶系统响应（m-c-k）
    ↓
实际位置 x_act(t) = H(s) × x_ref(s)
    ↓
位置误差 ε = x_act - x_ref
    ↓
【输出】转角处轨迹误差
```

### 粘结力因果链
```
G-code轨迹 → 喷嘴位置 (x,y,z)(t)
    ↓
挤出速度 + 流量 → 热输入 Q_in(t)
    ↓
移动热源边界条件
    ↓
热传导方程求解
    ↓
温度场 T(x,y,z,t)
    ↓
层间温度 T_interface + 冷却速率 dT/dt
    ↓
分子扩散系数 D(T) + 扩散时间 t
    ↓
【输出】层间粘结强度 σ_adhesion
```

## 💡 建议的仿真参数

### 快速测试（~2分钟）
```matlab
params.num_samples = 10;
params.trajectory_type = 'sine_wave';
```

### 标准仿真（~30分钟）
```matlab
params.num_samples = 200;
params.trajectory_type = 'random_rectangles';
params.num_corners = 20;
```

### 高质量数据（~2小时）
```matlab
params.num_samples = 1000;
params.trajectory_type = 'random_rectangles';
params.num_corners = 50;
```

## 🔧 参数校准建议

### 基于你的打印机型号
修改 `run_full_simulation.m` 中的参数：

**Ender-3 / Ender-3 V2**：
```matlab
params.mass_x = 0.5;           % kg
params.stiffness_x = 50000;    % N/m
params.damping_x = 20;         % N·s/m
params.print_speed = 50;       % mm/s
params.acceleration = 1500;    % mm/s^2
```

**Prusa i3 MK3**：
```matlab
params.mass_x = 0.3;           % kg（更轻）
params.stiffness_x = 80000;    % N/m（更硬）
params.damping_x = 25;         % N·s/m
params.print_speed = 80;       % mm/s（更快）
params.acceleration = 2000;    % mm/s^2
```

**Ultimaker**：
```matlab
params.mass_x = 0.4;           % kg
params.stiffness_x = 100000;   % N/m（工业级）
params.damping_x = 30;         % N·s/m
params.print_speed = 100;      % mm/s
```

## 📈 Python集成示例

```python
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

# 1. 加载数据
data = loadmat('output/3d_print_simulation_v1_data_python.mat')
X = data['X']
y = data['y']

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 训练PINN模型
# （这里使用你的PINN代码）
# model = create_pinn_model()
# model.fit(X_train, y_train)

# 4. 评估
# score = model.evaluate(X_test, y_test)
```

## ⚠️ 注意事项

1. **计算时间**：
   - 单个样本仿真时间：约3-10秒
   - 500个样本：约30-60分钟
   - 建议先用小样本测试（10-20个）

2. **内存使用**：
   - 温度场数据较大，注意可用内存
   - 可调整 `save_interval` 减少内存占用

3. **环境温度影响** ⭐：
   - `T_ambient` 对冷却速率影响显著
   - 建议生成不同环境温度的数据
   - 季节变化（夏天/冬天）会影响打印质量

4. **参数真实性**：
   - 建议根据实际打印机型号校准参数
   - 可通过实验测量实际轨迹误差
   - 热电偶测量实际温度分布

## 🐛 常见问题

### Q1: 运行时提示"Out of memory"
**A**: 减少 `num_samples` 或减小仿真网格尺寸：
```matlab
% 在 simulate_thermal_field.m 中
dx = 4;  % 增大网格间距（默认2）
```

### Q2: 轨迹误差过小/过大
**A**: 调整二阶系统参数：
```matlab
% 误差过大 → 增加刚度或阻尼
params.stiffness_x = 80000;  % N/m
params.damping_x = 30;       % N·s/m

% 误差过小 → 减小刚度或阻尼
params.stiffness_x = 30000;  % N/m
params.damping_x = 10;       % N·s/m
```

### Q3: 温度场计算不稳定
**A**: 减小时间步长或增大网格间距：
```matlab
% 在 simulate_thermal_field.m 中
dt = 0.005;  % 减小时间步长（默认0.01）
dx = 3;      % 增大网格间距（默认2）
```

### Q4: Python无法加载.mat文件
**A**: 确保使用MATLAB v7.3格式：
```matlab
save(filename, '-v7.3');  % 代码中已包含
```

## 📚 参考文献

1. **轨迹误差**：
   - 二阶系统控制理论
   - 传动系统弹性建模

2. **温度场**：
   - 移动热源理论（Rosenthal, 1941）
   - 有限差分法求解热传导方程

3. **粘结力**：
   - 分子扩散理论（Wool, 1995）
   - PLA材料热物性参数

## 🤝 贡献

如有问题或建议，请提交issue或pull request。

## 📄 许可

本项目代码仅供学术研究使用。

---

**最后更新**：2025-01-27
**作者**：自动生成
**联系方式**：通过GitHub Issues联系
