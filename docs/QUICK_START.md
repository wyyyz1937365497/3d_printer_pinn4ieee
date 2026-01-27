# 快速开始指南

## 5分钟上手

### 最快的方式（推荐）

```matlab
cd('F:\TJ\3d_print\3d_printer_pinn4ieee')
collect_data_optimized
```

**预期结果**:
- 100,000+ 训练样本
- 时间: 1.5 小时
- GPU: cuda1

---

## 三种使用模式

### 模式1：优化数据生成 ⭐推荐

```matlab
collect_data_optimized
```

- **适用**: 所有层形状相同
- **效率**: 30-40倍提升
- **数据量**: 100K+ 样本
- **时间**: 1.5 小时
- **GPU**: cuda1

### 模式2：单次测试

```matlab
addpath('matlab_simulation')

% CPU版本
data = run_full_simulation('Tremendous Hillar_PLA_17m1s.gcode');

% GPU版本
data = run_full_simulation_gpu('Tremendous Hillar_PLA_17m1s.gcode', ...
                               'output.mat', [], 1);
```

- **适用**: 测试、调试
- **时间**: 30秒 - 2分钟
- **数据量**: 2000-5000点

### 模式3：参数扫描

```matlab
% 扫描加速度
accelerations = [200, 300, 400, 500];
for i = 1:length(accelerations)
    params = physics_parameters();
    params.motion.max_accel = accelerations(i);

    data = run_full_simulation_gpu('Tremendous Hillar_PLA_17m1s.gcode', ...
                                   sprintf('output_%d.mat', i), [], 1);
end
```

---

## MATLAB → Python 转换

### 安装依赖

```bash
pip install numpy scipy h5py pandas
```

### 转换单个文件

```bash
python matlab_simulation/convert_matlab_to_python.py \
    "Tremendous Hillar_PLA_17m1s_simulation.mat" \
    training \
    -o training_data
```

### 批量转换

```bash
python matlab_simulation/convert_matlab_to_python.py \
    "data_simulation_layer25/*.mat" \
    training \
    -o training_data_combined
```

---

## GPU配置

### 检查GPU状态

```matlab
% 查看GPU数量
gpuDeviceCount  % 应该输出2

% 查看cuda1信息
gpuDevice(1)    % 显示名称、内存等

% 检查可用内存
gpu = gpuDevice(1);
fprintf('可用内存: %.2f GB\n', gpu.FreeMemory / 1e9);
```

### GPU使用建议

| 数据量 | 推荐 | 原因 |
|--------|------|------|
| < 1000点 | CPU | 传输开销 > 计算收益 |
| 1000-10000点 | GPU或CPU | 性能接近 |
| \> 10000点 | **GPU** | 4-13倍加速 |

---

## 常用参数修改

### 修改运动参数

```matlab
params = physics_parameters();

% 最大加速度 (默认500)
params.motion.max_accel = 400;  % mm/s²

% 最大速度 (默认500)
params.motion.max_velocity = 300;  % mm/s

% Jerk限制 (默认8)
params.motion.jerk_limit = 10;  % mm/s
```

### 修改热学参数

```matlab
params = physics_parameters();

% 环境温度 (默认20)
params.environment.ambient_temp = 25;  % °C

% 风扇对流系数 (默认44)
params.heat_transfer.h_convection_with_fan = 44;  % W/(m²·K)

% 喷嘴温度 (默认210)
params.material.nozzle_temp = 210;  % °C
```

---

## 输出数据结构

### MATLAB .mat文件

```matlab
simulation_data =
    time: [1×3000 double]           % 时间 (s)
    x_ref: [1×3000 double]          % 参考X位置 (mm)
    y_ref: [1×3000 double]          % 参考Y位置 (mm)
    error_x: [1×3000 double]        % X误差向量 (mm)
    error_y: [1×3000 double]        % Y误差向量 (mm)
    error_magnitude: [1×3000 double] % 误差幅值 (mm)
    T_interface: [1×3000 double]    % 层间温度 (°C)
    adhesion_ratio: [1×3000 double] % 粘结强度比 (0-1)
    ... (50+ 个变量)
```

### Python HDF5文件

```python
import h5py

with h5py.File('training_data.h5', 'r') as f:
    # 输入特征
    x_ref = f['inputs/x_ref'][:]
    vx_ref = f['inputs/vx_ref'][:]
    corner_angle = f['inputs/corner_angle'][:]

    # 目标输出
    error_x = f['outputs/error_x'][:]
    error_y = f['outputs/error_y'][:]
    adhesion = f['outputs/adhesion_ratio'][:]
```

---

## 典型工作流

```matlab
% === 步骤1: 快速测试 (5分钟) ===
addpath('matlab_simulation')
data = run_full_simulation_gpu('Tremendous Hillar_PLA_17m1s.gcode', ...
                               'test.mat', [], 1);

% === 步骤2: 检查结果 ===
load('test.mat');
fprintf('点数: %d\n', length(simulation_data.time));
fprintf('最大误差: %.3f mm\n', max(simulation_data.error_magnitude));

% === 步骤3: 生成完整数据集 (1.5小时) ===
collect_data_optimized

% === 步骤4: 转换为Python (命令行) ===
% python matlab_simulation/convert_matlab_to_python.py ...
```

---

## 故障排除速查

| 问题 | 解决方法 |
|------|----------|
| 找不到G-code | 使用绝对路径或检查`pwd` |
| GPU不可用 | 安装Parallel Computing Toolbox或用CPU |
| GPU内存不足 | 减少层数或用CPU版本 |
| 仿真太慢 | 使用GPU (`run_full_simulation_gpu`) |
| 缺少Python模块 | `pip install numpy scipy h5py pandas` |
| 转换错误 | 确保MATLAB文件用`-v7.3`标志保存 |

---

## 优化策略核心

**关键发现**: 每一层形状相同

```
原策略: 15层 × 少量参数 = 2-3天
优化策略: 1层 × 180种参数 + 3层验证 = 1.5小时

效率提升: 30-40倍
数据质量: 更好（参数覆盖更全面）
```

---

## 文件导航

```
matlab_simulation/
├── physics_parameters.m           # 物理参数定义
├── parse_gcode_improved.m         # G-code解析器
├── simulate_trajectory_error.m    # 轨迹误差模型
├── simulate_thermal_field.m       # 温度场模型
├── run_full_simulation_gpu.m      # GPU完整仿真
└── convert_matlab_to_python.py    # MATLAB→Python转换

根目录/
└── collect_data_optimized.m       # ⭐ 优化数据收集入口

docs/
├── THESIS_DOCUMENTATION.md        # 论文文献和理论
├── USER_GUIDE.md                  # 完整使用指南
└── QUICK_START.md                 # 本文件
```

---

## 下一步

1. **测试系统** (5分钟):
   ```matlab
   addpath('matlab_simulation')
   data = run_full_simulation_gpu('Tremendous Hillar_PLA_17m1s.gcode', 'test.mat', [], 1)
   ```

2. **生成数据** (1.5小时):
   ```matlab
   collect_data_optimized
   ```

3. **转换数据** (10分钟):
   ```bash
   python matlab_simulation/convert_matlab_to_python.py ...
   ```

4. **开始训练**:
   ```python
   python experiments/train_unified_model.py
   ```

---

**需要更多细节?** 查看 `docs/USER_GUIDE.md`

**论文写作?** 查看 `docs/THESIS_DOCUMENTATION.md`
