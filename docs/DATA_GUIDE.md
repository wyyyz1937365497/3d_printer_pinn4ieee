# 数据收集与使用指南

## 🎯 数据流程概述

```
G-code文件（针对Ender 3 V2切片）
    ↓
【MATLAB仿真】单参数配置（Ender 3 V2参数）
    ├─ 轨迹提取（每层1次）
    ├─ 误差仿真（GPU加速）
    ├─ 温度场仿真
    └─ 保存结果
    ↓
【数据文件】*.mat
    ├─ trajectory_data（轨迹）
    ├─ error_vectors（误差）
    └─ params（参数）
    ↓
【Python转换】convert_to_trajectory_features.py
    ├─ 提取轨迹特征（29维）
    ├─ 归一化
    └─ 合并所有数据
    ↓
【训练数据】trajectory_data.h5
    ├─ features: [N, 29]
    └─ targets: [N, 2]
    ↓
【模型训练】Transformer+BiLSTM
    ├─ 输入：轨迹特征序列
    ├─ 输出：偏移向量预测
    └─ 应用：路径补偿
```

---

## 📋 步骤1：数据收集

### Ender 3 V2参数配置

既然gcode是针对Ender 3 V2切片的，使用其真实参数（已在`physics_parameters.m`中配置）：

```matlab
% Ender 3 V2 真实参数（已在physics_parameters.m中）
params.motion.max_accel = 500;      % mm/s²（默认）
params.motion.max_velocity = 500;   % mm/s（最大速度）
params.motion.max_jerk = 10;        % mm/s³（急停限制）
params.dynamics.x.mass = 0.485;     % kg（X轴质量）
params.dynamics.y.mass = 0.650;     % kg（Y轴质量）
params.dynamics.x.stiffness = 150000; % N/m（皮带刚度）
params.dynamics.y.stiffness = 150000; % N/m
params.heat_transfer.h_convection_with_fan = 44;  % W/(m²·K)（风扇冷却）
params.environment.ambient_temp = 25;  % °C（室温）
params.printing.nozzle_temp = 220;   % °C（PLA）
```

### 运行数据收集

```bash
cd F:\TJ\3d_print\3d_printer_pinn4ieee

# 使用单参数配置收集数据
matlab -batch "collect_data_single_param" 2>&1 | tee data_collection.log
```

**预期结果：**
- 3DBenchy: 5层 × 1次 = 5次仿真
- 圆柱: 1层 × 1次 = 1次仿真
- 螺旋: 1层 × 1次 = 1次仿真
- **总计：7次仿真，约10-15分钟**

---

## 📋 步骤2：数据转换

### 转换为训练数据格式

```bash
python matlab_simulation/convert_to_trajectory_features.py \
    data_simulation_* \
    -o trajectory_data.h5
```

### 输入特征（29维）

| 类别 | 特征 | 维度 |
|------|------|------|
| 位置 | x, y, z | 3 |
| 速度 | vx, vy, vz, v_mag | 4 |
| 加速度 | ax, ay, az, a_mag | 4 |
| 加加速度 | jx, jy, jz, jerk_mag | 4 |
| 曲率 | curvature | 1 |
| 方向 | vx_norm, vy_norm, ax_norm, ay_norm | 4 |
| 相对位置 | dx_next, dy_next, dist_next, dist_prev | 4 |
| 变化率 | speed_change, direction_change | 2 |
| 标志 | is_corner, is_extruding, time | 3 |

**关键设计：**
- ✅ 不包含系统参数（推理时无法获得）
- ✅ 包含历史信息（时间序列）
- ✅ 物理信息完整（动量、转角、加速度）

### 输出目标（2维）

- `error_x`: X方向偏移（mm）
- `error_y`: Y方向偏移（mm）

---

## 📋 步骤3：验证数据

```bash
python matlab_simulation/test_conversion.py
```

生成可视化：
- `test_output/trajectory_overview.png` - 轨迹概览
- `test_output/feature_correlations.png` - 特征相关性
- `test_output/error_heatmap.png` - 误差热图

---

## 📋 步骤4：训练模型

### 数据加载示例

```python
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    def __init__(self, h5_file, sequence_length=10):
        with h5py.File(h5_file, 'r') as f:
            self.features = f['features'][:]  # [N, 29]
            self.targets = f['targets'][:]    # [N, 2]
        self.seq_len = sequence_length

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        return (
            self.features[idx:idx+self.seq_len],
            self.targets[idx+self.seq_len//2]
        )

# 使用
dataset = TrajectoryDataset('trajectory_data.h5', sequence_length=10)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 模型训练

```python
# 您的HybridDisplacementPredictor
model = HybridDisplacementPredictor(
    input_size=29,
    d_model=128,
    nhead=8,
    num_layers=2,
    output_size=2,
    sequence_length=10,
    dropout=0.1
)

for features, targets in dataloader:
    # features: [batch, 10, 29]
    # targets: [batch, 2]
    predictions = model(features)
    loss = criterion(predictions, targets)
    # ... 训练逻辑
```

---

## 🔧 高级选项

### 归一化方法

```bash
# 标准归一化（默认，Z-score）
python matlab_simulation/convert_to_trajectory_features.py \
    data_simulation_* -o trajectory_data.h5

# 鲁棒归一化（对异常值不敏感）
python matlab_simulation/convert_to_trajectory_features.py \
    data_simulation_* --norm-method robust -o trajectory_data_robust.h5

# Min-max归一化
python matlab_simulation/convert_to_trajectory_features.py \
    data_simulation_* --norm-method minmax -o trajectory_data_minmax.h5
```

### 不归一化

```bash
python matlab_simulation/convert_to_trajectory_features.py \
    data_simulation_* --no-normalize -o trajectory_data_raw.h5
```

---

## 📊 数据统计检查

运行转换后，检查输出确保正确：

```
Found 7 .mat files
Converting 7 MATLAB files to HDF5...

Dataset statistics:
  Total samples: ~8,500
  Feature dimension: 29
  Target dimension: 2

Target (error) statistics:
  X error: mean=0.000123, std=0.000456
  Y error: mean=-0.000089, std=0.000678
  Error magnitude: mean=0.000567, std=0.000345

✓ Conversion complete!
```

---

## ⚠️ 常见问题

### Q: 为什么只用单参数？

**A:** 因为gcode是针对Ender 3 V2切片的，打印机参数已经隐含在gcode的路径规划中。仿真的目的是用Ender 3 V2的物理参数计算这个gcode会产生什么误差，不是研究不同参数的影响。

### Q: 轨迹特征如何隐含参数信息？

**A:**
- 速度大小 → 隐含max_velocity约束
- 加速度大小 → 隐含max_accel约束
- 急转弯 → 需要高加速度

### Q: 为什么不直接用.mat文件？

**A:** HDF5格式：
- 读取速度更快
- 压缩率更好
- 支持部分加载
- 跨平台兼容

### Q: 如何处理不同长度的轨迹？

**A:** 使用滑动窗口（固定长度）：
```python
# 训练时：固定长度窗口
sequence_length = 10
features = data[i:i+sequence_length]  # [10, 29]
```

---

## 🎯 总结

### 关键原则

1. ✅ **单参数配置**：使用Ender 3 V2真实参数
2. ✅ **轨迹特征输入**：不包含系统参数
3. ✅ **偏移向量输出**：error_x, error_y
4. ✅ **时间序列格式**：支持Transformer+BiLSTM

### 文件清单

- `collect_data_single_param.m` - 数据收集（单参数）
- `convert_to_trajectory_features.py` - 数据转换
- `test_conversion.py` - 验证脚本
- `trajectory_data.h5` - 最终训练数据

---

## 📚 相关文档

- `DIVERSITY_RECOMMENDATIONS.md` - 为什么需要多个gcode文件
- `docs/TECHNICAL_DOCUMENTATION.md` - 仿真技术细节
