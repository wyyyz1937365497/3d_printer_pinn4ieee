# 3D Printer PINN - Physics-Informed Neural Network for Quality Prediction

基于物理信息的神经网络，用于FDM 3D打印质量预测和轨迹校正。

## 项目概述

本项目使用MATLAB仿真生成训练数据，Python训练PINN模型，实现：
- **轨迹误差预测**（基于二阶动力学系统）
- **质量特征预测**（内应力、孔隙率、尺寸精度等）
- **层间粘结强度预测**（基于热传导模型）
- **打印质量实时评估**

**关键特性**:
- 基于物理的仿真（Ender-3 V2 + PLA材料参数）
- GPU加速数据生成（2-5倍效率提升）
- 优化的数据生成策略（单层参数扫描 + 三层验证）
- 完整的MATLAB→Python数据转换流程

---

## 快速开始

### 1. 生成训练数据（MATLAB）

```matlab
cd('F:\TJ\3d_print\3d_printer_pinn4ieee')
collect_data
```

**预期结果**:
- ~109,000 训练样本
- 仿真时间: ~1.5 小时（GPU）

### 2. 训练模型（Python）

```bash
# 隐式状态推断模型（TCN）
python experiments/train_implicit_state_tcn.py \
    --data_dir data_simulation_3DBenchy_PLA_1h28m_layers_* \
    --epochs 50 \
    --batch_size 64

# 轨迹误差修正模型
python experiments/train_trajectory_correction.py \
    --data_dir data_simulation_3DBenchy_PLA_1h28m_layers_* \
    --epochs 50 \
    --batch_size 64
```

### 3. 评估模型

```bash
python experiments/evaluate_implicit_state_tcn.py \
    --model_path checkpoints/implicit_state_tcn/best_model.pth \
    --data_path data_simulation_3DBenchy_PLA_1h28m_layers_*/layer*.mat

python experiments/evaluate_trajectory_model.py \
    --model_path checkpoints/trajectory_correction/best_model.pth \
    --data_path data_simulation_3DBenchy_PLA_1h28m_layers_*/layer*.mat
```

---

## 文档导航

### 📚 核心文档

| 文档 | 用途 |
|------|------|
| **[docs/SIMULATION_DATA_GUIDE.md](docs/SIMULATION_DATA_GUIDE.md)** | 完整使用指南（数据生成→训练→评估） |
| **[matlab_simulation/README.md](matlab_simulation/README.md)** | MATLAB仿真系统文档 |
| **[docs/THESIS_DOCUMENTATION.md](docs/THESIS_DOCUMENTATION.md)** | 论文写作参考（文献综述、理论） ⭐ |
| **[docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md)** | 完整技术文档（公式、算法） ⭐ |

### 📖 快速查找

- **完整工作流程** → [docs/SIMULATION_DATA_GUIDE.md](docs/SIMULATION_DATA_GUIDE.md)
- **MATLAB仿真详解** → [matlab_simulation/README.md](matlab_simulation/README.md)
- **Python数据加载** → [docs/SIMULATION_DATA_GUIDE.md](docs/SIMULATION_DATA_GUIDE.md#python数据加载)
- **模型训练** → [docs/SIMULATION_DATA_GUIDE.md](docs/SIMULATION_DATA_GUIDE.md#模型训练)
- **模型评估** → [docs/SIMULATION_DATA_GUIDE.md](docs/SIMULATION_DATA_GUIDE.md#模型评估)
- **论文写作材料** → [docs/THESIS_DOCUMENTATION.md](docs/THESIS_DOCUMENTATION.md) ⭐
- **物理公式推导** → [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md) ⭐

---

## 项目结构

```
.
├── matlab_simulation/           # MATLAB仿真系统
│   ├── physics_parameters.m     # 物理参数（Ender-3 V2 + PLA）
│   ├── parse_gcode_improved.m   # G-code解析器
│   ├── simulate_trajectory_error.m   # 轨迹误差模型
│   ├── simulate_trajectory_error_gpu.m # GPU加速版
│   ├── simulate_thermal_field.m  # 温度场模型
│   ├── run_full_simulation_gpu.m # GPU完整仿真流程
│   └── convert_matlab_to_python.py   # MATLAB→Python转换
│
├── collect_data_optimized.m     # ⭐ 优化数据收集入口
│
├── docs/                        # 文档
│   ├── QUICK_START.md          # 快速开始
│   ├── USER_GUIDE.md            # 完整使用指南
│   └── THESIS_DOCUMENTATION.md # 论文写作参考
│
└── experiments/                 # Python训练实验
    ├── train_implicit_state_tcn.py
    ├── train_trajectory_correction.py
    ├── evaluate_implicit_state_tcn.py
    └── evaluate_trajectory_model.py
```

---

## 核心特性

### 1. 基于物理的仿真

**轨迹误差模型**（二阶动力学系统）:
- 运动方程: `m·x'' + c·x' + k·x = F_inertia`
- 输出向量误差: `error_x`, `error_y`
- 考虑皮带弹性、惯性力、阻尼

**温度场模型**（移动热源）:
- 热传导: `∂T/∂t = α·∇²T + Q_source - Q_cooling`
- Wool-O'Connor聚合物愈合模型
- 层间粘结强度预测

### 2. GPU加速

- 使用cuda1（不影响cuda0上的训练）
- 4-13倍加速（数据量>10K点）
- 自动CPU fallback

### 3. 优化数据生成策略

**关键发现**: 所有层形状相同

**优化效果**:
- 仿真时间: 2-3天 → 1.5小时（**30-40倍提升**）
- 数据质量: 6.9倍样本提升
- 参数覆盖: 180种配置 vs 原来的少量配置

**策略**: 单层参数扫描 + 三层验证
- 几何多样性: 从单层提取
- 物理多样性: 从参数变化获得
- 层间效应: 三层验证覆盖

---

## 系统要求

### MATLAB

- MATLAB R2020a或更高版本
- Parallel Computing Toolbox（GPU加速）
- Signal Processing Toolbox（可选）
- 统计与机器学习工具箱（可选）

### Python

- Python 3.8+
- PyTorch 1.12+
- NumPy, SciPy, h5py, pandas

### 硬件

- GPU: 至少1个（推荐2个）
- 内存: 8GB+
- 存储: 10GB+（用于数据集）

---

## 性能基准

| 数据量 | CPU时间 | GPU时间 | 样本数 |
|--------|---------|---------|--------|
| 单次仿真 | ~2分钟 | ~30秒 | 3K点 |
| 优化模式 | - | 1.5小时 | 100K+ |
| 原策略 | 2-3天 | - | 4.2K |

---

## 典型工作流

```matlab
% === 1. 快速测试 (5分钟) ===
addpath('matlab_simulation')
data = run_full_simulation_gpu('Tremendous Hillar_PLA_17m1s.gcode', ...
                               'test.mat', [], 1);

% === 2. 检查结果 ===
load('test.mat');
fprintf('点数: %d, 最大误差: %.3f mm\n', ...
        length(simulation_data.time), ...
        max(simulation_data.error_magnitude));

% === 3. 生成完整数据集 (1.5小时) ===
collect_data_optimized
```

```bash
# === 4. 转换为Python格式 (10分钟) ===
python matlab_simulation/convert_matlab_to_python.py \
    "data_simulation_layer25/*.mat" \
    training \
    -o training_data
```

```python
# === 5. 训练模型 ===
python experiments/train_unified_model.py
```

---

## 常见问题

### Q: 为什么优化策略这么快？

A: 因为发现每一层形状相同，不需要仿真所有50层。单层已包含所有几何特征，参数扫描提供物理多样性。详见 [docs/USER_GUIDE.md](docs/USER_GUIDE.md) 优化策略。

### Q: 如何使用GPU加速？

A: 默认的`collect_data_optimized`已自动使用cuda1。手动使用：
```matlab
data = run_full_simulation_gpu('file.gcode', 'output.mat', [], 1);
```
详见 [docs/USER_GUIDE.md](docs/USER_GUIDE.md) GPU加速。

### Q: 需要多少数据？

A: 对于5.69M参数的模型，推荐80,000-100,000样本。优化模式一次生成109,200样本（含增强）。详见 [docs/THESIS_DOCUMENTATION.md](docs/THESIS_DOCUMENTATION.md)。

### Q: 如何修改物理参数？

A: 编辑`matlab_simulation/physics_parameters.m`或在脚本中覆盖：
```matlab
params = physics_parameters();
params.motion.max_accel = 400;
```
详见 [docs/USER_GUIDE.md](docs/USER_GUIDE.md) 参数配置。

### Q: 转换Python时出错？

A: 确保安装依赖：`pip install numpy scipy h5py pandas`。详见 [docs/USER_GUIDE.md](docs/USER_GUIDE.md) 故障排除。

---

## 引用

如果本项目对你有帮助，请引用：

```bibtex
@misc{3d_printer_pinn,
  title={Physics-Informed Neural Network for 3D Printer Quality Prediction},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/3d_printer_pinn4ieee}
}
```

---

## 许可证

MIT License

---

## 更新日志

### v1.0 (2026-01-27)
- ✅ 完整MATLAB仿真系统
- ✅ GPU加速支持（cuda1）
- ✅ 优化数据生成策略（30-40倍提升）
- ✅ MATLAB→Python数据转换
- ✅ 完整文档系统

---

**开始使用**: [docs/QUICK_START.md](docs/QUICK_START.md)

**完整指南**: [docs/USER_GUIDE.md](docs/USER_GUIDE.md)

**论文参考**: [docs/THESIS_DOCUMENTATION.md](docs/THESIS_DOCUMENTATION.md)
