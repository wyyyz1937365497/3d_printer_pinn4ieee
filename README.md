# 3D Printer PINN - Real-Time Trajectory Error Correction

基于物理信息的神经网络，用于FDM 3D打印实时轨迹误差预测与补偿。

## 项目概述

本项目使用MATLAB仿真生成训练数据，Python训练轻量级LSTM模型，实现：
- **实时轨迹误差预测**（基于二阶动力学系统）
- **轻量级网络架构**（38K参数，<1ms推理）
- **固件级误差建模**（junction deviation, microstepping resonance, timer jitter）

**关键特性**:
- 基于物理的仿真（Ender-3 V2 + PLA材料参数）
- GPU加速数据生成（10-13倍效率提升）
- 纯LSTM架构（4维输入→2维输出，单步预测）
- 完整的MATLAB→Python训练→评估流程

---

## 快速开始

### 1. 生成训练数据（MATLAB）

```matlab
% 采样策略：每5层采样一次（推荐）
collect_3dbenchy('sampled:5');   % ~10 layers
collect_bearing5('sampled:5');   % ~10 layers
collect_nautilus('sampled:5');   % ~10 layers
collect_boat('sampled:5');       % ~10 layers
```

**预期结果**:
- ~40层 × 2分钟/层 = ~80分钟
- ~36,000样本点
- RMS误差: ~140 μm（固件增强仿真）

### 2. 准备训练数据（Python）

```bash
python data/scripts/prepare_training_data.py \
    --data_dirs data_simulation_* \
    --output_dir data/processed \
    --sequence_length 20 \
    --stride 4
```

**输出格式**:
- 特征: [batch, 20, 4] - [x_ref, y_ref, vx_ref, vy_ref]
- 标签: [batch, 2] - [error_x, error_y]

### 3. 训练模型（Python）

```bash
python experiments/train_realtime.py \
    --data_root data/processed \
    --batch_size 256 \
    --epochs 100 \
    --lr 1e-3 \
    --device cuda:0
```

**训练特性**:
- 混合精度训练 (FP16)
- AdamW优化器 + 余弦退火调度
- 早停机制 (patience=15)

**预期结果**:
- 训练时间: ~2小时 (GPU)
- MAE: ~0.015 mm
- R²: ~0.89

### 4. 评估模型

```bash
python experiments/evaluate_realtime.py \
    --checkpoint checkpoints/realtime_corrector/best_model.pth \
    --data_root data/processed/test
```

---

## 文档导航

### 📚 完整文档索引

详见: **[docs/README.md](docs/README.md)** ⭐

### 核心文档分类

**📘 理论基础** ([docs/theory/](docs/theory/))
- [公式库](docs/theory/formulas.md) ⭐ - 所有物理方程和LaTeX代码
- [轨迹动力学](docs/theory/trajectory_dynamics.md) - 二阶系统建模

**📗 方法实现** ([docs/methods/](docs/methods/))
- [仿真系统](docs/methods/simulation_system.md) - MATLAB仿真架构
- [固件效应](docs/methods/firmware_effects.md) - Marlin固件误差源
- [数据生成](docs/methods/data_generation.md) - 训练数据生成策略
- [神经网络](docs/methods/neural_network.md) - LSTM架构设计
- [训练流程](docs/methods/training_pipeline.md) - 端到端训练指南

**📙 实验设计** ([docs/experiments/](docs/experiments/))
- [实验设置](docs/experiments/setup.md) - 打印机配置和参数
- [数据集](docs/experiments/datasets.md) - 数据统计和格式
- [评估指标](docs/experiments/metrics.md) - 性能评估方法

**✍️ 论文写作** ([docs/writing/](docs/writing/))
- [结构模板](docs/writing/structure_template.md) ⭐ - IEEE论文模板
- [章节模板](docs/writing/section_templates/) - 各章节写作模板
- [LaTeX资源](docs/writing/latex/) - 自定义命令和参考文献
- [句式库](docs/writing/phrase_bank/) - 写作句式参考

---

## 系统架构

### 1. 物理仿真（MATLAB）

**二阶动力学系统**:
```
m·ẍ + c·ẋ + k·x = -m·a_ref(t)
```

**固件级误差建模**:
- Junction Deviation（转角偏差）
- Microstep Resonance（步进共振）
- Timer Jitter（定时器抖动）

**输出**: 参考轨迹 + 误差向量

### 2. 神经网络（PyTorch）

**轻量级LSTM架构**:
```
输入 [20, 4] → 编码器(32) → LSTM(56×2) → 输出(2)
```

**性能指标**:
- 参数量: ~38K
- 推理时间: 0.3-0.6 ms
- 满足实时要求 (< 1ms)

### 3. 训练流程

```
仿真数据 → 预处理 → 划分 → 训练 → 验证 → 测试
```

---

## 项目结构

```
3d_printer_pinn4ieee/
├── simulation/                 # MATLAB仿真系统
│   ├── +planner/              # 轨迹规划模块
│   ├── +stepper/              # 固件效应模块
│   ├── physics_parameters.m   # 物理参数配置
│   └── run_simulation.m       # 仿真入口
│
├── data/                      # Python数据处理
│   ├── realtime_dataset.py    # 4维数据集
│   └── scripts/               # 数据准备脚本
│
├── models/                    # 神经网络模型
│   └── realtime_corrector.py  # LSTM预测器
│
├── config/                    # 配置文件
│   └── realtime_config.py     # 训练配置
│
├── experiments/               # 训练和评估
│   ├── train_realtime.py      # 训练脚本
│   ├── evaluate_realtime.py   # 评估脚本
│   └── visualize_realtime.py  # 可视化脚本
│
├── docs/                      # 完整文档系统 ⭐
│   ├── theory/                # 理论文档
│   ├── methods/               # 方法论文档
│   ├── experiments/           # 实验文档
│   ├── writing/               # 论文写作资源
│   └── archives/              # 归档文档
│       ├── guides/            # 临时指南
│       ├── history/           # 历史记录
│       └── chinese_notes/     # 中文笔记
│
├── checkpoints/               # 模型检查点
│   └── realtime_corrector/    # 实时修正模型
│
├── evaluation_results/        # 评估结果
└── results/                   # 可视化输出
```

---

## 关键参数

### 物理参数（Ender-3 V2）

| 参数 | X轴 | Y轴 | 单位 |
|------|-----|-----|------|
| 质量 | 0.485 | 0.650 | kg |
| 刚度 | 150,000 | 150,000 | N/m |
| 阻尼 | 25 | 25 | N·s/m |

### 网络参数

| 参数 | 值 | 说明 |
|------|-----|------|
| input_size | 4 | [x_ref, y_ref, vx_ref, vy_ref] |
| hidden_size | 56 | LSTM隐藏单元 |
| num_layers | 2 | LSTM层数 |
| seq_len | 20 | 序列长度 (0.2s @ 100Hz) |
| pred_len | 1 | 单步预测 |
| dropout | 0.1 | Dropout比例 |

### 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 256 | 批大小 |
| epochs | 100 | 训练轮数 |
| lr | 1e-3 | 初始学习率 |
| weight_decay | 1e-4 | L2正则化 |

---

## 性能基准

### 数据生成

| 场景 | 层数 | 时间 | 样本数 |
|------|------|------|--------|
| 单模型 | ~10 | ~20 min | ~9,000 |
| 4模型 | ~40 | ~80 min | ~36,000 |

### 模型性能

| 指标 | 目标 | 实测 |
|------|------|------|
| 参数量 | < 50K | ~38K |
| 推理时间 | < 1ms | 0.3-0.6ms |
| MAE | < 0.02mm | ~0.015mm |
| R² | > 0.8 | ~0.89 |

### 训练效率

| 硬件 | 批大小 | 每轮时间 | 总时间 |
|------|--------|----------|--------|
| GPU (RTX 3080) | 256 | ~1 min | ~2小时 |
| GPU (GTX 1080) | 256 | ~2 min | ~3小时 |

---

## 系统要求

### MATLAB（数据生成）

- MATLAB R2020a或更高
- Parallel Computing Toolbox（GPU加速，可选）
- 推荐配置：8GB RAM，GPU（8GB+ VRAM）

### Python（模型训练）

- Python 3.8+
- PyTorch 1.10+
- NumPy, SciPy, Pandas, h5py
- Matplotlib, Seaborn（可视化）
- TensorBoard（可选，监控训练）

### 硬件要求

**最小配置**:
- CPU: 4核
- RAM: 8 GB
- GPU: 无（CPU模式，10×慢）

**推荐配置**:
- CPU: 8核
- RAM: 16 GB
- GPU: GTX 1080或更好（8GB VRAM）

**理想配置**:
- CPU: 16核
- RAM: 32 GB
- GPU: RTX 3080或更好（10GB+ VRAM）

---

## 引用

如果本项目对你有帮助，请引用：

```bibtex
@misc{3d_printer_pinn_realtime,
  title={Physics-Informed Neural Network for Real-Time Trajectory Error Correction in FDM 3D Printing},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/3d_printer_pinn4ieee}
}
```

---

## 许可证

MIT License

---

## 快速链接

- 📖 **完整文档**: [docs/README.md](docs/README.md)
- 🚀 **快速开始**: [docs/archives/quick_ref/QUICK_START_ENHANCED.md](docs/archives/quick_ref/QUICK_START_ENHANCED.md)
- 🔧 **仿真系统**: [docs/methods/simulation_system.md](docs/methods/simulation_system.md)
- 🧠 **网络架构**: [docs/methods/neural_network.md](docs/methods/neural_network.md)
- 📊 **训练流程**: [docs/methods/training_pipeline.md](docs/methods/training_pipeline.md)
- ✍️ **论文写作**: [docs/writing/structure_template.md](docs/writing/structure_template.md)

---

**当前分支**: `feature/realtime-correction`

**最后更新**: 2026-02-02
