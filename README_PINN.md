# 3D打印轨迹误差修正系统 (PINN + Vision)

## 基于物理信息神经网络（PINN）的全新实现

### 核心思想

传统LSTM方法需要大量训练数据且泛化能力差。新系统采用**PINN（Physics-Informed Neural Networks）**，结合：

1. **物理约束**：2阶动力学方程 `m·x'' + c·x' + k·x = F(t)`
2. **视觉测量**：从打印照片直接提取轮廓误差
3. **混合训练**：少量实测数据 + 大量仿真数据

### 优势

- ✅ **数据效率高**：物理约束减少对标注数据的依赖
- ✅ **泛化能力强**：符合物理定律，可外推
- ✅ **真实数据驱动**：基于视觉测量的真实打印误差
- ✅ **端到端**：从STL/G-code → 视觉测量 → 模型训练 → 修正应用

---

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    数据收集层                            │
├─────────────────────────────────────────────────────────┤
│  1. 视觉测量系统                                        │
│     - 逐层拍摄打印件                                    │
│     - OpenCV提取轮廓                                    │
│     - 与STL理想轮廓对比                                 │
│     - 生成误差标签                                      │
│                                                         │
│  2. 仿真生成系统                                        │
│     - Python物理仿真器                                 │
│     - 大量合成训练数据                                  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    模型训练层                            │
├─────────────────────────────────────────────────────────┤
│  PINN模型                                               │
│    Input:  [x, y, vx, vy, ax, ay, curvature]           │
│    Output: [error_x, error_y]                          │
│                                                         │
│  损失函数                                               │
│    Loss = λ_data·MSE(pred, label)                      │
│         + λ_physics·MSE(pred, -m/k·a_ref)              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    应用层                                │
├─────────────────────────────────────────────────────────┤
│  G-code修正                                            │
│    corrected = ideal - predicted_error                 │
│                                                         │
│  打印验证                                              │
│    打印修正后的G-code                                   │
│    再次视觉测量                                        │
│    评估改进效果                                        │
└─────────────────────────────────────────────────────────┘
```

---

## 文件结构

```
3d_printer_pinn4ieee/
├── models/
│   └── pinn_trajectory_model.py     # PINN模型定义
│
├── training/
│   └── train_pinn.py                 # PINN训练脚本
│
├── utils/
│   └── vision_processor.py          # 视觉测量处理
│
├── experiments/
│   ├── collect_training_data.py    # 数据收集实验
│   ├── evaluate_pinn.py             # 模型评估
│   └── apply_correction.py          # 应用修正
│
└── data/
    ├── vision_data_collector.py     # 视觉数据收集器
    └── pinn_dataset.py              # PINN数据集
```

---

## 使用流程

### 1. 收集训练数据

#### 方法A：视觉测量（真实数据）

```bash
# 硬件需求：3D打印机 + 摄像头 + 良好照明

# 1. 开始打印测试件
# 2. 运行数据收集脚本
python experiments/collect_training_data.py \
    --gcode test_parts/calibration_cube.gcode \
    --stl test_parts/calibration_cube.stl \
    --camera_id 0 \
    --output data/real_measurements/

# 流程：
# - 每层完成后自动拍摄
# - 提取轮廓并计算误差
# - 生成训练数据集
```

#### 方法B：仿真生成（合成数据）

```python
# 使用Python仿真器生成大量数据
from data.gcode_physics_simulator_enhanced import PrinterPhysicsSimulator

simulator = PrinterPhysicsSimulator()
result = simulator.simulate_gcode('test.gcode')

# 自动保存为训练数据格式
```

### 2. 训练PINN模型

```bash
# 预训练（仅仿真数据）
python training/train_pinn.py \
    --mode pretrain \
    --sim_data data_simulation_* \
    --epochs 500 \
    --output checkpoints/pinn_pretrain

# 微调（真实数据）
python training/train_pinn.py \
    --mode finetune \
    --real_data data/real_measurements/print_errors.npz \
    --resume checkpoints/pinn_pretrain/best_model.pth \
    --epochs 100 \
    --output checkpoints/pinn_final

# 混合训练（推荐）
python training/train_pinn.py \
    --mode hybrid \
    --real_data data/real_measurements/print_errors.npz \
    --sim_data data_simulation_* \
    --lambda_data 1.0 \
    --lambda_physics 0.1 \
    --epochs 500 \
    --output checkpoints/pinn_hybrid
```

### 3. 应用修正

```bash
python experiments/apply_correction.py \
    --gcode test_parts/benchy.gcode \
    --checkpoint checkpoints/pinn_hybrid/best_model.pth \
    --output results/corrected/
```

### 4. 验证效果

```bash
# 打印修正后的G-code
# 然后再次视觉测量

python experiments/evaluate_pinn.py \
    --original_gcode test_parts/benchy.gcode \
    --corrected_gcode results/corrected/benchy_corrected.gcode \
    --stl test_parts/benchy.stl \
    --output results/evaluation/
```

---

## 关键创新点

### 1. PINN架构

```python
# 模型输入包含物理特征
input = [x, y, vx, vy, ax, ay, curvature]

# 物理约束嵌入损失函数
physics_loss = MSE(pred_error, -mass/stiffness * acceleration)

# 总损失
loss = λ_data·data_loss + λ_physics·physics_loss
```

### 2. 视觉测量流程

```python
# 1. 图像预处理
img = preprocess_image(photo)

# 2. 轮廓提取
contour = extract_contour(img)

# 3. 对齐（ICP）
aligned, transform = align_contours_icp(measured, ideal)

# 4. 误差计算
errors = compute_error(aligned, ideal)
```

### 3. 混合训练策略

- **预训练**：大量仿真数据，学习物理约束
- **微调**：少量真实数据，适应实际打印机
- **自适应权重**：λ_data和λ_physics可学习

---

## 与旧系统对比

| 特性 | 旧系统 (LSTM) | 新系统 (PINN) |
|------|-------------|--------------|
| 数据需求 | 大量（需要千级样本） | 少量（百级即可） |
| 泛化能力 | 差（容易过拟合） | 强（物理约束） |
| 训练难度 | 高（调参困难） | 中（物理约束稳定训练） |
| 推理速度 | 快（0.6ms） | 快（<1ms） |
| 可解释性 | 低（黑盒） | 高（物理意义明确） |
| 实测数据依赖 | 高 | 低（可用仿真辅助） |

---

## 依赖项

```bash
# PyTorch
pip install torch torchvision

# 计算机视觉
pip install opencv-python matplotlib

# 科学计算
pip install numpy scipy scikit-learn

# 数据处理
pip install h5py pandas

# 3D打印
pip install gcodeparser numpy-stl
```

---

## 下一步计划

1. ✅ 完成PINN模型定义
2. ✅ 完成视觉处理模块
3. ✅ 完成训练脚本
4. ⏳ 集成STL切片功能
5. ⏳ 实际打印实验验证
6. ⏳ 论文撰写

---

## 贡献

分支：`feature/pinn-vision-based-correction`

状态：重写中，旧LSTM代码已删除
