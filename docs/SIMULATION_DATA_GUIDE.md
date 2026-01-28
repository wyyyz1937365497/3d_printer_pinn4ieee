# 仿真数据生成与训练完整指南

**最后更新**: 2026-01-28
**版本**: 3.0

---

## 📋 目录

1. [系统架构总览](#系统架构总览)
2. [MATLAB数据生成](#matlab数据生成)
3. [Python数据加载](#python数据加载)
4. [模型训练](#模型训练)
5. [完整工作流程](#完整工作流程)

---

## 系统架构总览

### 数据流

```
┌─────────────────────┐
│   G-code文件        │
│  (Tremendous Hillar) │
└──────────┬──────────┘
           │
           ↓
┌──────────────────────────────────────┐
│  MATLAB仿真 (collect_data.m)         │
│  1. 轨迹重建 (reconstruct_trajectory) │
│  2. 热场仿真 (simulate_thermal)       │
│  3. 质量评估 (calculate_quality) ✨NEW │
│     ← 仅基于理想轨迹+热场             │
│  4. 误差仿真 (simulate_trajectory)    │
│     ← 动力学仿真产生误差              │
└──────────────────┬───────────────────┘
                   │
                   ↓
          ┌─────────────────┐
          │  .mat 文件       │
          │  (完整仿真数据)   │
          └────────┬────────┘
                   │
                   ↓
┌──────────────────────────────────────┐
│  Python数据加载 (dataset.py) ✨NEW   │
│  ├─ 12个输入特征（理想轨迹+显式测量） │
│  ├─ 2个误差向量输出（动力学仿真）     │
│  └─ 5个质量特征输出（理想轨迹计算）   │
└──────────────────┬───────────────────┘
                   │
                   ↓
┌──────────────────────────────────────┐
│  PINN模型训练                         │
│  ├─ Trajectory Correction Head       │
│  └─ Quality Prediction Head          │
└──────────────────────────────────────┘
```

**关键设计原则**:
- 质量特征（adhesion, stress, porosity, accuracy, score）基于**理想轨迹**计算
- 误差向量（error_x, error_y）由**动力学仿真**产生
- 两者独立计算，同时作为神经网络的学习目标

---

## MATLAB数据生成

### 1️⃣ 运行完整仿真

```matlab
% 在MATLAB中运行
cd('F:\TJ\3d_print\3d_printer_pinn4ieee')
collect_data
```

**输出**:
- `data_simulation_layer25/` - 100个参数配置的仿真
- `validation_layer1/`, `validation_layer25/`, `validation_layer50/` - 验证数据

### 2️⃣ 生成的数据字段

#### ✅ 输入特征（12个）

**理想轨迹** (6个):
- `x_ref, y_ref, z_ref` - 参考位置
- `vx_ref, vy_ref, vz_ref` - 参考速度

**显式测量量** (6个):
- `T_nozzle` - 喷嘴温度
- `T_interface` - 层间界面温度
- `F_inertia_x, F_inertia_y` - 惯性力
- `cooling_rate` - 冷却速率
- `layer_num` - 层号

#### ✅ 输出标签

**误差向量** (2个):
- `error_x, error_y` - X/Y误差

**质量特征** (5个):
- `adhesion_ratio` - 粘结强度比 (0-1)
- `internal_stress` - 内应力 (MPa)
- `porosity` - 孔隙率 (0-100%)
- `dimensional_accuracy` - 尺寸误差 (mm)
- `quality_score` - 综合质量评分 (0-1)

---

## Python数据加载

### 1️⃣ 快速开始

```python
from data.simulation import create_dataloaders

# 创建dataloaders
train_loader, val_loader, test_loader, scaler = create_dataloaders(
    train_dir='data_simulation_layer25',
    val_dir='validation_layer25',
    test_dir='validation_layer50',
    batch_size=64,
    seq_len=200,
    pred_len=50,
    stride=10
)

# 训练循环
for batch in train_loader:
    inputs = batch['input_features']          # [batch, seq_len, 12]
    trajectory_targets = batch['trajectory_targets']  # [batch, pred_len, 2]
    quality_targets = batch['quality_targets']        # [batch, 5]

    # 训练模型...
```

### 2️⃣ 数据集类

```python
from data.simulation import PrinterSimulationDataset

# 创建数据集
dataset = PrinterSimulationDataset(
    data_files='data_simulation_layer25/*.mat',
    seq_len=200,      # 输入序列长度
    pred_len=50,      # 预测序列长度
    stride=10,        # 序列滑动步长
    mode='train',
    scaler=None,      # 自动fit scaler
    fit_scaler=True
)

# 访问样本
sample = dataset[0]
print(sample.keys())
# dict_keys(['input_features', 'trajectory_targets', 'quality_targets'])
```

---

## 模型训练

### 1️⃣ 快速训练（测试）

```bash
python experiments/quick_train_simulation.py \
    --data_dir data_simulation_layer25 \
    --epochs 10 \
    --batch_size 32
```

### 2️⃣ 完整训练

```bash
python experiments/train_unified_model.py \
    --config unified \
    --data_dir data_simulation_layer25 \
    --epochs 100 \
    --batch_size 64
```

### 3️⃣ 自定义训练

```python
from models import UnifiedPINNSeq3D
from config import get_config
from training import Trainer
from data.simulation import create_dataloaders

# 1. 加载配置
config = get_config(preset='unified')

# 2. 创建dataloaders
train_loader, val_loader, test_loader, scaler = create_dataloaders(
    train_dir='data_simulation_layer25',
    val_dir='validation_layer25',
    test_dir='validation_layer50',
    batch_size=config.training.batch_size,
    seq_len=config.data.seq_len,
    pred_len=config.data.pred_len,
    stride=config.data.stride
)

# 3. 创建模型
model = UnifiedPINNSeq3D(config.model)

# 4. 创建trainer
trainer = Trainer(model, config)

# 5. 训练
history = trainer.train(
    train_loader,
    val_loader,
    epochs=config.training.num_epochs,
    save_dir='checkpoints/my_experiment'
)
```

---

## 完整工作流程

### Step 1: 生成MATLAB仿真数据

```matlab
% 在MATLAB中
collect_data
```

**时间**: ~1.5小时（GPU）
**输出**: ~100个.mat文件

### Step 2: 转换为Python格式（可选）

```bash
python data/scripts/prepare_training_data.py \
    --mat_dir data_simulation_layer25 \
    --output_dir data/processed
```

### Step 3: 训练模型

```bash
python experiments/quick_train_simulation.py \
    --data_dir data_simulation_layer25 \
    --epochs 100 \
    --batch_size 64
```

### Step 4: 评估模型

```bash
python experiments/evaluate_model.py \
    --checkpoint checkpoints/quick_train/best_model.pth \
    --test_dir validation_layer50
```

---

## 新增功能说明

### ✨ 质量特征计算（NEW）

**文件**: `matlab_simulation/calculate_quality_metrics.m`

**功能**:
- 计算内应力（基于热收缩模型）
- 估算孔隙率（温度+速度+粘结度）
- 评估尺寸精度（误差+热收缩）
- 综合质量评分（加权组合）

**调用方式**:
```matlab
quality_data = calculate_quality_metrics(trajectory_data, thermal_data, params);
```

### ✨ Python数据集类（NEW）

**文件**: `data/simulation/dataset.py`

**功能**:
- 直接加载MATLAB .mat文件
- 自动归一化（StandardScaler）
- 滑动窗口序列生成
- 与PyTorch DataLoader无缝集成

### ✨ 配置文件更新（NEW）

**文件**: `config/base_config.py`

**更新内容**:
- 明确定义12个输入特征
- 明确定义2个轨迹输出
- 明确定义5个质量输出

---

## 常见问题

### Q1: 如何使用已生成的.mat文件？

**A**: 直接在Python中加载：
```python
from data.simulation import PrinterSimulationDataset

dataset = PrinterSimulationDataset('data_simulation_layer25/*.mat')
```

### Q2: 缺少质量特征怎么办？

**A**: Dataset会自动处理：
- 如果MATLAB文件有新特征，使用它们
- 如果没有，设置为0或从现有特征推导

### Q3: 如何调整输入特征？

**A**: 编辑以下文件：
1. `data/simulation/dataset.py` - 修改`INPUT_FEATURES`列表
2. `config/base_config.py` - 修改`input_features`列表
3. 确保两边一致！

### Q4: 数据增强怎么做？

**A**: 在`dataset.py`中的`_create_sequences`方法中添加：
```python
# 可选：添加噪声
noise = np.random.normal(0, 0.01, input_features.shape)
input_features = input_features + noise
```

---

## 数据统计

| 指标 | 值 |
|------|-----|
| 原始G-code点数 | 33点/层 |
| 重建后点数 | 2000-5000点/层 |
| 采样率 | 100 Hz |
| 输入特征数 | 12 |
| 输出特征数 | 7 (2误差+5质量) |
| 训练样本数 | ~109,200 (含增强) |

---

## 模型评估

### 评估指标

#### 轨迹误差预测 (2个输出)
```python
# 误差向量预测
- RMSE (Root Mean Squared Error): 目标 < 0.05 mm
- MAE (Mean Absolute Error): 目标 < 0.03 mm
- 最大误差: 目标 < 0.1 mm
```

#### 质量特征预测 (5个输出)
```python
# 1. 粘结强度比 (adhesion_ratio, 0-1)
- R² score: 目标 > 0.85
- RMSE: 目标 < 0.1

# 2. 内应力 (internal_stress, MPa)
- R² score: 目标 > 0.80
- RMSE: 目标 < 5 MPa

# 3. 孔隙率 (porosity, %)
- R² score: 目标 > 0.75
- RMSE: 目标 < 2%

# 4. 尺寸精度 (dimensional_accuracy, mm)
- R² score: 目标 > 0.80
- RMSE: 目标 < 0.1 mm

# 5. 质量评分 (quality_score, 0-1)
- Binary Accuracy (good/bad): 目标 > 0.85
- R² score (regression): 目标 > 0.85
```

### 评估脚本

```bash
# 完整评估pipeline
python experiments/full_evaluation_pipeline.py \
    --model_path checkpoints/best_model.pth \
    --output_dir results/evaluation \
    --data_dir validation_layer50

# 快速评估
python experiments/evaluate_model.py \
    --checkpoint checkpoints/best_model.pth \
    --test_dir validation_layer50
```

### 可视化结果

评估后生成的图表：
- `quality_predictions.png` - 质量特征预测vs真实值散点图
- `trajectory_comparison.png` - 参考轨迹vs实际轨迹对比
- `error_distribution.png` - 误差分布直方图
- `confusion_matrix.png` - 质量分类混淆矩阵

---

## 下一步

1. **运行MATLAB仿真**生成完整数据
2. **测试Python数据加载**确保能正确读取
3. **开始训练**使用quick_train脚本
4. **监控训练**使用TensorBoard
5. **评估模型**使用评估脚本
6. **优化模型**调整超参数

---

**最后更新**: 2026-01-28
**维护者**: 3D Printer PINN Project Team
