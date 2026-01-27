# 完整项目实施指南：从数据收集到模型评估

## 📋 目录

1. [数据收集方案](#1-数据收集方案)
2. [数据预处理](#2-数据预处理)
3. [模型训练](#3-模型训练)
4. [模型评估](#4-模型评估)
5. [常见问题](#5-常见问题)

---

## 1. 数据收集方案

### 1.1 总体策略

#### 核心概念
```
打印任务 → 记录传感器数据 → 打印完成 → 质量测试 → 数据配对
(可观测)     (可观测序列)       (不可观测)   (配对训练)
```

#### 数据要求
- **最少样本数**：100-200个打印样件（建议300+以获得更好效果）
- **数据配对**：每个打印任务需要传感器数据 + 质量测试结果
- **多样性**：覆盖不同的打印参数（温度、速度、材料等）

### 1.2 传感器数据收集（可观测参数）

#### 需要的传感器

| 传感器类型 | 测量参数 | 采样频率 | 必需性 | 替代方案 |
|-----------|---------|---------|--------|---------|
| **热电偶** | 喷嘴温度 | 10 Hz | ✅ 必需 | 打印机自带 |
| **热电偶** | 床温 | 10 Hz | ✅ 必需 | 打印机自带 |
| **加速度计** | X/Y/Z振动 | 1000 Hz | ✅ 必需 | 可选（后期） |
| **电流传感器** | 电机电流 | 100 Hz | ⚠️ 建议 | 可选 |
| **压力传感器** | 挤出压力 | 100 Hz | ⚠️ 建议 | 可选 |

#### 最小配置（低成本方案）
如果预算有限，从以下开始：
1. **喷嘴温度** - 打印机自带（通过串口/G-code读取）
2. **床温** - 打印机自带
3. **打印速度** - 控制器变量
4. **位置** - 打印机坐标

#### 完整配置（推荐方案）
1. **温度** × 2（喷嘴、床）
2. **振动** × 3（X、Y、Z轴加速度计）
3. **电机电流** × 3（X、Y、Z轴）
4. **打印速度**（控制器）
5. **位置** × 3（X、Y、Z坐标）

### 1.3 质量测试数据收集（不可观测参数）

每个打印完成后，需要测量以下质量参数：

#### 层间粘合力测试
```python
测试方法：拉伸试验
设备：万能材料试验机
标准：ASTM D3039 或类似

测试步骤：
1. 打印测试样件（例如：狗骨形拉伸试样）
2. 沿层间方向进行拉伸
3. 记录最大断裂力
4. 计算粘结强度：σ = F_max / A

目标值：15-30 MPa（PLA材料）
```

#### 内应力测试
```python
测试方法1：X射线衍射
- 测量晶格畸变
- 计算残余应力

测试方法2：轮廓法
- 切割零件
- 测量变形
- 反推应力

测试方法3：简化法（适合初学者）
- 打印弯曲悬臂梁
- 测量翘曲程度
- 间接评估应力
```

#### 孔隙率测试
```python
测试方法：CT扫描或密度法

CT扫描法：
1. 对打印件进行CT扫描
2. 重建内部结构
3. 计算孔隙体积占比
4. 获得孔隙率百分比

密度法（简化）：
1. 测量实际质量 m
2. 计算理论质量 m_0 = ρ × V
3. 孔隙率 = (1 - m/m_0) × 100%
```

#### 尺寸精度测试
```python
测试方法：三坐标测量机或卡尺

步骤：
1. 设计标准几何体（例如：20mm × 20mm × 20mm 立方体）
2. 打印测试件
3. 测量实际尺寸
4. 计算误差 = |实际尺寸 - 设计尺寸|
```

#### 综合质量评分
```python
基于上述参数计算：

Quality_Score = (
    0.35 × sigmoid(adhesion / 30.0) +    # 粘结力贡献35%
    0.25 × sigmoid(-stress / 20.0) +     # 低应力贡献25%
    0.20 × sigmoid(-porosity / 5.0) +    # 低孔隙率贡献20%
    0.20 × sigmoid(-accuracy / 0.1)      # 低误差贡献20%
)

或使用专家评分法（更简单）：
- 由专家根据各项指标给出0-1分
- 或使用分级系统（优秀/良好/合格/不合格）
```

### 1.4 数据收集实施步骤

#### 第1步：准备阶段
```bash
# 1. 设计实验
确定变量范围：
- 温度：190-240°C（间隔10°C）
- 速度：30-80 mm/s（间隔10 mm/s）
- 层高：0.1-0.3 mm
- 材料：PLA, ABS等

# 2. 设计测试样件
必需样件：
- 拉伸试样（测粘结力）× 5个/组
- 立方体（测尺寸、孔隙率）× 3个/组
- 悬臂梁（测应力）× 3个/组

# 3. 准备数据记录系统
见下文数据收集脚本
```

#### 第2步：设置传感器数据记录
```python
# 使用我们提供的脚本（见下一节）
python data/scripts/collect_sensor_data.py \
    --printer_type "your_printer_model" \
    --output_dir "data/raw/" \
    --duration_minutes 30
```

#### 第3步：执行打印测试
```python
# 打印矩阵设计
from experiments.design_printing_matrix import generate_print_matrix

# 生成打印任务列表
print_matrix = generate_print_matrix(
    temperatures=[200, 210, 220, 230],
    speeds=[40, 60, 80],
    repetitions=3
)

# 总打印任务数：4 × 3 × 3 = 36个
# 每个任务包括多个样件
```

#### 第4步：质量测试
```bash
# 打印完成后，进行破坏性测试
# 记录结果到CSV文件

python data/scripts/record_quality_test.py \
    --test_type "adhesion" \
    --sample_id "print_001" \
    --result 25.5  # MPa
```

#### 第5步：数据配对
```python
# 自动配对传感器数据和质量数据
python data/scripts/pair_sensor_quality_data.py \
    --sensor_dir "data/raw/sensor_data/" \
    --quality_dir "data/raw/quality_data/" \
    --output "data/processed/paired_data.pkl"
```

---

## 2. 数据预处理

### 2.1 数据格式要求

#### 传感器数据格式
```python
# 每个打印任务的传感器数据
{
    'sample_id': 'print_001',
    'print_parameters': {
        'temperature': 220,  # °C
        'speed': 60,         # mm/s
        'layer_height': 0.2, # mm
        'material': 'PLA',
    },
    'sensor_data': {
        # 时间序列数据 [timesteps, features]
        'nozzle_temp': np.array([[...], [t=1], [t=2], ...]),  # [T, 1]
        'bed_temp': np.array([...]),                          # [T, 1]
        'vibration_x': np.array([...]),                       # [T, 1]
        'vibration_y': np.array([...]),                       # [T, 1]
        'vibration_z': np.array([...]),                       # [T, 1]
        'motor_current_x': np.array([...]),                   # [T, 1]
        'motor_current_y': np.array([...]),                   # [T, 1]
        'motor_current_z': np.array([...]),                   # [T, 1]
        'print_speed': np.array([...]),                       # [T, 1]
        'position_x': np.array([...]),                        # [T, 1]
        'position_y': np.array([...]),                        # [T, 1]
        'position_z': np.array([...]),                        # [T, 1]
        'timestamp': np.array([...]),                         # [T, 1]
    },
    'sequence_length': T,  # 实际序列长度
    'sampling_rate': 1000,  # Hz
}
```

#### 质量数据格式
```python
{
    'sample_id': 'print_001',
    'quality_metrics': {
        'adhesion_strength': 25.5,    # MPa
        'internal_stress': 12.3,      # MPa
        'porosity': 3.2,              # %
        'dimensional_accuracy': 0.05, # mm
        'quality_score': 0.85,        # [0-1]
    },
    'test_info': {
        'test_date': '2024-01-27',
        'test_method': 'tensile_test',
        'tester': 'your_name',
    }
}
```

### 2.2 数据预处理脚本

我们提供了完整的预处理脚本：

```python
# 使用预处理脚本
python data/scripts/preprocess_data.py \
    --raw_dir "data/raw/" \
    --output_dir "data/processed/" \
    --seq_len 200 \
    --normalize \
    --train_val_test_split 0.7 0.15 0.15
```

### 2.3 数据增强（可选）

如果数据量不足，可以使用数据增强：

```python
from data.augmentation import augment_sensor_data

# 添加噪声
augmented_data = augment_sensor_data(
    sensor_data,
    noise_level=0.01,
    time_shift=True,
    scaling=True
)
```

---

## 3. 模型训练

### 3.1 准备训练环境

#### 环境配置
```bash
# 1. 确保已安装所有依赖
pip install -r requirements.txt

# 2. 检查GPU（推荐）
python -c "import torch; print(torch.cuda.is_available())"

# 3. 设置数据目录
export DATA_DIR="data/processed/"
export CHECKPOINT_DIR="checkpoints/"
```

### 3.2 训练配置

#### 基础训练（小数据集）
```bash
python experiments/train_unified_model.py \
    --data_dir "data/processed/" \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --config_preset "unified" \
    --device "cuda" \
    --checkpoint_dir "checkpoints/unified_model/"
```

#### 完整训练（大数据集）
```bash
python experiments/train_unified_model.py \
    --data_dir "data/processed/" \
    --batch_size 64 \
    --num_epochs 200 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --gradient_accumulation 2 \
    --mixed_precision \
    --early_stopping_patience 20 \
    --save_every 5 \
    --config_preset "unified" \
    --device "cuda"
```

### 3.3 训练监控

#### 使用TensorBoard
```bash
# 启动TensorBoard
tensorboard --logdir logs/

# 在浏览器中打开
# http://localhost:6006
```

#### 监控指标
观察以下指标：
- **训练损失**：应该持续下降
- **验证损失**：应该下降但不应该过拟合
- **学习率**：应该在合理范围内
- **梯度范数**：不应该出现梯度爆炸

### 3.4 训练技巧

#### 技巧1：逐步训练
```python
# 第1阶段：训练编码器（冻结解码器）
# 第2阶段：训练解码器（冻结编码器）
# 第3阶段：端到端微调
```

#### 技巧2：损失权重调整
```python
# 如果某些任务学习困难，调整损失权重
config.lambda_quality = 2.0      # 增加质量预测权重
config.lambda_fault = 1.0
config.lambda_trajectory = 0.5   # 减少轨迹校正权重
config.lambda_physics = 0.2      # 增加物理约束权重
```

#### 技巧3：学习率调度
```python
# 使用warmup + cosine annealing
config.training.warmup_epochs = 10
config.training.scheduler = "cosine"
```

---

## 4. 模型评估

### 4.1 基础评估

```bash
# 运行评估脚本
python experiments/evaluate_model.py \
    --model_path "checkpoints/unified_model/best_model.pth" \
    --data_dir "data/processed/" \
    --output_dir "results/evaluation/"
```

### 4.2 完整评估流程

```bash
# 运行完整评估管道
python experiments/full_evaluation_pipeline.py \
    --model_path "checkpoints/unified_model/best_model.pth" \
    --output_dir "results/full_eval/" \
    --batch_size 64 \
    --device "cuda"
```

### 4.3 评估结果解读

#### 输出文件
```
results/full_eval/
├── metrics.txt              # 人类可读的指标
├── metrics.json             # 机器可读的指标
├── summary_report.txt       # 综合总结报告
├── figures/                 # 可视化图表
│   ├── adhesion_prediction.png
│   ├── stress_prediction.png
│   ├── porosity_prediction.png
│   ├── accuracy_prediction.png
│   └── error_distributions.png
└── benchmarks/
    └── full_comparison.json  # 与基线对比
```

#### 关键指标解读

**隐式质量参数预测**：
```python
# 层间粘合力
Adhesion RMSE: 2.5 MPa  # 目标：< 3 MPa
Adhesion R²: 0.92       # 目标：> 0.85

# 内应力
Stress RMSE: 1.8 MPa    # 目标：< 2 MPa
Stress R²: 0.88         # 目标：> 0.85

# 孔隙率
Porosity RMSE: 0.8%     # 目标：< 1%
Porosity R²: 0.90       # 目标：> 0.85

# 尺寸精度
Accuracy RMSE: 0.04 mm  # 目标：< 0.05 mm
Accuracy R²: 0.95       # 目标：> 0.90
```

### 4.4 模型改进

如果性能不理想，可以尝试：

#### 方法1：增加数据量
```python
# 收集更多打印样件
# 目标：300-500个样件
```

#### 方法2：调整模型
```python
# 增加模型容量
config.model.d_model = 512
config.model.num_layers = 8
```

#### 方法3：调整损失权重
```python
# 加强物理约束
config.lambda_physics = 0.3

# 调整任务权重
config.lambda_quality = 2.0
```

#### 方法4：数据清洗
```python
# 移除异常样本
# 检查数据质量
# 标准化测量流程
```

---

## 5. 常见问题

### Q1: 数据收集需要多长时间？

**答**：
- 最小方案（100个样件）：约2-3周
- 推荐方案（300个样件）：约6-8周
- 每个样件打印时间：30分钟到2小时

### Q2: 没有昂贵的测试设备怎么办？

**答**：
1. **粘结力**：可以简化为剥离试验
2. **内应力**：可以使用悬臂梁翘曲测试
3. **孔隙率**：可以使用密度法代替CT
4. **尺寸精度**：普通卡尺即可

**重点**：保持测试方法的一致性比绝对精度更重要

### Q3: 训练需要多长时间？

**答**：
- 100个样件：GPU上约2-4小时
- 300个样件：GPU上约6-12小时
- CPU上训练时间：约5-10倍

### Q4: 内存不足怎么办？

**答**：
```python
# 减小batch size
config.training.batch_size = 16

# 使用梯度累积
config.training.accumulation_steps = 4

# 使用混合精度训练
config.training.mixed_precision = True

# 减小序列长度
config.data.seq_len = 100
```

### Q5: 如何判断模型是否过拟合？

**答**：
```python
# 观察训练和验证损失
if train_loss << val_loss:
    # 过拟合
    solutions = [
        "增加数据量",
        "使用dropout",
        "添加正则化",
        "减小模型容量"
    ]
```

### Q6: 物理约束如何调试？

**答**：
```python
# 检查物理约束损失
L_physics = model.compute_physics_constraints(predictions, inputs)

# 如果约束损失过大：
if L_physics > 0.1:
    # 降低学习率
    # 增加warmup轮数
    # 检查数据质量
```

---

## 📅 实施时间线

### 阶段1：数据收集（4-8周）
- 周1-2：设置传感器，准备测试样件
- 周3-6：执行打印测试，收集传感器数据
- 周7-8：质量测试，数据配对

### 阶段2：数据预处理（1周）
- 数据清洗
- 格式转换
- 归一化

### 阶段3：模型训练（1-2周）
- 初步训练
- 超参数调优
- 模型选择

### 阶段4：评估和部署（1周）
- 完整评估
- 可视化结果
- 部署测试

**总计：7-12周**（根据数据量）

---

## 🎯 成功标准

### 最低要求（可发表论文）
- ✅ 100+个打印样件
- ✅ 粘结力预测RMSE < 5 MPa
- ✅ 内应力预测RMSE < 3 MPa
- ✅ R²分数 > 0.80
- ✅ 早停准确率 > 70%

### 推荐目标（高质量论文）
- ✅ 300+个打印样件
- ✅ 粘结力预测RMSE < 3 MPa
- ✅ 内应力预测RMSE < 2 MPa
- ✅ R²分数 > 0.90
- ✅ 早停准确率 > 85%

---

## 📞 下一步

1. **阅读数据收集指南**
   → `docs/DATA_COLLECTION_GUIDE.md`

2. **设置数据收集系统**
   → `data/scripts/collect_sensor_data.py`

3. **开始第一个打印测试**
   → `experiments/start_first_print.py`

4. **训练第一个模型**
   → `experiments/train_unified_model.py`

5. **评估模型性能**
   → `experiments/full_evaluation_pipeline.py`

---

**准备好了吗？让我们开始吧！** 🚀
