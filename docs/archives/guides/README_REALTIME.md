# 实时轨迹修正系统

轻量级实时3D打印轨迹误差预测与补偿系统

## 系统特性

- **输入**: 4维 [x_ref, y_ref, vx_ref, vy_ref]
- **输出**: 2维 [error_x, error_y]
- **推理时间**: < 1ms (实测: 0.4-0.6ms)
- **参数量**: 46K
- **目标性能**: MAE < 0.05mm, R² > 0.8

## 快速开始

### 1. 测试系统

```bash
python scripts/test_realtime.py
```

预期输出:
```
✓ 所有测试通过!
系统已就绪,可以开始训练
```

### 2. 训练模型

```bash
python experiments/train_realtime.py \
    --data_dir "data_simulation_*" \
    --seq_len 20 \
    --batch_size 256 \
    --epochs 100
```

训练特性:
- 混合精度训练 (FP16)
- 梯度累积 (effective_batch_size = 256 × 2 = 512)
- AdamW优化器 (lr=1e-3)
- 余弦退火学习率调度
- 早停 (patience=15)

预期训练时间: 30-45分钟 (100 epochs)

### 3. 评估性能

```bash
python experiments/evaluate_realtime.py \
    --checkpoint checkpoints/realtime_corrector/best_model.pth \
    --data_dir "data_simulation_*"
```

输出指标:
- R² Score
- MAE / RMSE
- 相关系数
- 推理时间

### 4. 生成可视化

```bash
python experiments/visualize_realtime.py \
    --checkpoint checkpoints/realtime_corrector/best_model.pth \
    --data_dir "data_simulation_*" \
    --output_dir results/realtime_visualization
```

生成的图表:
1. 训练曲线 (loss, lr)
2. 预测vs真实散点图
3. 误差时间序列
4. 误差分布直方图
5. 2D误差热图
6. 推理性能分析

## 系统架构

### 数据层 (`data/realtime_dataset.py`)
- 从.mat文件加载4维特征和2维标签
- 滑动窗口序列创建 (seq_len=20, pred_len=1)
- StandardScaler归一化
- 70%/15%/15% 训练/验证/测试划分

### 模型层 (`models/realtime_corrector.py`)
```
输入 (4维) → 编码器 (32维) → LSTM (56×2) → 解码器 (2维)
```
- 参数量: 46,034
- 推理时间: 0.4-0.6ms
- 满足实时性要求 (< 1ms)

### 训练层 (`experiments/train_realtime.py`)
- MAE损失 (L1Loss)
- 混合精度 + 梯度累积
- 学习率调度
- 早停机制

### 评估层 (`experiments/evaluate_realtime.py`)
- 多指标评估 (R², MAE, RMSE, Correlation)
- 推理性能测试
- 结果保存 (JSON格式)

### 可视化层 (`experiments/visualize_realtime.py`)
- 6种可视化类型
- 高DPI输出 (150 DPI)
- 一键生成所有图表

## 文件结构

```
3d_printer_pinn4ieee/
├── data/
│   └── realtime_dataset.py          # 4维数据集
├── models/
│   └── realtime_corrector.py         # 轻量级模型
├── config/
│   └── realtime_config.py            # 配置文件
├── experiments/
│   ├── train_realtime.py            # 训练脚本
│   ├── evaluate_realtime.py         # 评估脚本
│   └── visualize_realtime.py        # 可视化脚本
├── scripts/
│   └── test_realtime.py              # 测试脚本
├── checkpoints/
│   └── realtime_corrector/           # 模型保存目录
└── results/
    ├── realtime_evaluation/          # 评估结果
    └── realtime_visualization/       # 可视化图表
```

## 性能指标

### 模型性能
- **参数量**: 46,034 (< 50K ✓)
- **推理时间**: 0.4-0.6ms (< 1ms ✓)
- **吞吐量**: ~2500 inf/s (单样本)

### 预期训练性能
- **验证损失**: < 0.02
- **MAE**: < 0.05mm
- **R²**: > 0.8

## 关键设计决策

### 1. 为什么使用4维输入?
- `x_ref, y_ref`: 位置信息
- `vx_ref, vy_ref`: 速度信息
- 最小化输入以满足实时性要求

### 2. 为什么seq_len=20?
- 100Hz采样下,20步 = 0.2秒历史
- 匹配3D打印机机械系统响应时间 (0.1-0.2秒)
- 避免长序列引入的噪声和梯度问题

### 3. 为什么pred_len=1?
- 单步预测 (10ms提前)
- 满足实时补偿要求
- 避免多步预测的累积误差

### 4. 为什么使用LSTM?
- 轻量级: 46K参数
- 高效: 0.4-0.6ms推理
- 适合捕获时序动态

## 配置参数

所有配置在 `config/realtime_config.py`:

```python
DATA_CONFIG = {
    'seq_len': 20,
    'pred_len': 1,
}

MODEL_CONFIG = {
    'input_size': 4,
    'hidden_size': 56,
    'num_layers': 2,
    'dropout': 0.1,
}

TRAINING_CONFIG = {
    'batch_size': 256,
    'epochs': 100,
    'lr': 1e-3,
}
```

## Git提交建议

每个阶段完成后提交:

```bash
# 数据层
git add data/realtime_dataset.py
git commit -m "feat(data): 实现4维实时数据集"

# 模型层
git add models/realtime_corrector.py
git commit -m "feat(model): 实现轻量级实时修正器 (46K参数, 0.4ms推理)"

# 训练层
git add experiments/train_realtime.py
git commit -m "feat(train): 实现训练脚本 (混合精度+梯度累积)"

# 评估层
git add experiments/evaluate_realtime.py
git commit -m "feat(eval): 实现评估脚本"

# 可视化层
git add experiments/visualize_realtime.py
git commit -m "feat(viz): 实现6种可视化"

# 清理
git add -A
git commit -m "chore: 删除旧代码和文档"

# 最终提交
git add -A
git commit -m "feat: 完成实时轨迹修正系统重建"
```

## 已知问题

- 当前使用仿真数据训练,需要在真实打印数据上验证
- 物理参数基于文献,可能需要针对具体打印机调优

## 参考文献

物理参数来源:
- Wozniak et al., Applied Sciences 2025 (质量, 阻尼)
- Wang et al., Robotics 2018 (刚度)
- Grgić et al., Processes 2023 (Ender-3精度)

## 许可证

本项目用于学术研究目的。

## 版本历史

- v1.0 (2025-02-01): 初始实现
  - 4维输入特征
  - 46K参数模型
  - 0.4ms推理时间
  - 完整训练/评估/可视化pipeline
