# 评估脚本修复记录

## 问题描述

**现象**：训练loss很低（0.001871），但评估R²很差（0.0948）

**原因**：评估脚本与训练脚本的目标计算方式不一致

## 问题详情

### 训练时的计算方式（train_trajectory_correction_working.py）

```python
traj_targets = batch['trajectory_targets']  # [batch, pred_len, 2]

targets = {
    'displacement_x_seq': traj_targets[:, :, 0:1],  # 整个序列
    'displacement_y_seq': traj_targets[:, :, 1:2],
    'displacement_x': traj_targets[:, :, 0:1].mean(dim=1, keepdim=True),  # 序列平均
    'displacement_y': traj_targets[:, :, 1:2].mean(dim=1, keepdim=True),  # 序列平均
}
```

训练时：
- 序列loss使用整个预测序列
- 标量目标使用序列的平均值

### 原评估脚本的错误（evaluate_trajectory_model.py）

**错误1：目标维度不匹配**
```python
# 预测只取最后1个时间步
pred_x = outputs['displacement_x_seq'][:, -1:, :].cpu().numpy()  # [batch, 1, 1]

# 但目标使用整个序列
target_x = trajectory_targets[:, :, 0:1].cpu().numpy()  # [batch, pred_len, 1]

# flatten后：1个预测值对应50个目标值，完全错误！
```

**错误2：Loss计算维度不匹配**
```python
loss_x = criterion(
    outputs['displacement_x_seq'][:, -1:, :],  # [batch, 1, 1]
    trajectory_targets[:, :, 0:1].to(device)   # [batch, pred_len, 1]
)  # 维度不匹配！
```

**错误3：评估指标无意义**
- 预测值：129,638个（每个样本1个值）
- 目标值：129,638个（每个样本50个值flatten）
- R²计算时，1个预测值被重复比较50次，完全不准确

## 修复方案

### 修复1：统一目标维度（默认 'last' 模式）

```python
# 预测最后1个时间步
pred_x = pred_x_seq[:, -1:, :].cpu().numpy()  # [batch, 1, 1]

# 目标也用最后1个时间步
target_x = trajectory_targets[:, -1:, 0:1].cpu().numpy()  # [batch, 1, 1]

# 维度匹配，比较公平
```

### 修复2：添加多种评估模式

新增 `--eval_mode` 参数，支持三种模式：

1. **'last'（默认）**：比较最后时间步
   - 最相关实时修正场景
   - 命令：`python experiments/evaluate_trajectory_model.py --eval_mode last`

2. **'mean'**：比较序列平均值
   - 与训练的标量目标一致
   - 命令：`python experiments/evaluate_trajectory_model.py --eval_mode mean`

3. **'all'**：比较整个序列
   - 最全面的评估
   - 命令：`python experiments/evaluate_trajectory_model.py --eval_mode all`

## 预期效果

### 修复前（错误）
```
Average Performance:
  R²:           0.0948
  Correlation:  0.3079
  Norm. MAE:    0.1651

Overall Rating: POOR - Model does not learn meaningful trajectory patterns
```

### 修复后（预期）
R²应该显著提升到0.3-0.6之间，因为：
- 预测和目标维度匹配
- Loss计算正确
- 评估指标有意义

## 使用建议

### 标准评估
```bash
python experiments/evaluate_trajectory_model.py \
    --checkpoint checkpoints/trajectory_correction/best_model.pth \
    --data_dir "data_simulation_*" \
    --eval_mode last
```

### 全面评估
建议运行三种模式对比：
```bash
# 1. 最后时间步（实时修正最相关）
python experiments/evaluate_trajectory_model.py --eval_mode last

# 2. 序列平均（与训练标量目标一致）
python experiments/evaluate_trajectory_model.py --eval_mode mean

# 3. 整个序列（最全面）
python experiments/evaluate_trajectory_model.py --eval_mode all
```

## 技术细节

### 数据维度说明
- `input_features`: [batch, seq_len=200, 9]
- `trajectory_targets`: [batch, pred_len=50, 2]
- `outputs['displacement_x_seq']`: [batch, pred_len=50, 1]

### 评估模式对齐

| 评估模式 | 预测维度 | 目标维度 | 说明 |
|---------|---------|---------|------|
| 'last' | [batch, 1, 1] | [batch, 1, 1] | 最后时间步，最相关修正 |
| 'mean' | [batch, 1, 1] | [batch, 1, 1] | 序列平均，与训练标量目标一致 |
| 'all' | [batch, 50, 1] | [batch, 50, 1] | 整个序列，最全面 |

## 文件修改

- `experiments/evaluate_trajectory_model.py`
  - 修复目标维度匹配（line 134-141）
  - 添加eval_mode参数（line 324-326）
  - 实现三种评估模式（line 94-179）

## 验证方法

运行评估后，检查：
1. Test loss应该与训练时的val loss相近（0.001871）
2. R²应该>0.3（如果是训练良好的模型）
3. 预测值方差应该与目标值方差相近（不是平坦值）

## 后续优化

如果R²仍然较低（<0.3），可能需要：
1. 继续训练（使用--resume参数）
2. 调整学习率（降低到1e-5）
3. 增加数据增强
4. 调整模型架构

---

**修复日期**：2025-02-01
**修复人**：Claude Code
**状态**：✅ 已修复
