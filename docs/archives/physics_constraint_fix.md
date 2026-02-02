# 物理约束损失修复记录

## 问题描述

**症状**：模型训练时loss很低（0.0009），但R²很差（0.0948），预测标准差只有0.007（目标标准差0.05）

**根本原因**：物理约束损失使用了错误的参数和公式，严重限制了模型的学习能力

---

## 错误分析

### 错误1：错误的刚度值

**配置文件中的值**（旧）：
```python
# config/base_config.py (旧)
stiffness = 150000  # N/m - 理论GT2皮带刚度
```

**MATLAB仿真实际使用的值**：
```matlab
% matlab_simulation/physics_parameters.m line 98
params.dynamics.x.stiffness = 8000;  % N/m - 有效刚度（调整以生成±0.1mm误差）
params.dynamics.y.stiffness = 8000;  % N/m - 有效刚度
```

**影响**：刚度值相差**18.75倍**（150000 vs 8000）！

### 错误2：错误的物理方程

**旧实现**（training/losses.py line 276）：
```python
# 错误：稳态公式 error = F / k
x_theory = F_inertia_x / k  # k=150000
physics_loss_x = F.smooth_l1_loss(error_x, x_theory)
```

**MATLAB真实物理模型**（simulate_trajectory_error.m line 101-107）：
```matlab
% 状态空间: [x; v] 其中x是位置误差，v是速度误差
% x' = v
% v' = -(c/m)*v - (k/m)*x - a_ref

% 系统矩阵
A = [0, 1;
     -k/m, -c/m]
B = [0; -1]  % 输入是加速度
```

**关键理解**（line 105-107）：
```matlab
% 注意：a_ref上的负号是因为误差定义为：
% error = actual - reference
% 实际位置响应于惯性力 F = -m*a_ref
```

### 错误3：力的平衡理解错误

**完整动力学方程**：
```
m·x'' + c·x' + k·x = -m·a_ref
```

**稳态时**（x'=x''=0）：
```
k·x = -m·a_ref
x = -(m/k)·a_ref = -F_inertia / k
```

**旧实现错误**：
```python
x_theory = F_inertia_x / k  # 缺少负号！
```

**正确实现**：
```python
error_theory_x = -F_inertia_x / k  # 添加负号
```

---

## 修复方案

### 1. 更新物理参数（config/base_config.py）

```python
class PhysicsConfig:
    """物理约束配置 - 基于MATLAB仿真的真实参数"""
    # 参数来源：matlab_simulation/physics_parameters.m line 96-110

    # X轴动力学
    mass_x: float = 0.35                # kg - X轴质量
    stiffness_x: float = 8000.0         # N/m - X轴刚度（关键修复！）
    damping_x: float = 15.0             # N·s/m - X轴阻尼

    # Y轴动力学
    mass_y: float = 0.45                # kg - Y轴质量
    stiffness_y: float = 8000.0         # N/m - Y轴刚度
    damping_y: float = 15.0             # N·s/m - Y轴阻尼
```

### 2. 修正物理约束损失（training/losses.py）

```python
def physics_loss(self, predictions, inputs, physics_config=None):
    """
    基于真实二阶动力学系统计算物理约束损失

    完整动力学方程（来自MATLAB仿真）：
        m·x'' + c·x' + k·x = -m·a_ref

    稳态近似（忽略速度项）：
        k·error ≈ -m·a_ref
        error ≈ -(m/k)·a_ref
    """
    # 真实物理参数（来自physics_parameters.m）
    m_x = physics_config.get('mass_x', 0.35)         # kg
    m_y = physics_config.get('mass_y', 0.45)         # kg
    k_x = physics_config.get('stiffness_x', 8000)    # N/m
    k_y = physics_config.get('stiffness_y', 8000)    # N/m

    # X轴动力学约束
    if 'displacement_x' in predictions and 'F_inertia_x' in inputs:
        error_x = predictions['displacement_x']
        F_inertia_x = inputs['F_inertia_x']

        # 稳态物理约束：k*error ≈ -F_inertia
        error_theory_x = -F_inertia_x / k_x  # 正确的负号！

        physics_loss_x = F.smooth_l1_loss(error_x, error_theory_x)
        total_physics_loss += physics_loss_x
```

### 3. 调整物理约束权重

```python
# experiments/train_trajectory_correction_working.py
criterion = MultiTaskLoss(
    lambda_quality=0.0,
    lambda_fault=0.0,
    lambda_trajectory=1.0,
    lambda_physics=0.1,  # 降低权重，避免过度约束
)
```

**权重选择理由**：
- 太高（如5.0）：会限制模型学习，R²很低
- 太低（如0.0）：失去物理约束的意义
- **0.1**：提供软约束，引导模型但不限制学习

---

## MATLAB物理模型详解

### 动力学方程（simulate_trajectory_error.m）

```matlab
% 状态向量: [位置误差; 速度误差]
x' = v
v' = -(c/m)*v - (k/m)*x - a_ref

% 矩阵形式
A = [0, 1; -k/m, -c/m]
B = [0; -1]
```

### 参数验证（physics_parameters.m line 96-110）

```matlab
% X轴动力学
params.dynamics.x.mass = 0.35;           % kg
params.dynamics.x.stiffness = 8000;      % N/m (关键！)
params.dynamics.x.damping = 15.0;        % N·s/m
params.dynamics.x.natural_freq = sqrt(k/m);  % ≈ 151 rad/s (24 Hz)

% Y轴动力学
params.dynamics.y.mass = 0.45;           % kg
params.dynamics.y.stiffness = 8000;      % N/m
params.dynamics.y.damping = 15.0;        % N·s/m
params.dynamics.y.natural_freq = sqrt(k/m);  % ≈ 133 rad/s (21 Hz)
```

### 力的平衡（simulate_trajectory_error.m line 182-194）

```matlab
% 惯性力: F = m * a_ref
F_inertia_x = mx * ax_ref_uniform;

% 弹性力（皮带拉伸）: F = k * error
F_elastic_x = kx * error_x_uniform;

% 阻尼力: F = c * v_error
F_damping_x = cx * v_error_x;
```

---

## 数值影响分析

### 旧参数 vs 新参数

| 参数 | 旧值（错误） | 新值（正确） | 变化 |
|------|------------|------------|------|
| 刚度 k | 150000 N/m | 8000 N/m | ÷18.75 |
| 质量 m_x | 0.485 kg | 0.35 kg | ÷1.39 |
| 理论误差 | F/150000 | -F/8000 | ×18.75 |

### 理论误差范围

**假设加速度 a = 500 mm/s²（最大加速度）**

旧参数：
```
F_inertia = 0.485 × 500 × 10⁻³ = 0.24 N
error_theory = 0.24 / 150000 = 0.0000016 mm (1.6 nm)
```

新参数：
```
F_inertia = 0.35 × 500 × 10⁻³ = 0.175 N
error_theory = -0.175 / 8000 = -0.0000219 mm (-21.9 μm)
```

**实际误差范围**（从数据统计）：
- X轴误差标准差：约0.05 mm = 50 μm
- Y轴误差标准差：约0.058 mm = 58 μm

**结论**：新参数的理论值（21.9 μm）更接近实际值（50 μm），且在同一数量级！

---

## 验证方法

### 1. 理论验证

检查理论误差与实际误差的量级是否匹配：

```python
# 计算理论误差范围
F_inertia_max = m_x * max_accel  # max_accel ≈ 500 mm/s²
error_theory_max = F_inertia_max / k_x

# 应该接近实际误差标准差（0.05 mm）
print(f"Theoretical max error: {error_theory_max:.6f} mm")
# Expected: ≈ 0.02-0.03 mm
```

### 2. 训练验证

重新训练后检查：
- R² > 0.3（之前0.0948）
- 预测标准差 ≈ 目标标准差（之前0.007 vs 0.05）
- Test loss ≈ Val loss（之前0.007 vs 0.0009）

### 3. 物理一致性验证

检查预测误差与惯性力的关系：

```python
# 稳态约束检查
expected_ratio = -1.0 / k_x  # error / F_inertia
actual_ratio_x = pred_x.mean() / F_inertia_x.mean()

# 应该接近expected_ratio
print(f"Expected ratio: {expected_ratio:.6e}")
print(f"Actual ratio: {actual_ratio_x:.6e}")
```

---

## 重新训练指南

### 为什么必须从头训练？

之前的模型在错误的物理约束下训练了50个epoch，被引导到错误的最小值（预测接近0）。继续训练只会停留在局部最优。

### 训练命令

```bash
# 从头训练（推荐）
python experiments/train_trajectory_correction_working.py \
    --data_dir "data_simulation_*" \
    --epochs 50 \
    --batch_size 512 \
    --lr 1e-4

# 如果之前训练过，删除旧checkpoint
rm checkpoints/trajectory_correction/best_model.pth
rm checkpoints/trajectory_correction/last_model.pth
```

### 预期效果

- **训练loss**：应该收敛到0.001-0.002左右
- **R²分数**：应该达到0.3-0.6（之前0.0948）
- **预测方差**：应该接近目标方差（之前0.007 vs 0.05）
- **Test vs Val loss**：差异应该<2倍（之前8倍）

---

## 关键要点

1. **参数必须与仿真一致**：模型训练使用的数据来自MATLAB仿真，物理约束必须使用相同的参数
2. **刚度是关键参数**：8000 N/m vs 150000 N/m，相差18.75倍！
3. **负号很重要**：`error = -F_inertia/k` 不是 `error = F_inertia/k`
4. **权重需要平衡**：太强限制学习，太弱失去约束意义
5. **从头训练**：错误的物理约束会引导到错误的最小值

---

## 参考文件

1. **MATLAB仿真脚本**：
   - `matlab_simulation/simulate_trajectory_error.m` - 动力学仿真
   - `matlab_simulation/physics_parameters.m` - 物理参数

2. **Python实现**：
   - `training/losses.py` - 物理约束损失
   - `config/base_config.py` - 物理参数配置
   - `experiments/train_trajectory_correction_working.py` - 训练脚本

3. **数据集**：
   - 输入特征包含F_inertia_x, F_inertia_y（来自MATLAB仿真）
   - 目标值error_x, error_y（来自MATLAB动力学求解）

---

**修复日期**：2025-02-01
**修复人**：Claude Code
**状态**：✅ 已修复，等待重新训练验证
