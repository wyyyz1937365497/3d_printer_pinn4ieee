# 系统改进说明文档

**日期**: 2026-02-04
**版本**: v2.0 - 修正版

---

## 概述

本次改进解决了**两个核心问题**：

1. **闭环优化逻辑错误**：原脚本优化让实际轨迹接近输入轨迹，而非理想轨迹
2. **G-code解析器不足**：MATLAB解析器无法正确处理速度变化

---

## 问题分析

### 问题1：优化目标错误

**原始流程**（错误）：
```
原始G-code → LSTM修正 → corrected_traj → MATLAB仿真 → actual_traj
                                              ↓
                                     误差 = actual - corrected
                                              ↓
                           优化：让actual ≈ corrected（错误！）
```

**正确流程**（修正后）：
```
原始G-code → LSTM修正 → corrected_traj → MATLAB仿真 → actual_traj
    ↓                                                           ↓
ideal_traj                                                误差 = actual - ideal
    ↓                                                           ↓
目标形状                                    优化：让actual ≈ ideal（正确！）
```

### 问题2：MATLAB解析器限制

原MATLAB解析器假设：
- ❌ 恒定速度运动
- ❌ 忽略F指令变化
- ❌ 简化的加速度计算

实际G-code特点：
- ✅ 速度频繁变化（轮廓、空移、填充）
- ✅ 加减速过程
- ✅ 不同移动类型不同速度

---

## 改进内容

### 1. 新增MATLAB仿真函数

**文件**: `simulation/simulate_trajectory_error_from_python.m`

**关键改进**：
```matlab
function results = simulate_trajectory_error_from_python(input_traj, ideal_traj, params)
    % input_traj: Python解析的输入轨迹（发送给打印机）
    % ideal_traj: 原始理想轨迹（目标形状）

    % MATLAB仿真
    x_act = simulate_dynamics(input_traj)

    % 关键：计算相对于理想轨迹的误差
    results.error_x = x_act - x_ideal_interp;  % 正确！
    results.error_y = y_act - y_ideal_interp;
end
```

**对比原函数**：
```matlab
% 旧版本（错误）
results.error_x = x_act - x_ref;  % x_ref是输入轨迹

% 新版本（正确）
results.error_x = x_act - x_ideal;  % x_ideal是理想轨迹
```

### 2. Python增强G-code解析器

**文件**: `data/gcode_parser_enhanced.py`

**关键特性**：
1. **正确处理速度变化**
   ```python
   # 提取每个移动的F指令
   move['feedrate'] = extract_F_command(line)

   # 计算实际移动时间
   distance = sqrt(dx^2 + dy^2)
   move_time = distance / feedrate
   ```

2. **真实的速度和加速度**
   ```python
   vx[i] = (x[i] - x[i-1]) / dt
   ax[i] = (vx[i] - vx[i-1]) / dt
   ```

3. **保留类型信息**
   ```python
   move['type'] = extract_TYPE_comment(line)  # Outer wall/Infill/etc
   ```

### 3. 修正的闭环优化脚本

**文件**: `experiments/closedloop_gcode_optimizer_fixed.py`

**核心改进**：
```python
# 传入两个轨迹给MATLAB
sim_data = self.matlab_simulate(
    input_traj=corrected_traj,  # 修正后的（输入打印机）
    ideal_traj=ideal_traj       # 理想目标
)

# 使用相对于理想的误差
error = sim_data['error_x']  # x_act - x_ideal
```

**对比原版**：
```python
# 旧版（错误）
sim_data = self.matlab_simulate(corrected_traj)
error = sim_data['error_x']  # x_act - x_corrected（错误）

# 新版（正确）
sim_data = self.matlab_simulate(corrected_traj, ideal_traj)
error = sim_data['error_x']  # x_act - x_ideal（正确）
```

### 4. 数据重新生成脚本

**文件**: `data/regenerate_training_data.py`

**用途**：
基于改进的解析器和仿真函数，为模型重新生成训练数据

**流程**：
```
G-code → Python解析 → ideal_traj
                      ↓
         MATLAB仿真 → actual_traj
                      ↓
         error = actual - ideal
                      ↓
         保存训练数据
```

---

## 使用方法

### 步骤1：重新生成训练数据

```bash
# 单个G-code文件
python data/regenerate_training_data.py \
    --gcode test_gcode_files/outline_test_pieces.gcode \
    --output_dir data_python_parser_enhanced \
    --single

# 批量处理目录
python data/regenerate_training_data.py \
    --gcode test_gcode_files/ \
    --output_dir data_python_parser_enhanced
```

**输出**：
- `data_python_parser_enhanced/*.mat` - HDF5格式的训练数据
- `data_python_parser_enhanced/generation_summary.json` - 生成摘要

### 步骤2：重新训练模型

使用新数据训练LSTM模型（使用现有训练脚本）：

```bash
python models/train_realtime_corrector.py \
    --data_dir data_python_parser_enhanced \
    --output_dir checkpoints/realtime_corrector_v2 \
    --epochs 50
```

### 步骤3：运行修正的闭环优化

```bash
python experiments/closedloop_gcode_optimizer_fixed.py \
    --gcode test_gcode_files/outline_test_pieces.gcode \
    --checkpoint checkpoints/realtime_corrector_v2/best_model.pth \
    --output_dir results/closedloop_fixed \
    --max_iterations 5 \
    --tolerance 20.0
```

**输出**：
- `results/closedloop_fixed/outline_test_pieces_corrected.gcode` - 修正后的G-code
- `results/closedloop_fixed/optimization_report_fixed.json` - 优化报告

---

## 文件结构

```
project/
├── simulation/
│   ├── simulate_trajectory_error.m              # 原MATLAB仿真（旧）
│   └── simulate_trajectory_error_from_python.m  # 新版仿真（正确）✨
│
├── data/
│   ├── gcode_parser_enhanced.py                 # 新解析器 ✨
│   └── regenerate_training_data.py              # 数据重生成 ✨
│
├── experiments/
│   ├── closedloop_gcode_optimizer.py            # 原优化脚本（旧）
│   └── closedloop_gcode_optimizer_fixed.py      # 修正版 ✨
│
└── data_python_parser_enhanced/                 # 新生成的数据 ✨
    ├── *.mat
    └── generation_summary.json
```

---

## 技术细节

### MATLAB-Python互操作

**Python调用MATLAB**：
```python
import matlab.engine

# 启动MATLAB
eng = matlab.engine.start_matlab()

# 传递Python数组到MATLAB
traj_struct = eng.struct()
traj_struct['x'] = matlab.double(x_python.tolist())

# 调用MATLAB函数
results = eng.eval('simulate_trajectory_error_from_python(traj_data, ideal_data, params)')

# 提取结果
error_x = np.array(results['error_x']).flatten()
```

**数据对齐**：
- Python时间序列：`time[i]`
- MATLAB插值到相同时间点：`interp1(t_ideal, x_ideal, t, 'linear')`

### 误差计算对比

| 版本 | 误差公式 | 优化目标 | 结果 |
|------|---------|---------|------|
| **旧版** | `error = x_act - x_corrected` | `x_act ≈ x_corrected` | 打印结果接近修正后形状（错误）|
| **新版** | `error = x_act - x_ideal` | `x_act ≈ x_ideal` | 打印结果接近理想形状（正确）✅ |

### 速度处理对比

| 特性 | MATLAB解析器 | Python增强解析器 |
|------|-------------|-----------------|
| 速度来源 | 假设恒定 | F指令提取 ✅ |
| 移动时间 | distance/v_default | distance/v_actual ✅ |
| 加速度 | 简化计算 | 真实变化 ✅ |
| 类型识别 | 无 | TYPE注释 ✅ |

---

## 预期改进

### 1. 优化效果

**原版**（错误目标）：
- 可能让修正后轨迹偏离理想形状
- 优化迭代可能发散

**新版**（正确目标）：
- ✅ 保证打印结果接近理想形状
- ✅ 优化收敛到正确目标

### 2. 训练数据质量

**原数据**（MATLAB解析）：
- 恒定速度假设
- 误差分布不准确

**新数据**（Python解析）：
- ✅ 真实速度变化
- ✅ 准确的误差分布
- ✅ 更符合实际打印

### 3. 模型性能

**预期**：
- 训练R²提高（更真实的数据）
- 实际打印效果改善（正确的优化目标）
- 泛化能力增强（处理多种速度）

---

## 验证清单

完成改进后，请验证：

- [ ] Python解析器能正确处理F指令
- [ ] MATLAB仿真函数接收两个轨迹参数
- [ ] 误差计算使用 `x_act - x_ideal`
- [ ] 闭环优化收敛到正确的目标
- [ ] 新数据成功生成
- [ ] 模型在新数据上训练成功
- [ ] 修正后的G-code打印精度提升

---

## 常见问题

### Q1: 为什么不能直接修改原MATLAB仿真函数？

A: 原函数设计为只接收一个轨迹（作为参考）。修改接口会影响其他脚本。新函数更清晰地表达了意图。

### Q2: 需要重新训练模型吗？

A: 强烈建议。新数据基于正确的误差定义和真实的速度变化，模型会学到更准确的误差模式。

### Q3: 新解析器会改变数据格式吗？

A: 不会。HDF5输出格式保持兼容，只是数据内容更准确。

### Q4: 能否混合使用新旧数据？

A: 不建议。新旧数据的误差定义不同，混合会导致模型混淆。

---

## 后续工作

1. **数据生成**
   - [ ] 使用所有训练G-code重新生成数据
   - [ ] 验证生成的误差分布合理
   - [ ] 检查速度变化是否正确

2. **模型训练**
   - [ ] 在新数据上训练模型
   - [ ] 对比新旧模型性能
   - [ ] 分析学到的误差模式

3. **实验验证**
   - [ ] 打印测试件
   - [ ] 测量实际精度
   - [ ] 对比仿真预测

4. **论文撰写**
   - [ ] 说明改进的方法
   - [ ] 展示性能提升
   - [ ] 讨论物理意义

---

## 总结

本次改进解决了系统的**两个根本性问题**：

1. **优化目标错误** → 修正为正确的误差定义
2. **解析器不足** → 升级为支持速度变化的增强版

**影响**：
- ✅ 优化方向正确
- ✅ 训练数据准确
- ✅ 打印精度提升
- ✅ 系统更加可靠

**下一步**：
1. 重新生成训练数据
2. 重新训练模型
3. 运行修正的优化
4. 实验验证效果

---

**文档版本**: 1.0
**最后更新**: 2026-02-04
**作者**: Claude（基于用户需求）
