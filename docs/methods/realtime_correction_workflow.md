# 实时轨迹修正仿真系统

## 概述

这是一个完整的实时轨迹修正仿真系统，完全模拟真实3D打印过程：
- **MATLAB逐点仿真**：每一步都调用LSTM模型预测误差
- **实时修正**：在发送给"电机"之前修正轨迹指令
- **物理仿真**：修正后的指令经过二阶动力学系统得到实际位置

## 工作原理

### 真实的3D打印实时修正过程

```
对于每个轨迹点 t:
  1. 读取参考轨迹 r_ref(t) = [x_ref, y_ref, vx_ref, vy_ref]
  2. LSTM模型预测误差: e_pred = model(history_20points)
  3. 修正轨迹: r_corrected = r_ref - e_pred  ← 在执行前修正
  4. 发送给"电机": 执行修正后的轨迹
  5. 物理仿真: 实际位置 r_actual = dynamics(r_corrected)
  6. 实际误差: e_actual = r_actual - r_ref
  7. 继续下一个点
```

### 关键点

1. **预测误差**：模型预测的是**参考轨迹**和**实际轨迹**的差
2. **修正方法**：从参考轨迹中**减去**预测的误差
   ```
   修正后的位置 = 参考位置 - 预测误差
   x_corrected = x_ref - error_x_predicted
   ```
3. **实时性**：每个点的推理时间 < 1ms，满足100Hz控制频率

## 使用方法

### 安装依赖

```bash
# 1. 安装MATLAB Engine API for Python
cd "matlabroot/extern/engines/python"
python setup.py install

# 2. 验证安装
python -c "import matlab.engine; print(matlab.__version__)"
```

### 运行仿真

```bash
python experiments/visualize_realtime_correction.py \
    --checkpoint checkpoints/realtime_corrector/best_model.pth \
    --gcode test_gcode_files/3DBenchy_PLA_1h28m.gcode \
    --layer 25 \
    --output_dir results/realtime_correction
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--checkpoint` | 模型检查点路径 | 必需 |
| `--gcode` | G-code文件路径 | test_gcode_files/3DBenchy_PLA_1h28m.gcode |
| `--layer` | 要仿真的层编号 | 25 |
| `--output_dir` | 输出目录 | results/realtime_correction |

## 输出结果

### 1. 可视化图表

- **heatmap_comparison.png** - 修正前后误差热图对比
- **three_trajectories.png** - 三轨迹对比（参考、修正指令、实际位置）
- **detailed_statistics.png** - 详细统计（4个子图）
- **time_series.png** - 误差时间序列

### 2. 数据文件

- **realtime_correction_layer_N.mat** - 完整仿真数据（MATLAB格式）
- **correction_report.json** - 统计报告（JSON格式）

### 3. 性能指标

- 平均误差（μm）
- 最大误差（μm）
- 改善率（%）
- 推理时间（ms）
- 吞吐量（predictions/s）

## 系统架构

```
Python脚本
    ↓
[启动MATLAB引擎]
    ↓
[MATLAB仿真循环]
  for each trajectory point:
    ├─ 读取参考轨迹点
    ├─ Python LSTM预测误差
    ├─ 修正轨迹: r_corr = r_ref - e_pred
    ├─ 物理仿真: r_act = dynamics(r_corr)
    ├─ 计算误差
    └─ 更新历史缓冲区
    ↓
[保存结果]
    ↓
[Python加载结果]
    ↓
[生成可视化]
```

## 技术细节

### MATLAB端（simulate_realtime_correction.m）

- 解析G-code获取参考轨迹
- 加载PyTorch LSTM模型
- 逐点循环：预测 → 修正 → 仿真
- 计算误差并保存结果

### Python端（visualize_realtime_correction.py）

- 启动MATLAB引擎
- 调用MATLAB仿真函数
- 加载仿真结果
- 生成对比可视化

### 物理仿真

使用二阶系统模型：
```
m·ẍ + c·ẋ + k·x = -m·a_ref
```

每一步使用RK4积分，时间步长dt = 0.01s (100Hz)

## 常见问题

### Q1: MATLAB Engine安装失败

```bash
# 找到MATLAB安装目录
matlabroot

# 安装Python引擎
cd "matlabroot/extern/engines/python"
python setup.py install
```

### Q2: Python环境问题

确保MATLAB可以找到Python环境：
```matlab
% 在MATLAB中检查
pe = pyenv;
pe.Version
```

### Q3: 模型加载失败

确保：
1. 模型文件存在
2. 模型架构与保存时一致
3. Python环境已安装PyTorch

## 性能优化

### 加速仿真

1. **使用更少的点**：选择层数较少的模型
2. **减少历史长度**：将seq_len从20减到10
3. **GPU加速**：如果MATLAB支持，启用GPU

### 减少内存占用

1. 逐层仿真（不要一次仿真多层）
2. 降低轨迹点采样率
3. 使用单精度（float32）

## 引用

如果本项目对你有帮助，请引用：

```bibtex
@misc{3d_printer_realtime_correction,
  title={Physics-Informed Neural Network for Real-Time Trajectory Error Correction in FDM 3D Printing},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/3d_printer_pinn4ieee}
}
```

---

**最后更新**: 2026-02-02
