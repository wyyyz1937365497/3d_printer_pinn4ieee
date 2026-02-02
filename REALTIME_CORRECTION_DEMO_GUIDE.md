# 实时轨迹修正演示指南

## 概述

`visualize_realtime_correction_demo.py` 是一个演示实时轨迹误差修正的脚本，它模拟真实打印过程：
1. 运行MATLAB仿真获取原始轨迹（未修正）
2. 使用Python LSTM模型实时预测误差
3. 应用修正到参考轨迹
4. 再次运行MATLAB仿真验证修正效果
5. 生成对比可视化

## 工作流程

```
G-code文件
    ↓
┌─────────────────────────────────────────┐
│ 步骤1: MATLAB仿真（原始轨迹）            │
│   - 运行完整物理仿真                     │
│   - 得到原始误差                         │
│   - 保存为 original_simulation.mat       │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 步骤2: Python实时预测                    │
│   - 逐点读取参考轨迹                     │
│   - LSTM模型预测误差                     │
│   - 应用修正（补偿）                     │
│   - 保存修正后轨迹                       │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 步骤3: MATLAB仿真（修正轨迹）            │
│   - 加载修正后的轨迹                     │
│   - 运行动力学仿真                       │
│   - 得到修正后误差                       │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 步骤4: 可视化对比                        │
│   - 误差热图对比                         │
│   - 统计分析                             │
│   - 改善率计算                           │
└─────────────────────────────────────────┘
```

## 系统要求

### 必需组件

1. **MATLAB**
   - MATLAB R2020a或更高版本
   - Parallel Computing Toolbox（GPU加速，可选）

2. **MATLAB Engine API for Python**
   ```bash
   cd "matlabroot/extern/engines/python"
   python setup.py install
   ```

3. **Python环境**
   - Python 3.8+
   - PyTorch 1.10+
   - 其他依赖见 requirements.txt

### 检查MATLAB Engine安装

```python
import matlab.engine
print(matlab.__version__)  # 应该显示版本号
```

## 使用方法

### 完整模式（使用MATLAB仿真）

这是推荐的模式，会运行真实的MATLAB仿真：

```bash
python experiments/visualize_realtime_correction_demo.py \
    --checkpoint checkpoints/realtime_corrector/best_model.pth \
    --gcode test_gcode_files/3DBenchy_PLA_1h28m.gcode \
    --layer 25 \
    --output_dir results/realtime_correction_demo \
    --device cuda
```

**参数说明**：
- `--checkpoint`: 模型检查点路径（必需）
- `--gcode`: G-code文件路径（默认：test_gcode_files/3DBenchy_PLA_1h28m.gcode）
- `--layer`: 要仿真的层编号（默认：25）
- `--output_dir`: 输出目录（默认：results/realtime_correction_demo）
- `--device`: 计算设备（默认：cuda如果可用，否则cpu）

### 演示模式（跳过MATLAB仿真）

如果不想安装MATLAB Engine API，可以使用现有数据演示：

```bash
python experiments/visualize_realtime_correction_demo.py \
    --checkpoint checkpoints/realtime_corrector/best_model.pth \
    --skip_matlab \
    --output_dir results/demo
```

**注意**：此模式使用现有的仿真数据，不会运行新的MATLAB仿真。

## 输出结果

脚本会在输出目录生成以下文件：

### 1. 可视化图表

- `realtime_correction_comparison.png` - 修正前后误差热图对比
  - 左图：原始仿真（未修正）
  - 右图：修正后仿真

- `realtime_correction_stats.png` - 详细统计图
  - 误差分布对比
  - X/Y轴误差相关性
  - 改善率分布

### 2. 统计报告

`correction_stats.json` - JSON格式的统计报告：

```json
{
  "original": {
    "mean_error_mm": 0.127,
    "max_error_mm": 0.452,
    "std_error_mm": 0.095,
    "x_mean_error_mm": 0.089,
    "y_mean_error_mm": 0.091
  },
  "corrected": {
    "mean_error_mm": 0.063,
    "max_error_mm": 0.234,
    "std_error_mm": 0.048,
    "x_mean_error_mm": 0.044,
    "y_mean_error_mm": 0.046
  },
  "improvement": {
    "mean_improvement_percent": 50.2,
    "median_improvement_percent": 48.7,
    "mean_error_reduction_percent": 50.4
  }
}
```

### 3. 中间数据文件

- `original_simulation.mat` - 原始仿真数据
- `corrected_trajectory.mat` - 修正后的轨迹数据
- `corrected_simulation.mat` - 修正后重新仿真的数据

## 实时预测过程

脚本使用滑动窗口方法进行实时预测：

```python
# 序列长度
seq_len = 20  # 0.2秒历史 @ 100Hz

# 滑动窗口
history = deque(maxlen=seq_len)

# 对每个时间步
for i in range(n_points):
    # 获取当前特征 [x, y, vx, vy]
    features = [x_ref[i], y_ref[i], vx_ref[i], vy_ref[i]]

    # 更新历史
    history.append(features)

    # 如果历史不足，跳过预测
    if len(history) < seq_len:
        predicted_error = [0, 0]
    else:
        # 准备输入序列 [seq_len, 4]
        seq = np.array(history)

        # 归一化
        seq_norm = scaler.transform(seq)

        # LSTM预测
        pred = model(seq_norm)  # [1, 2]
        predicted_error = pred[0]

    # 应用修正
    corrected_x[i] = x_ref[i] - predicted_error[0]
    corrected_y[i] = y_ref[i] - predicted_error[1]
```

## 常见问题

### Q1: MATLAB Engine安装失败

**错误**：`ImportError: No module named 'matlab'`

**解决方案**：
```bash
# 找到MATLAB安装目录
matlabroot

# 进入Python引擎目录
cd "matlabroot/extern/engines/python"

# 安装
python setup.py install

# 如果有多个Python版本，指定版本
python3 setup.py install
```

### Q2: MATLAB引擎启动失败

**错误**：`MatlabEngineError`

**解决方案**：
1. 确保MATLAB已正确安装并激活
2. 检查MATLAB版本（需要R2020a或更高）
3. 尝试重新安装MATLAB Engine API

### Q3: 内存不足

**错误**：`Out of memory`

**解决方案**：
1. 使用更小的层（`--layer 1` 而不是 `--layer 25`）
2. 减少仿真的层数
3. 关闭其他应用程序

### Q4: GPU相关错误

**解决方案**：
```bash
# 使用CPU模式
python experiments/visualize_realtime_correction_demo.py \
    --checkpoint checkpoints/realtime_corrector/best_model.pth \
    --device cpu
```

## 性能优化

### 加速MATLAB仿真

1. **使用GPU加速**：
   ```matlab
   % 在脚本中设置
   use_gpu = true;
   ```

2. **减少仿真层数**：
   ```bash
   # 只仿真一层
   --layer 1
   ```

3. **使用更简单的模型**：
   - 选择层数较少的G-code（如simple_boat5）

### 加速Python预测

1. **使用更大的步幅**：
   ```python
   # 修改脚本中的stride参数
   stride = 4  # 而不是1
   ```

2. **使用GPU**：
   ```bash
   --device cuda
   ```

## 验证结果

检查修正效果：

```python
import json
import matplotlib.pyplot as plt

# 加载统计报告
with open('results/realtime_correction_demo/correction_stats.json') as f:
    stats = json.load(f)

# 打印关键指标
print(f"原始平均误差: {stats['original']['mean_error_mm']:.4f} mm")
print(f"修正平均误差: {stats['corrected']['mean_error_mm']:.4f} mm")
print(f"改善率: {stats['improvement']['mean_error_reduction_percent']:.1f}%")

# 查看可视化
import cv2
img = cv2.imread('results/realtime_correction_demo/realtime_correction_comparison.png')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
```

## 扩展功能

### 测试多个层

```bash
# 测试不同层
for layer in 1 10 25 48; do
    python experiments/visualize_realtime_correction_demo.py \
        --checkpoint checkpoints/realtime_corrector/best_model.pth \
        --layer $layer \
        --output_dir results/layer_$layer
done
```

### 测试不同模型

```bash
# 对比不同模型
for model in checkpoints/*/best_model.pth; do
    python experiments/visualize_realtime_correction_demo.py \
        --checkpoint $model \
        --output_dir results/$(basename $(dirname $model))
done
```

## 相关文档

- [训练流程](docs/methods/training_pipeline.md) - 如何训练模型
- [评估指南](docs/experiments/metrics.md) - 评估指标说明
- [仿真系统](docs/methods/simulation_system.md) - MATLAB仿真详情

## 技术支持

遇到问题时：
1. 检查输出日志中的错误信息
2. 确认所有依赖已正确安装
3. 查看本文档的常见问题部分
4. 提交Issue到项目仓库

---

**最后更新**: 2026-02-02
