# 快速开始指南

## 改进后的工作流程

### 1. 重新生成训练数据

```bash
# 进入项目目录
cd F:\TJ\3d_print\3d_printer_pinn4ieee

# 激活环境
/g/Miniconda3/envs/3dprint/python.exe

# 生成数据（使用所有G-code文件）
python data/regenerate_training_data.py \
    --gcode test_gcode_files/*.gcode \
    --output_dir data_python_parser_enhanced
```

### 2. 训练新模型

```bash
python models/train_realtime_corrector.py \
    --data_dir data_python_parser_enhanced \
    --output_dir checkpoints/realtime_corrector_v2 \
    --epochs 50 \
    --batch_size 256
```

### 3. 运行修正的闭环优化

```bash
python experiments/closedloop_gcode_optimizer_fixed.py \
    --gcode test_gcode_files/outline_test_pieces.gcode \
    --checkpoint checkpoints/realtime_corrector_v2/best_model.pth \
    --output_dir results/closedloop_fixed \
    --max_iterations 5 \
    --tolerance 20.0
```

### 4. 打印测试

```bash
# 原始版本
# 打印 test_gcode_files/outline_test_pieces.gcode

# 修正版本
# 打印 results/closedloop_fixed/outline_test_pieces_corrected.gcode
```

---

## 关键文件

| 文件 | 用途 |
|------|------|
| `simulation/simulate_trajectory_error_from_python.m` | 新版MATLAB仿真 |
| `data/gcode_parser_enhanced.py` | Python增强解析器 |
| `experiments/closedloop_gcode_optimizer_fixed.py` | 修正的闭环优化 |
| `data/regenerate_training_data.py` | 数据重生成 |

---

## 验证改进

### 检查Python解析器

```bash
python data/gcode_parser_enhanced.py test_gcode_files/outline_test_pieces.gcode
```

应该看到：
```
解析G-code文件: test_gcode_files/outline_test_pieces.gcode
  提取了 1675 个移动指令

计算轨迹（考虑速度变化）...
  轨迹点数: 1675
  总时间: 128.48 s
  速度范围: [50.0, 150.0] mm/s  # 确认速度在变化
```

### 检查MATLAB仿真

确保MATLAB能找到新函数：
```matlab
% 在MATLAB中
cd simulation
which simulate_trajectory_error_from_python
% 应该显示: simulate_trajectory_error_from_python.m
```

---

## 故障排除

### 问题1：MATLAB引擎启动失败

```bash
# 检查MATLAB引擎是否安装
cd "G:\MATLAB\extern\engines\python"
python setup.py install
```

### 问题2：找不到新函数

```matlab
% 在MATLAB中运行
rehash toolboxcache
which simulate_trajectory_error_from_python
```

### 问题3：数据生成失败

```bash
# 测试单个文件
python data/regenerate_training_data.py \
    --gcode test_gcode_files/outline_test_pieces.gcode \
    --output_dir test_output \
    --single
```

---

## 核心改进总结

### 修正1：优化目标

**❌ 旧版**：
```python
# 优化让 actual ≈ corrected
error = x_act - x_corrected
```

**✅ 新版**：
```python
# 优化让 actual ≈ ideal
error = x_act - x_ideal
```

### 修正2：速度处理

**❌ 旧版**：
```python
# MATLAB假设恒定速度
v = constant
```

**✅ 新版**：
```python
# Python提取实际F指令
v = extract_F_command(line) / 60  # mm/min → mm/s
```

---

## 下一步

1. ✅ 运行数据重生成
2. ✅ 训练新模型
3. ✅ 测试闭环优化
4. ✅ 打印实物验证
5. ✅ 撰写论文

---

**需要帮助？** 查看详细文档：`docs/IMPROVEMENTS_README.md`
