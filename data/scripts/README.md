# 数据脚本说明

## 概述

此目录包含与数据处理相关的脚本。由于项目重点转移到基于MATLAB的物理仿真，Python数据生成脚本已被移除。

## 当前数据流程

### 1. 物理仿真数据生成

使用MATLAB进行高保真物理仿真：

```matlab
% 运行完整仿真
simulation_data = run_full_simulation('your_gcode_file.gcode');
```

仿真包含:
- 轨迹误差模型（二阶系统动力学）
- 温度场模型（移动热源）
- 层间粘结强度模型
- G-code特征提取

### 2. MATLAB到Python数据转换

将MATLAB仿真结果转换为Python训练数据:

```bash
python matlab_simulation/convert_matlab_to_python.py \
    "simulation_output.mat" \
    training \
    -o training_data.h5
```

### 3. 数据目录结构

- `raw/` - 原始数据（如果可用）
- `simulation/` - 仿真数据
- `processed/` - 处理后的训练数据

## 数据处理流程

1. 使用MATLAB仿真生成物理一致性数据
2. 使用[convert_matlab_to_python.py](file:///TJ/3d_print/3d_printer_pinn4ieee/matlab_simulation/convert_matlab_to_python.py)转换为Python格式
3. 数据自动分为训练集、验证集和测试集
4. 特征标准化处理