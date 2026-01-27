# MATLAB仿真运行说明

本项目使用MATLAB进行物理仿真，以生成高保真度的3D打印数据。本文档说明如何使用根目录下的G-code文件的特定层（例如第10层外墙）作为输入轨迹运行仿真。

## 环境要求

- MATLAB R2020a 或更高版本
- Statistics and Machine Learning Toolbox
- Signal Processing Toolbox

## 运行仿真：使用第10层外墙作为输入

要运行仿真并使用[Tremendous Hillar_PLA_17m1s.gcode](file:///TJ/3d_print/3d_printer_pinn4ieee/examples/Tremendous%20Hillar_PLA_17m1s.gcode)文件的第10层外墙作为输入轨迹，请按以下步骤操作：

### 1. 启动MATLAB

打开MATLAB并确保当前工作目录设置为项目的根目录：

```matlab
cd('f:\TJ\3d_print\3d_printer_pinn4ieee')
```

### 2. 添加路径

```matlab
addpath(genpath(fullfile(pwd, 'matlab_simulation')))
```

### 3. 运行仿真

使用以下MATLAB代码运行仿真，专门提取第10层的外墙轨迹：

```matlab
% 设置仿真选项
options = struct();
options.layers = 10;                                    % 指定第10层
options.include_type = {'Outer wall'};                  % 仅包含外墙
options.include_skirt = false;                          % 不包含裙边
options.use_improved = true;                            % 使用改进的解析器

% 运行完整仿真
gcode_file = 'Tremendous Hillar_PLA_17m1s.gcode';
output_file = 'layer10_outer_wall_simulation.mat';

simulation_data = run_full_simulation(gcode_file, output_file, options);
```

或者，您可以使用单行命令：

```matlab
% 直接指定选项的单行命令
simulation_data = run_full_simulation('Tremendous Hillar_PLA_17m1s.gcode', 'layer10_outer_wall_simulation.mat', struct('layers', 10, 'include_type', {'Outer wall'}, 'use_improved', true));
```

### 4. 转换数据供Python使用

仿真完成后，使用提供的转换脚本将MATLAB数据转换为Python格式：

```bash
python matlab_simulation/convert_matlab_to_python.py layer10_outer_wall_simulation.mat training -o training_data_layer10_outer_wall.h5
```

## 详细说明

- `options.layers = 10`：指定仅处理第10层
- `options.include_type = {'Outer wall'}`：仅提取外墙轨迹，忽略内墙、填充等
- `options.use_improved = true`：使用改进的G-code解析器，更适合2D轨迹提取
- 仿真将生成包含轨迹误差、热场、层间粘结强度等物理特性的综合数据集

## 可选参数

您可以根据需要调整以下参数：

- `options.layers`：可以是单个数字（如10）、向量（如[5,10,15]）、'first'（第一层）或'all'（所有层）
- `options.include_type`：可以包含多种类型，如{'Outer wall', 'Inner wall'}
- `options.min_segment`：最小线段长度（单位mm），过滤掉过短的线段

## 仿真输出

仿真完成后，您将得到：

1. `.mat`文件：包含完整的仿真数据结构
2. 可视化图表：显示各种物理参数随时间的变化
3. 转换后的`.h5`文件：可用于训练神经网络的格式

## 注意事项

- 确保MATLAB环境中安装了必要的工具箱
- 仿真时间取决于轨迹复杂度，可能需要几分钟到几十分钟
- 生成的仿真数据包含50多个物理变量，可用于训练PINN模型