# 3D Printer PINN-Seq3D Framework

> 基于物理信息神经网络和序列模型的3D打印质量预测与轨迹优化系统

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🎯 项目简介

本项目提出了一种统一的深度学习框架，结合**物理信息神经网络（PINN）**和**序列模型**，解决3D打印中的两个核心问题：

1. **质量预测与早停系统**：实时监测打印状态，预测打印质量指标，实现早期故障检测和打印终止决策
2. **轨迹优化与误差补偿**：针对快速转角等关键位置预测误差，实时调整打印轨迹，提升打印精度

## ✨ 核心特性

- 🔬 **物理信息引导**：将热力学、振动动力学、能量守恒等物理定律嵌入神经网络
- 🧠 **混合序列建模**：Transformer + BiLSTM 捕捉多尺度时序依赖关系
- 🎭 **多任务学习**：统一框架同时处理质量预测、故障分类和轨迹优化
- 📦 **模块化设计**：清晰的代码结构，易于扩展和复用
- ⚡ **高性能**：支持混合精度训练、多GPU并行，推理速度 >100Hz

## 🏗️ 项目架构

```
输入：传感器时序数据 [batch, seq_len, features]
         ↓
    共享编码器 (PINN-Guided Transformer)
         ↓
    ┌─────┴─────┬─────────┬─────────┐
    ↓           ↓         ↓         ↓
质量预测    故障分类   轨迹校正   物理场重建
    ↓           ↓         ↓         ↓
  RUL、温度   4类故障   误差补偿   状态重建
```

详细的项目结构说明请查看 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## 📊 性能指标

### 质量预测模块
- ✅ 温度预测 RMSE: < 0.5°C
- ✅ 振动预测 RMSE: < 0.02mm
- ✅ RUL预测 RMSE: < 50s
- ✅ 故障分类准确率: > 95%

### 轨迹校正模块
- ✅ 转角误差减少: > 90%
- ✅ 平均预测误差: < 5mm
- ✅ 实时推理速度: > 100Hz

## 🚀 快速开始

### 环境安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/3d_printer_pinn4ieee.git
cd 3d_printer_pinn4ieee

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

```bash
# 生成物理仿真数据（质量预测）
python data/scripts/generate_physics_data.py

# 生成轨迹数据（轨迹校正）
python data/scripts/generate_trajectory_data.py
```

### 训练模型

```bash
# 训练统一模型（推荐）
python experiments/train_unified_model.py --config config/base_config.py

# 或分别训练各个模块
python experiments/train_quality_model.py
python experiments/train_trajectory_model.py
```

### 推理预测

```python
from inference.predictor import UnifiedPredictor

# 加载模型
predictor = UnifiedPredictor.load_from_checkpoint(
    'checkpoints/unified_model/best_model.pth'
)

# 实时预测
results = predictor.predict(sensor_data)

# 获取结果
quality_metrics = results['quality']      # 质量指标
fault_prediction = results['fault']       # 故障分类
trajectory_correction = results['trajectory']  # 轨迹校正
```

## 📁 目录结构

```
3d_printer_pinn4ieee/
├── config/              # 配置文件
├── data/                # 数据目录
├── models/              # 模型定义
│   ├── encoders/        # 编码器
│   ├── decoders/        # 解码器
│   └── physics/         # 物理约束
├── training/            # 训练模块
├── evaluation/          # 评估模块
├── inference/           # 推理模块
├── utils/               # 工具函数
└── experiments/         # 实验脚本
```

## 🔬 技术细节

### 物理约束

模型嵌入以下物理定律：

1. **热力学方程**：`∂T/∂t = α∇²T + Q(x,t)`
2. **振动动力学**：`m·d²x/dt² + c·dx/dt + k·x = F(t)`
3. **能量守恒**：`dE/dt = P_in - P_out - P_loss`
4. **电机耦合**：`I_motor ∝ acceleration + vibration_load`

### 损失函数

```python
total_loss = λ_quality × L_quality +
             λ_fault × L_fault +
             λ_trajectory × L_trajectory +
             λ_physics × L_physics
```

## 📈 实验结果

我们的方法在多个数据集上进行了验证：

| 数据集 | 任务 | 指标 | 性能 |
|--------|------|------|------|
| 仿真数据 | 故障分类 | 准确率 | 100% |
| 仿真数据 | RUL预测 | RMSE | 48.38s |
| 实测数据 | 轨迹校正 | 误差减少 | 96.87% |

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📝 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📧 联系方式

- 作者：Your Name
- 邮箱：your.email@example.com
- GitHub Issues：[提交问题](https://github.com/yourusername/3d_printer_pinn4ieee/issues)

## 🙏 致谢

感谢以下项目和资源的启发：
- Physics-Informed Neural Networks (PINN)
- Transformer架构
- 3D打印开源社区

## 📚 参考文献

```bibtex
@article{raissi2019physics,
  title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George E},
  journal={Journal of Computational Physics},
  year={2019}
}

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}
```

---

⭐ 如果这个项目对你有帮助，请给我们一个星标！
