# 3D打印轨迹误差修正系统 (PINN + Vision)

基于**物理信息神经网络（PINN）**和**计算机视觉**的3D打印轨迹误差自动修正系统。

## 🎯 核心思想

传统方法需要大量训练数据且泛化能力差。本系统采用：

1. **PINN模型** - 结合物理约束，减少数据需求
2. **视觉测量** - 自动从打印照片提取误差
3. **完全自动化** - Klipper + ESP-CAM实现无人值守数据收集

## 📊 系统架构

```
┌─────────────────┐
│  Klipper固件    │ ← Ender 3 V2打印机
│  (每层触发)      │
└────────┬────────┘
         │ HTTP
         ↓
┌─────────────────┐
│  Python服务     │ ← Raspberry Pi 4
│  - 调用ESP-CAM  │
│  - 处理图像     │
│  - 计算误差     │
└────────┬────────┘
         │ WiFi
         ↓
┌─────────────────┐
│  ESP32-CAM      │ ← 正上方拍摄
│  (拍照)         │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  蓝色PLA打印件  │
└─────────────────┘
```

## 🚀 快速开始

### 1. 硬件准备

| 组件 | 价格 | 用途 |
|------|------|------|
| Raspberry Pi 4 | ¥250 | 主机 |
| ESP32-CAM | ¥80 | 摄像头 |
| 蓝色PLA | ¥50 | 打印材料 |

**总成本**: 约¥350（不含打印机）

### 2. 安装系统

```bash
# 安装Klipper固件
参考：docs/SYSTEM_SETUP.md

# 安装Python依赖
pip install -r requirements.txt
```

### 3. 测试硬件

```bash
python experiments/setup_hardware.py --test all
```

### 4. 收集数据

```bash
# 启动服务
python experiments/auto_data_collector.py

# 开始打印，自动收集数据
```

### 5. 训练模型

```bash
python training/train_pinn.py \
    --mode hybrid \
    --real_data data/collected_photos/dataset.npz \
    --sim_data "data_simulation_*" \
    --epochs 500
```

### 6. 应用修正

```bash
python experiments/apply_correction.py \
    --gcode test.gcode \
    --checkpoint checkpoints/pinn/best_model.pth
```

## 📖 详细文档

- **QUICKSTART.md** - 5分钟快速上手
- **docs/SYSTEM_SETUP.md** - 完整硬件安装指南
- **docs/AUTO_DATA_COLLECTION_GUIDE.md** - 数据收集使用指南

## 🎁 核心优势

| 特性 | 传统方法 | 本系统 |
|------|---------|--------|
| 数据需求 | 大量（千级样本） | 少量（百级） |
| 泛化能力 | 差 | 强（物理约束） |
| 人工干预 | 全手动 | 完全自动 |
| 成本 | 低 | 中（¥350） |
| 准确度 | 依赖模型 | 真实数据驱动 |

## 📂 项目结构

```
├── models/          # PINN模型定义
├── training/        # 训练脚本
├── utils/           # 视觉处理工具
├── experiments/     # 实验脚本
├── data/            # 数据存储
└── docs/            # 详细文档
```

## 🔬 技术栈

- **深度学习**: PyTorch
- **计算机视觉**: OpenCV
- **3D打印**: Klipper + G-code
- **硬件**: Raspberry Pi + ESP32-CAM

## 📈 预期效果

- RMS误差减少：**30-50%**
- 从±100um → ±50-70um
- 转角处改善更明显

## 🛠️ 故障排查

```bash
# 查看日志
tail -f data/collection.log

# 测试连接
python experiments/setup_hardware.py --test klipper
python experiments/setup_hardware.py --test espcam
```

## 🤝 贡献

欢迎提Issue和Pull Request！

## 📄 许可证

MIT License

---

**状态**: 开发中
**分支**: `feature/pinn-vision-based-correction`
**最后更新**: 2025-02-05
