# 🚀 快速开始指南

## 场景1：只想测试系统（无硬件）

如果你想先测试PINN模型和仿真器：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 测试PINN模型
python models/pinn_trajectory_model.py

# 3. 测试物理仿真器
python data/gcode_physics_simulator_enhanced.py

# 4. 测试视觉处理（如果有测试图像）
python utils/vision_processor.py
```

---

## 场景2：已安装Klipper，想快速收集数据

如果你已经配置好Klipper + ESP-CAM：

```bash
# 1. 测试硬件连接
python experiments/setup_hardware.py --test all

# 2. 校准摄像头
python experiments/setup_hardware.py --test calibration

# 3. 启动数据收集服务
python experiments/auto_data_collector.py \
    --espcam http://192.168.1.100 \
    --output data/my_collection

# 4. 在Mainsail开始打印，数据自动收集
```

---

## 场景3：已有数据，想训练模型

```bash
# 使用仿真数据预训练
python training/train_pinn.py \
    --mode pretrain \
    --sim_data "data_simulation_*" \
    --epochs 500 \
    --output checkpoints/pinn_pretrain

# 使用真实数据微调
python training/train_pinn.py \
    --mode finetune \
    --real_data data/real_measurements/dataset.npz \
    --resume checkpoints/pinn_pretrain/best_model.pth \
    --epochs 100 \
    --output checkpoints/pinn_final
```

---

## 场景4：模型已训练，想应用修正

```bash
python experiments/apply_correction.py \
    --gcode test_parts/benchy.gcode \
    --checkpoint checkpoints/pinn_final/best_model.pth \
    --output results/corrected/
```

---

## 完整工作流程（推荐）

### 第1步：硬件设置（4-6小时）

```bash
# 1.1 安装Klipper固件
# 参考：docs/SYSTEM_SETUP.md

# 1.2 安装Moonraker和Mainsail
# 参考：docs/SYSTEM_SETUP.md

# 1.3 配置ESP-CAM
# 参考：docs/SYSTEM_SETUP.md
```

### 第2步：硬件测试（30分钟）

```bash
# 测试所有组件
python experiments/setup_hardware.py --test all

# 如有失败，单独测试
python experiments/setup_hardware.py --test klipper
python experiments/setup_hardware.py --test espcam
python experiments/setup_hardware.py --test vision
```

### 第3步：校准（15分钟）

```bash
# 打印20x20mm校准方块
python experiments/setup_hardware.py --test calibration
```

### 第4步：收集数据（数小时到数天）

```bash
# 启动服务
python experiments/auto_data_collector.py

# 打印多个测试件：
# - 20mm立方
# - 3DBenchy
# - 圆形测试件
# - 尖角测试件
```

### 第5步：训练模型（2-4小时）

```bash
# 混合训练
python training/train_pinn.py \
    --mode hybrid \
    --real_data data/collected_photos/dataset_*.npz \
    --sim_data "data_simulation_*" \
    --epochs 500
```

### 第6步：验证效果（数小时）

```bash
# 应用修正
python experiments/apply_correction.py \
    --gcode test_parts/benchy.gcode \
    --checkpoint checkpoints/pinn/best_model.pth

# 打印修正后的G-code
# 再次视觉测量
# 评估改进
```

---

## 文件结构

```
3d_printer_pinn4ieee/
├── docs/
│   ├── SYSTEM_SETUP.md           # 系统设置详细指南
│   └── AUTO_DATA_COLLECTION_GUIDE.md  # 数据收集使用指南
│
├── models/
│   └── pinn_trajectory_model.py  # PINN模型定义
│
├── training/
│   └── train_pinn.py             # 训练脚本
│
├── utils/
│   └── vision_processor.py       # 视觉处理工具
│
├── experiments/
│   ├── auto_data_collector.py    # 自动数据收集服务
│   ├── setup_hardware.py         # 硬件测试工具
│   ├── apply_correction.py       # 应用修正
│   └── evaluate_pinn.py          # 评估模型
│
├── data/
│   └── collected_photos/         # 收集的照片和数据
│
└── checkpoints/
    └── pinn/                     # 训练好的模型
```

---

## 常用命令

### 系统服务管理

```bash
# Klipper
sudo systemctl start klipper
sudo systemctl stop klipper
sudo systemctl restart klipper
sudo journalctl -u klipper -f

# Moonraker
sudo systemctl start moonraker
sudo systemctl stop moonraker
sudo systemctl restart moonraker

# Mainsail
sudo systemctl restart nginx
```

### 数据收集

```bash
# 测试连接
curl http://192.168.1.100/capture -o test.jpg

# 测试API服务
curl http://localhost:5000/status

# 手动触发拍照
curl -X POST http://localhost:5000/capture \
    -H "Content-Type: application/json" \
    -d '{"layer": 1, "filename": "test.gcode"}'
```

### 模型训练

```bash
# 监控训练（另一个终端）
tensorboard --logdir checkpoints/pinn/logs

# 继续训练
python training/train_pinn.py \
    --resume checkpoints/pinn/best_model.pth \
    --epochs 500
```

---

## 故障排查速查表

| 问题 | 检查项 | 解决方案 |
|------|-------|---------|
| Klipper无法连接 | 服务状态 | `sudo systemctl restart klipper` |
| ESP-CAM离线 | WiFi/电源 | 检查电源，重启ESP-CAM |
| 照片全黑 | 曝光设置 | 调整ESP-CAM曝光或增加照明 |
| 无法提取轮廓 | 材料颜色 | 使用蓝色/白色PLA |
| 训练无显存 | Batch size | 减小`--batch_size` |
| 修正后误差更大 | 模型预测 | 检查验证集R²分数 |

---

## 预期效果

### 数据收集

- 每层约5-10秒处理时间
- 单层照片约2-3MB
- 100层打印约200-300MB数据

### 模型训练

- 预训练（仅仿真）：约2小时（500 epochs）
- 微调（真实数据）：约30分钟（100 epochs）

### 修正效果

- 预期RMS误差减少：30-50%
- 从±100um → ±50-70um
- 转角处改善更明显

---

## 下一步

完成数据收集和模型训练后：

1. **部署到生产环境**
   - 将修正后的G-code用于实际打印
   - 持续收集数据并迭代模型

2. **优化和改进**
   - 尝试不同的网络结构
   - 调整物理约束权重
   - 添加更多特征（曲率、加速度变化率）

3. **论文撰写**
   - 整理实验数据
   - 绘制对比图表
   - 撰写技术论文

---

## 技术支持

- 文档：`docs/`目录
- 日志：`data/collection.log`
- Issues：GitHub Issues

有问题请先查看日志和文档！
