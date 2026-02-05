# 针对现有Klipper + IP摄像头的完整启动指南

## 您的当前配置

- **Klipper**: 10.168.1.123:19255
- **Windows PC**: 10.168.1.118 (数据收集服务运行在此机器)
- **IP摄像头**: 10.168.1.129:8080
  - 快照URL: `http://10.168.1.129:8080/shot.jpg`
  - MJPEG流: `http://10.168.1.129:8080/video`

---

## 第一步：测试硬件连接（5分钟）

### 1.1 测试IP摄像头

```bash
# 在浏览器中打开
http://10.168.1.129:8080/video

# 或使用命令行测试
curl http://10.168.1.129:8080/shot.jpg -o test_camera.jpg
```

如果能看到图像，摄像头工作正常。

### 1.2 测试Klipper API

```bash
# 测试Klipper连接
curl http://10.168.1.123:19255/printer/info

# 应该返回打印机信息JSON
```

### 1.3 使用Python测试工具

```bash
# 在Windows PC上
cd F:\TJ\3d_print\3d_printer_pinn4ieee
python experiments/test_existing_setup.py
```

这会自动测试两个组件并显示结果。

---

## 第二步：配置Klipper宏（10分钟）

### 2.1 编辑printer.cfg

在Mainsail界面：
1. 点击右上角 "配置"
2. 找到 `printer.cfg` 文件
3. 在文件末尾添加以下内容：

```ini
# 数据收集宏
[gcode_macro LAYER_COMPLETE]
description: "每层完成时触发拍照"
gcode:
    {action_respond_info("Layer {printer.gcode_move.position.z} complete, capturing...")}
    {% set http_ok = True %}
    {% if http_ok %}
        {action_call_http(
            method="POST",
            url="http://10.168.1.118:5000/capture",
            body={"layer": printer.gcode_move.position.z|int,
                   "filename": printer.print_stats.filename}
        )}
    {% endif %}
```

### 2.2 重启Klipper

在Mainsail界面：
1. 点击 "重启 Klipper"
2. 等待重启完成

或使用命令行：
```bash
sudo systemctl restart klipper
```

---

## 第三步：配置Slicer后处理（5分钟）

### 选项A：Cura（推荐）

1. 打开Cura
2. 打印设置 → 后处理脚本
3. 添加 "Pause at height" 或 "Post-processing"
4. 在每层结束后插入：`LAYER_COMPLETE`

### 选项B：PrusaSlicer

1. 打开PrusaSlicer
2. 打印设置 → 后处理
3. 创建新后处理脚本
4. 在G-code中插入：`LAYER_COMPLETE`

### 选项C：SuperSlicer（同PrusaSlicer）

### 选项D：手动添加（最简单）

1. 切片G-code
2. 打开G-code文件（文本编辑器）
3. 使用查找替换：
   - 查找：`;LAYER:`
   - 替换为：`;LAYER:\nLAYER_COMPLETE`
   - 或在每个 `;TYPE:` 后添加 `LAYER_COMPLETE`

---

## 第四步：安装Python依赖（5分钟）

```bash
# 在Windows PC上，打开命令提示符或PowerShell

# 进入项目目录
cd F:\TJ\3d_print\3d_printer_pinn4ieee

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import cv2, requests, flask; print('依赖安装成功')"
```

---

## 第五步：启动数据收集服务（1分钟）

### 5.1 在Windows PC上运行

```bash
# 打开命令提示符或PowerShell

# 进入项目目录
cd F:\TJ\3d_print\3d_printer_pinn4ieee

# 启动服务
python experiments/auto_data_collector_existing.py \
    --klipper-host 10.168.1.123 \
    --klipper-port 19255 \
    --camera-host 10.168.1.129 \
    --camera-port 8080 \
    --output data/collected_photos
```

### 5.2 或者在后台运行（Windows）

```bash
# 使用start命令在后台运行
start /B python experiments/auto_data_collector_existing.py ^
    --klipper-host 10.168.1.123 ^
    --klipper-port 19255 ^
    --camera-host 10.168.1.129 ^
    --camera-port 8080 ^
    --output data/collected_photos ^
    > data\collection.log 2>&1

# 查看日志（另一个终端）
type data\collection.log
```

您会看到：
```
==================================================
自动数据收集服务启动
==================================================
  Klipper: 10.168.1.123:19255
  摄像头: 10.168.1.129:8080
  HTTP端口: 5000
  监听地址: http://10.168.1.118:5000
  输出目录: data/collected_photos

API端点:
  POST http://localhost:5000/capture
  GET  http://localhost:5000/status
  POST http://localhost:5000/save
  POST http://localhost:5000/stop

等待Klipper请求...
```

---

## 第六步：打印测试件并收集数据（数小时）

### 6.1 准备测试G-code

建议测试件：
1. **20mm立方** - 验证尺寸精度
2. **3DBenchy小船** - 综合测试
3. **圆形测试件** - 圆度精度

### 6.2 开始打印

在Mainsail中：
1. 上传测试G-code
2. 切片（如果还没切片）
3. 开始打印

### 6.3 观察数据收集

打印过程中，每层完成时会：
1. Klipper触发 `LAYER_COMPLETE` 宏
2. 发送HTTP请求到数据收集服务
3. 服务调用IP摄像头拍照
4. 处理图像并计算误差
5. 保存数据

您会看到类似输出：
```
INFO - 处理层 1
INFO -   图像已保存: test_layer001_20250205_143022.jpg
INFO -   处理成功: 1234点, RMS=45.23um
```

---

## 第七步：保存和管理数据

### 7.1 实时查看状态

在Windows PC的浏览器或终端：
```bash
# 查看收集状态
curl http://localhost:5000/status

# 或从其他机器
curl http://10.168.1.118:5000/status
```

返回：
```json
{
  "job": {
    "filename": "test.gcode",
    "start_time": "2025-02-05T14:30:22",
    "layers_collected": 15
  },
  "layers_collected": 15,
  "output_dir": "data/collected_photos"
}
```

### 7.2 手动保存数据集

打印完成后或打印过程中：
```bash
curl -X POST http://localhost:5000/save
```

或：
```bash
curl -X POST http://localhost:5000/stop
```

数据会保存为：
```
data/collected_photos/
├── dataset_20250205_143022.npz     # 训练数据
├── metadata_20250205_143022.json   # 元数据
├── test_layer001_*.jpg             # 原始照片
├── test_layer002_*.jpg
└── ...
```

---

## 常见问题

### Q1: LAYER_COMPLETE宏没有触发

**可能原因**：
- G-code中没有插入 `LAYER_COMPLETE`
- 宏配置有语法错误

**解决方案**：
1. 检查Klipper日志：`sudo journalctl -u klipper -f`
2. 手动测试：在Mainsail控制台输入 `LAYER_COMPLETE`
3. 确认slicer后处理已配置

### Q2: HTTP请求失败

**检查**：
1. 数据收集服务是否运行：检查Windows任务管理器
2. IP地址是否正确：`10.168.1.118:5000`
3. 防火墙是否阻止

**解决**：
```bash
# 在Windows PC上测试连接
curl http://localhost:5000/status

# 如果失败，检查服务日志
type data\collection.log
```

### Q3: 照片全黑或过曝

**解决**：
1. 检查IP摄像头焦距（转动镜头环）
2. 调整摄像头位置（30-40cm高度）
3. 改善照明（增加环境光或LED）

### Q4: 轮廓提取失败

**可能原因**：
- 材料颜色（需要蓝色/白色PLA）
- 照明不足
- 对焦不准

**解决**：
1. 使用蓝色PLA打印
2. 确保良好照明
3. 调整摄像头对焦

### Q5: 打印速度变慢

**可能原因**：
- HTTP请求延迟
- 图像处理耗时

**优化**：
1. 使用有线网络连接（而非WiFi）
2. 降低拍照频率（每5层拍1次）
3. 减小图像分辨率

---

## 数据查看和验证

### 查看收集的照片

```bash
ls -lh data/collected_photos/*.jpg
```

### 查看数据集信息

```python
import numpy as np

# 加载数据集
data = np.load('data/collected_photos/dataset_20250205_143022.npz')

# 查看信息
print(f"总层数: {len(data['layers'])}")
print(f"G-code文件: {data['gcode_files']}")

# 查看误差统计
for i, stats in enumerate(data['stats']):
    print(f"层{data['layers'][i]}: RMS={stats['rms_um']:.2f}um, Max={stats['max_um']:.2f}um")
```

---

## 下一步

数据收集完成后：

### 1. 训练PINN模型

```bash
python training/train_pinn.py \
    --mode finetune \
    --real_data data/collected_photos/dataset_*.npz \
    --epochs 200
```

### 2. 应用修正

```bash
python experiments/apply_correction.py \
    --gcode test.gcode \
    --checkpoint checkpoints/pinn/best_model.pth
```

### 3. 验证效果

打印修正后的G-code，再次测量误差，评估改进。

---

## 快速参考

### 启动服务（在Windows PC上）
```bash
python experiments/auto_data_collector_existing.py \
    --klipper-host 10.168.1.123 \
    --camera-host 10.168.1.129 \
    --output data/collected_photos
```

### 测试硬件
```bash
python experiments/test_existing_setup.py
```

### 查看日志（Windows）
```bash
type data\collection.log
```

### 保存数据
```bash
curl -X POST http://localhost:5000/save
```

---

## 需要帮助？

查看完整文档：
- `docs/SYSTEM_SETUP.md` - 硬件设置
- `docs/AUTO_DATA_COLLECTION_GUIDE.md` - 数据收集指南
- `QUICKSTART.md` - 快速开始
