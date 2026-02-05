# 完全自动化数据收集系统使用指南

## 系统概述

这是一个基于**Klipper + ESP-CAM + Python**的完全自动化数据收集系统，用于训练PINN模型进行3D打印轨迹误差修正。

### 核心特点

- ✅ **完全自动化**：每层打印完成时自动拍照
- ✅ **高精度测量**：计算机视觉提取轮廓误差
- ✅ **零人工干预**：从打印到数据处理全自动化
- ✅ **批量数据收集**：可连续收集多个打印任务的数据

---

## 硬件配置

### 必需组件

| 组件 | 型号/规格 | 用途 | 参考价格 |
|------|----------|------|---------|
| Raspberry Pi | 4B (2GB/4GB) | 主机+Klipper运行 | ¥200-300 |
| MicroSD卡 | 32GB Class 10 | 系统存储 | ¥30 |
| ESP32-CAM | OV2640摄像头 | 图像采集 | ¥80 |
| Ender 3 V2 | 原厂 | 3D打印机 | 已有 |
| PLA耗材 | 蓝色1.75mm | 打印材料 | ¥50 |

### 总成本

约**¥350-400**（不含打印机）

---

## 安装步骤

### 第一阶段：准备Raspberry Pi（30分钟）

#### 1. 烧录系统

```bash
# 下载Raspberry Pi Imager
# https://www.raspberrypi.com/software/

# 选择系统
OS: Raspberry Pi OS Lite (64-bit)
Storage: MicroSD卡

# 高级设置
✅ 启用SSH
✅ 设置用户名/密码
✅ 配置WiFi
```

#### 2. SSH连接并更新

```bash
ssh pi@<pi-ip-address>

# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y python3 python3-pip python3-venv git
sudo apt install -y build-essential python3-dev libncurses-dev
```

---

### 第二阶段：安装Klipper固件（1小时）

#### 1. 编译Klipper

```bash
cd ~
git clone https://github.com/Klipper3d/klipper.git
cd klipper
make menuconfig

# 配置：
# Microcontroller: STM32F103
# 250000 baud
# Disable console勾选（可选）

make
```

#### 2. 刷入Ender 3 V2

```bash
# 1. 断电，短接Boot0
# 2. 连接USB到Pi
# 3. 刷入固件
dfu-util -a 0 -D klipper/out/klipper.bin

# 4. 移除Boot0短接，重启
```

#### 3. 配置Klipper

```bash
mkdir -p ~/printer_config

# 下载Ender 3 V2配置
cd ~/printer_config
wget https://www.klipper3d.org/Config/cartesian-20201016.zip
unzip cartesian-20201016.zip

# 或使用提供的配置文件
nano ~/printer_config/printer.cfg
```

#### 4. 添加数据收集宏

在`printer.cfg`末尾添加：

```ini
[gcode_macro LAYER_COMPLETE]
# 每层完成时自动调用
gcode:
    {action_call_http(
        method="POST",
        url="http://localhost:5000/capture",
        body={"layer": {printer.gcode_move.position.z},
               "filename": "{printer.print_stats.filename}"}
    )}
```

#### 5. 启动服务

```bash
# 创建服务
sudo nano /etc/systemd/system/klipper.service
```

```ini
[Unit]
Description=Klipper
After=network.target

[Service]
Type=simple
User=pi
ExecStart=/home/pi/klipper/klippy/klippy.py /home/pi/printer_config/printer.cfg
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable klipper
sudo systemctl start klipper
```

---

### 第三阶段：安装Moonraker和Mainsail（30分钟）

#### 1. 安装Moonraker

```bash
cd ~
git clone https://github.com/Arksine/moonraker.git
cd moonraker

python3 -m venv moonmoon-env
source moonmoon-env/bin/activate
pip install -r requirements.txt

mkdir -p ~/.moonraker
nano ~/.moonraker/moonraker.conf
```

配置内容见`docs/SYSTEM_SETUP.md`

#### 2. 安装Mainsail

```bash
cd ~
git clone https://github.com/mainsail-crew/mainsail.git

sudo apt install -y nginx
sudo ln -s ~/mainsail /var/www/mainsail
sudo systemctl restart nginx
```

访问：`http://<pi-ip-address>`

---

### 第四阶段：配置ESP-CAM（45分钟）

#### 1. 安装Arduino IDE

在PC或Pi上安装Arduino IDE

#### 2. 烧录ESP-CAM

打开：`File` → `Examples` → `ESP32` → `Camera` → `CameraWebServer`

修改：
```cpp
#define CAMERA_MODEL_AI_THINKER
const char* ssid = "YourWiFi";
const char* password = "YourPassword";
```

烧录到ESP32-CAM

#### 3. 测试ESP-CAM

```bash
# 在浏览器访问
http://<esp32-ip-address>/

# 或命令行
curl http://<esp32-ip-address>/capture -o test.jpg
```

---

### 第五阶段：安装Python环境（15分钟）

```bash
# 克隆项目
cd ~
git clone <your-repo-url>
cd 3d_printer_pinn4ieee

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

---

## 使用流程

### 1. 硬件测试

```bash
# 运行完整测试
python experiments/setup_hardware.py --test all

# 单独测试某个组件
python experiments/setup_hardware.py --test espcam
python experiments/setup_hardware.py --test vision
```

### 2. 校准摄像头

```bash
python experiments/setup_hardware.py --test calibration

# 按提示：
# 1. 打印20x20mm校准方块
# 2. 放置在打印中心
# 3. 输入实际尺寸
# 4. 自动校准像素比例
```

### 3. 启动数据收集服务

```bash
# 在一个终端窗口
cd ~/3d_printer_pinn4ieee
source venv/bin/activate
python experiments/auto_data_collector.py \
    --espcam http://192.168.1.100 \
    --port 5000 \
    --output data/collected_photos
```

你会看到：
```
Starting auto data collection service...
  HTTP port: 5000
  ESP-CAM: http://192.168.1.100
  Output: data/collected_photos

Waiting for Klipper requests...
```

### 4. 开始打印

在Mainsail界面：
1. 上传测试G-code
2. 开始打印
3. **每层完成时会自动拍照**
4. 观察终端输出：
```
Processing layer 1
  Saved: data/collected_photos/test_layer001_20250205_143022.jpg
  Layer 1 processed:
    Points: 1234
    RMS error: 45.23 um
```

### 5. 停止并保存

打印完成后：
```bash
# 方法1：在浏览器访问
http://<pi-ip-address>:5000/stop

# 方法2：Ctrl+C停止服务（会自动保存）
```

数据会自动保存为：
```
data/collected_photos/
├── dataset_20250205_143022.npz
├── metadata_20250205_143022.json
├── test_layer001_*.jpg
├── test_layer002_*.jpg
└── ...
```

---

## 数据格式

### NPZ数据集

```python
data = np.load('dataset_20250205_143022.npz')

# 包含：
data['layers']         # 层数列表
data['gcode_files']    # G-code文件名
data['image_paths']    # 图像路径
data['contours']       # 轮廓点 [n_layers, n_points, 2]
data['errors']         # 误差向量 [n_layers, n_points, 2]
```

### JSON元数据

```json
{
  "job": {
    "filename": "test.gcode",
    "start_time": "2025-02-05T14:30:22",
    "layers_collected": 50
  },
  "total_layers": 50,
  "output_file": "data/collected_photos/dataset_20250205_143022.npz"
}
```

---

## Klipper G-code集成

### 自动触发每层拍照

在G-code文件中添加（或使用 slicer 后处理）：

```gcode
; 在每层开始时
LAYER_CHANGE
; 在每层结束时
LAYER_COMPLETE

; Klipper会自动调用LAYER_COMPLETE宏
```

### 自定义层触发间隔

如果不需要每层都拍照，可以修改：

```ini
[gcode_macro LAYER_N]
# 每5层触发一次
gcode:
    {% set layer_z = printer.gcode_move.position.z %}
    {% set layer_num = (layer_z / 0.2) | int %}  # 0.2mm层高
    {% if layer_num % 5 == 0 %}
        {action_call_http(
            method="POST",
            url="http://localhost:5000/capture",
            body={"layer": layer_num, "filename": "test"}
        )}
    {% endif %}
```

---

## 常见问题

### Q1: ESP-CAM无法连接

```bash
# 检查ESP-CAM是否在线
ping 192.168.1.100

# 检查防火墙
sudo ufw status

# 如果防火墙阻止
sudo ufw allow 5000/tcp
```

### Q2: 照片全黑或过曝

调整ESP-CAM曝光：
```cpp
// 在ESP-CAM代码中
s->set_exposure_control(10);  // 0-120，越大越亮
```

或使用外部LED补光

### Q3: 轮廓提取失败

可能原因：
1. 材料颜色不对（用蓝色PLA，不要用黑色）
2. 照明不足（增加环境光或补光）
3. 摄像头太高/太低（调整到30-40cm高度）
4. 对焦不准（转动ESP-CAM镜头对焦）

### Q4: Klipper宏不执行

检查日志：
```bash
sudo journalctl -u klipper -f
```

确认宏已定义在`printer.cfg`中

---

## 优化建议

### 提高数据质量

1. **固定照明条件**
   - 使用LED灯带固定在打印仓内
   - 避免日光直射

2. **控制环境**
   - 关闭门窗，避免气流干扰
   - 控制温度（PLA最佳20-25°C）

3. **定期校准**
   - 每周重新校准一次像素比例
   - 检查ESP-CAM是否松动

### 提高收集效率

1. **批量打印**
   - 准备多个测试件G-code
   - 连续打印，自动收集

2. **多种测试件**
   - 20mm立方（尺寸精度）
   - 3DBenchy（综合测试）
   - 圆形测试件（圆度）
   - 尖角测试件（转角误差）

---

## 下一步

数据收集完成后：

1. **训练PINN模型**
   ```bash
   python training/train_pinn.py \
       --mode hybrid \
       --real_data data/collected_photos/dataset_*.npz \
       --sim_data data_simulation_* \
       --epochs 500
   ```

2. **应用修正**
   ```bash
   python experiments/apply_correction.py \
       --gcode test.gcode \
       --checkpoint checkpoints/pinn/best_model.pth
   ```

3. **验证效果**
   - 打印修正后的G-code
   - 再次测量误差
   - 评估改进效果

---

## 技术支持

### 日志位置

- Klipper: `sudo journalctl -u klipper -f`
- Moonraker: `sudo journalctl -u moonraker -f`
- 数据收集: `tail -f data/collection.log`

### 配置文件

- Klipper: `~/printer_config/printer.cfg`
- Moonraker: `~/.moonraker/moonraker.conf`

### 重启服务

```bash
sudo systemctl restart klipper
sudo systemctl restart moonraker
```
