# 完全自动化数据收集系统设置指南

## 硬件清单

### 必需组件
- [x] Raspberry Pi 4 (2GB或4GB)
- [x] MicroSD卡 (32GB, Class 10)
- [x] ESP32-CAM开发板 (OV2640摄像头)
- [x] Ender 3 V2 打印机
- [x] 蓝色PLA耗材 (1.75mm)
- [x] USB数据线 (连接Pi和打印机)
- [x] 5V电源 (给ESP-CAM供电)

### 可选组件
- [ ] Raspberry Pi散热片/风扇
- [ ] 摄像头支架 (3D打印或购买)
- [ ] LED补光灯 (改善照明)

---

## 第一步：准备Raspberry Pi

### 1. 安装操作系统

```bash
# 1. 下载Raspberry Pi Imager
# https://www.raspberrypi.com/software/

# 2. 选择系统
# OS: Raspberry Pi OS Lite (64-bit)
# Storage: 选择MicroSD卡

# 3. 高级设置
# 启用SSH
# 设置用户名和密码
# 配置WiFi (如果使用WiFi)

# 4. 写入SD卡并启动
```

### 2. 基础配置

```bash
# SSH连接到Pi
ssh pi@<pi-ip-address>

# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装依赖
sudo apt install -y python3 python3-pip python3-venv
sudo apt install -y git libopenblas-dev libatlas-base-dev

# 安装Klipper依赖
sudo apt install -y build-essential
sudo apt install -y python3-dev libncurses-dev
```

---

## 第二步：安装Klipper固件

### 1. 克隆Klipper仓库

```bash
# 创建工作目录
mkdir -p ~/klipper
cd ~/klipper

# 克隆Klipper
git clone https://github.com/Klipper3d/klipper.git
cd klipper

# 编译固件
make menuconfig

# 配置选项：
#   → Microcontroller Architecture
#     → STM32F103
#   → 250000 baud
#   → Disable console (可选)

make
```

### 2. 刷入Ender 3 V2主板

```bash
# 1. 断开打印机电源
# 2. 短接Boot0跳线（参考打印机手册）
# 3. 连接Pi到打印机USB

# 刷入固件
sudo service klipper stop  # 如果已运行
dfu-util -a 0 -D klipper/out/klipper.bin

# 4. 移除Boot0短接
# 5. 重启打印机
```

---

## 第三步：配置Klipper

### 1. 创建配置文件

```bash
# 创建配置目录
mkdir -p ~/printer_config

# Ender 3 V2配置
nano ~/printer_config/printer.cfg
```

**printer.cfg 基础配置**：

```ini
# Ender 3 V2 Klipper配置
# 参考：https://www.klipper3d.org/Config_Reference.html

# ==================================================
# MCU配置
# ==================================================
[mcu]
serial: /dev/ttyUSB0
baud: 250000
restart_method: command

# ==================================================
# 步进电机配置
# ==================================================

# X轴
[stepper_x]
step_pin: PB5
dir_pin: !PB4
enable_pin: !PD8
microsteps: 16
rotation_distance: 40
endstop_pin: !PC0
position_endstop: 0
position_max: 235
homing_speed: 50

# Y轴
[stepper_y]
step_pin: PA1
dir_pin: !PA0
enable_pin: !PD8
microsteps: 16
rotation_distance: 40
endstop_pin: !PE1
position_endstop: 0
position_max: 235
homing_speed: 50

# Z轴
[stepper_z]
step_pin: PA3
dir_pin: !PA2
enable_pin: !PD8
microsteps: 16
rotation_distance: 8
endstop_pin: !PA15
position_max: 250

# 挤出机
[extruder]
step_pin: PB6
dir_pin: !PB7
enable_pin: !PD8
microsteps: 16
rotation_distance: 33.5
nozzle_diameter: 0.400
filament_diameter: 1.750
heater_pin: PB0
sensor_pin: PC5
control: pid
pid_Kp: 21.527
pid_Ki: 1.063
pid_Kd: 108.982
min_temp: 0
max_temp: 260

# ==================================================
# 热床
# ==================================================
[heater_bed]
heater_pin: PA1
sensor_pin: PC4
control: pid
pid_Kp: 54.027
pid_Ki: 0.770
pid_Kd: 948.182
min_temp: 0
max_temp: 130

# ==================================================
# 风扇
# ==================================================
[fan]
pin: PB8

# ==================================================
# 运动设置
# ==================================================
[printer]
kinematics: cartesian
max_velocity: 500
max_accel: 500
max_z_velocity: 25
max_z_accel: 100

# ==================================================
# 自动保存配置
# ==================================================
[save_variables]
filename: ~/printer_config/variables.cfg

# ==================================================
# 数据收集G-code宏（关键！）
# ==================================================
[gcode_macro LAYER_COMPLETE]
# 每层完成时调用此宏
gcode:
    # 获取当前层号
    {% set layer = printer["gcode_move"].position.z %}
    {% set filename = printer.print_stats.filename %}

    # 发送HTTP请求触发拍照
    {% set http_ok = true %}
    {% if http_ok %}
        {action_call_http(
            method="POST",
            url="http://localhost:5000/capture",
            body={"layer": layer, "filename": filename}
        )}
    {% endif %}

    # 打印日志
    {action_respond_info("Layer {layer} complete, photo captured")}

# ==================================================
# 响应代码（用于自动触发）
# ==================================================
[gcode_macro START_DATA_COLLECTION]
description: "开始数据收集"
gcode:
    # 启动Python数据收集服务
    {action_respond_info("Data collection service started")}
```

### 2. 启动Klipper服务

```bash
# 创建服务文件
sudo nano /etc/systemd/system/klipper.service
```

```ini
[Unit]
Description=Klipper 3D Printer Firmware
After=network.target

[Service]
Type=simple
User=pi
ExecStart=/home/pi/klipper/klipper/klippy/klippy.py /home/pi/printer_config/printer.cfg
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 启用并启动服务
sudo systemctl enable klipper
sudo systemctl start klipper

# 查看日志
sudo journalctl -u klipper -f
```

---

## 第四步：安装Moonraker

### 1. 安装Moonraker

```bash
# 克隆Moonraker
cd ~
git clone https://github.com/Arksine/moonraker.git
cd moonraker

# 安装Python依赖
python3 -m venv moonmoon-env
source moonmoon-env/bin/activate
pip install -r requirements.txt

# 创建配置文件
mkdir -p ~/.moonraker
nano ~/.moonraker/moonraker.conf
```

**moonraker.conf 配置**：

```ini
[server]
host: 0.0.0.0
port: 7125

[printer_config]
folder: ~/printer_config

[authorization]
force_logins: false
cors_domains:
    http://*.local
    http://192.168.*.*

[octoprint_compat]
topic: /octoprint

# 数据收集API（自定义）
[data_collection]
host: 0.0.0.0
port: 5000
```

### 2. 创建Moonraker服务

```bash
sudo nano /etc/systemd/system/moonraker.service
```

```ini
[Unit]
Description=Moonraker API Server
After=network.target klipper.service

[Service]
Type=simple
User=pi
ExecStart=/home/pi/moonraker/moonmoon-env/bin/python /home/pi/moonraker/moonraker/moonraker.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable moonraker
sudo systemctl start moonraker
```

---

## 第五步：安装Mainsail（Web界面）

```bash
cd ~
git clone https://github.com/mainsail-crew/mainsail.git

# 配置nginx
sudo apt install -y nginx
sudo ln -s ~/mainsail /var/www/mainsail

sudo systemctl restart nginx
```

访问：`http://<pi-ip-address>`

---

## 第六步：配置ESP-CAM

### 1. 安装Arduino IDE

```bash
# 在Pi上安装Arduino IDE（可选）
sudo apt install -y arduino

# 或在电脑上安装Arduino IDE
# 下载：https://www.arduino.cc/en/software
```

### 2. 烧录ESP-CAM代码

在Arduino IDE中打开：`File` → `Examples` → `ESP32` → `Camera` → `CameraWebServer`

修改配置：

```cpp
#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

const char* ssid = "YourWiFiSSID";
const char* password = "YourPassword";

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = YAML_GPIO_NUM_5;
  config.pin_d1 = YAML_GPIO_NUM_18;
  // ... (其他引脚配置)

  // 初始化摄像头
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  // 启动Web服务器
  startCameraServer();
}

void loop() {
  // 空循环
}
```

烧录到ESP-CAM。

### 3. 测试ESP-CAM

```bash
# 在浏览器中测试
http://<esp32-ip-address>/

# 或使用curl
curl http://<esp32-ip-address>/capture -o test.jpg
```

---

## 第七步：连接系统

### 物理连接

```
Raspberry Pi 4
    │
    ├── USB ──→ Ender 3 V2 (MCU)
    │
    └── WiFi ──→ ESP-CAM (摄像头)
```

### ESP-CAM安装位置

```
        [ESP-CAM]
            ↓
    ┌─────────────┐
    │  打印区域   │
    │             │
    │  [蓝色PLA]  │
    │             │
    └─────────────┘
        Ender 3 V2
```

**建议**：
- 高度：距热床约30-40cm
- 位置：正对打印中心
- 固定：使用3D打印支架

---

## 第八步：完整测试

### 1. 测试Klipper

```bash
# 在Mainsail界面中
# 1. Home所有轴
# 2. 测试移动：G1 X100 Y100 F3000
# 3. 测试挤出
```

### 2. 测试数据收集流程

```bash
# 在Pi上运行数据收集服务（稍后创建）
cd ~/3d_printer_pinn4ieee
python3 experiments/auto_data_collector.py
```

### 3. 打印测试件

```bash
# 在Mainsail中上传test.gcode
# 开始打印
# 观察每层完成时的自动拍照
```

---

## 故障排查

### Klipper无法连接
```bash
# 检查串口
ls /dev/ttyUSB*

# 检查权限
sudo usermod -a -G dialout pi

# 重启服务
sudo systemctl restart klipper
```

### ESP-CAM无法连接
```bash
# 检查WiFi
ping <esp32-ip-address>

# 检查防火墙
sudo ufw allow 5000/tcp
```

### 照片无法保存
```bash
# 检查目录权限
ls -la ~/3d_printer_pinn4ieee/data/collected_photos/

# 查看日志
tail -f ~/3d_printer_pinn4ieee/data/collection.log
```

---

## 下一步

完成硬件设置后，运行：
```bash
python experiments/setup_hardware.py
```

这将自动检测硬件并完成最终配置。
