# Klipper HTTP触发替代方案

## 问题诊断

错误信息：`'action_call_http' is undefined`

**原因**：
- `action_call_http` 是Klipper的较新功能
- 需要Klipper >= v0.11.0
- 需要Moonraker的HTTP client支持

---

## 🔍 方案1：检查您的Klipper和Moonraker版本

### 在Mainsail中查看版本

1. 打开Mainsail界面
2. 查看右上角的版本信息
3. 记录Klipper和Moonraker版本

**或通过命令**：
```
HELP
```

查看输出中的版本信息。

### 如果版本支持但功能未启用

在Moonraker配置中（通常是 `moonraker.conf`）添加：

```ini
[http_client]
```

然后重启Moonraker：
```bash
sudo systemctl restart moonraker
```

---

## ✅ 方案2：手动触发方式（推荐，无需配置）

### 修改Klipper宏为信息提示模式

```ini
[gcode_macro TEST_PHOTO]
description: "显示拍照触发信息"
gcode:
    {% set z_pos = printer.toolhead.position.z %}
    {action_respond_info("================================")}
    {action_respond_info("触发拍照命令")}
    {action_respond_info("================================")}
    {action_respond_info("在Windows PC上执行：")}
    {action_respond_info("  curl -X POST http://10.168.1.118:5000/capture")}
    {action_respond_info("    -H \"Content-Type: application/json\"")}
    {action_respond_info("    -d '{{\"layer\": %d, \"filename\": \"manual_test\"}}' % (z_pos * 1000|Int))}
    {action_respond_info("================================")}
```

### 使用步骤

1. 在Mainsail控制台输入：`TEST_PHOTO`
2. 会显示curl命令
3. 在Windows PC的命令提示符中执行显示的命令

**或者创建Windows批处理脚本**：

创建文件 `trigger_photo.bat`：
```batch
@echo off
curl -X POST http://10.168.1.118:5000/capture ^
  -H "Content-Type: application/json" ^
  -d "{\"layer\": %1, \"filename\": \"manual_test\"}"
```

使用方法：
```
trigger_photo.bat 10200
```

---

## ✅ 方案3：使用Moonraker的gcode_shell_command（推荐）

### 3.1 检查是否有Moonraker

如果您的Klipper是通过Mainsail访问的，很可能已经有Moonraker。

### 3.2 在Moonraker配置中添加shell命令

编辑 `moonraker.conf`，添加：

```ini
[gcode_shell_command TRIGGER_PHOTO]
command: curl -X POST http://10.168.1.118:5000/capture -H "Content-Type: application/json" -d '{"layer": {z_pos}, "filename": "{filename}"}'
timeout: 10.0
verbose: True
```

### 3.3 在Klipper宏中使用

```ini
[gcode_macro TEST_PHOTO]
description: "通过shell命令触发拍照"
gcode:
    {% set z_pos = printer.toolhead.position.z|int %}
    {action_respond_info("Triggering photo at Z=%d..." % z_pos)}
    TRIGGER_PHOTO Z_POS={z_pos} FILENAME=manual_test
    {action_respond_info("Photo triggered, check data/collection.log")}
```

---

## ✅ 方案4：使用Python脚本定时器（全自动）

### 创建Python脚本监控Klipper状态

**文件**：`experiments/klipper_monitor.py`

```python
"""
Klipper状态监控器 - 自动触发拍照

功能：定期查询Klipper状态，检测Z高度变化，触发拍照
"""

import time
import requests
import json

KLIPPER_API = "http://10.168.1.123:19255"
CAPTURE_API = "http://10.168.1.118:5000/capture"

last_z = 0.0
layer_threshold = 0.2  # 层高阈值

print("Klipper监控器启动...")
print(f"  Klipper API: {KLIPPER_API}")
print(f"  捕获API: {CAPTURE_API}")
print("  监控Z高度变化...\n")

while True:
    try:
        # 查询Klipper状态
        response = requests.get(f"{KLIPPER_API}/printer/objects/query?toolhead")
        data = response.json()

        # 获取当前Z位置
        z_pos = data['result']['status']['toolhead']['position'][3]

        # 检测新的层
        if abs(z_pos - last_z) >= layer_threshold and z_pos > 0:
            print(f"检测到新层: Z={z_pos:.3f}")

            # 触发拍照
            capture_data = {
                "layer": int(z_pos * 1000),
                "filename": "auto_monitor"
            }

            resp = requests.post(CAPTURE_API, json=capture_data)
            print(f"  拍照触发: {resp.json()}")

            last_z = z_pos

    except Exception as e:
        print(f"错误: {e}")

    time.sleep(2)  # 每2秒检查一次
```

### 使用方法

在Windows PC上：
```bash
python experiments/klipper_monitor.py
```

**优点**：
- 完全自动，无需修改Klipper配置
- 实时监控Z高度变化
- 自动触发拍照

**缺点**：
- 需要Python脚本持续运行
- 依赖Klipper API

---

## ✅ 方案5：使用G-code后处理（最简单）

### 不依赖宏，直接在G-code中插入命令

#### 5.1 使用Cura后处理

1. 打开Cura
2. 设置 → 后处理脚本
3. 添加"Pause at height"插件
4. 或创建自定义后处理脚本

#### 5.2 手动编辑G-code

在每层结束后插入：
```gcode
; LAYER_CHANGE
LAYER_COMPLETE
```

然后创建一个简单的Windows脚本监听这些事件。

---

## 🎯 推荐方案选择

| 方案 | 难度 | 优点 | 缺点 | 推荐度 |
|------|------|------|------|--------|
| **方案2：手动触发** | ⭐ | 最简单，无需配置 | 需要手动执行 | ⭐⭐⭐⭐ |
| **方案3：Moonraker shell** | ⭐⭐ | 半自动，在宏中调用 | 需要配置Moonraker | ⭐⭐⭐⭐⭐ |
| **方案4：Python监控** | ⭐⭐⭐ | 完全自动 | 需要额外脚本 | ⭐⭐⭐ |
| **方案5：G-code后处理** | ⭐⭐ | 一次性设置 | 需要重新切片 | ⭐⭐⭐⭐ |

---

## 📝 立即可用的方案（无需配置）

### 方案A：创建Windows批处理脚本

**文件**：`trigger_photo.bat`
```batch
@echo off
echo 触发拍照: Z=%1
curl -X POST http://10.168.1.118:5000/capture -H "Content-Type: application/json" -d "{\"layer\": %1, \"filename\": \"manual\"}"
echo 完成! 检查 data\collection.log
pause
```

**使用**：
```
trigger_photo.bat 10200
```

### 方案B：Python一键脚本

**文件**：`trigger_photo.py`
```python
import requests
import sys

z_pos = int(sys.argv[1]) if len(sys.argv) > 1 else 0

resp = requests.post(
    'http://10.168.1.118:5000/capture',
    json={'layer': z_pos, 'filename': 'manual'}
)

print(f"触发拍照: Z={z_pos}um")
print(f"响应: {resp.json()}")
```

**使用**：
```bash
python trigger_photo.py 10200
```

---

## 🔧 下一步

1. **检查Klipper版本**：在Mainsail中查看版本信息
2. **尝试方案3**：如果支持Moonraker，使用shell命令
3. **使用方案A/B**：创建Windows脚本，手动触发
4. **考虑方案4**：如果需要完全自动化

需要我帮您创建具体的脚本文件吗？
