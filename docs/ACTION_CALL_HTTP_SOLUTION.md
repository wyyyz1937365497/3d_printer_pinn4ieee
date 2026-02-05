# action_call_http 完整解决方案

## 问题诊断

### 根本原因

经过查阅Klipper官方文档发现：

**标准的`action`命令只有4个**：
1. `action_respond_info(msg)`
2. `action_raise_error(msg)`
3. `action_emergency_stop(msg)`
4. `action_call_remote_method(method_name)`

**没有`action_call_http`**！

这说明：
- `action_call_http` 可能不是Klipper的标准功能
- 或者是特定发行版的扩展功能
- 或者需要非常新的Klipper版本

---

## ✅ 解决方案：使用 gcode_shell_command

### 原理

不使用不存在的`action_call_http`，而是使用Klipper的`gcode_shell_command`功能调用`curl`命令。

### 优点

- ✅ 使用Klipper标准功能
- ✅ 无需action_call_http
- ✅ 简单可靠
- ✅ Klipper 0.13.0完全支持

---

## 📝 配置步骤

### 步骤1：确认gcode_shell_command已启用

在Moonraker配置文件中检查：
```ini
[gcode_shell_command TRIGGER_PHOTO]
```

默认情况下这个功能是启用的。

### 步骤2：添加配置到printer.cfg

打开 `docs/KLIPPER_MACROS_WORKING.cfg`，复制内容到您的`printer.cfg`。

**关键部分**：
```ini
# 定义shell命令
[gcode_shell_command TRIGGER_PHOTO]
command: curl -s -X POST http://10.168.1.118:5000/capture -H "Content-Type: application/json" -d '{"layer": {layer}, "filename": "{filename}"}'
timeout: 10.0
verbose: false

# 在宏中调用
[gcode_macro TEST_PHOTO]
gcode:
    {% set z_pos = printer.toolhead.position.z %}
    {% set layer_num = (z_pos * 1000)|int %}
    TRIGGER_PHOTO LAYER={layer_num} FILENAME=manual_test
```

### 步骤3：重启Klipper

在Mainsail界面点击"重启Klipper"

### 步骤4：测试

在Mainsail控制台输入：
```gcode
TEST_PHOTO
```

---

## 🔍 工作原理

```
Klipper宏
   ↓
调用 gcode_shell_command
   ↓
执行 curl 命令
   ↓
HTTP POST 到 Flask服务 (10.168.1.118:5000)
   ↓
Flask处理请求
   ↓
从IP摄像头获取照片
   ↓
处理并保存
```

---

## 📊 完整示例

### 测试宏

```ini
[gcode_macro TEST_PHOTO]
description: "测试拍照功能"
gcode:
    # 获取Z位置
    {% set z_pos = printer.toolhead.position.z %}
    {% set layer_num = (z_pos * 1000)|int %}

    # 显示信息
    {action_respond_info("测试拍照: Z=%.3f mm" % z_pos)}

    # 调用shell命令
    TRIGGER_PHOTO LAYER={layer_num} FILENAME=manual_test

    {action_respond_info("拍照完成")}
```

### 层完成自动拍照

```ini
[gcode_macro LAYER_COMPLETE]
description: "每层完成时自动拍照"
gcode:
    {% set z_pos = printer.toolhead.position.z %}
    {% set layer_num = (z_pos * 1000)|int %}
    {% set filename = printer.print_stats.filename|default("unknown") %}

    {action_respond_info("Layer %.3f complete..." % z_pos)}

    TRIGGER_PHOTO LAYER={layer_num} FILENAME={filename}

    {action_respond_info("Capture complete")}
```

---

## ⚠️ 重要说明

### 参数传递

在`gcode_shell_command`中定义的参数（`{layer}`, `{filename}`）会被宏调用时传递的值替换：

```ini
# 定义时使用占位符
command: ... -d '{"layer": {layer}, "filename": "{filename}"}'

# 调用时传递实际值
TRIGGER_PHOTO LAYER=10200 FILENAME=test.gcode
```

### curl选项说明

- `-s`: 静默模式（不显示进度条）
- `-X POST`: HTTP POST方法
- `-H "Content-Type: application/json"`: 设置JSON头
- `-d '...'`: POST数据

---

## 🚀 立即测试

### 准备工作

1. **启动Flask服务**（终端1）：
   ```bash
   cd F:\TJ\3d_print\3d_printer_pinn4ieee
   python experiments/auto_data_collector_existing.py \
       --klipper-host 10.168.1.123 \
       --camera-host 10.168.1.129 \
       --output data/collected_photos
   ```

2. **添加配置到printer.cfg**：
   - 复制 `docs/KLIPPER_MACROS_WORKING.cfg` 内容
   - 粘贴到 `printer.cfg` 末尾

3. **重启Klipper**

### 测试命令

```gcode
# 测试1：基本拍照
TEST_PHOTO

# 测试2：移动Z轴后拍照
G1 Z10
TEST_PHOTO

# 测试3：显示配置
SHOW_NETWORK_CONFIG
```

---

## 🎯 预期结果

### Klipper控制台
```
========================================
测试拍照功能
========================================
Z位置: 0.000 mm
层号: 0 um
目标: http://10.168.1.118:5000
========================================
拍照命令已发送，检查data\collection.log
```

### Flask服务终端
```
INFO - 收到拍照请求: 层0, 文件manual_test
INFO - 处理层 0
INFO -   图像已保存: manual_test_layer000_20250205_*.jpg
INFO -   处理成功: XXX点, RMS=XX.XXum
```

---

## 🔧 故障排除

### 问题1：shell命令未启用

**错误**：`gcode_shell_command not enabled`

**解决**：
在Moonraker配置中添加：
```ini
[gcode_shell_command TRIGGER_PHOTO]
```

重启Moonraker：`sudo systemctl restart moonraker`

### 问题2：curl命令不存在

**错误**：`curl: command not found`

**解决**：
安装curl：
```bash
sudo apt-get install curl
```

### 问题3：连接超时

**错误**：`curl: (7) Failed to connect`

**解决**：
1. 检查Flask服务是否运行
2. 检查Windows防火墙
3. 测试连接：`curl http://10.168.1.118:5000/status`

---

## 📝 总结

| 特性 | action_call_http | gcode_shell_command |
|-----|-----------------|-------------------|
| **Klipper支持** | ❓ 非标准/新版 | ✅ 标准功能 |
| **可靠性** | ❓ 不确定 | ✅ 稳定 |
| **配置难度** | ⭐⭐ | ⭐ |
| **推荐度** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**推荐使用 `gcode_shell_command` 方案**！

---

**最后更新**: 2025-02-05
