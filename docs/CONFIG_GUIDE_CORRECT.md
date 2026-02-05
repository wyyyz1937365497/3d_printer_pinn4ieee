# 正确配置Moonraker gcode_shell_command的完整指南

## 🔑 关键区别

### Klipper vs Moonraker 配置

| 配置项 | 所属系统 | 配置文件 |
|-------|---------|---------|
| `gcode_macro` | **Klipper** | `printer.cfg` |
| `gcode_shell_command` | **Moonraker** | `moonraker.conf` |

---

## 📁 配置文件位置

### 常见位置

**Klipper配置**：
- `~/printer.cfg`
- `/home/pi/printer.cfg`
- `~/printer_data/config/printer.cfg`

**Moonraker配置**：
- `~/moonraker.conf`
- `/etc/moonraker.conf`
- `~/printer_data/config/moonraker.conf`

### 查找配置文件

在Mainsail界面：
1. 点击"配置"
2. 左侧会显示所有配置文件
3. 找到 `printer.cfg` 和 `moonraker.conf`

---

## ✅ 正确的配置步骤

### 步骤1：编辑 moonraker.conf

在Mainsail中：
1. 点击"配置"
2. 找到并编辑 `moonraker.conf`
3. 在文件**末尾**添加：

```ini
# 数据收集shell命令

[gcode_shell_command TRIGGER_PHOTO]
command: curl -s -X POST http://10.168.1.118:5000/capture -H "Content-Type: application/json" -d '{"layer": {layer}, "filename": "{filename}"}'
timeout: 10.0
verbose: false

[gcode_shell_command SAVE_DATASET]
command: curl -s -X POST http://10.168.1.118:5000/save
timeout: 5.0
verbose: false

[gcode_shell_command CHECK_SERVICE]
command: curl -s http://10.168.1.118:5000/status
timeout: 5.0
verbose: false
```

4. 保存文件

### 步骤2：重启Moonraker（不是Klipper）

**重要**：修改Moonraker配置后需要重启Moonraker，不是Klipper！

```bash
# SSH到Klipper机器
ssh pi@10.168.1.123

# 重启Moonraker
sudo systemctl restart moonraker

# 查看日志
sudo journalctl -u moonraker -f
```

**或在Mainsail中**：
1. 有些版本可能有"重启Moonraker"按钮
2. 或者重启整个系统

### 步骤3：编辑 printer.cfg

在Mainsail中：
1. 点击"配置"
2. 找到并编辑 `printer.cfg`
3. 在文件**末尾**添加：

```ini
# 数据收集宏

[gcode_macro TEST_PHOTO]
description: "测试拍照功能"
gcode:
    {% set z_pos = printer.toolhead.position.z %}
    {% set layer_num = (z_pos * 1000)|int %}

    {action_respond_info("测试拍照: Z=%.3f mm" % z_pos)}

    TRIGGER_PHOTO LAYER={layer_num} FILENAME=manual_test

    {action_respond_info("拍照命令已发送")}

[gcode_macro LAYER_COMPLETE]
description: "每层完成时自动拍照"
gcode:
    {% set z_pos = printer.toolhead.position.z %}
    {% set layer_num = (z_pos * 1000)|int %}
    {% set filename = printer.print_stats.filename|default("unknown") %}

    {action_respond_info("Layer %.3f complete..." % z_pos)}

    TRIGGER_PHOTO LAYER={layer_num} FILENAME={filename}

    {action_respond_info("Capture complete")}

[gcode_macro SAVE_DATASET]
description: "保存数据集"
gcode:
    {action_respond_info("保存数据集...")}
    SAVE_DATASET
    {action_respond_info("保存完成")}

[gcode_macro CHECK_SERVICE]
description: "检查服务状态"
gcode:
    {action_respond_info("检查服务...")}
    CHECK_SERVICE
    {action_respond_info("服务: http://10.168.1.118:5000")}

[gcode_macro SHOW_CONFIG]
description: "显示配置信息"
gcode:
    {action_respond_info("========================================")}
    {action_respond_info("系统配置")}
    {action_respond_info("Klipper: 10.168.1.123")}
    {action_respond_info("Windows PC: 10.168.1.118")}
    {action_respond_info("IP摄像头: 10.168.1.129")}
    {action_respond_info("========================================")}
```

4. 保存文件

### 步骤4：重启Klipper

在Mainsail界面点击"重启Klipper"

---

## 🧪 测试步骤

### 1. 确保Flask服务正在运行

在Windows PC上：
```bash
cd F:\TJ\3d_print\3d_printer_pinn4ieee
python experiments/auto_data_collector_existing.py \
    --klipper-host 10.168.1.123 \
    --camera-host 10.168.1.129 \
    --output data/collected_photos
```

### 2. 测试宏

在Mainsail控制台输入：
```gcode
TEST_PHOTO
```

### 3. 观察输出

**Klipper控制台**：
```
测试拍照: Z=0.000 mm
拍照命令已发送
```

**Flask服务终端**：
```
INFO - 收到拍照请求: 层0, 文件manual_test
INFO - 处理层 0
INFO -   图像已保存: manual_test_layer000_*.jpg
```

---

## 🔍 验证配置

### 检查Moonraker是否识别shell命令

在Mainsail控制台输入：
```gcode
HELP
```

查找输出中的 `TRIGGER_PHOTO`、`SAVE_DATASET`、`CHECK_SERVICE`

或者通过API查询：
```bash
curl http://10.168.1.123:19255/server/gcode_shell_command/list
```

应该返回：
```json
{
  "result": {
    "TRIGGER_PHOTO": {...},
    "SAVE_DATASET": {...},
    "CHECK_SERVICE": {...}
  }
}
```

---

## ⚠️ 常见错误

### 错误1：`gcode_shell_command` 在 printer.cfg 中

**错误**：
```
Section 'gcode_shell_command TRIGGER_PHOTO' is not a valid config section
```

**原因**：放错配置文件了

**解决**：
- 从 `printer.cfg` 中删除 `[gcode_shell_command ...]` 段落
- 添加到 `moonraker.conf` 中

### 错误2：命令未找到

**错误**：
```
gcode_shell_command: TRIGGER_PHOTO not found
```

**原因**：Moonraker未重启或配置有语法错误

**解决**：
1. 检查Moonraker日志：`sudo journalctl -u moonraker -n 50`
2. 确保配置格式正确
3. 重启Moonraker：`sudo systemctl restart moonraker`

### 错误3：curl命令不存在

**错误**：
```
curl: command not found
```

**解决**：
```bash
# SSH到Klipper机器
ssh pi@10.168.1.123

# 安装curl
sudo apt-get update
sudo apt-get install curl
```

---

## 📊 配置文件对照表

| 文件 | 系统作用 | 内容 |
|-----|---------|------|
| **moonraker.conf** | 定义shell命令 | `[gcode_shell_command TRIGGER_PHOTO]` |
| **printer.cfg** | 定义G-code宏 | `[gcode_macro TEST_PHOTO]` |

**工作流程**：
```
printer.cfg中的宏
  → 调用命令
  → Moonraker执行shell命令
  → curl发送HTTP请求
  → Flask服务处理
```

---

## 🎯 快速检查清单

### Moonraker配置
- [ ] `[gcode_shell_command TRIGGER_PHOTO]` 在 `moonraker.conf` 中
- [ ] 命令格式正确
- [ ] Moonraker已重启
- [ ] 命令可通过API查询到

### Klipper配置
- [ ] `[gcode_macro TEST_PHOTO]` 在 `printer.cfg` 中
- [ ] 宏调用 `TRIGGER_PHOTO`
- [ ] Klipper已重启

### 网络
- [ ] Flask服务运行在 10.168.1.118:5000
- [ ] Klipper机器能访问 10.168.1.118
- [ ] curl已安装

---

## 💡 提示

1. **配置位置很重要**：
   - Shell命令 → Moonraker
   - G-code宏 → Klipper

2. **重启顺序**：
   - 修改Moonraker配置 → 重启Moonraker
   - 修改Klipper配置 → 重启Klipper

3. **测试方法**：
   - 先测试简单命令：`CHECK_SERVICE`
   - 再测试拍照：`TEST_PHOTO`

---

**最后更新**: 2025-02-05
