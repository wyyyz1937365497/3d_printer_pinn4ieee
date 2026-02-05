# 完整安装和配置指南

## 📋 前提条件

- ✅ Klipper 0.13.0
- ✅ Moonraker v0.10.0
- ✅ 已安装 KIAUH

---

## 步骤1：安装 gcode_shell_command 扩展

### 方法A：使用 KIAUH（推荐）

```bash
# SSH到Klipper机器
ssh pi@10.168.1.123

# 运行KIAUH
cd ~/kiauh
./kiauh.sh

# 选择：
# [Install Extensions]
# → [gcode-shell-command]
```

### 方法B：手动安装

```bash
# 克隆扩展仓库
cd ~/klipper/klippy/extras
git clone https://github.com/Arksine/gcode_shell_command.git

# 重启Klipper
sudo systemctl restart klipper
```

### 验证安装

在Mainsail控制台输入：
```gcode
HELP
```

查找输出中是否有 `RUN_SHELL_COMMAND` 命令。

---

## 步骤2：配置 Shell 命令

### 编辑 printer.cfg

在Mainsail界面：
1. 点击"配置"
2. 编辑 `printer.cfg`
3. 在文件**末尾**添加：

```ini
# ==================================================
# gcode_shell_command 扩展配置
# ==================================================

[gcode_shell_command TRIGGER_PHOTO]
command: curl -s -X POST http://10.168.1.118:5000/capture -H "Content-Type: application/json" -d '{"layer": %d, "filename": "%s"}'
timeout: 10.0
verbose: false

[gcode_shell_command SAVE_DATASET]
command: curl -s -X POST http://10.168.1.118:5000/save
timeout: 5.0
verbose: false

[gcode_shell_command CHECK_SERVICE]
command: curl -s http://10.168.1.118:5000/status
timeout: 5.0
verbose: true

# ==================================================
# G-code宏
# ==================================================

[gcode_macro TEST_PHOTO]
description: "测试拍照功能"
gcode:
    {% set z_pos = printer.toolhead.position.z %}
    {% set layer_num = (z_pos * 1000)|int %}
    {action_respond_info("测试拍照: Z=%.3f mm" % z_pos)}
    RUN_SHELL_COMMAND CMD=TRIGGER_PHOTO PARAMS={layer_num} PARAMS=manual_test
    {action_respond_info("拍照命令已发送")}

[gcode_macro LAYER_COMPLETE]
description: "每层完成时自动拍照"
gcode:
    {% set z_pos = printer.toolhead.position.z %}
    {% set layer_num = (z_pos * 1000)|int %}
    {% set filename = printer.print_stats.filename|default("unknown") %}
    {action_respond_info("Layer %.3f complete..." % z_pos)}
    RUN_SHELL_COMMAND CMD=TRIGGER_PHOTO PARAMS={layer_num} PARAMS={filename}
    {action_respond_info("Capture complete")}

[gcode_macro SAVE_DATASET]
description: "保存数据集"
gcode:
    {action_respond_info("保存数据集...")}
    RUN_SHELL_COMMAND CMD=SAVE_DATASET
    {action_respond_info("保存完成")}
```

4. 保存文件

---

## 步骤3：重启 Klipper

在Mainsail界面点击"重启Klipper"

---

## 步骤4：安装 curl（如果还没有）

```bash
# SSH到Klipper机器
ssh pi@10.168.1.123

# 安装curl
sudo apt-get update
sudo apt-get install curl

# 验证安装
curl --version
```

---

## 步骤5：启动 Flask 服务

**在Windows PC上**：

```bash
cd F:\TJ\3d_print\3d_printer_pinn4ieee
python experiments/auto_data_collector_existing.py \
    --klipper-host 10.168.1.123 \
    --camera-host 10.168.1.129 \
    --output data/collected_photos
```

---

## 步骤6：测试

### 测试1：基本拍照

在Mainsail控制台输入：
```gcode
TEST_PHOTO
```

**预期输出**：
```
测试拍照: Z=0.000 mm
拍照命令已发送
```

**Flask服务应该显示**：
```
INFO - 收到拍照请求: 层0, 文件manual_test
INFO - 处理层 0
INFO -   图像已保存: manual_test_layer000_*.jpg
INFO -   处理成功: XXX点, RMS=XX.XXum
```

### 测试2：移动Z轴后拍照

```gcode
G1 Z10
TEST_PHOTO
```

### 测试3：检查服务状态

```gcode
CHECK_SERVICE
```

应该返回Flask服务的状态JSON。

---

## 🔧 参数传递说明

### 单参数命令

```ini
[gcode_shell_command SIMPLE_CMD]
command: echo "Parameter: %d"
timeout: 2.0

[gcode_macro TEST_SIMPLE]
gcode:
    RUN_SHELL_COMMAND CMD=SIMPLE_CMD PARAMS=123
```

### 多参数命令

```ini
[gcode_shell_command MULTI_CMD]
command: curl -X POST http://example.com -d '{"layer": %d, "name": "%s"}'
timeout: 10.0

[gcode_macro TEST_MULTI]
gcode:
    {% set layer = 100 %}
    {% set name = "test" %}
    RUN_SHELL_COMMAND CMD=MULTI_CMD PARAMS={layer} PARAMS={name}
```

**注意**：
- `%d` 用于整数
- `%s` 用于字符串
- 参数按顺序传递

---

## ⚠️ 常见问题

### 问题1：扩展未安装

**错误**：
```
RUN_SHELL_COMMAND: command not found
```

**解决**：
1. 通过KIAUH安装gcode_shell_command扩展
2. 重启Klipper

### 问题2：curl未安装

**错误**：
```
curl: command not found
```

**解决**：
```bash
sudo apt-get install curl
```

### 问题3：参数传递错误

**错误**：
```
gcode_shell_command: incorrect number of parameters
```

**解决**：
检查command中的占位符数量与传递的PARAMS数量是否匹配。

---

## 📊 配置总结

| 组件 | 配置文件 | 说明 |
|-----|---------|------|
| **gcode_shell_command扩展** | 需要安装 | Klipper扩展 |
| **[gcode_shell_command ...]** | `printer.cfg` | 定义shell命令 |
| **[gcode_macro ...]** | `printer.cfg` | 定义G-code宏 |
| **RUN_SHELL_COMMAND** | 在宏中调用 | 执行shell命令 |

---

## 🎯 快速检查清单

- [ ] gcode_shell_command扩展已安装（通过KIAUH）
- [ ] curl已安装
- [ ] [gcode_shell_command ...] 在printer.cfg中定义
- [ ] [gcode_macro ...] 在printer.cfg中定义
- [ ] Klipper已重启
- [ ] Flask服务正在运行
- [ ] 网络连接正常

---

**最后更新**: 2025-02-05
