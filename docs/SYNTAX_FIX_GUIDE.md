# Klipper配置语法修复指南

## 问题诊断

错误信息：
```
line 6: {action_respond_info("在Windows PC查看: curl http://localhost:5000/status")
# expected token 'end of print statement', got '{'
```

### 根本原因

1. **中文引号问题**：复制粘贴时可能使用了中文引号 `""` 而不是英文引号 `""`
2. **URL中的特殊字符**：某些字符可能干扰Jinja2解析器
3. **Jinja2模板语法冲突**：`{` 和 `}` 在Jinja2中有特殊含义

---

## ✅ 解决方案

### 关键修复点

1. **使用英文引号**：`"string"` 而不是 `"string"`
2. **简化字符串内容**：避免复杂的URL在同一个字符串中
3. **拆分为多个action_respond_info**：每行一个信息

### 修复前后对比

#### ❌ 错误示例

```jinja
{action_respond_info("在Windows PC查看: curl http://localhost:5000/status")}
```

**问题**：
- 可能包含中文引号
- URL太长，可能干扰解析

#### ✅ 正确示例

```jinja
{action_respond_info("Service: http://10.168.1.118:5000")}
{action_respond_info("On Windows PC check: curl http://localhost:5000/status")}
```

---

## 📋 完整的正确配置

### 复制到 printer.cfg

```ini
# ==================================================
# 数据收集系统配置
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
description: Test photo capture
gcode:
    {% set z_pos = printer.toolhead.position.z %}
    {% set layer_num = (z_pos * 1000)|int %}
    {action_respond_info("Testing photo: Z=%.3f mm" % z_pos)}
    RUN_SHELL_COMMAND CMD=TRIGGER_PHOTO PARAMS={layer_num} PARAMS=manual_test
    {action_respond_info("Photo command sent")}

[gcode_macro LAYER_COMPLETE]
description: Auto capture on layer complete
gcode:
    {% set z_pos = printer.toolhead.position.z %}
    {% set layer_num = (z_pos * 1000)|int %}
    {% set filename = printer.print_stats.filename|default("unknown") %}
    {action_respond_info("Layer %.3f complete, capturing..." % z_pos)}
    RUN_SHELL_COMMAND CMD=TRIGGER_PHOTO PARAMS={layer_num} PARAMS={filename}
    {action_respond_info("Capture complete")}

[gcode_macro SAVE_DATASET]
description: Save collected dataset
gcode:
    {action_respond_info("Saving dataset...")}
    RUN_SHELL_COMMAND CMD=SAVE_DATASET
    {action_respond_info("Dataset save command sent")}

[gcode_macro CHECK_SERVICE]
description: Check service status
gcode:
    {action_respond_info("Checking service status...")}
    RUN_SHELL_COMMAND CMD=CHECK_SERVICE
    {action_respond_info("Service: http://10.168.1.118:5000")}
    {action_respond_info("Check on Windows: curl http://localhost:5000/status")}

[gcode_macro SHOW_CONFIG]
description: Show configuration
gcode:
    {action_respond_info("========================================")}
    {action_respond_info("Data Collection System")}
    {action_respond_info("========================================")}
    {action_respond_info("Klipper: 10.168.1.123")}
    {action_respond_info("Windows PC: 10.168.1.118")}
    {action_respond_info("IP Camera: 10.168.1.129")}
    {action_respond_info("========================================")}
```

---

## 🔍 语法检查要点

### 1. 引号必须使用英文

❌ 错误：
```jinja
{action_respond_info("字符串")}  # 中文引号
```

✅ 正确：
```jinja
{action_respond_info("string")}   # 英文引号
```

### 2. 复杂内容拆分

❌ 错误：
```jinja
{action_respond_info("Line 1: http://example.com/api?key=value&param2=data")}
```

✅ 正确：
```jinja
{action_respond_info("Line 1: http://example.com/api")}
{action_respond_info("Line 2: Check service for details")}
```

### 3. 字符串格式化

✅ 正确：
```jinja
{action_respond_info("Z=%.3f mm" % z_pos)}
{action_respond_info("Layer %d" % layer_num)}
```

### 4. 变量使用

✅ 正确：
```jinja
{% set z_pos = printer.toolhead.position.z %}
{% set layer_num = (z_pos * 1000)|int %}
RUN_SHELL_COMMAND CMD=TRIGGER_PHOTO PARAMS={layer_num}
```

---

## ⚠️ 常见Jinja2语法错误

### 错误1：未关闭的括号

❌ 错误：
```jinja
{action_respond_info("text"
```

✅ 正确：
```jinja
{action_respond_info("text")}
```

### 错误2：混合中英文标点

❌ 错误：
```jinja
{action_respond_info("Text，more text"}  # 中文逗号
```

✅ 正确：
```jinja
{action_respond_info("Text, more text")}   # 英文逗号
```

### 错误3：转义字符处理不当

❌ 错误：
```jinja
{action_respond_info("Path: C:\folder\file")}  # 反斜杠问题
```

✅ 正确：
```jinja
{action_respond_info("Path: C:/folder/file")}   # 使用正斜杠
```

---

## 🛠️ 调试技巧

### 1. 逐步添加配置

先添加一个简单的宏测试：
```ini
[gcode_macro TEST_SIMPLE]
description: Simple test
gcode:
    {action_respond_info("Test message")}
```

重启Klipper，如果成功，再逐步添加其他功能。

### 2. 检查日志

```bash
# SSH到Klipper机器
ssh pi@10.168.1.123

# 查看Klipper日志
sudo journalctl -u klipper -n 50
```

### 3. 在线Jinja2验证

使用在线工具验证Jinja2语法：
https://jinja2Live-playground.herokuapp.com/

---

## 📝 复制粘贴注意事项

### 从文档复制时的风险

1. **中文引号**：某些编辑器会自动转换
2. **不可见字符**：可能有零宽字符
3. **编码问题**：UTF-8 BOM等

### 安全做法

1. **使用纯文本编辑器**：
   - VS Code（设置UTF-8编码）
   - Notepad++
   - vim/nano

2. **避免Word等富文本编辑器**

3. **检查字符编码**：
   ```bash
   file printer.cfg
   # 应该显示: UTF-8 Unicode text
   ```

---

## 🎯 快速修复步骤

1. **删除现有配置**：
   - 从 `printer.cfg` 中删除之前添加的所有内容

2. **使用新配置**：
   - 打开 `docs/PRINTER_CFG_FINAL.cfg`
   - 全选复制
   - 粘贴到 `printer.cfg` 末尾

3. **保存并重启**：
   - 保存文件
   - 在Mainsail中重启Klipper

4. **验证**：
   - 在控制台输入：`TEST_PHOTO`
   - 检查是否正常工作

---

**最后更新**: 2025-02-05
