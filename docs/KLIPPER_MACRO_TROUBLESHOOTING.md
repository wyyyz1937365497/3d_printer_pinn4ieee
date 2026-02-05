# Klipper宏配置快速修复指南

## 问题诊断

您遇到的错误：
```
Error evaluating 'gcode_macro PHOTO_CAPTURE:gcode':
jinja2.exceptions.UndefinedError: 'dict object' has no attribute 'position_z'
```

**原因**：Klipper中访问Z位置的语法错误

---

## ✅ 解决方案

### 方法1：使用修复后的配置文件（推荐）

1. **删除旧的宏配置**
   - 打开Mainsail界面
   - 进入"配置" → 编辑`printer.cfg`
   - 删除之前添加的`LAYER_COMPLETE`和`PHOTO_CAPTURE`宏

2. **复制修复后的配置**

打开 `docs/KLIPPER_MACROS_SIMPLE.cfg`，复制全部内容到`printer.cfg`末尾

3. **重启Klipper**
   - 在Mainsail界面点击"重启Klipper"
   - 或使用命令：`sudo systemctl restart klipper`

4. **测试宏**
   ```
   # 在Mainsail控制台输入
   TEST_PHOTO
   ```

---

### 方法2：手动修复现有配置

如果不想重新复制，只需修改以下两处：

#### 修复 `LAYER_COMPLETE` 宏

**查找**：
```ini
{action_respond_info("Layer {printer.gcode_move.position_z} complete...")}
body={"layer": printer.gcode_move.position_z|int,
```

**替换为**：
```ini
{% set z_pos = printer.toolhead.position.z %}
{action_respond_info("Layer %.3f complete..." % z_pos)}
body={"layer": (z_pos * 1000)|int,
```

#### 修复 `PHOTO_CAPTURE` 宏

**查找**：
```ini
body={"layer": printer.gcode_move.position_z|int,
```

**替换为**：
```ini
{% set z_pos = printer.toolhead.position.z %}
body={"layer": (z_pos * 1000)|int,
```

---

## 🔍 关键变更说明

### ❌ 错误的语法
```jinja
printer.gcode_move.position_z
```

### ✅ 正确的语法
```jinja
{% set z_pos = printer.toolhead.position.z %}
```

或者使用：
```jinja
printer['gcode_move'].gcode_position[2]
```

---

## 📋 完整的修复后LAYER_COMPLETE宏

```ini
[gcode_macro LAYER_COMPLETE]
description: "每层完成时自动拍照"
gcode:
    # 获取当前Z位置（单位：毫米）
    {% set z_pos = printer.toolhead.position.z %}

    # 显示信息
    {action_respond_info("Layer %.3f complete, capturing..." % z_pos)}

    # 发送HTTP请求到数据收集服务
    {% set http_ok = True %}
    {% if http_ok %}
        {action_call_http(
            method="POST",
            url="http://10.168.1.129:5000/capture",
            body={"layer": (z_pos * 1000)|int,
                   "filename": printer.print_stats.filename}
        )}
    {% endif %}

    {action_respond_info("Capture complete")}
```

---

## 🧪 测试步骤

### 步骤1：确保数据收集服务正在运行

```bash
# 在Windows上
python experiments/auto_data_collector_existing.py \
    --klipper-host 10.168.1.123 \
    --camera-host 10.168.1.129 \
    --output data/collected_photos
```

### 步骤2：在Mainsail控制台测试

```
TEST_PHOTO
```

### 步骤3：检查输出

**预期输出**：
```
Manually capturing photo at Z=10.200...
Photo captured! Check data/collection.log
```

### 步骤4：查看日志

```bash
# 在另一个终端
tail -f data/collection.log
```

**应该看到**：
```
INFO - 收到拍照请求: 层10200, 文件manual_test
INFO - 处理层 10200
INFO -   图像已保存: manual_test_layer10200_20250205_*.jpg
INFO -   处理成功: XXX点, RMS=XX.XXum
```

---

## ⚠️ 常见问题

### Q1: 重启Klipper后报错"Unknown macro"

**原因**：宏定义有语法错误

**解决**：
1. 检查Klipper日志：`sudo journalctl -u klipper -f`
2. 确保没有多余的空格或特殊字符
3. 尝试逐个添加宏，找出有问题的那个

### Q2: HTTP请求失败

**原因**：数据收集服务未运行或IP地址错误

**解决**：
```bash
# 测试服务是否运行
curl http://10.168.1.129:5000/status

# 如果失败，检查：
# 1. Python进程是否运行
# 2. IP地址是否正确
# 3. 防火墙是否阻止
```

### Q3: 照片全黑或无法提取轮廓

**原因**：
- 摄像头对焦不准
- 照明不足
- 材料对比度不够

**解决**：
1. 转动摄像头镜头环调整对焦
2. 增加环境光或LED照明
3. 使用蓝色PLA材料（推荐）

---

## 🎯 下一步

修复宏配置后：

1. ✅ 测试`TEST_PHOTO`宏
2. ✅ 确认照片质量良好
3. ✅ 确认轮廓提取成功
4. ✅ 开始打印测试件
5. ✅ 自动收集每层数据

---

## 📞 需要帮助？

如果问题仍然存在：

1. 查看完整日志：`data/collection.log`
2. 检查Klipper日志：`sudo journalctl -u klipper -n 50`
3. 确认服务运行：`ps aux | grep auto_data_collector`

---

**最后更新**: 2025-02-05
**修复版本**: v1.1
