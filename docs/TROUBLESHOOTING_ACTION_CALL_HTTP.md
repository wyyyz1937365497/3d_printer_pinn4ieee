# Klipper action_call_http 故障排除指南

## 问题：'action_call_http' is undefined

### 原因分析

您遇到这个错误可能是因为：

1. **Klipper版本太旧** - action_call_http在v0.11.0+才支持
2. **宏配置语法错误** - Jinja2模板语法问题
3. **Moonraker配置问题** - http_client未正确配置

---

## ✅ 解决步骤

### 步骤1：确认Klipper版本

**方法A：在Mainsail中查看**
1. 打开Mainsail界面
2. 右上角显示Klipper版本

**方法B：通过命令查询**
在Mainsail控制台输入：
```gcode
HELP
```

**方法C：通过API查询**
```bash
curl http://10.168.1.123:19255/server/info
```

查看返回的JSON中的`klipper_version`字段。

**需要的版本**：>= v0.11.0

---

### 步骤2：如果版本支持但仍然报错

#### 检查宏配置语法

**正确示例**：
```ini
[gcode_macro TEST_PHOTO]
gcode:
    {% set z_pos = printer.toolhead.position.z %}

    {% set http_ok = True %}
    {% if http_ok %}
        {action_call_http(
            method="POST",
            url="http://10.168.1.118:5000/capture",
            body={"layer": (z_pos * 1000)|int}
        )}
    {% endif %}
```

**常见错误**：
1. ❌ `printer.gcode_move.position_z` → ✅ `printer.toolhead.position.z`
2. ❌ `body={"layer": z_pos|int}` → ✅ `body={"layer": (z_pos * 1000)|int}`
3. ❌ `{% if True %}` → ✅ `{% set http_ok = True %}{% if http_ok %}`

---

### 步骤3：测试Moonraker http_client

#### 方法A：直接测试Moonraker API

```bash
# 测试http_client是否工作
curl -X POST http://10.168.1.123:19255/server/http_client/request \
  -H "Content-Type: application/json" \
  -d '{
    "url": "http://10.168.1.118:5000/status",
    "method": "GET"
  }'
```

**预期返回**：Flask服务的状态JSON

#### 方法B：使用Moonraker的webhook

测试http_client功能：
```bash
curl -X POST http://10.168.1.123:19255/webhooks/test_http_client
```

---

### 步骤4：如果Klipper版本太旧

#### 选项A：升级Klipper

```bash
# SSH到Klipper机器
ssh pi@10.168.1.123

# 备份当前配置
cp ~/klipper/klippy/env ~/klipper_backup

# 更新Klipper
cd ~/klipper
git pull
./scripts/install-octopi.sh  # 或根据您的安装方式

# 重启Klipper
sudo systemctl restart klipper
```

#### 选项B：使用替代方案

**如果无法升级Klipper，使用我们的Python监控脚本**：

```bash
# 在Windows PC上
python experiments/klipper_monitor.py
```

这个脚本：
- ✅ 无需修改Klipper
- ✅ 完全自动监控Z高度
- ✅ 自动触发拍照
- ✅ 实时显示进度

---

## 🔍 诊断检查清单

### Moonraker配置

- [ ] Moonraker版本 >= v0.7.0 ✅ (您有v0.10.0)
- [ ] http_client组件已启用 ✅ (确认)
- [ ] http_client可以发送HTTP请求 (需要测试)

### Klipper配置

- [ ] Klipper版本 >= v0.11.0 (需要确认)
- [ ] 宏使用正确的Jinja2语法
- [ ] 宏使用正确的printer对象访问方式

### 网络连接

- [ ] Klipper机器能访问Windows PC (10.168.1.118)
- [ ] Windows PC能访问IP摄像头 (10.168.1.129)
- [ ] Flask服务正在Windows PC上运行

---

## 🎯 推荐的测试顺序

### 1. 测试Moonraker http_client（最简单）

```bash
curl -X POST http://10.168.1.123:19255/server/http_client/request \
  -H "Content-Type: application/json" \
  -d '{
    "url": "http://10.168.1.118:5000/status",
    "method": "GET"
  }'
```

如果这个成功，说明http_client工作正常。

### 2. 测试简化的宏

使用最简单的宏配置：
```ini
[gcode_macro TEST_SIMPLE]
gcode:
    {action_respond_info("Testing...")}
    {action_call_http(
        method="GET",
        url="http://10.168.1.118:5000/status"
    )}
```

### 3. 如果仍然失败

使用Python监控脚本（最可靠）：
```bash
python experiments/klipper_monitor.py
```

---

## 📊 版本对照表

| Klipper版本 | action_call_http | 推荐方案 |
|------------|-----------------|---------|
| < v0.11.0 | ❌ 不支持 | Python监控脚本 |
| v0.11.0+ | ✅ 支持 | Klipper宏 |

---

## 💡 快速解决方案

### 立即可用：Python监控脚本

```bash
# 终端1：启动Flask服务
python experiments/auto_data_collector_existing.py \
    --klipper-host 10.168.1.123 \
    --camera-host 10.168.1.129 \
    --output data/collected_photos

# 终端2：启动监控
python experiments/klipper_monitor.py

# 开始打印，监控自动拍照
```

**优点**：
- ✅ 无需升级或配置Klipper
- ✅ 立即可用
- ✅ 完全自动
- ✅ 可靠性高

---

## 📞 需要帮助？

如果问题仍然存在，请提供：

1. Klipper版本号
2. Moonraker http_client测试结果
3. 完整的错误信息
4. 使用的宏配置

---

**最后更新**: 2025-02-05
