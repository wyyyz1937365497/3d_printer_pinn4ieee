# 网络配置说明

## 🌐 您的实际网络配置

```
┌─────────────────────────────────────────────────────┐
│ 设备1: Klipper主机                                  │
│ IP: 10.168.1.123                                    │
│ - Klipper服务 (端口 19255)                          │
│ - Moonraker API                                     │
└─────────────────────────────────────────────────────┘
                    ↓ HTTP POST
┌─────────────────────────────────────────────────────┐
│ 设备2: Windows PC ⭐ 您的机器                       │
│ IP: 10.168.1.118                                    │
│ - Flask数据收集服务 (端口 5000)  ← 服务运行在这里   │
│ - 项目代码: F:\TJ\3d_print\3d_printer_pinn4ieee    │
└─────────────────────────────────────────────────────┘
                    ↓ HTTP GET
┌─────────────────────────────────────────────────────┐
│ 设备3: IP摄像头                                     │
│ IP: 10.168.1.129                                    │
│ - MJPEG流: http://10.168.1.129:8080/video          │
│ - 快照接口: http://10.168.1.129:8080/shot.jpg      │
└─────────────────────────────────────────────────────┘
```

---

## 📝 关键配置

### 1. Klipper宏配置（在Klipper机器上）

在 `printer.cfg` 中，宏需要指向**Windows PC的IP**：

```ini
[gcode_macro TEST_PHOTO]
description: "测试拍照"
gcode:
    {% set z_pos = printer.toolhead.position.z %}
    {action_respond_info("Testing photo at Z=%.3f..." % z_pos)}

    {% set http_ok = True %}
    {% if http_ok %}
        {action_call_http(
            method="POST",
            url="http://10.168.1.118:5000/capture",  # ← Windows PC IP
            body={"layer": (z_pos * 1000)|int,
                   "filename": "manual_test"}
        )}
    {% endif %}

    {action_respond_info("Photo captured!")}
```

**完整配置文件**: `docs/KLIPPER_MACROS_SIMPLE.cfg`（已更新为正确IP）

---

### 2. Flask服务启动（在Windows PC上）

**在Windows PC的命令提示符或PowerShell中运行**：

```bash
# 进入项目目录
cd F:\TJ\3d_print\3d_printer_pinn4ieee

# 启动数据收集服务
python experiments/auto_data_collector_existing.py \
    --klipper-host 10.168.1.123 \
    --klipper-port 19255 \
    --camera-host 10.168.1.129 \
    --camera-port 8080 \
    --output data/collected_photos
```

**服务启动后会监听在**: `http://10.168.1.118:5000`

---

## 🔧 验证配置

### 步骤1：在Windows PC上启动服务

```bash
cd F:\TJ\3d_print\3d_printer_pinn4ieee
python experiments/auto_data_collector_existing.py \
    --klipper-host 10.168.1.123 \
    --camera-host 10.168.1.129 \
    --output data/collected_photos
```

**预期输出**：
```
==================================================
自动数据收集服务启动
==================================================
  Klipper: 10.168.1.123:19255
  摄像头: 10.168.1.129:8080
  HTTP端口: 5000
  监听地址: http://10.168.1.118:5000
  输出目录: data/collected_photos

等待Klipper请求...
```

### 步骤2：在Windows PC上测试服务

**打开新的命令提示符窗口**：
```bash
# 测试服务状态
curl http://localhost:5000/status

# 预期返回：
# {
#   "job": {...},
#   "layers_collected": 0,
#   "output_dir": "data/collected_photos"
# }
```

### 步骤3：在Mainsail中测试宏

1. 打开Mainsail界面（`http://10.168.1.123`）
2. 进入控制台
3. 输入：`TEST_PHOTO`
4. 按回车

**预期输出**：
```
Testing photo capture at Z=10.200...
Photo captured! Check data/collection.log
```

### 步骤4：检查日志

在Windows PC上：
```bash
type data\collection.log
```

**应该看到**：
```
INFO - 收到拍照请求: 层10200, 文件manual_test
INFO - 处理层 10200
INFO -   图像已保存: manual_test_layer10200_20250205_*.jpg
INFO -   处理成功: XXX点, RMS=XX.XXum
```

---

## ⚠️ 常见错误

### 错误1：连接被拒绝

**错误信息**：
```
Failed to establish connection to 10.168.1.118:5000
```

**原因**：Windows PC上的Flask服务未运行

**解决**：
1. 检查Windows PC上是否有Python进程在运行（任务管理器）
2. 重新启动Flask服务
3. 检查Windows防火墙是否阻止了端口5000

### 错误2：超时

**错误信息**：
```
Timeout waiting for HTTP response
```

**原因**：Flask服务运行中但无法访问IP摄像头

**解决**：
1. 在Windows PC上测试IP摄像头：
   ```bash
   curl http://10.168.1.129:8080/shot.jpg -o test.jpg
   ```
2. 检查网络连通性
3. 检查IP摄像头是否在线

### 错误3：IP地址配置错误

**症状**：Klipper宏执行但没有任何反应

**原因**：Klipper宏中的IP地址错误

**解决**：
确保Klipper宏中使用的是Windows PC的IP：
- ✅ `http://10.168.1.118:5000/capture`
- ❌ `http://10.168.1.129:5000/capture`（这是IP摄像头的IP）

---

## 📊 数据流向

```
1. 打印层完成
   ↓
2. Klipper (10.168.1.123) 发送HTTP POST
   → http://10.168.1.118:5000/capture
   ↓
3. Flask服务 (Windows PC) 收到请求
   ↓
4. Flask向IP摄像头发送HTTP GET
   → http://10.168.1.129:8080/shot.jpg
   ↓
5. IP摄像头返回JPEG图像
   ↓
6. Flask处理图像（提取轮廓、计算误差）
   ↓
7. 保存到Windows PC磁盘:
   - data/collected_photos/*.jpg (原始照片)
   - data/collected_photos/dataset_*.npz (训练数据)
```

---

## 🎯 快速配置检查清单

### 在Windows PC上

- [ ] Python已安装
- [ ] 依赖已安装：`pip install -r requirements.txt`
- [ ] Flask服务正在运行：`http://localhost:5000`
- [ ] 可以访问IP摄像头：`curl http://10.168.1.129:8080/shot.jpg`

### 在Klipper机器上

- [ ] `printer.cfg` 中已添加宏
- [ ] 宏中使用正确的IP：`http://10.168.1.118:5000`
- [ ] Klipper已重启
- [ ] 可以访问Windows PC：`curl http://10.168.1.118:5000/status`

---

## 💡 提示

1. **Windows防火墙**：首次运行可能需要允许Python通过防火墙
2. **网络稳定性**：确保三台设备在同一局域网且网络稳定
3. **IP地址固定**：建议为Windows PC和IP摄像头设置静态IP
4. **测试顺序**：
   - 先测试IP摄像头能否访问
   - 再启动Flask服务
   - 最后测试Klipper宏

---

**配置文件位置**：
- Klipper宏：`docs/KLIPPER_MACROS_SIMPLE.cfg`（已更新为正确IP）
- 启动指南：`docs/START_WITH_EXISTING_SETUP.md`（已更新）
- 完整文档：`README_EXISTING_SETUP.md`

**最后更新**: 2025-02-05
**Windows PC IP**: 10.168.1.118
