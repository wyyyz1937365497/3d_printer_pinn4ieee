# 打印参数对比与补充

**生成日期**: 2026-02-04
**参考文件**:
- G-code: `Tremendous Hillar_PLA_10m22s.gcode` (Creality_Print切片)
- MATLAB参数: `simulation/physics_parameters.m`

---

## 1. 参数对比表

### 耗材参数

| 参数 | G-code值 | MATLAB值 | 状态 | 说明 |
|------|----------|-----------|------|------|
| 耗材类型 | PLA | PLA | ✅ 匹配 | - |
| 耗材直径 | 1.75 mm | 1.75 mm | ✅ 匹配 | - |
| 耗材密度 | 1.24 g/cm³ | 1.24 g/cm³ | ✅ 匹配 | - |
| 材料名称 | - | PLA | ✅ 匹配 | G-code未明确，但PLA参数一致 |

### 温度参数

| 参数 | G-code值 | MATLAB值 | 状态 | 说明 |
|------|----------|-----------|------|------|
| **喷嘴温度** | 220°C | 220°C | ✅ 匹配 | M104 S220 |
| **热床温度** | 60°C | 60°C | ✅ 匹配 | M140 S60 |
| 环境温度 | - | 25°C | ⚠️ 未指定 | MATLAB默认值 |
| 腔室温度 | - | 25°C | ⚠️ 未指定 | Ender-3 V2开放式结构 |
| 风扇启动温度 | - | 220°C | ⚠️ 未指定 | MATLAB设定 |

### 运动参数

| 参数 | G-code值 | MATLAB值 | 状态 | 说明 |
|------|----------|-----------|------|------|
| **最大速度** | - | 500 mm/s | ⚠️ 未指定 | MATLAB M203设定 |
| **最大加速度** | - | 500 mm/s² | ⚠️ 未指定 | MATLAB M201设定 |
| **急停(Jerk)** | X10 Y10 mm/s | 10 mm/s | ✅ 匹配 | M205 X10 Y10 |
| **Z轴jerk** | 0.40 mm/s | - | ⚠️ MATLAB未指定 | M205 Z0.40 |
| **E轴jerk** | 5.00 mm/s | - | ⚠️ MATLAB未指定 | M205 E5.00 |

### 挤出参数

| 参数 | G-code值 | MATLAB值 | 状态 | 说明 |
|------|----------|-----------|------|------|
| **喷嘴直径** | - | 0.4 mm | ⚠️ 未指定 | MATLAB默认 |
| **层高** | - | 0.2 mm | ⚠️ 未指定 | MATLAB默认 |
| **挤出宽度** | - | 0.45 mm | ⚠️ 未指定 | MATLAB计算值 |
| 线材直径 | 1.75 mm | 1.75 mm | ✅ 匹配 | - |
| 线材密度 | 1.24 g/cm³ | 1.24 g/cm³ | ✅ 匹配 | - |

### 打印范围

| 参数 | G-code值 | 说明 |
|------|----------|------|
| X范围 | 85.24 - 139.43 mm | 打印宽度: 54.19 mm |
| Y范围 | 84.84 - 140.56 mm | 打印深度: 55.72 mm |
| Z范围 | 0.00 - 11.49 mm | 打印高度: 11.49 mm |
| 总层数 | 57 层 | - |

---

## 2. 缺失参数分析

### 关键缺失参数

1. **打印速度** ✅ **已确认**
   - G-code实际速度:
     - **打印移动**: 40-50 mm/s (中位数)
     - **空移移动**: 60 mm/s (中位数)
     - **最大速度**: 120 mm/s
   - MATLAB: 500 mm/s (最大速度)
   - **结论**: 实际打印速度远低于MATLAB最大值设定
   - **影响**: 速度直接影响动态误差幅值
   - **测试件建议**: 使用50 mm/s (符合实际打印)

2. **加速度** ⚠️
   - G-code: 未明确指定
   - MATLAB: 500 mm/s² (最大加速度)
   - **影响**: 加速度变化影响惯性力
   - **建议**: 使用MATLAB默认值或从Marlin配置读取

3. **层高** ⚠️
   - G-code: 未在头部明确
   - MATLAB: 0.2 mm (默认)
   - **影响**: 影响Z轴分辨率和打印时间
   - **建议**: 从Z轴移动计算 (11.49mm / 57层 ≈ 0.20mm)

4. **喷嘴直径** ⚠️
   - G-code: 未指定
   - MATLAB: 0.4 mm (标准Ender-3配置)
   - **影响**: 影响挤出宽度和线宽

---

## 3. 推荐的G-code生成参数补充

基于physics_parameters.m，建议在G-code头部添加以下注释：

```gcode
; ========================================
; 打印参数说明（基于physics_parameters.m）
; ========================================
;
; --- 耗材参数 ---
; MATERIAL: PLA
; FILAMENT_DIAMETER: 1.75 mm
; FILAMENT_DENSITY: 1.24 g/cm³
;
; --- 温度参数 ---
; NOZZLE_TEMP: 220 °C
; BED_TEMP: 60 °C
; CHAMBER_TEMP: 25 °C (环境温度)
;
; --- 运动参数 ---
; MAX_VELOCITY: 500 mm/s
; MAX_ACCEL: 500 mm/s²
; MAX_JERK_XY: 10 mm/s
; MAX_JERK_Z: 0.4 mm/s
; MAX_JERK_E: 5 mm/s
;
; --- 挤出参数 ---
; NOZZLE_DIAMETER: 0.4 mm
; LAYER_HEIGHT: 0.2 mm
; EXTRUSION_WIDTH: 0.45 mm
;
; --- 动力学参数（MATLAB仿真用） ---
; X_AXIS_MASS: 0.35 kg
; X_AXIS_STIFFNESS: 8000 N/m
; X_AXIS_DAMPING: 15.0 N·s/m
; X_AXIS_NATURAL_FREQ: 151.19 rad/s (24.06 Hz)
; X_AXIS_DAMPING_RATIO: 0.1417
;
; Y_AXIS_MASS: 0.45 kg
; Y_AXIS_STIFFNESS: 8000 N/m
; Y_AXIS_DAMPING: 15.0 N·s/m
; Y_AXIS_NATURAL_FREQ: 133.33 rad/s (21.22 Hz)
; Y_AXIS_DAMPING_RATIO: 0.1250
;
; --- 预期误差 ---
; EXPECTED_RMS_ERROR: ~300-400 um (未修正)
; EXPECTED_MAX_ERROR: ~1500-2000 um (未修正)
;
; ========================================
```

---

## 4. 参数验证建议

### 方法1：从G-code统计实际速度

```python
# 分析G-code中的F指令，统计实际打印速度
import re

def analyze_gcode_speed(gcode_file):
    with open(gcode_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    speeds = []
    for line in lines:
        if line.startswith('G1') or line.startswith('G0'):
            match = re.search(r'F(\d+\.?\d*)', line)
            if match:
                speed_mm_min = float(match.group(1))
                speed_mm_s = speed_mm_min / 60.0
                speeds.append(speed_mm_s)

    print(f"速度统计:")
    print(f"  最小速度: {min(speeds):.1f} mm/s")
    print(f"  最大速度: {max(speeds):.1f} mm/s")
    print(f"  平均速度: {np.mean(speeds):.1f} mm/s")
    print(f"  中位速度: {np.median(speeds):.1f} mm/s")
```

### 方法2：从切片软件配置读取

Creality Print / Cura / PrusaSlicer 的配置文件中包含：
- 打印速度
- 加速度设置
- 层高
- 初始层参数
- 支撑和填充设置

---

## 5. 对测试件G-code的参数建议

### 推荐的打印参数

基于physics_parameters.m和实际G-code分析，测试件应使用以下参数：

```python
# 测试件打印参数
TEST_PRINT_PARAMS = {
    # 耗材
    'material': 'PLA',
    'filament_diameter': 1.75,  # mm
    'filament_density': 1.24,   # g/cm³

    # 温度
    'nozzle_temp': 220,         # °C
    'bed_temp': 60,             # °C
    'chamber_temp': 25,         # °C (环境)

    # 运动
    'max_velocity': 50,         # mm/s (测试用，比实际慢)
    'max_accel': 500,           # mm/s²
    'jerk_xy': 10,              # mm/s
    'jerk_z': 0.4,              # mm/s
    'jerk_e': 5,                # mm/s

    # 挤出
    'nozzle_diameter': 0.4,     # mm
    'layer_height': 0.2,        # mm
    'extrusion_width': 0.45,    # mm

    # 质量（用于MATLAB仿真）
    'mass_x': 0.35,             # kg
    'mass_y': 0.45,             # kg
    'stiffness': 8000,          # N/m
    'damping': 15.0,            # N·s/m
}
```

### 为什么测试件用较低速度（50 mm/s）？

```
原因分析：
├─ 实际打印中常见速度: 30-60 mm/s (轮廓)
├─ 高速打印: 80-100 mm/s (快速原型)
└─ 最高速: 120+ mm/s (牺牲质量)

选择50 mm/s的原因：
✓ 代表典型轮廓打印速度
✓ 误差明显且稳定
✓ 便于测量对比
✓ 符合实际应用场景
✓ 与训练数据的速度范围匹配

如果测试100 mm/s：
- 误差更大，模型可能未充分训练
- 测量难度增加
- 脱离实际应用
```

---

## 6. 完整的参数补充方案

### 方案A：在G-code头部添加参数注释

```gcode
; ========================================
; Outline Test Pieces - Print Parameters
; ========================================
;
; 基于simulation/physics_parameters.m确认的参数
; 生成日期: 2026-02-04
;
; --- 耗材 ---
; MATERIAL: PLA (Polylactic Acid)
; FILAMENT_DIAMETER: 1.75 mm
; FILAMENT_DENSITY: 1.24 g/cm³
; MELTING_POINT: 171 °C
; GLASS_TRANSITION: 60 °C
;
; --- 温度 ---
; NOZZLE_TEMP: 220 °C
; BED_TEMP: 60 °C
; CHAMBER_TEMP: 25 °C
; FAN_SPEED: 255 (100%)
;
; --- 运动 ---
; PRINT_SPEED: 50 mm/s (轮廓)
; TRAVEL_SPEED: 150 mm/s (空移)
; MAX_VELOCITY: 500 mm/s
; MAX_ACCEL: 500 mm/s²
; JERK_XY: 10 mm/s
; JERK_Z: 0.4 mm/s
;
; --- 挤出 ---
; NOZZLE_DIAMETER: 0.4 mm
; LAYER_HEIGHT: 0.2 mm
; EXTRUSION_WIDTH: 0.45 mm
; EXTRUSION_MULTIPLIER: 1.0
;
; --- 质量（动力学） ---
; MASS_X: 0.35 kg
; MASS_Y: 0.45 kg
; STIFFNESS_X: 8000 N/m
; STIFFNESS_Y: 8000 N/m
; DAMPING_X: 15.0 N·s/m
; DAMPING_Y: 15.0 N·s/m
;
; --- 预期性能（基于MATLAB仿真） ---
; UNCORRECTED_RMS_ERROR: ~350-400 um
; UNCORRECTED_MAX_ERROR: ~1800-2000 um
; EXPECTED_IMPROVEMENT: 30-50%
;
; ========================================

G28 ; Home all axes
G1 Z15 F3000 ; Move to travel height
G90 ; Absolute positioning
G21 ; Units in millimeters
... (rest of G-code)
```

### 方案B：创建参数配置文件

创建 `test_print_config.json`:

```json
{
  "material": {
    "name": "PLA",
    "diameter_mm": 1.75,
    "density_g_cm3": 1.24,
    "type": "PLA"
  },
  "temperatures": {
    "nozzle_c": 220,
    "bed_c": 60,
    "chamber_c": 25,
    "fan_speed_percent": 100
  },
  "motion": {
    "print_speed_mm_s": 50,
    "travel_speed_mm_s": 150,
    "max_velocity_mm_s": 500,
    "max_accel_mm_s2": 500,
    "jerk_xy_mm_s": 10.0,
    "jerk_z_mm_s": 0.4,
    "jerk_e_mm_s": 5.0
  },
  "extrusion": {
    "nozzle_diameter_mm": 0.4,
    "layer_height_mm": 0.2,
    "extrusion_width_mm": 0.45,
    "extrusion_multiplier": 1.0
  },
  "dynamics": {
    "mass_x_kg": 0.35,
    "mass_y_kg": 0.45,
    "stiffness_n_m": 8000,
    "damping_n_s_m": 15.0,
    "natural_freq_x_hz": 24.06,
    "natural_freq_y_hz": 21.22,
    "damping_ratio_x": 0.1417,
    "damping_ratio_y": 0.1250
  },
  "expected_errors": {
    "uncorrected_rms_um": 380,
    "uncorrected_max_um": 1900,
    "target_rss_um": 200,
    "expected_improvement_percent": 47
  }
}
```

---

## 7. 实施建议

### 立即行动项

1. **验证层高**
   ```bash
   # 从G-code计算
   z_moves = [line for line in gcode if 'G1 Z' in line]
   # Z从0到11.49mm，共57层
   # 层高 ≈ 11.49 / 57 ≈ 0.20 mm ✓
   ```

2. **统计实际打印速度**
   ```python
   # 使用上面的analyze_gcode_speed函数
   python analyze_gcode_speed.py \
       --gcode test_gcode_files/Tremendous\ Hillar_PLA_10m22s.gcode
   ```

3. **更新测试件G-code生成脚本**
   ```python
   # 在generate_outline_test_pieces.py中
   # 添加参数注释头部
   # 确保速度设置为50 mm/s
   ```

### 中期优化

1. **从切片软件导出完整配置**
   - Creality Print: 导出配置文件
   - 读取关键参数
   - 对比MATLAB参数

2. **创建参数验证脚本**
   ```python
   # validate_print_parameters.py
   # - 读取G-code
   # - 读取MATLAB参数
   # - 生成对比报告
   # - 标注不匹配项
   ```

---

## 8. 总结

### 参数完整性评估

| 类别 | 完整度 | 优先级 | 行动 |
|------|--------|--------|------|
| 耗材参数 | ✅ 完整 | 高 | 无需补充 |
| 温度参数 | ✅ 基本完整 | 高 | 添加环境温度说明 |
| 运动参数 | ⚠️ 缺少速度 | 高 | 统计实际速度，添加到MATLAB |
| 挤出参数 | ⚠️ 缺少层高等 | 中 | 从Z轴移动计算层高 |
| 动力学参数 | ✅ MATLAB完整 | - | 已在physics_parameters.m中 |

### 关键发现

1. ✅ **主要参数已匹配**
   - 耗材: PLA
   - 温度: 220°C (喷嘴), 60°C (床)
   - Jerk: 10 mm/s (XY)

2. ⚠️ **需要补充的参数**
   - 打印速度: 建议从G-code统计
   - 层高: 建议从Z范围计算 (~0.2mm)
   - 喷嘴直径: 使用MATLAB默认 (0.4mm)

3. ⚠️ **参数不一致风险**
   - MATLAB最大速度500 mm/s vs 实际打印可能较慢
   - **建议**: 统计实际G-code中的F指令，确定真实速度范围

### 下一步

1. 运行速度统计分析
2. 更新`physics_parameters.m`添加实际速度参数
3. 在测试件G-code中添加完整参数注释
4. 创建参数配置文档供切片软件参考

---

## 9. 实际速度分析结果

基于`analyze_gcode_speed.py`对`Tremendous Hillar_PLA_10m22s.gcode`的分析：

### 速度统计

| 移动类型 | 总指令数 | 最小速度 | 最大速度 | 平均速度 | 中位速度 |
|---------|---------|---------|---------|---------|---------|
| **所有移动** | 2164 | 10.0 mm/s | 120.0 mm/s | 65.2 mm/s | 50.0 mm/s |
| **仅打印移动** | 691 | 25.0 mm/s | 50.0 mm/s | 44.7 mm/s | 40.0 mm/s |
| **仅空移移动** | 1472 | 10.0 mm/s | 120.0 mm/s | 74.9 mm/s | 60.0 mm/s |

### 关键发现

1. **打印速度集中范围**: 40-50 mm/s (中位数40 mm/s)
   - 占所有打印移动的54.8%
   - 这是最常见的轮廓打印速度

2. **速度分布特点**:
   - 10%分位: 40 mm/s
   - 25%分位: 50 mm/s
   - 50%分位: 50 mm/s
   - 75%分位: 96 mm/s
   - 95%分位: 120 mm/s

3. **与MATLAB参数对比**:
   - MATLAB最大速度: 500 mm/s
   - G-code实际最大: 120 mm/s
   - **结论**: 实际打印远未达到MATLAB设定的理论最大值

4. **测试件速度建议**:
   - ✅ **使用50 mm/s** - 符合实际打印中位数
   - ✅ 误差特征明显，便于测量
   - ✅ 代表典型应用场景

### 图表输出

- 速度分布图: `test_gcode_files/analysis/speed_distribution_analysis.png`
- 详细报告: `test_gcode_files/analysis/speed_analysis_report.json`

---

**文档版本**: 1.1
**最后更新**: 2026-02-04
**作者**: Claude (基于用户G-code和MATLAB参数分析)
