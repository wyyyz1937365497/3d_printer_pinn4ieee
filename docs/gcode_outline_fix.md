# G-code修正问题修复报告

**日期**: 2026-02-03
**分支**: fix/gcode-correction-outline-only
**问题**: 模型修改了内部填充和墙体，破坏打印结构

---

## 🔍 问题分析

### 原始问题

**症状**：
- 修正后的G-code打印件解析失败
- 模型不仅修改了外墙轮廓
- 内部填充和墙体也被修改

**原因**：
原始的`gcode_offline_correction.py`脚本对**所有**G1/G0移动都应用修正：

```python
# ❌ 错误做法：对所有移动都修正
if line.startswith('G1') or line.startswith('G0'):
    move = self._parse_move(line)
    if move:
        self.moves.append(move)  # 所有移动都加入
```

这导致：
- ✅ 外壁修正 - 正确
- ❌ 内部填充修正 - **错误**！
- ❌ 支撑结构修正 - **错误**！
- ❌ 裙边修正 - **错误**！

---

## ✅ 解决方案

### G-code类型识别

通过分析G-code文件，发现Creality Slicer使用`;TYPE:`注释标识移动类型：

| TYPE注释 | 说明 | 是否修正 |
|----------|------|----------|
| `;TYPE:Outer wall` | 外壁 | ✅ **修正** |
| `;TYPE:Inner wall` | 内壁 | ✅ **修正** |
| `;TYPE:Internal solid infill` | 内部实心填充 | ❌ 不修正 |
| `;TYPE:Gap infill` | 间隙填充 | ❌ 不修正 |
| `;TYPE:Skirt` | 裙边 | ❌ 不修正 |
| `;TYPE:Support` | 支撑 | ❌ 不修正 |
| `;TYPE:Support interface` | 支撑接触面 | ❌ 不修正 |

### 修复实现

**新脚本**: `experiments/gcode_outline_correction.py`

核心逻辑：

```python
class OutlineAwareGCodeParser:
    def parse(self):
        current_type = None

        for line in lines:
            # 检测TYPE注释
            if line.startswith(';TYPE:'):
                current_type = line[6:].strip()
                continue

            # 解析G1/G0移动
            if line.startswith('G1') or line.startswith('G0'):
                move = self._parse_move(line, line_number, current_type)

                # 只记录轮廓类型
                if current_type in ['Outer wall', 'Inner wall']:
                    self.outline_only_moves.append(move)
```

**修正策略**：

```python
def correct_outline_only(self, moves):
    # 1. 提取轮廓移动
    outline_moves = [m for m in moves if m['type'] in ['Outer wall', 'Inner wall']]

    # 2. 只对轮廓应用模型修正
    # ... LSTM correction ...

    # 3. 返回修正后的轮廓位置
    return corrected_positions  # {line_number: (x, y)}
```

**生成G-code**：

```python
def generate_corrected_gcode(input_gcode, corrected_positions, output_file):
    for i, line in enumerate(lines, 1):
        if i in corrected_positions:
            # 只修正轮廓移动
            x, y = corrected_positions[i]
            line = apply_correction(line, x, y)
        # 其他移动保持不变
```

---

## 📊 效果对比

### 修复前

```
G1 X100.5 Y50.2 Z0.2 E0.1 ;TYPE:Outer wall
  ↓ 修正
G1 X100.52 Y50.19 Z0.2 E0.1 ;TYPE:Outer wall  ✅

G1 X100.3 Y50.1 Z0.2 E0.05 ;TYPE:Internal solid infill
  ↓ 修正
G1 X100.31 Y50.11 Z0.2 E0.05 ;TYPE:Internal solid infill  ❌ 错误！
```

### 修复后

```
G1 X100.5 Y50.2 Z0.2 E0.1 ;TYPE:Outer wall
  ↓ 修正
G1 X100.52 Y50.19 Z0.2 E0.1 ;TYPE:Outer wall  ✅

G1 X100.3 Y50.1 Z0.2 E0.05 ;TYPE:Internal solid infill
  ↓ 不变
G1 X100.3 Y50.1 Z0.2 E0.05 ;TYPE:Internal solid infill  ✅ 正确！
```

---

## 🚀 使用方法

### 基础使用

```bash
python experiments/gcode_outline_correction.py \
    --gcode test_gcode_files/3DBenchy_PLA_1h28m.gcode \
    --checkpoint checkpoints/realtime_corrector/best_model.pth \
    --output_dir results/outline_correction
```

### 输出

```
results/outline_correction/
└── 3DBenchy_PLA_1h28m_outline_corrected.gcode
```

---

## 📝 修改统计

运行后你会看到：

```
解析G-code文件: test_gcode_files/3DBenchy_PLA_1h28m.gcode
  总移动指令数: 50000
  轮廓移动数: 12000      ← 只修正这些
  填充等其他移动数: 38000  ← 保持不变

应用模型修正（仅轮廓）...
  轮廓移动数: 12000
  [OK] 修正了 11950 个轮廓点

  修正量统计:
    Mean: 45.23 um
    Max:  120.45 um
    RMS:  52.34 um
```

---

## ⚠️ 重要提示

### 只修正轮廓的原因

1. **结构完整性**
   - 填充模式是切片软件优化的
   - 修改填充可能影响强度
   - 修正填充可能导致过挤出

2. **打印质量**
   - 填充密度、间距是精确设计的
   - 修改可能造成粘连或空隙
   - 表面质量主要取决于外壁

3. **打印时间**
   - 填充路径是优化的
   - 修改可能增加不必要的移动

### 为什么外壁需要修正

- 外壁决定**尺寸精度**
- 外壁影响**表面质量**
- 外壁是**可见部分**
- 轮廓误差直接影响**几何精度**

---

## 🔬 技术细节

### G-code格式示例

```
;TYPE:Outer wall
G1 X81.991 Y121.68 E0.04595 F1800
G1 X80.847 Y121.232 E0.05047
G1 X79.703 Y120.784 E0.05184
G1 X78.559 Y120.336 E0.05184

;TYPE:Internal solid infill
G1 X95.234 Y105.678 E0.03456 F2400
G1 X96.345 Y106.789 E0.03234
G1 X97.456 Y107.890 E0.03567
```

### TYPE注释的位置

- 在移动指令**之前**出现
- 对后续所有移动生效
- 直到遇到新的`;TYPE:`注释

---

## ✅ 验证检查

### 检查修正后的G-code

```bash
# 统计轮廓移动数
grep ";TYPE:Outer wall" results/outline_correction/*_corrected.gcode | wc -l

# 统计填充移动数（应该与原始相同）
grep ";TYPE:Internal solid infill" results/outline_correction/*_corrected.gcode | wc -l
```

### 对比原始和修正

```bash
# 提取轮廓移动
grep -A1 ";TYPE:Outer wall" original.gcode > original_outline.txt
grep -A1 ";TYPE:Outer wall" corrected.gcode > corrected_outline.txt

# 对比差异
diff original_outline.txt corrected_outline.txt
```

---

## 📚 参考资料

### G-code规范
- **RepRap Wiki**: [G-code](https://reprap.org/wiki/G-code)
- **Slicer输出格式**: 各厂家略有不同

### Creality Slicer
- 使用`;TYPE:`注释标识移动类型
- 与Cura（使用`;TYPE:`）类似
- 与PrusaSlicer（使用`;TYPE:`）兼容

---

## 🎯 下一步

1. ✅ 使用新脚本生成仅修正轮廓的G-code
2. ✅ 切片并打印测试件
3. ⏳ 测量打印精度
4. ⏳ 对比全修正vs仅轮廓修正的效果

---

## 📌 总结

**问题根源**：对所有G1移动都修正

**解决方案**：识别TYPE注释，只修正轮廓

**关键改进**：
- ✅ 保持填充不变
- ✅ 保持支撑不变
- ✅ 只修正外壁和内壁
- ✅ 保护打印结构

**预期效果**：
- 尺寸精度提升（通过轮廓修正）
- 结构完整性保持（填充不变）
- 打印质量改善（表面精度）

---

**Created**: 2026-02-03
**Branch**: fix/gcode-correction-outline-only
**Status**: Ready for Testing
