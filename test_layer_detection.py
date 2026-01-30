#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试：检测所有gcode文件的层数
"""

import os
import re
import sys

# 设置Windows兼容的编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

gcode_files = [
    'test_gcode_files/3DBenchy_PLA_1h28m.gcode',
    'test_gcode_files/bearing5_PLA_2h27m.gcode',
    'test_gcode_files/Nautilus_Gears_Plate_PLA_3h36m.gcode',
    'test_gcode_files/simple_boat5_PLA_4h4m.gcode'
]

print("=" * 80)
print("Gcode文件层数检测")
print("=" * 80)
print()

total_layers = 0

for gcode_file in gcode_files:
    if not os.path.exists(gcode_file):
        print(f"[X] 文件不存在: {gcode_file}")
        continue

    # 方法1: 查找 "total layer number" 注释
    max_layer = 0
    with open(gcode_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(r'total layer number:\s*(\d+)', line, re.IGNORECASE)
            if match:
                max_layer = int(match.group(1))
                break

    if max_layer > 0:
        print(f"[OK] {os.path.basename(gcode_file)}")
        print(f"  总层数: {max_layer}")

        # 计算采样后的层数（间隔5，最多100层）
        if max_layer <= 100:
            layers_to_collect = max_layer
            sampling_method = "全部"
        else:
            layers_to_collect = min(100, len(range(1, max_layer + 1, 5)))
            sampling_method = f"采样间隔5 (收集{layers_to_collect}/{max_layer} = {layers_to_collect/max_layer*100:.0f}%)"

        print(f"  收集层数: {layers_to_collect} ({sampling_method})")

        total_layers += layers_to_collect
        print()
    else:
        print(f"[!] {os.path.basename(gcode_file)}: 无法检测层数")
        print()

print("=" * 80)
print(f"总仿真层数: {total_layers}")
print(f"预计时间 (GPU): {total_layers * 0.5:.1f} 分钟 ({total_layers * 0.5 / 60:.2f} 小时)")
print(f"预计时间 (CPU): {total_layers * 1.5:.1f} 分钟 ({total_layers * 1.5 / 60:.2f} 小时)")
print("=" * 80)
