"""
生成外轮廓精度测试件G-code

包含5个测试件：
1. 简单立方体（20×20×5mm） - 测试X/Y轴直线精度
2. 3棱锥（底20×20，高15mm） - 测试对称斜面精度
3. 4棱锥（底30×30，高18mm） - 测试锐角转向精度
4. 带孔圆柱（外20×内10×5mm） - 测试内外轮廓对比
5. 扁平三角形（底20×高20×3mm） - 测试45°斜边精度

关键特性：
- 添加;TYPE:注释以便outline correction脚本识别
- 仅包含外轮廓，无填充（纯粹测试外轮廓精度）
- 使用标准的G-code格式
- 适合直接用于修正模型测试

使用方法：
    python test_gcode_files/generate_outline_test_pieces.py

输出：
    test_gcode_files/outline_test_pieces.gcode
"""

import numpy as np
import math
from pathlib import Path


def write_header(f):
    """写入G-code头部"""
    f.write("; 外轮廓精度测试件 - Outline Accuracy Test Pieces\n")
    f.write("; 包含5个测试件：立方体、3棱锥、4棱锥、带孔圆柱、三角形\n")
    f.write("; 用途：测试外轮廓优化模型的效果\n\n")

    f.write("; G-code生成器: 自定义Python脚本\n")
    f.write("; 生成日期: 2026-02-04\n\n")

    # 标准G-code头部
    f.write("G28 ; Home all axes\n")
    f.write("G1 Z15 F3000 ; Move to travel height\n")
    f.write("G90 ; Use absolute coordinates\n")
    f.write("G21 ; Units in millimeters\n\n")

    # 加热指令（根据实际打印机调整）
    f.write("; --- 加热 ---\n")
    f.write("M104 S200 ; Set extruder temp to 200°C\n")
    f.write("M140 S60 ; Set bed temp to 60°C\n")
    f.write("M190 S60 ; Wait for bed temp\n")
    f.write("M109 S200 ; Wait for extruder temp\n")
    f.write("G28 ; Home again after heating\n")
    f.write("G1 Z15 F3000 ; Move to travel height\n")
    f.write("G92 E0 ; Reset extruder\n")
    f.write("\n; --- 开始打印 ---\n\n")


def write_simple_cube(f, x_offset, y_offset, size=20, height=5, layer_height=0.5, speed=50):
    """
    生成简单立方体（仅外轮廓）

    关键：添加;TYPE:Outer wall注释
    """
    f.write(f"; ========================================\n")
    f.write(f"; 测试件1: 简单立方体 {size}×{size}×{height}mm\n")
    f.write(f"; 位置: ({x_offset}, {y_offset})\n")
    f.write(f"; 用途: 测试X/Y轴直线精度（基准）\n")
    f.write(f"; ========================================\n\n")

    layers = int(height / layer_height)

    for layer in range(layers):
        z = (layer + 1) * layer_height
        f.write(f"; Layer {layer+1}/{layers}\n")
        f.write(f"G1 Z{z:.3f} F3000\n")

        # 标注：外壁（需要修正）
        f.write(";TYPE:Outer wall\n")
        f.write(f";MOVE_TYPE:OUTLINE\n")
        f.write(f";TEST_PIECE:SIMPLE_CUBE\n")
        f.write(f";LAYER:{layer+1}\n")

        # 外轮廓 - 逆时针，从左下角开始
        f.write(f"G1 X{x_offset:.3f} Y{y_offset:.3f} F{speed*60}\n")
        f.write(f"G1 X{x_offset+size:.3f} Y{y_offset:.3f} E{size*0.04:.5f} ; Bottom edge\n")
        f.write(f"G1 X{x_offset+size:.3f} Y{y_offset+size:.3f} E{size*0.04:.5f} ; Right edge\n")
        f.write(f"G1 X{x_offset:.3f} Y{y_offset+size:.3f} E{size*0.04:.5f} ; Top edge\n")
        f.write(f"G1 X{x_offset:.3f} Y{y_offset:.3f} E{size*0.04:.5f} ; Left edge\n")

    f.write("\n")


def write_triangular_pyramid(f, x_offset, y_offset, base_size=20, height=15, layer_height=0.5, speed=50):
    """
    生成3棱锥（正四面体）

    每一层是正三角形，尺寸逐渐缩小
    """
    f.write(f"; ========================================\n")
    f.write(f"; 测试件2: 3棱锥（正四面体）\n")
    f.write(f"; 底边: {base_size}×{base_size}mm, 高: {height}mm\n")
    f.write(f"; 位置: ({x_offset}, {y_offset})\n")
    f.write(f"; 用途: 测试对称斜面精度\n")
    f.write(f"; ========================================\n\n")

    layers = int(height / layer_height)

    # 正三角形顶点计算（底边在下）
    # 顶点A（左下）、B（右下）、C（上中）
    A = (x_offset, y_offset)
    B = (x_offset + base_size, y_offset)
    # C点：等边三角形的高度 = base_size * sqrt(3)/2
    triangle_height = base_size * np.sqrt(3) / 2
    C = (x_offset + base_size / 2, y_offset + triangle_height)

    for layer in range(layers):
        z = (layer + 1) * layer_height

        # 计算当前层的缩放比例（从底到顶线性缩小）
        scale = 1.0 - (layer / layers) * 0.8  # 顶部保留20%

        # 计算当前层顶点
        current_A = (
            A[0] + (C[0] - A[0]) * (1 - scale),
            A[1] + (C[1] - A[1]) * (1 - scale)
        )
        current_B = (
            B[0] + (C[0] - B[0]) * (1 - scale),
            B[1] + (C[1] - B[1]) * (1 - scale)
        )
        current_C = C

        f.write(f"; Layer {layer+1}/{layers}\n")
        f.write(f"G1 Z{z:.3f} F3000\n")

        # 标注：外壁
        f.write(";TYPE:Outer wall\n")
        f.write(f";MOVE_TYPE:OUTLINE\n")
        f.write(f";TEST_PIECE:TRIANGULAR_PYRAMID\n")
        f.write(f";LAYER:{layer+1}\n")

        # 外轮廓：A → B → C → A
        f.write(f"G1 X{current_A[0]:.3f} Y{current_A[1]:.3f} F{speed*60}\n")

        # 边AB
        edge_ab = np.sqrt((current_B[0] - current_A[0])**2 + (current_B[1] - current_A[1])**2)
        f.write(f"G1 X{current_B[0]:.3f} Y{current_B[1]:.3f} E{edge_ab*0.04:.5f} ; Edge AB\n")

        # 边BC
        edge_bc = np.sqrt((current_C[0] - current_B[0])**2 + (current_C[1] - current_B[1])**2)
        f.write(f"G1 X{current_C[0]:.3f} Y{current_C[1]:.3f} E{edge_bc*0.04:.5f} ; Edge BC\n")

        # 边CA
        edge_ca = np.sqrt((current_A[0] - current_C[0])**2 + (current_A[1] - current_C[1])**2)
        f.write(f"G1 X{current_A[0]:.3f} Y{current_A[1]:.3f} E{edge_ca*0.04:.5f} ; Edge CA\n")

    f.write("\n")


def write_square_pyramid(f, x_offset, y_offset, base_size=30, height=18, layer_height=0.5, speed=50):
    """
    生成4棱锥（正方形底）

    关键：测试锐角（约50°底角）
    """
    f.write(f"; ========================================\n")
    f.write(f"; 测试件3: 4棱锥（正方形底）\n")
    f.write(f"; 底边: {base_size}×{base_size}mm, 高: {height}mm\n")
    f.write(f"; 底角: ≈{np.degrees(np.arctan(height/(base_size/2))):.1f}°\n")
    f.write(f"; 位置: ({x_offset}, {y_offset})\n")
    f.write(f"; 用途: 测试锐角转向精度（关键测试）\n")
    f.write(f"; ========================================\n\n")

    layers = int(height / layer_height)

    # 正方形底边顶点
    half_base = base_size / 2
    center_x = x_offset + half_base
    center_y = y_offset + half_base

    for layer in range(layers):
        z = (layer + 1) * layer_height

        # 计算当前层的缩放比例（线性缩小）
        scale = 1.0 - (layer / layers) * 0.85  # 顶部保留15%

        # 当前层边长
        current_size = base_size * scale

        # 当前层顶点（围绕中心缩放）
        half_current = current_size / 2
        A = (center_x - half_current, center_y - half_current)  # 左下
        B = (center_x + half_current, center_y - half_current)  # 右下
        C = (center_x + half_current, center_y + half_current)  # 右上
        D = (center_x - half_current, center_y + half_current)  # 左上

        f.write(f"; Layer {layer+1}/{layers}\n")
        f.write(f"G1 Z{z:.3f} F3000\n")

        # 标注：外壁
        f.write(";TYPE:Outer wall\n")
        f.write(f";MOVE_TYPE:OUTLINE\n")
        f.write(f";TEST_PIECE:SQUARE_PYRAMID\n")
        f.write(f";LAYER:{layer+1}\n")
        f.write(f";FEATURE:SHARP_ANGLE\n")

        # 外轮廓：A → B → C → D → A
        f.write(f"G1 X{A[0]:.3f} Y{A[1]:.3f} F{speed*60}\n")

        f.write(f"G1 X{B[0]:.3f} Y{B[1]:.3f} E{current_size*0.04:.5f} ; Edge AB (bottom)\n")
        f.write(f"G1 X{C[0]:.3f} Y{C[1]:.3f} E{current_size*0.04:.5f} ; Edge BC (right)\n")
        f.write(f"G1 X{D[0]:.3f} Y{D[1]:.3f} E{current_size*0.04:.5f} ; Edge CD (top)\n")
        f.write(f"G1 X{A[0]:.3f} Y{A[1]:.3f} E{current_size*0.04:.5f} ; Edge DA (left)\n")

    f.write("\n")


def write_cylinder_with_hole(f, x_center, y_center, outer_diameter=20, inner_diameter=10, height=5, layer_height=0.5, speed=50, segments=64):
    """
    生成带孔圆柱

    关键：同时生成外轮廓和内轮廓，测试内外对比
    """
    f.write(f"; ========================================\n")
    f.write(f"; 测试件4: 带孔圆柱\n")
    f.write(f"; 外径: {outer_diameter}mm, 内径: {inner_diameter}mm, 高: {height}mm\n")
    f.write(f"; 圆心: ({x_center}, {y_center})\n")
    f.write(f"; 用途: 测试内外轮廓对比（关键测试）\n")
    f.write(f"; ========================================\n\n")

    layers = int(height / layer_height)
    outer_radius = outer_diameter / 2
    inner_radius = inner_diameter / 2

    for layer in range(layers):
        z = (layer + 1) * layer_height
        f.write(f"; Layer {layer+1}/{layers}\n")
        f.write(f"G1 Z{z:.3f} F3000\n")

        # === 外轮廓（Outer wall） ===
        f.write(";TYPE:Outer wall\n")
        f.write(f";MOVE_TYPE:OUTLINE\n")
        f.write(f";TEST_PIECE:CYLINDER_OUTER\n")
        f.write(f";LAYER:{layer+1}\n")

        start_x = x_center + outer_radius
        start_y = y_center
        f.write(f"G1 X{start_x:.3f} Y{start_y:.3f} F{speed*60}\n")

        # 生成外圆周
        circumference = np.pi * outer_diameter
        total_e = circumference * 0.04

        for i in range(1, segments + 1):
            angle = 2 * np.pi * i / segments
            x = x_center + outer_radius * np.cos(angle)
            y = y_center + outer_radius * np.sin(angle)
            e = total_e * i / segments
            f.write(f"G1 X{x:.3f} Y{y:.3f} E{e:.5f}\n")

        # === 内轮廓（Inner wall） ===
        f.write(";TYPE:Inner wall\n")
        f.write(f";MOVE_TYPE:OUTLINE\n")
        f.write(f";TEST_PIECE:CYLINDER_INNER\n")
        f.write(f";LAYER:{layer+1}\n")

        # 移动到内圆起点（不挤出）
        inner_start_x = x_center + inner_radius
        inner_start_y = y_center
        f.write(f"G1 X{inner_start_x:.3f} Y{inner_start_y:.3f} F{speed*60}\n")

        # 生成内圆周（顺时针，为了正确的挤出方向）
        inner_circumference = np.pi * inner_diameter
        inner_total_e = inner_circumference * 0.04

        for i in range(1, segments + 1):
            angle = 2 * np.pi * i / segments
            x = x_center + inner_radius * np.cos(angle)
            y = y_center + inner_radius * np.sin(angle)
            e = inner_total_e * i / segments
            f.write(f"G1 X{x:.3f} Y{y:.3f} E{e:.5f}\n")

    f.write("\n")


def write_flat_triangle(f, x_offset, y_offset, base=20, height=20, thickness=3, layer_height=0.5, speed=50):
    """
    生成扁平等腰直角三角形

    关键：扁平设计便于卡尺测量斜边
    """
    f.write(f"; ========================================\n")
    f.write(f"; 测试件5: 扁平等腰直角三角形\n")
    f.write(f"; 底边: {base}mm, 高: {height}mm, 厚度: {thickness}mm\n")
    f.write(f"; 位置: ({x_offset}, {y_offset})\n")
    f.write(f"; 用途: 测试45°斜边精度（便于测量）\n")
    f.write(f"; ========================================\n\n")

    layers = int(thickness / layer_height)

    # 等腰直角三角形顶点
    A = (x_offset, y_offset)  # 直角顶点（左下）
    B = (x_offset + base, y_offset)  # 底边右顶点
    C = (x_offset, y_offset + height)  # 垂边顶顶点

    # 斜边长度
    hypotenuse = np.sqrt(base**2 + height**2)

    for layer in range(layers):
        z = (layer + 1) * layer_height
        f.write(f"; Layer {layer+1}/{layers}\n")
        f.write(f"G1 Z{z:.3f} F3000\n")

        # 标注：外壁
        f.write(";TYPE:Outer wall\n")
        f.write(f";MOVE_TYPE:OUTLINE\n")
        f.write(f";TEST_PIECE:FLAT_TRIANGLE\n")
        f.write(f";LAYER:{layer+1}\n")
        f.write(f";FEATURE:HYPOTENUSE\n")

        # 外轮廓：A → B → C → A（逆时针）
        f.write(f"G1 X{A[0]:.3f} Y{A[1]:.3f} F{speed*60}\n")

        f.write(f"G1 X{B[0]:.3f} Y{B[1]:.3f} E{base*0.04:.5f} ; Edge AB (bottom)\n")
        f.write(f"G1 X{C[0]:.3f} Y{C[1]:.3f} E{hypotenuse*0.04:.5f} ; Edge BC (hypotenuse)\n")
        f.write(f"G1 X{A[0]:.3f} Y{A[1]:.3f} E{height*0.04:.5f} ; Edge CA (vertical)\n")

    f.write("\n")


def write_footer(f):
    """写入G-code尾部"""
    f.write("\n; ========================================\n")
    f.write("; 所有测试件打印完成\n")
    f.write("; ========================================\n\n")

    f.write("; --- 移动到安全位置 ---\n")
    f.write("G1 Z15 F3000 ; Move to travel height\n")
    f.write("G1 X0 Y0 F3000 ; Home XY\n")
    f.write("\n; --- 关闭加热器 ---\n")
    f.write("M104 S0 ; Turn off extruder\n")
    f.write("M140 S0 ; Turn off bed\n")
    f.write("\n; --- 禁用电机 ---\n")
    f.write("G28 X0 Y0 ; Home XY\n")
    f.write("M84 ; Disable motors\n")

    f.write("\n; ========================================\n")
    f.write("; 测量指南\n")
    f.write("; ========================================\n")
    f.write(";\n")
    f.write("; 测试件1: 简单立方体 (20×20×5mm)\n")
    f.write(";   测量: 4条边的中点（避开转角3-5mm）\n")
    f.write(";   期望: X边=20.00mm, Y边=20.00mm\n")
    f.write(";   目的: 建立X/Y轴精度基准\n")
    f.write(";\n")
    f.write("; 测试件2: 3棱锥（底20×20，高15mm）\n")
    f.write(";   测量: 底边和斜边\n")
    f.write(";   期望: 底边=20.00mm, 斜边≈21.21mm\n")
    f.write(";   目的: 测试对称斜面精度\n")
    f.write(";\n")
    f.write("; 测试件3: 4棱锥（底30×30，高18mm）\n")
    f.write(";   测量: 底边和斜边，目视检查锐角清晰度\n")
    f.write(";   期望: 底边=30.00mm, 斜边≈23.43mm, 底角≈50°\n")
    f.write(";   目的: 测试锐角转向精度（关键！）\n")
    f.write(";\n")
    f.write("; 测试件4: 带孔圆柱（外20×内10×5mm）\n")
    f.write(";   测量: 外径和内孔径（多个方向）\n")
    f.write(";   期望: 外径=20.00mm, 内径=10.00mm\n")
    f.write(";   目的: 内外轮廓对比（验证优化边界）\n")
    f.write(";\n")
    f.write("; 测试件5: 扁平三角形（底20×高20×3mm）\n")
    f.write(";   测量: 底边、垂边、斜边\n")
    f.write(";   期望: 底边=20.00mm, 垂边=20.00mm, 斜边≈28.28mm\n")
    f.write(";   目的: 测试45°复合运动精度\n")
    f.write(";\n")


def generate_outline_test_pieces(output_file):
    """生成完整的外轮廓测试件G-code"""
    print(f"生成外轮廓测试件G-code: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        write_header(f)

        # 打印参数
        layer_height = 0.5
        speed = 50  # mm/s

        # 测试件布局（打印床220×220mm，中心在110,110）
        # 立方体: (80, 80)
        # 3棱锥: (80, 110)
        # 4棱锥: (120, 110)
        # 圆柱: (80, 140)
        # 三角形: (140, 80)

        print("\n生成测试件...")

        # 测试件1: 简单立方体
        print("  [1/5] 简单立方体 20×20×5mm @ (80, 80)")
        write_simple_cube(f, x_offset=80, y_offset=80, size=20, height=5,
                         layer_height=layer_height, speed=speed)

        # 测试件2: 3棱锥
        print("  [2/5] 3棱锥（底20×20，高15mm）@ (80, 110)")
        write_triangular_pyramid(f, x_offset=80, y_offset=110, base_size=20, height=15,
                                layer_height=layer_height, speed=speed)

        # 测试件3: 4棱锥
        print("  [3/5] 4棱锥（底30×30，高18mm）@ (120, 110)")
        write_square_pyramid(f, x_offset=120, y_offset=110, base_size=30, height=18,
                            layer_height=layer_height, speed=speed)

        # 测试件4: 带孔圆柱
        print("  [4/5] 带孔圆柱（外20×内10×5mm）@ (80, 140)")
        write_cylinder_with_hole(f, x_center=90, y_center=140,
                                outer_diameter=20, inner_diameter=10, height=5,
                                layer_height=layer_height, speed=speed)

        # 测试件5: 扁平三角形
        print("  [5/5] 扁平三角形（底20×高20×3mm）@ (140, 80)")
        write_flat_triangle(f, x_offset=140, y_offset=80, base=20, height=20, thickness=3,
                          layer_height=layer_height, speed=speed)

        write_footer(f)

    print(f"\n[OK] 已保存: {output_file}")
    print(f"\n测试件总览:")
    print(f"  1. 简单立方体 20×20×5mm @ (80, 80)")
    print(f"  2. 3棱锥（底20×20，高15mm）@ (80, 110)")
    print(f"  3. 4棱锥（底30×30，高18mm）@ (120, 110)")
    print(f"  4. 带孔圆柱（外20×内10×5mm）@ (90, 140)")
    print(f"  5. 扁平三角形（底20×高20×3mm）@ (140, 80)")
    print(f"\n预计打印时间: 15-20分钟")
    print(f"\nG-code特性:")
    print(f"  ✓ 所有轮廓都有;TYPE:Outer wall/Inner wall注释")
    print(f"  ✓ 仅包含外轮廓，无填充")
    print(f"  ✓ 适合直接用于outline correction测试")
    print(f"\n下一步:")
    print(f"  1. 打印原始版本（此G-code）")
    print(f"  2. 使用修正模型生成修正后G-code:")
    print(f"     python experiments/gcode_outline_correction.py \\")
    print(f"         --gcode {output_file} \\")
    print(f"         --checkpoint checkpoints/realtime_corrector/best_model.pth \\")
    print(f"         --output_dir results/outline_test_pieces")
    print(f"  3. 打印修正版本")
    print(f"  4. 测量对比")


if __name__ == '__main__':
    import os
    from pathlib import Path

    # 创建输出目录
    output_dir = Path("test_gcode_files")
    output_dir.mkdir(exist_ok=True)

    # 生成G-code
    output_file = output_dir / "outline_test_pieces.gcode"
    generate_outline_test_pieces(str(output_file))

    print(f"\n" + "="*70)
    print("生成完成！")
    print("="*70)
    print(f"\n文件已保存到: {output_file}")
    print(f"\n使用说明:")
    print(f"  1. 将此文件导入到 slicer（如Cura）或直接打印")
    print(f"  2. 使用outline correction脚本生成修正版本")
    print(f"  3. 对比原始和修正后的测量结果")
