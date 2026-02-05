"""
硬件设置和测试脚本

功能：
1. 检测Klipper连接
2. 检测ESP-CAM连接
3. 校准摄像头
4. 测试完整流程
"""

import os
import sys
import requests
import cv2
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.vision_processor import VisionProcessor


def test_klipper_connection():
    """测试Klipper连接"""
    print("\n" + "="*70)
    print("测试 1: Klipper连接")
    print("="*70)

    try:
        # Moonraker API
        response = requests.get('http://localhost:7125/server/info', timeout=5)

        if response.status_code == 200:
            info = response.json()
            print("✅ Klipper连接成功")
            print(f"   Klipper版本: {info.get('software_version', 'Unknown')}")
            return True
        else:
            print(f"❌ Klipper连接失败: HTTP {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到Klipper (Moonraker)")
        print("   请检查：")
        print("   1. Moonraker是否正在运行：sudo systemctl status moonraker")
        print("   2. 是否在正确的IP地址上")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_espcam_connection(espcam_url='http://192.168.1.100'):
    """测试ESP-CAM连接"""
    print("\n" + "="*70)
    print("测试 2: ESP-CAM连接")
    print("="*70)

    try:
        # 测试首页
        response = requests.get(espcam_url, timeout=5)

        if response.status_code == 200:
            print("✅ ESP-CAM连接成功")
            print(f"   URL: {espcam_url}")

            # 测试拍照
            capture_url = f"{espcam_url}/capture"
            response = requests.get(capture_url, timeout=10)

            if response.status_code == 200:
                # 保存测试照片
                test_image = Path('data/espcam_test.jpg')
                test_image.parent.mkdir(parents=True, exist_ok=True)

                with open(test_image, 'wb') as f:
                    f.write(response.content)

                print(f"   ✅ 拍照测试成功")
                print(f"   测试照片已保存: {test_image}")

                # 验证照片
                img = cv2.imread(str(test_image))
                if img is not None:
                    print(f"   ✅ 照片验证成功")
                    print(f"   分辨率: {img.shape[1]}x{img.shape[0]}")
                    return True
                else:
                    print(f"   ❌ 照片损坏")
                    return False
            else:
                print(f"   ❌ 拍照失败: HTTP {response.status_code}")
                return False
        else:
            print(f"❌ ESP-CAM连接失败: HTTP {response.status_code}")
            print("   请检查：")
            print("   1. ESP-CAM是否连接到WiFi")
            print("   2. IP地址是否正确")
            print(f"   3. ESP-CAM是否在运行: {espcam_url}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到ESP-CAM: {espcam_url}")
        print("   请检查：")
        print("   1. ESP-CAM是否通电")
        print("   2. WiFi连接是否正常")
        print("   3. 防火墙是否阻止连接")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_vision_processing():
    """测试视觉处理"""
    print("\n" + "="*70)
    print("测试 3: 视觉处理")
    print("="*70)

    test_image = Path('data/espcam_test.jpg')

    if not test_image.exists():
        print(f"❌ 测试照片不存在: {test_image}")
        print("   请先运行ESP-CAM连接测试")
        return False

    try:
        processor = VisionProcessor()

        # 1. 预处理
        print("   预处理图像...")
        processed = processor.preprocess_image(str(test_image))

        # 2. 提取轮廓
        print("   提取轮廓...")
        contour = processor.extract_contour(processed)

        if len(contour) == 0:
            print("   ❌ 未能提取轮廓")
            print("   建议：")
            print("   1. 确保使用蓝色PLA")
            print("   2. 改善照明条件")
            print("   3. 调整摄像头高度和角度")
            return False

        print(f"   ✅ 提取到 {len(contour)} 个轮廓点")

        # 3. 转换为毫米
        contour_mm = processor.pixel_to_mm_scale(contour)
        print(f"   ✅ 轮廓尺寸:")
        print(f"      X: {contour_mm[:, 0].min():.1f} ~ {contour_mm[:, 0].max():.1f} mm")
        print(f"      Y: {contour_mm[:, 1].min():.1f} ~ {contour_mm[:, 1].max():.1f} mm")

        # 4. 可视化
        output_dir = Path('data/test_results')
        output_dir.mkdir(parents=True, exist_ok=True)

        # 绘制轮廓
        img_visualized = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_visualized, [contour.astype(np.int32)], -1, (0, 255, 0), 2)

        output_path = output_dir / 'contour_test.jpg'
        cv2.imwrite(str(output_path), img_visualized)

        print(f"   ✅ 可视化结果已保存: {output_path}")
        return True

    except Exception as e:
        print(f"   ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pixel_calibration():
    """测试像素校准"""
    print("\n" + "="*70)
    print("测试 4: 像素到毫米校准")
    print("="*70)

    print("\n准备校准工具：")
    print("1. 打印一个20x20mm的校准方块")
    print("2. 放置在打印中心")
    print("3. 确保完全在摄像头视野内")
    print("4. 输入实际尺寸（mm）")

    actual_size = input("\n请输入校准方块的实际边长（mm）[默认20.0]: ")
    actual_size = float(actual_size) if actual_size else 20.0

    print("\n开始校准...")

    try:
        processor = VisionProcessor()

        # 拍摄校准方块
        print("正在拍摄校准方块...")
        espcam_url = 'http://192.168.1.100'
        response = requests.get(f"{espcam_url}/capture", timeout=10)

        if response.status_code != 200:
            print("❌ 拍照失败")
            return False

        # 保存照片
        calib_image = Path('data/calibration.jpg')
        with open(calib_image, 'wb') as f:
            f.write(response.content)

        # 处理
        processed = processor.preprocess_image(str(calib_image))
        contour = processor.extract_contour(processed)

        if len(contour) == 0:
            print("❌ 未能检测到校准方块")
            return False

        # 计算像素尺寸
        x_min, y_min = np.min(contour, axis=0)
        x_max, y_max = np.max(contour, axis=0)
        pixel_size = max(x_max - x_min, y_max - y_min)

        # 计算比例
        pixel_to_mm = actual_size / pixel_size

        print(f"\n✅ 校准成功！")
        print(f"   实际尺寸: {actual_size} mm")
        print(f"   像素尺寸: {pixel_size:.2f} px")
        print(f"   比例: 1 px = {pixel_to_mm:.4f} mm")
        print(f"   分辨率: {1/pixel_to_mm:.2f} px/mm")

        # 保存校准结果
        calib_file = Path('data/pixel_calibration.json')
        import json
        with open(calib_file, 'w') as f:
            json.dump({
                'pixel_to_mm': pixel_to_mm,
                'actual_size_mm': actual_size,
                'pixel_size': float(pixel_size),
                'calibration_date': str(datetime.now())
            }, f, indent=2)

        print(f"\n   校准数据已保存: {calib_file}")

        # 更新处理器
        processor.pixel_to_mm = pixel_to_mm

        return True

    except Exception as e:
        print(f"❌ 校准失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_workflow():
    """测试完整工作流程"""
    print("\n" + "="*70)
    print("测试 5: 完整工作流程")
    print("="*70)

    print("\n准备测试打印：")
    print("1. 切片一个简单的测试件（如20mm立方）")
    print("2. 上传到Klipper")
    print("3. 准备好蓝色PLA")
    print("4. 确保ESP-CAM位置正确")

    input("\n按Enter开始测试...")

    try:
        # 启动数据收集服务（测试模式）
        print("\n启动数据收集服务...")

        from experiments.auto_data_collector import AutoDataCollector

        collector = AutoDataCollector(
            espcam_url='http://192.168.1.100',
            output_dir='data/test_collection'
        )

        # 模拟层完成
        print("\n模拟第1层完成...")
        result = collector.process_layer(layer_num=1, gcode_filename='test_cube.gcode')

        if result['success']:
            print("✅ 测试成功！")
            print(f"   收集到 {result['data']['stats']['n_points']} 个数据点")
            print(f"   RMS误差: {result['data']['stats']['rms_um']:.2f} um")

            # 保存数据
            collector.save_dataset()
            return True
        else:
            print(f"❌ 测试失败: {result.get('error')}")
            return False

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='硬件设置和测试')
    parser.add_argument('--espcam', type=str, default='http://192.168.1.100',
                       help='ESP-CAM URL')
    parser.add_argument('--test', type=str, default='all',
                       choices=['all', 'klipper', 'espcam', 'vision', 'calibration', 'workflow'],
                       help='测试项目')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("3D打印自动数据收集系统 - 硬件测试")
    print("="*70)

    results = {}

    if args.test in ['all', 'klipper']:
        results['klipper'] = test_klipper_connection()

    if args.test in ['all', 'espcam']:
        results['espcam'] = test_espcam_connection(args.espcam)

    if args.test in ['all', 'vision']:
        if results.get('espcam', True):
            results['vision'] = test_vision_processing()
        else:
            print("\n⚠️ 跳过视觉处理测试（ESP-CAM未连接）")

    if args.test in ['all', 'calibration']:
        results['calibration'] = test_pixel_calibration()

    if args.test in ['all', 'workflow']:
        results['workflow'] = test_full_workflow()

    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)

    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 所有测试通过！系统已准备就绪。")
        print("\n下一步：")
        print("  1. 启动数据收集服务：")
        print("     python experiments/auto_data_collector.py")
        print("  2. 在Klipper中开始打印")
        print("  3. 观察自动数据收集过程")
    else:
        print("\n⚠️ 部分测试失败，请先解决上述问题。")
        print("   可以单独测试某个项目：")
        print(f"   python experiments/setup_hardware.py --test <test_name>")


if __name__ == '__main__':
    from datetime import datetime
    main()
