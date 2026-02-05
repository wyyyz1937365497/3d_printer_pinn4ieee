"""
针对现有Klipper + IP摄像头的配置适配器

自动检测并使用用户现有的Klipper和IP摄像头配置
"""

import os
import sys
import json
import requests
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.vision_processor import VisionProcessor

# 默认配置（基于用户提供的实际配置）
DEFAULT_CONFIG = {
    'klipper': {
        'host': '10.168.1.123',
        'port': 19255,
        'api_url': 'http://10.168.1.123:19255'
    },
    'camera': {
        'host': '10.168.1.129',
        'port': 8080,
        'base_url': 'http://10.168.1.129:8080',
        'snapshot_url': 'http://10.168.1.129:8080/shot.jpg',
        'mjpeg_url': 'http://10.168.1.129:8080/video'
    }
}


class ExistingKlipperCollector:
    """
    适配现有Klipper + IP摄像头的数据收集器
    """

    def __init__(self, config=None):
        """
        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config or DEFAULT_CONFIG

        # Klipper API
        self.klipper_url = self.config['klipper']['api_url']

        # IP摄像头
        self.camera_snapshot_url = self.config['camera']['snapshot_url']
        self.camera_stream_url = self.config['camera']['mjpeg_url']

        # 初始化视觉处理器
        self.vision = VisionProcessor()

        # 当前任务信息
        self.current_job = {
            'filename': None,
            'start_time': None,
            'layers_collected': 0
        }

        # 数据存储
        self.collected_data = []

        # 输出目录
        self.output_dir = Path('data/collected_photos')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\n=== 数据收集器初始化 ===")
        print(f"Klipper API: {self.klipper_url}")
        print(f"摄像头快照: {self.camera_snapshot_url}")
        print(f"输出目录: {self.output_dir}")

    def test_klipper_connection(self):
        """测试Klipper连接"""
        try:
            # 测试打印机信息
            response = requests.get(f"{self.klipper_url}/printer/info", timeout=5)

            if response.status_code == 200:
                info = response.json()
                print("\n✅ Klipper连接成功")
                print(f"   打印机名称: {info.get('hostname', 'Unknown')}")
                print(f"   Klipper版本: {info.get('software_version', 'Unknown')}")
                print(f"   MCU: {info.get('mcu', 'Unknown')}")
                return True
            else:
                print(f"\n❌ Klipper连接失败: HTTP {response.status_code}")
                return False

        except requests.exceptions.ConnectionError:
            print(f"\n❌ 无法连接到Klipper: {self.klipper_url}")
            print("   请检查：")
            print("   1. Klipper是否在运行")
            print("   2. IP地址是否正确: 10.168.1.123:19255")
            return False
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            return False

    def test_camera_connection(self):
        """测试IP摄像头连接"""
        try:
            # 测试获取快照
            response = requests.get(self.camera_snapshot_url, timeout=10)

            if response.status_code == 200:
                # 测试解码
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                if image is not None:
                    # 保存测试图片
                    test_path = self.output_dir / 'camera_test.jpg'
                    cv2.imwrite(str(test_path), image)

                    print("\n✅ IP摄像头连接成功")
                    print(f"   分辨率: {image.shape[1]}x{image.shape[0]}")
                    print(f"   测试照片已保存: {test_path}")
                    return True
                else:
                    print("\n❌ 无法解码图像")
                    return False
            else:
                print(f"\n❌ IP摄像头连接失败: HTTP {response.status_code}")
                return False

        except requests.exceptions.ConnectionError:
            print(f"\n❌ 无法连接到IP摄像头: {self.camera_snapshot_url}")
            print("   请检查：")
            print("   1. IP摄像头是否在运行")
            print("   2. IP地址是否正确: 10.168.1.129:8080")
            return False
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            return False

    def capture_photo(self):
        """
        从IP摄像头获取照片

        Returns:
            image: OpenCV图像或None
        """
        try:
            response = requests.get(self.camera_snapshot_url, timeout=10)

            if response.status_code == 200:
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                if image is not None:
                    return image

            return None

        except Exception as e:
            print(f"拍照失败: {e}")
            return None

    def process_layer(self, layer_num, gcode_filename):
        """
        处理单个层的数据收集

        Args:
            layer_num: 层号
            gcode_filename: G-code文件名

        Returns:
            result: 处理结果字典
        """
        print(f"\n处理层 {layer_num}...")

        # 1. 拍摄
        image = self.capture_photo()

        if image is None:
            return {'success': False, 'error': '拍照失败'}

        # 2. 保存原始图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{Path(gcode_filename).stem}_layer{layer_num:03d}_{timestamp}.jpg"
        image_path = self.output_dir / image_filename
        cv2.imwrite(str(image_path), image)

        print(f"  图像已保存: {image_path}")

        # 3. 图像预处理
        processed = self.vision.preprocess_image(str(image_path))

        # 4. 提取轮廓
        measured_contour = self.vision.extract_contour(processed)

        if len(measured_contour) == 0:
            print(f"  ⚠️ 未能提取轮廓")
            # 仍然保存图像以便后续分析
            return {
                'success': True,
                'warning': '轮廓提取失败',
                'image_path': str(image_path)
            }

        # 5. 转换为毫米
        contour_mm = self.vision.pixel_to_mm_scale(measured_contour)

        # 6. 生成理想轮廓（临时使用平滑轮廓）
        ideal_contour = self._generate_ideal_contour(measured_contour)

        # 7. 对齐
        aligned, transform, error = self.vision.align_contours_icp(
            measured_contour, ideal_contour
        )

        # 8. 计算误差
        errors, stats = self.vision.compute_errors(aligned, ideal_contour)

        # 9. 保存数据
        layer_data = {
            'layer': layer_num,
            'gcode_file': gcode_filename,
            'image_path': str(image_path),
            'contour': aligned.tolist(),
            'errors': errors.tolist(),
            'stats': stats,
            'timestamp': timestamp
        }

        self.collected_data.append(layer_data)
        self.current_job['layers_collected'] += 1

        print(f"  ✅ 处理完成:")
        print(f"     点数: {stats['n_points']}")
        print(f"     RMS误差: {stats['rms_um']:.2f} um")
        print(f"     最大误差: {stats['max_um']:.2f} um")

        return {'success': True, 'data': layer_data}

    def _generate_ideal_contour(self, measured):
        """生成理想轮廓（临时使用平滑版本）"""
        if len(measured) < 10:
            return measured

        # 使用高斯滤波平滑
        smoothed = cv2.GaussianBlur(measured, (5, 5), 0)
        return smoothed

    def save_dataset(self):
        """保存收集的数据集"""
        if len(self.collected_data) == 0:
            print("\n⚠️  没有数据可保存")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存NPZ
        npz_file = self.output_dir / f"dataset_{timestamp}.npz"

        data_dict = {
            'layers': [d['layer'] for d in self.collected_data],
            'gcode_files': [d['gcode_file'] for d in self.collected_data],
            'image_paths': [d['image_path'] for d in self.collected_data],
            'contours': [d['contour'] for d in self.collected_data],
            'errors': [d['errors'] for d in self.collected_data],
            'stats': [d['stats'] for d in self.collected_data],
        }

        np.savez_compressed(npz_file, **data_dict)

        # 保存JSON元数据
        json_file = self.output_dir / f"metadata_{timestamp}.json"

        metadata = {
            'job': self.current_job,
            'total_layers': len(self.collected_data),
            'dataset_file': str(npz_file),
            'config': self.config
        }

        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ 数据集已保存:")
        print(f"   NPZ: {npz_file}")
        print(f"   JSON: {json_file}")
        print(f"   总层数: {len(self.collected_data)}")

    def get_status(self):
        """获取状态"""
        return {
            'current_job': self.current_job,
            'layers_collected': self.current_job['layers_collected'],
            'output_dir': str(self.output_dir),
            'config': self.config
        }


def main():
    """测试现有配置"""
    print("="*70)
    print("测试现有Klipper + IP摄像头配置")
    print("="*70)

    # 创建收集器
    collector = ExistingKlipperCollector()

    # 测试Klipper
    klipper_ok = collector.test_klipper_connection()

    # 测试摄像头
    camera_ok = collector.test_camera_connection()

    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    print(f"Klipper连接: {'✅ 成功' if klipper_ok else '❌ 失败'}")
    print(f"摄像头连接: {'✅ 成功' if camera_ok else '❌ 失败'}")

    if klipper_ok and camera_ok:
        print("\n🎉 所有组件正常！")
        print("\n下一步：")
        print("1. 在Klipper配置中添加层完成宏（见下方）")
        print("2. 启动数据收集服务")
        print("3. 开始打印")
    else:
        print("\n⚠️  请先解决上述问题")

    # 显示Klipper宏配置建议
    print("\n" + "="*70)
    print("Klipper宏配置建议")
    print("="*70)
    print("""
在您的printer.cfg中添加：

[gcode_macro LAYER_COMPLETE]
description: "触发数据收集拍照"
gcode:
    {action_call_http(
        method="POST",
        url="http://10.168.1.129:5000/capture",
        body={"layer": {printer.gcode_move.position.z},
               "filename": "{printer.print_stats.filename}"}
    )}
    {action_respond_info("Layer {layer} captured")}

然后在Mainsail中重启Klipper。
    """)


if __name__ == '__main__':
    main()
