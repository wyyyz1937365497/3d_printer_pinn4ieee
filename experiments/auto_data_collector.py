"""
自动化数据收集服务

功能：
1. HTTP服务器监听Klipper的拍照请求
2. 调用ESP-CAM获取图像
3. 自动处理图像并提取轮廓
4. 计算与STL理想轮廓的误差
5. 保存结构化训练数据

架构：
    Klipper (layer_complete) → HTTP POST → Python Server
                                                    ↓
                                              ESP-CAM capture
                                                    ↓
                                              Image processing
                                                    ↓
                                              Error calculation
                                                    ↓
                                              Save data
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from threading import Thread
import traceback

from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.vision_processor import VisionProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AutoDataCollector:
    """
    自动化数据收集器
    """

    def __init__(self,
                 espcam_url: str,
                 output_dir: str = "data/collected_photos",
                 stl_dir: str = "test_stl"):
        """
        Args:
            espcam_url: ESP-CAM的HTTP地址
            output_dir: 输出目录
            stl_dir: STL文件目录
        """
        self.espcam_url = espcam_url
        self.output_dir = Path(output_dir)
        self.stl_dir = Path(stl_dir)

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化视觉处理器
        self.vision = VisionProcessor()

        # 当前打印任务信息
        self.current_job = {
            'filename': None,
            'start_time': None,
            'layers_collected': 0
        }

        # 数据存储
        self.collected_data = []

        logger.info("AutoDataCollector initialized")
        logger.info(f"  ESP-CAM URL: {espcam_url}")
        logger.info(f"  Output dir: {self.output_dir}")

    def capture_photo(self) -> np.ndarray:
        """
        从ESP-CAM获取照片

        Returns:
            image: OpenCV图像
        """
        try:
            # 方法1：HTTP GET capture
            capture_url = f"{self.espcam_url}/capture"
            response = requests.get(capture_url, timeout=10)

            if response.status_code == 200:
                # 转换为OpenCV格式
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                logger.info("Photo captured from ESP-CAM")
                return image
            else:
                logger.error(f"Failed to capture photo: HTTP {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error capturing photo: {e}")
            return None

    def get_ideal_contour(self, stl_filename: str, layer_z: float):
        """
        从STL获取理想轮廓

        Args:
            stl_filename: STL文件名
            layer_z: 层高度

        Returns:
            contour: [n, 2] 理想轮廓点（mm）
        """
        # TODO: 集成STL切片功能
        # 这里先返回占位数据

        logger.warning(f"STL contour generation not yet implemented: {stl_filename}")
        return None

    def process_layer(self, layer_num: int, gcode_filename: str) -> dict:
        """
        处理单个层的数据收集

        Args:
            layer_num: 层数
            gcode_filename: G-code文件名

        Returns:
            result: 处理结果
        """
        logger.info(f"\nProcessing layer {layer_num}")

        # 1. 拍摄
        image = self.capture_photo()
        if image is None:
            return {'success': False, 'error': 'Failed to capture photo'}

        # 2. 保存原始图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{gcode_filename}_layer{layer_num:03d}_{timestamp}.jpg"
        image_path = self.output_dir / image_filename
        cv2.imwrite(str(image_path), image)
        logger.info(f"  Saved: {image_path}")

        # 3. 提取轮廓
        processed = self.vision.preprocess_image(str(image_path))
        measured_contour = self.vision.extract_contour(processed)

        if len(measured_contour) == 0:
            return {'success': False, 'error': 'Failed to extract contour'}

        # 4. 获取理想轮廓
        # TODO: 从STL生成
        # ideal_contour = self.get_ideal_contour(gcode_filename, layer_num)

        # 临时：使用轮廓的中心作为理想轮廓（测试用）
        # 实际应用中应该从STL切片获取
        ideal_contour = self._generate_test_ideal_contour(measured_contour)

        # 5. 对齐
        aligned, transform, error = self.vision.align_contours_icp(
            measured_contour, ideal_contour
        )

        # 6. 计算误差
        errors, stats = self.vision.compute_errors(aligned, ideal_contour)

        # 7. 保存数据
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

        logger.info(f"  Layer {layer_num} processed:")
        logger.info(f"    Points: {stats['n_points']}")
        logger.info(f"    RMS error: {stats['rms_um']:.2f} um")
        logger.info(f"    Max error: {stats['max_um']:.2f} um")

        return {'success': True, 'data': layer_data}

    def _generate_test_ideal_contour(self, measured_contour: np.ndarray) -> np.ndarray:
        """
        生成测试用理想轮廓（占位函数）

        实际应用中应该从STL切片获取

        Args:
            measured_contour: 测量轮廓

        Returns:
            理想轮廓（使用测量轮廓的平滑版本）
        """
        # 简单平滑处理
        if len(measured_contour) < 10:
            return measured_contour

        # 使用高斯滤波平滑
        smoothed = cv2.GaussianBlur(measured_contour, (5, 5), 0)

        return smoothed

    def save_dataset(self):
        """保存收集的数据集"""
        if len(self.collected_data) == 0:
            logger.warning("No data to save")
            return

        # 保存为NPZ格式
        output_file = self.output_dir / f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"

        data_dict = {
            'layers': [d['layer'] for d in self.collected_data],
            'gcode_files': [d['gcode_file'] for d in self.collected_data],
            'image_paths': [d['image_path'] for d in self.collected_data],
            'contours': [d['contour'] for d in self.collected_data],
            'errors': [d['errors'] for d in self.collected_data],
            'stats': [d['stats'] for d in self.collected_data],
        }

        np.savez_compressed(output_file, **data_dict)

        logger.info(f"\nDataset saved: {output_file}")
        logger.info(f"  Total layers: {len(self.collected_data)}")

        # 保存JSON元数据
        metadata_file = self.output_dir / f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'job': self.current_job,
                'total_layers': len(self.collected_data),
                'output_file': str(output_file)
            }, f, indent=2)

        logger.info(f"  Metadata saved: {metadata_file}")

    def get_status(self):
        """获取收集状态"""
        return {
            'current_job': self.current_job,
            'layers_collected': self.current_job['layers_collected'],
            'output_dir': str(self.output_dir)
        }


# Flask应用
app = Flask(__name__)
collector = None


@app.route('/capture', methods=['POST'])
def capture_layer():
    """
    处理Klipper的拍照请求

    请求格式：
    {
        "layer": 10,
        "filename": "test.gcode"
    }
    """
    try:
        data = request.get_json()
        layer = data.get('layer')
        filename = data.get('filename')

        if layer is None or filename is None:
            return jsonify({'error': 'Missing layer or filename'}), 400

        # 更新当前任务信息
        if collector.current_job['filename'] != filename:
            # 新任务
            collector.current_job['filename'] = filename
            collector.current_job['start_time'] = datetime.now().isoformat()
            collector.current_job['layers_collected'] = 0
            collector.collected_data = []  # 清空之前的数据

        # 处理层
        result = collector.process_layer(int(layer), filename)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in /capture: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/status', methods=['GET'])
def get_status():
    """获取状态"""
    return jsonify(collector.get_status())


@app.route('/save', methods=['POST'])
def save_dataset():
    """保存数据集"""
    collector.save_dataset()
    return jsonify({'success': True})


@app.route('/stop', methods=['POST'])
def stop_collection():
    """停止收集并保存"""
    collector.save_dataset()
    return jsonify({'success': True, 'message': 'Data collection stopped'})


def main():
    import argparse

    parser = argparse.ArgumentParser(description='自动数据收集服务')
    parser.add_argument('--espcam', type=str, default='http://192.168.1.100',
                       help='ESP-CAM URL')
    parser.add_argument('--port', type=int, default=5000,
                       help='HTTP server port')
    parser.add_argument('--output', type=str, default='data/collected_photos',
                       help='Output directory')

    args = parser.parse_args()

    # 创建收集器
    global collector
    collector = AutoDataCollector(
        espcam_url=args.espcam,
        output_dir=args.output
    )

    # 启动服务器
    logger.info(f"\nStarting auto data collection service...")
    logger.info(f"  HTTP port: {args.port}")
    logger.info(f"  ESP-CAM: {args.espcam}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"\nAPI endpoints:")
    logger.info(f"  POST http://localhost:{args.port}/capture")
    logger.info(f"  GET  http://localhost:{args.port}/status")
    logger.info(f"  POST http://localhost:{args.port}/save")
    logger.info(f"  POST http://localhost:{args.port}/stop")
    logger.info("\nWaiting for Klipper requests...\n")

    app.run(host='0.0.0.0', port=args.port, debug=False)


if __name__ == '__main__':
    main()
