"""
适配现有Klipper + IP摄像头的自动数据收集服务

根据用户实际配置：
- Klipper: 10.168.1.123:19255
- IP摄像头: 10.168.1.129:8080
  - 快照: http://10.168.1.129:8080/shot.jpg

使用方法：
    python experiments/auto_data_collector_existing.py \
        --klipper-host 10.168.1.123 \
        --klipper-port 19255 \
        --camera-host 10.168.1.129 \
        --camera-port 8080 \
        --output data/collected_photos
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests
import cv2
import numpy as np

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


class ExistingSetupCollector:
    """
    适配现有Klipper + IP摄像头的数据收集器
    """

    def __init__(self, klipper_config, camera_config, output_dir):
        """
        Args:
            klipper_config: Klipper配置
            camera_config: 摄像头配置
            output_dir: 输出目录
        """
        self.klipper_config = klipper_config
        self.camera_config = camera_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # API URLs
        self.camera_snapshot_url = f"http://{camera_config['host']}:{camera_config['port']}/shot.jpg"

        # 初始化视觉处理器
        self.vision = VisionProcessor()

        # 当前任务
        self.current_job = {
            'filename': None,
            'start_time': None,
            'layers_collected': 0
        }

        # 数据存储
        self.collected_data = []

        logger.info("数据收集器初始化完成")
        logger.info(f"  Klipper: {klipper_config['host']}:{klipper_config['port']}")
        logger.info(f"  摄像头: {camera_config['host']}:{camera_config['port']}")
        logger.info(f"  输出目录: {self.output_dir}")

    def capture_from_camera(self):
        """从IP摄像头获取照片"""
        try:
            logger.debug("正在拍照...")
            response = requests.get(self.camera_snapshot_url, timeout=10)

            if response.status_code == 200:
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                if image is not None:
                    logger.debug(f"拍照成功: {image.shape}")
                    return image

            logger.error("拍照失败: 无法解码图像")
            return None

        except Exception as e:
            logger.error(f"拍照失败: {e}")
            return None

    def process_layer(self, layer_num, gcode_filename):
        """处理单层数据"""
        logger.info(f"处理层 {layer_num}")

        # 1. 拍照
        image = self.capture_from_camera()
        if image is None:
            return {'success': False, 'error': '拍照失败'}

        # 2. 保存原图
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = Path(gcode_filename).stem
        image_filename = f"{safe_filename}_layer{layer_num:03d}_{timestamp}.jpg"
        image_path = self.output_dir / image_filename
        cv2.imwrite(str(image_path), image)

        logger.info(f"  图像已保存: {image_filename}")

        # 3. 处理图像
        try:
            processed = self.vision.preprocess_image(str(image_path))
            measured = self.vision.extract_contour(processed)

            if len(measured) == 0:
                logger.warning("未能提取轮廓")
                return {
                    'success': True,
                    'warning': '轮廓提取失败',
                    'image_path': str(image_path)
                }

            # 转换单位
            measured_mm = self.vision.pixel_to_mm_scale(measured)

            # 生成理想轮廓（临时）
            ideal = self._generate_ideal_contour(measured)

            # 对齐
            aligned, transform, error = self.vision.align_contours_icp(measured, ideal)

            # 计算误差
            errors, stats = self.vision.compute_errors(aligned, ideal)

            # 保存
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

            logger.info(f"  处理成功: {stats['n_points']}点, RMS={stats['rms_um']:.2f}um")

            return {'success': True, 'data': layer_data}

        except Exception as e:
            logger.error(f"处理失败: {e}")
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}

    def _generate_ideal_contour(self, measured):
        """生成理想轮廓（临时）"""
        if len(measured) < 10:
            return measured
        return cv2.GaussianBlur(measured, (5, 5), 0)

    def save_dataset(self):
        """保存数据集"""
        if len(self.collected_data) == 0:
            logger.warning("没有数据可保存")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # NPZ
        npz_file = self.output_dir / f"dataset_{timestamp}.npz"
        np.savez_compressed(
            npz_file,
            layers=[d['layer'] for d in self.collected_data],
            gcode_files=[d['gcode_file'] for d in self.collected_data],
            image_paths=[d['image_path'] for d in self.collected_data],
            contours=[d['contour'] for d in self.collected_data],
            errors=[d['errors'] for d in self.collected_data],
            stats=[d['stats'] for d in self.collected_data]
        )

        # JSON
        json_file = self.output_dir / f"metadata_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'job': self.current_job,
                'total_layers': len(self.collected_data),
                'dataset': str(npz_file),
                'timestamp': timestamp
            }, f, indent=2)

        logger.info(f"数据集已保存: {npz_file.name}")
        logger.info(f"  总层数: {len(self.collected_data)}")

    def get_status(self):
        """获取状态"""
        return {
            'job': self.current_job,
            'layers_collected': self.current_job['layers_collected'],
            'output_dir': str(self.output_dir)
        }


# Flask应用
app = Flask(__name__)
collector = None


@app.route('/capture', methods=['POST'])
def capture_layer():
    """
    接收Klipper的拍照请求

    请求格式：
    {
        "layer": 10,
        "filename": "test.gcode"
    }
    """
    try:
        # 记录原始请求数据用于调试
        logger.info(f"收到/capture请求")
        logger.info(f"  Content-Type: {request.content_type}")
        logger.info(f"  Raw data: {request.get_data(as_text=True)[:200]}")

        # 尝试获取JSON数据
        data = request.get_json(silent=True)
        logger.info(f"  Parsed JSON: {data}")

        # 如果JSON解析失败，尝试从表单数据获取
        if data is None:
            layer = request.form.get('layer')
            filename = request.form.get('filename')
            logger.info(f"  Form data - layer: {layer}, filename: {filename}")
        else:
            layer = data.get('layer')
            filename = data.get('filename')
            logger.info(f"  JSON data - layer: {layer}, filename: {filename}")

        # 如果还是没有，尝试从URL参数获取
        if layer is None:
            layer = request.args.get('layer')
        if filename is None:
            filename = request.args.get('filename', 'unknown')

        # 转换layer为整数
        if layer is not None:
            try:
                layer = int(layer)
            except (ValueError, TypeError):
                pass

        if layer is None:
            logger.warning("缺少layer参数")
            return jsonify({'error': '缺少layer参数', 'received_data': str(request.get_data(as_text=True))}), 400

        if filename is None:
            filename = 'unknown'

        logger.info(f"最终参数 - 层{layer}, 文件{filename}")

        # 更新任务
        if collector.current_job['filename'] != filename:
            collector.current_job['filename'] = filename
            collector.current_job['start_time'] = datetime.now().isoformat()
            collector.current_job['layers_collected'] = 0
            collector.collected_data = []

        # 处理
        result = collector.process_layer(int(layer), filename)

        return jsonify(result)

    except Exception as e:
        logger.error(f"/capture错误: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/status', methods=['GET'])
def get_status():
    """获取状态"""
    return jsonify(collector.get_status())


@app.route('/save', methods=['POST'])
def save_dataset():
    """保存数据集"""
    try:
        collector.save_dataset()
        return jsonify({'success': True, 'message': '数据集已保存'})
    except Exception as e:
        logger.error(f"保存失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/stop', methods=['POST'])
def stop_collection():
    """停止并保存"""
    try:
        collector.save_dataset()
        return jsonify({
            'success': True,
            'message': '数据收集已停止',
            'layers_collected': collector.current_job['layers_collected']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    import argparse

    parser = argparse.ArgumentParser(description='自动数据收集服务（现有配置）')
    parser.add_argument('--klipper-host', type=str, default='10.168.1.123',
                       help='Klipper主机地址')
    parser.add_argument('--klipper-port', type=int, default=19255,
                       help='Klipper端口')
    parser.add_argument('--camera-host', type=str, default='10.168.1.129',
                       help='IP摄像头主机地址')
    parser.add_argument('--camera-port', type=int, default=8080,
                       help='IP摄像头端口')
    parser.add_argument('--port', type=int, default=5000,
                       help='HTTP服务端口')
    parser.add_argument('--output', type=str, default='data/collected_photos',
                       help='输出目录')

    args = parser.parse_args()

    # 配置
    klipper_config = {
        'host': args.klipper_host,
        'port': args.klipper_port
    }

    camera_config = {
        'host': args.camera_host,
        'port': args.camera_port
    }

    # 创建收集器
    global collector
    collector = ExistingSetupCollector(
        klipper_config=klipper_config,
        camera_config=camera_config,
        output_dir=args.output
    )

    # 启动服务器
    logger.info("\n" + "="*70)
    logger.info("自动数据收集服务启动")
    logger.info("="*70)
    logger.info(f"  Klipper: {args.klipper_host}:{args.klipper_port}")
    logger.info(f"  摄像头: {args.camera_host}:{args.camera_port}")
    logger.info(f"  HTTP端口: {args.port}")
    logger.info(f"  输出目录: {args.output}")
    logger.info("\nAPI端点:")
    logger.info(f"  POST http://localhost:{args.port}/capture")
    logger.info(f"  GET  http://localhost:{args.port}/status")
    logger.info(f"  POST http://localhost:{args.port}/save")
    logger.info(f"  POST http://localhost:{args.port}/stop")
    logger.info("\n等待Klipper请求...\n")

    app.run(host='0.0.0.0', port=args.port, debug=False)


if __name__ == '__main__':
    main()
