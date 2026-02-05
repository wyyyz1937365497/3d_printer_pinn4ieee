"""
实验数据收集方案 - 基于视觉测量的轨迹误差

流程：
1. 逐层打印测试件
2. 每层完成后拍摄照片
3. 使用OpenCV提取轮廓
4. 与STL理想轮廓对比
5. 生成训练数据

优势：
- 真实数据，包含实际打印机的所有误差源
- 数据标注自动完成（无需手动测量）
- 可以收集多种打印条件下的数据
"""

import numpy as np
import cv2
import json
from pathlib import Path
from typing import List, Dict, Tuple
import subprocess


class RealPrintDataCollector:
    """
    真实打印数据收集器

    需要硬件：
    - 3D打印机（Ender 3 V2或其他）
    - 相机（固定在打印仓上方）
    - 良好的照明
    """

    def __init__(self, camera_id: int = 0):
        """
        Args:
            camera_id: 摄像头ID
        """
        self.camera_id = camera_id
        self.camera = None

        # 打印参数
        self.print_settings = {
            'nozzle_temp': 200,
            'bed_temp': 60,
            'print_speed': 50,
            'layer_height': 0.2
        }

    def setup_camera(self):
        """初始化摄像头"""
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            raise RuntimeError(f"无法打开摄像头 {self.camera_id}")

        # 设置分辨率
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        print("摄像头初始化成功")

    def capture_layer(self, layer_num: int, output_dir: str) -> str:
        """
        拍摄当前层

        Args:
            layer_num: 层数
            output_dir: 输出目录

        Returns:
            保存的图像路径
        """
        if self.camera is None:
            self.setup_camera()

        # 拍照
        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("无法读取摄像头")

        # 保存
        output_path = Path(output_dir) / f"layer_{layer_num:04d}.png"
        cv2.imwrite(str(output_path), frame)

        print(f"  拍摄层 {layer_num}: {output_path}")
        return str(output_path)

    def extract_contour(self, image_path: str) -> np.ndarray:
        """
        从图像中提取打印轮廓

        Args:
            image_path: 图像路径

        Returns:
            contour: [n, 2] - 轮廓点坐标
        """
        # 读取图像
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 二值化
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            print(f"  [WARNING] 未检测到轮廓: {image_path}")
            return np.array([])

        # 选择最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)

        # 简化轮廓（减少点数）
        epsilon = 0.001 * cv2.arcLength(largest_contour, True)
        simplified = cv2.approxPolyDP(largest_contour, epsilon, True)

        # 转换为numpy数组
        contour = simplified[:, 0, :].astype(np.float32)

        return contour

    def load_ideal_contour(self, stl_file: str, layer_height: float) -> np.ndarray:
        """
        从STL文件加载理想轮廓

        Args:
            stl_file: STL文件路径
            layer_height: 层高

        Returns:
            contour: [n, 2] - 理想轮廓点坐标
        """
        # 使用slicer获取该层的轮廓
        # 这里简化处理，实际需要使用slicing软件
        # 可以使用Cura的命令行接口或Python库

        # 占位：直接返回一个示例轮廓
        # 实际实现需要集成slicer

        print(f"  [TODO] 从STL生成理想轮廓: {stl_file}")
        return np.array([])

    def align_contours(self, measured: np.ndarray, ideal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对齐测量轮廓和理想轮廓

        使用ICP（Iterative Closest Point）算法

        Returns:
            aligned_measured, transform_matrix
        """
        # 简化版本：仅平移对齐
        # 完整版本应该使用ICP

        if len(measured) == 0 or len(ideal) == 0:
            return measured, np.eye(3)

        # 计算中心点偏移
        center_measured = np.mean(measured, axis=0)
        center_ideal = np.mean(ideal, axis=0)

        translation = center_ideal - center_measured

        # 应用平移
        aligned = measured + translation

        return aligned, np.array([])

    def compute_error(self, measured: np.ndarray, ideal: np.ndarray) -> np.ndarray:
        """
        计算轮廓误差

        Args:
            measured: 测量轮廓 [n, 2]
            ideal: 理想轮廓 [m, 2]

        Returns:
            errors: [n, 2] - 每个测量点的误差向量
        """
        if len(measured) == 0 or len(ideal) == 0:
            return np.array([])

        # 对每个测量点，找到最近理想点
        from scipy.spatial import cKDTree

        tree = cKDTree(ideal)
        distances, indices = tree.query(measured)

        # 计算误差向量
        errors = np.zeros_like(measured)
        for i, (measured_point, ideal_idx) in enumerate(zip(measured, indices)):
            ideal_point = ideal[ideal_idx]
            errors[i] = ideal_point - measured_point

        return errors

    def collect_print_run(self, gcode_file: str, stl_file: str,
                         output_dir: str) -> str:
        """
        完整的打印数据收集流程

        Args:
            gcode_file: G-code文件
            stl_file: STL文件
            output_dir: 输出目录

        Returns:
            数据集路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 子目录
        images_dir = output_path / "images"
        images_dir.mkdir(exist_ok=True)

        # 1. 开始打印（使用打印机控制软件）
        print("\n1. 开始打印...")
        print(f"  G-code: {gcode_file}")
        print(f"  等待打印机就绪...")

        # 这里需要集成打印机控制
        # 可以使用OctoPrint API或直接串口控制
        # 暂时手动操作

        input("  [手动] 请开始打印，按Enter继续...")

        # 2. 逐层拍摄
        print("\n2. 逐层拍摄...")

        layer_num = 0
        while True:
            cont = input(f"  层 {layer_num} 完成了吗？(y/n/q): ")
            if cont.lower() == 'q':
                break
            if cont.lower() != 'y':
                continue

            # 拍摄
            image_path = self.capture_layer(layer_num, str(images_dir))
            layer_num += 1

        # 3. 处理图像
        print("\n3. 提取轮廓...")
        layer_errors = {}

        for image_file in sorted(images_dir.glob("layer_*.png")):
            layer_idx = int(image_file.stem.split('_')[1])

            # 提取测量轮廓
            measured = self.extract_contour(str(image_file))

            # 加载理想轮廓
            ideal = self.load_ideal_contour(stl_file, self.print_settings['layer_height'])

            # 对齐
            aligned, _ = self.align_contours(measured, ideal)

            # 计算误差
            errors = self.compute_error(aligned, ideal)

            if len(errors) > 0:
                layer_errors[layer_idx] = errors

        # 4. 保存数据
        print("\n4. 保存数据...")

        data_file = output_path / "print_errors.npz"
        np.savez_compressed(
            data_file,
            layer_errors=layer_errors,
            print_settings=self.print_settings
        )

        print(f"  数据已保存: {data_file}")

        # 5. 生成统计报告
        print("\n5. 生成报告...")

        all_errors = np.vstack([err for err in layer_errors.values()])
        error_mag = np.sqrt(all_errors[:, 0]**2 + all_errors[:, 1]**2)

        report = {
            'n_layers': len(layer_errors),
            'n_points': len(all_errors),
            'error_stats': {
                'mean_um': float(np.mean(error_mag) * 1000),
                'std_um': float(np.std(error_mag) * 1000),
                'rms_um': float(np.sqrt(np.mean(error_mag**2)) * 1000),
                'max_um': float(np.max(error_mag) * 1000)
            }
        }

        report_file = output_path / "report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n报告:")
        print(f"  层数: {report['n_layers']}")
        print(f"  总点数: {report['n_points']}")
        print(f"  RMS误差: {report['error_stats']['rms_um']:.2f} um")

        return str(data_file)


def create_synthetic_training_data(real_data_file: str,
                                  simulation_data_dir: str,
                                  output_file: str):
    """
    混合真实数据和仿真数据创建训练集

    Args:
        real_data_file: 真实打印数据
        simulation_data_dir: 仿真数据目录
        output_file: 输出文件
    """
    print("创建混合训练数据集...")

    # 1. 加载真实数据
    real_data = np.load(real_data_file, allow_pickle=True)
    layer_errors = real_data['layer_errors'].item()

    # 2. 加载仿真数据
    import glob
    sim_files = glob.glob(f"{simulation_data_dir}/*.mat")

    # 3. 合并
    all_data = []

    # 添加真实数据
    for layer_idx, errors in layer_errors.items():
        # 标记为真实数据
        all_data.append({
            'errors': errors,
            'source': 'real',
            'layer': layer_idx
        })

    # 添加仿真数据
    for sim_file in sim_files:
        # 加载并处理仿真数据
        pass

    # 4. 保存
    np.savez_compressed(output_file, data=all_data)
    print(f"  保存到: {output_file}")


if __name__ == '__main__':
    print("实验数据收集系统")
    print("="*70)
    print("\n硬件需求:")
    print("  - 3D打印机")
    print("  - 摄像头（固定在打印仓上方）")
    print("  - 良好照明")
    print("\n流程:")
    print("  1. 开始打印测试件")
    print("  2. 每层完成后拍摄")
    print("  3. 自动提取轮廓并计算误差")
    print("  4. 生成训练数据")
    print("\n准备就绪后，运行:")
    print("  python collect_real_print_data.py")

    # 示例使用
    collector = RealPrintDataCollector()

    # 测试摄像头
    # collector.setup_camera()
    # test_image = collector.capture_layer(0, "test_output")
