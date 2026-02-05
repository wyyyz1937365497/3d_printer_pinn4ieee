"""
视觉测量模块 - 从打印件照片提取轨迹误差

功能：
1. 图像预处理（去噪、二值化）
2. 轮廓提取（OpenCV）
3. 轮廓匹配（ICP算法）
4. 误差计算

使用方法：
    processor = VisionProcessor()
    measured_contour = processor.extract_contour(image_path)
    ideal_contour = processor.load_stl_contour(stl_path, layer_z)
    aligned, transform = processor.align_contours(measured, ideal)
    errors = processor.compute_errors(aligned, ideal)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import json


class VisionProcessor:
    """
    视觉测量处理器
    """

    def __init__(self,
                 image_resolution: Tuple[int, int] = (1920, 1080),
                 pixel_to_mm: float = 0.05):  # 需要标定
        """
        Args:
            image_resolution: 图像分辨率
            pixel_to_mm: 像素到毫米的转换比例
        """
        self.image_resolution = image_resolution
        self.pixel_to_mm = pixel_to_mm

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        图像预处理

        步骤：
        1. 读取图像
        2. 转灰度
        3. 降噪
        4. 增强对比度

        Args:
            image_path: 图像路径

        Returns:
            processed_image: 预处理后的图像
        """
        # 读取
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 转灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 降噪（双边滤波）
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # CLAHE增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)

        return enhanced

    def extract_contour(self,
                       image: np.ndarray,
                       min_area: int = 1000,
                       simplify_epsilon: float = 0.001) -> np.ndarray:
        """
        提取打印轮廓

        Args:
            image: 预处理后的图像
            min_area: 最小轮廓面积
            simplify_epsilon: 轮廓简化参数

        Returns:
            contour: [n, 2] - 轮廓点坐标（像素）
        """
        # 二值化（自适应阈值）
        binary = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        # 形态学操作（去噪）
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if len(contours) == 0:
            print("[WARNING] 未检测到轮廓")
            return np.array([])

        # 选择最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)

        # 过滤小面积
        if cv2.contourArea(largest_contour) < min_area:
            print(f"[WARNING] 轮廓面积过小: {cv2.contourArea(largest_contour)}")
            return np.array([])

        # 简化轮廓（减少点数）
        perimeter = cv2.arcLength(largest_contour, True)
        simplified = cv2.approxPolyDP(
            largest_contour,
            simplify_epsilon * perimeter,
            True
        )

        # 转换格式
        contour = simplified[:, 0, :].astype(np.float32)

        return contour

    def load_stl_contour(self,
                        stl_path: str,
                        layer_z: float,
                        resolution: float = 0.1) -> np.ndarray:
        """
        从STL生成指定层的理想轮廓

        Args:
            stl_path: STL文件路径
            layer_z: 层高度
            resolution: 采样分辨率（mm）

        Returns:
            contour: [n, 2] - 理想轮廓点坐标（mm）
        """
        # TODO: 集成STL切片库
        # 可以使用：
        # - PySLA
        # - Cura命令行
        # - 自定义切片算法

        # 占位实现
        print(f"[TODO] 从STL生成轮廓: {stl_path}, layer_z={layer_z}")

        # 临时返回空数组
        return np.array([])

    def pixel_to_mm_scale(self, contour_pixels: np.ndarray) -> np.ndarray:
        """
        像素坐标转换为毫米坐标

        Args:
            contour_pixels: [n, 2] - 像素坐标

        Returns:
            contour_mm: [n, 2] - 毫米坐标
        """
        return contour_pixels * self.pixel_to_mm

    def align_contours_icp(self,
                          source: np.ndarray,
                          target: np.ndarray,
                          max_iterations: int = 100,
                          tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        使用ICP算法对齐两个轮廓

        Args:
            source: 源轮廓 [n, 2]
            target: 目标轮廓 [m, 2]
            max_iterations: 最大迭代次数
            tolerance: 收敛容差

        Returns:
            aligned_source: 对齐后的源轮廓
            transform_matrix: 3x3变换矩阵
            error: 最终对齐误差
        """
        if len(source) == 0 or len(target) == 0:
            return source, np.eye(3), 0.0

        # 初始平移对齐（中心对齐）
        source_center = np.mean(source, axis=0)
        target_center = np.mean(target, axis=0)
        initial_translation = target_center - source_center

        source_aligned = source + initial_translation

        # ICP迭代
        transform = np.eye(3)
        prev_error = float('inf')

        for i in range(max_iterations):
            # 寻找最近点
            from scipy.spatial import cKDTree
            tree = cKDTree(target)
            distances, indices = tree.query(source_aligned)

            # 计算当前误差
            error = np.mean(distances)

            # 检查收敛
            if abs(prev_error - error) < tolerance:
                break

            prev_error = error

            # 计算变换（仅平移，简化版本）
            # 完整版本应该包含旋转和缩放
            matched_target = target[indices]

            # 计算平移
            translation = np.mean(matched_target - source_aligned, axis=0)

            # 应用变换
            source_aligned = source_aligned + translation

            # 更新变换矩阵
            T = np.eye(3)
            T[:2, 2] = translation
            transform = T @ transform

        return source_aligned, transform, error

    def compute_errors(self,
                      measured: np.ndarray,
                      ideal: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        计算轮廓误差

        Args:
            measured: 测量轮廓 [n, 2]
            ideal: 理想轮廓 [m, 2]

        Returns:
            errors: [n, 2] - 每个测量点的误差向量
            stats: 误差统计
        """
        if len(measured) == 0 or len(ideal) == 0:
            return np.array([]), {}

        # 对每个测量点找最近理想点
        from scipy.spatial import cKDTree
        tree = cKDTree(ideal)
        distances, indices = tree.query(measured)

        # 计算误差向量
        errors = np.zeros_like(measured)
        for i, (measured_point, ideal_idx) in enumerate(zip(measured, indices)):
            ideal_point = ideal[ideal_idx]
            errors[i] = ideal_point - measured_point

        # 统计
        error_mag = np.sqrt(errors[:, 0]**2 + errors[:, 1]**2)

        stats = {
            'n_points': len(errors),
            'mean_um': float(np.mean(error_mag) * 1000),
            'std_um': float(np.std(error_mag) * 1000),
            'rms_um': float(np.sqrt(np.mean(error_mag**2)) * 1000),
            'max_um': float(np.max(error_mag) * 1000)
        }

        return errors, stats

    def process_layer_image(self,
                           image_path: str,
                           ideal_contour: np.ndarray,
                           output_dir: Optional[str] = None) -> Dict:
        """
        处理单层图像的完整流程

        Args:
            image_path: 图像路径
            ideal_contour: 理想轮廓
            output_dir: 输出目录（可选）

        Returns:
            result: 处理结果字典
        """
        # 1. 预处理
        img_processed = self.preprocess_image(image_path)

        # 2. 提取轮廓
        contour_pixels = self.extract_contour(img_processed)

        if len(contour_pixels) == 0:
            return {'success': False, 'error': '未检测到轮廓'}

        # 3. 转换单位
        contour_mm = self.pixel_to_mm_scale(contour_pixels)

        # 4. 对齐
        aligned, transform, error = self.align_contours_icp(contour_mm, ideal_contour)

        # 5. 计算误差
        errors, stats = self.compute_errors(aligned, ideal_contour)

        # 6. 可视化（如果需要）
        if output_dir:
            self.visualize_result(
                img_processed, contour_mm, ideal_contour, errors,
                Path(output_dir) / Path(image_path).name
            )

        return {
            'success': True,
            'contour': aligned,
            'errors': errors,
            'transform': transform,
            'stats': stats
        }

    def visualize_result(self,
                         image: np.ndarray,
                         measured: np.ndarray,
                         ideal: np.ndarray,
                         errors: np.ndarray,
                         output_path: str):
        """
        可视化处理结果
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # 左图：原始图像+轮廓
        axes[0].imshow(image, cmap='gray')
        if len(measured) > 0:
            axes[0].plot(measured[:, 0], measured[:, 1], 'r-', linewidth=1, label='Measured')
        if len(ideal) > 0:
            axes[0].plot(ideal[:, 0], ideal[:, 1], 'g--', linewidth=1, label='Ideal')
        axes[0].set_title('Contour Extraction')
        axes[0].legend()
        axes[0].axis('equal')

        # 右图：误差分布
        if len(errors) > 0:
            error_mag = np.sqrt(errors[:, 0]**2 + errors[:, 1]**2) * 1000  # um
            axes[1].hist(error_mag, bins=50, alpha=0.7, edgecolor='black')
            axes[1].axvline(np.mean(error_mag), color='r', linestyle='--',
                           label=f"Mean: {np.mean(error_mag):.2f} um")
            axes[1].axvline(np.std(error_mag), color='g', linestyle='--',
                           label=f"Std: {np.std(error_mag):.2f} um")
            axes[1].set_xlabel('Error Magnitude (um)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Error Distribution')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def calibrate_pixel_scale(self,
                             reference_object_size_mm: float,
                             reference_image_path: str) -> float:
        """
        标定像素到毫米的比例

        Args:
            reference_object_size_mm: 参考物体实际尺寸（mm）
            reference_image_path: 参考物体图像

        Returns:
            pixel_to_mm: 像素到毫米的比例
        """
        # 提取参考物体
        img = self.preprocess_image(reference_image_path)
        contour = self.extract_contour(img)

        if len(contour) == 0:
            raise ValueError("无法提取参考物体")

        # 计算像素尺寸（用边界框）
        x_min, y_min = np.min(contour, axis=0)
        x_max, y_max = np.max(contour, axis=0)
        pixel_size = max(x_max - x_min, y_max - y_min)

        # 计算比例
        self.pixel_to_mm = reference_object_size_mm / pixel_size

        print(f"标定结果: 1 pixel = {self.pixel_to_mm:.4f} mm")

        return self.pixel_to_mm


if __name__ == '__main__':
    # 测试代码
    print("Vision Processor Test")
    print("="*70)

    processor = VisionProcessor()

    # 测试图像处理（如果有测试图像）
    test_image = "test_print_layer.png"

    if Path(test_image).exists():
        result = processor.process_layer_image(
            test_image,
            ideal_contour=np.random.rand(100, 2) * 100,  # 临时理想轮廓
            output_dir="test_output"
        )

        if result['success']:
            print(f"\n处理成功:")
            print(f"  检测到 {result['stats']['n_points']} 个点")
            print(f"  RMS误差: {result['stats']['rms_um']:.2f} um")
            print(f"  最大误差: {result['stats']['max_um']:.2f} um")
    else:
        print(f"\n测试图像不存在: {test_image}")
        print("请提供打印层照片进行测试")
