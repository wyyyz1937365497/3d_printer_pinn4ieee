"""
视觉处理器 - 用于从照片中提取打印轮廓和计算误差

核心功能：
1. 图像预处理（去噪、增强对比度）
2. 轮廓提取（自适应阈值、形态学操作）
3. 轮廓对齐（ICP算法）
4. 误差计算（点对点距离）
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
from scipy.spatial import cKDTree


class VisionProcessor:
    """3D打印视觉处理器"""

    def __init__(self, pixel_to_mm_ratio: float = 1.0):
        """
        Args:
            pixel_to_mm_ratio: 像素到毫米的转换比例（需要标定）
        """
        self.pixel_to_mm_ratio = pixel_to_mm_ratio

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        图像预处理流程

        Args:
            image_path: 图像文件路径

        Returns:
            预处理后的灰度图像
        """
        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 双边滤波去噪（保持边缘）
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # CLAHE对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)

        return enhanced

    def extract_contour(self,
                       image: np.ndarray,
                       min_area: int = 1000,
                       simplify_epsilon: float = 0.001) -> np.ndarray:
        """
        从图像中提取打印轮廓

        Args:
            image: 预处理后的灰度图像
            min_area: 最小轮廓面积（像素）
            simplify_epsilon: 轮廓简化精度（0-1）

        Returns:
            轮廓点集，形状为(N, 2)的numpy数组
        """
        # 自适应阈值二值化
        binary = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # 形态学操作去噪
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 查找轮廓
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if len(contours) == 0:
            return np.array([])

        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)

        # 检查面积
        area = cv2.contourArea(largest_contour)
        if area < min_area:
            return np.array([])

        # 轮廓简化（Ramer-Douglas-Peucker算法）
        perimeter = cv2.arcLength(largest_contour, True)
        simplified = cv2.approxPolyDP(
            largest_contour,
            simplify_epsilon * perimeter,
            True
        )

        # 返回点集 (N, 2)
        return simplified[:, 0, :].astype(np.float32)

    def pixel_to_mm_scale(self, contour: np.ndarray) -> np.ndarray:
        """
        将像素坐标转换为毫米坐标

        Args:
            contour: 像素坐标轮廓

        Returns:
            毫米坐标轮廓
        """
        return contour * self.pixel_to_mm_ratio

    def align_contours_icp(self,
                          source: np.ndarray,
                          target: np.ndarray,
                          max_iterations: int = 100,
                          tolerance: float = 1e-6) -> Tuple[np.ndarray, Dict, float]:
        """
        使用ICP算法对齐两个轮廓（仅平移）

        Args:
            source: 源轮廓
            target: 目标轮廓
            max_iterations: 最大迭代次数
            tolerance: 收敛阈值

        Returns:
            aligned: 对齐后的源轮廓
            transform: 变换参数 {'translation': np.array([dx, dy])}
            error: 最终对齐误差
        """
        if len(source) == 0 or len(target) == 0:
            return source, {}, float('inf')

        # 初始平移对齐（质心对齐）
        source_center = np.mean(source, axis=0)
        target_center = np.mean(target, axis=0)

        source_aligned = source + (target_center - source_center)

        # ICP迭代
        prev_error = float('inf')
        transform = {'translation': target_center - source_center}

        for i in range(max_iterations):
            # 找到最近点
            tree = cKDTree(target)
            distances, indices = tree.query(source_aligned)

            # 计算匹配点
            matched_target = target[indices]

            # 计算平移
            translation = np.mean(matched_target - source_aligned, axis=0)
            source_aligned = source_aligned + translation
            transform['translation'] += translation

            # 计算误差
            error = np.mean(distances)

            # 检查收敛
            if abs(prev_error - error) < tolerance:
                break

            prev_error = error

        return source_aligned, transform, error

    def compute_errors(self,
                      measured: np.ndarray,
                      ideal: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        计算测量轮廓与理想轮廓之间的误差

        Args:
            measured: 测量轮廓
            ideal: 理想轮廓

        Returns:
            errors: 误差向量 (N, 2)
            stats: 统计信息字典
        """
        if len(measured) == 0 or len(ideal) == 0:
            return np.array([]), {
                'n_points': 0,
                'mean_um': 0.0,
                'std_um': 0.0,
                'rms_um': 0.0,
                'max_um': 0.0
            }

        # 为每个测量点找到理想轮廓上的最近点
        tree = cKDTree(ideal)
        distances, indices = tree.query(measured)

        # 计算误差向量
        errors = np.zeros_like(measured)
        for i, (measured_point, ideal_idx) in enumerate(zip(measured, indices)):
            errors[i] = ideal[ideal_idx] - measured_point

        # 转换为微米
        error_mag = np.sqrt(errors[:, 0]**2 + errors[:, 1]**2) * 1000

        stats = {
            'n_points': len(errors),
            'mean_um': float(np.mean(error_mag)),
            'std_um': float(np.std(error_mag)),
            'rms_um': float(np.sqrt(np.mean(error_mag**2))),
            'max_um': float(np.max(error_mag))
        }

        return errors, stats

    def calibrate_pixel_to_mm(self,
                              calibration_image_path: str,
                              known_width_mm: float) -> float:
        """
        使用已知尺寸的标定物计算像素到毫米的比例

        Args:
            calibration_image_path: 标定物照片
            known_width_mm: 标定物实际宽度（毫米）

        Returns:
            pixel_to_mm_ratio: 像素到毫米的比例
        """
        # 预处理图像
        processed = self.preprocess_image(calibration_image_path)

        # 提取轮廓
        contours, _ = cv2.findContours(
            processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            raise ValueError("未检测到标定物轮廓")

        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)

        # 计算边界框
        x, y, w, h = cv2.boundingRect(largest_contour)

        # 假设宽度对应已知宽度
        pixel_to_mm_ratio = known_width_mm / w

        print(f"标定结果:")
        print(f"  标定物宽度: {known_width_mm} mm")
        print(f"  检测宽度: {w} pixels")
        print(f"  比例: {pixel_to_mm_ratio:.6f} mm/pixel")

        self.pixel_to_mm_ratio = pixel_to_mm_ratio

        return pixel_to_mm_ratio

    def visualize_comparison(self,
                            measured: np.ndarray,
                            ideal: np.ndarray,
                            output_path: str,
                            figsize: Tuple[int, int] = (12, 10)):
        """
        可视化测量轮廓与理想轮廓的对比

        Args:
            measured: 测量轮廓
            ideal: 理想轮廓
            output_path: 输出图像路径
            figsize: 图像大小
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 轮廓对比
        axes[0, 0].plot(ideal[:, 0], ideal[:, 1], 'b-', linewidth=1,
                       label='Ideal', alpha=0.7)
        axes[0, 0].plot(measured[:, 0], measured[:, 1], 'r-', linewidth=1,
                       label='Measured', alpha=0.7)
        axes[0, 0].set_xlabel('X (mm)', fontsize=12)
        axes[0, 0].set_ylabel('Y (mm)', fontsize=12)
        axes[0, 0].set_title('Contour Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].axis('equal')
        axes[0, 0].grid(True, alpha=0.3)

        # 误差向量图
        if len(measured) > 0 and len(ideal) > 0:
            errors, stats = self.compute_errors(measured, ideal)

            # 绘制误差向量（每10个点绘制一个）
            step = max(1, len(measured) // 50)
            for i in range(0, len(measured), step):
                axes[0, 1].arrow(
                    measured[i, 0], measured[i, 1],
                    errors[i, 0], errors[i, 1],
                    color='red', alpha=0.5, head_width=0.5
                )

            axes[0, 1].plot(ideal[:, 0], ideal[:, 1], 'b-', linewidth=1,
                           label='Ideal', alpha=0.5)
            axes[0, 1].set_xlabel('X (mm)', fontsize=12)
            axes[0, 1].set_ylabel('Y (mm)', fontsize=12)
            axes[0, 1].set_title('Error Vectors', fontsize=14, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].axis('equal')
            axes[0, 1].grid(True, alpha=0.3)

            # 误差分布直方图
            error_mag = np.sqrt(errors[:, 0]**2 + errors[:, 1]**2) * 1000
            axes[1, 0].hist(error_mag, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(stats['mean_um'], color='r', linestyle='--',
                              linewidth=2, label=f"Mean: {stats['mean_um']:.2f} um")
            axes[1, 0].axvline(stats['rms_um'], color='g', linestyle='--',
                              linewidth=2, label=f"RMS: {stats['rms_um']:.2f} um")
            axes[1, 0].set_xlabel('Error Magnitude (um)', fontsize=12)
            axes[1, 0].set_ylabel('Frequency', fontsize=12)
            axes[1, 0].set_title('Error Distribution', fontsize=14, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')

            # 统计信息
            info_text = f"""
            Statistics:
            Points: {stats['n_points']}
            Mean Error: {stats['mean_um']:.2f} um
            Std Error: {stats['std_um']:.2f} um
            RMS Error: {stats['rms_um']:.2f} um
            Max Error: {stats['max_um']:.2f} um
            """
            axes[1, 1].text(0.1, 0.5, info_text, fontsize=12,
                           verticalalignment='center')
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"可视化结果已保存: {output_path}")
