"""
用于模型性能评估的评估指标
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, roc_curve
)


class RegressionMetrics:
    """
    用于质量预测和轨迹校正的回归指标
    """

    @staticmethod
    def compute(predictions: np.ndarray,
                targets: np.ndarray,
                prefix: str = '') -> Dict[str, float]:
        """
        计算回归指标

        参数:
            predictions: 预测值 [n_samples] 或 [n_samples, 1]
            targets: 真实值 [n_samples] 或 [n_samples, 1]
            prefix: 指标名称前缀

        返回:
            包含指标名称和值的字典
        """
        # 如需要则展平
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        if targets.ndim > 1:
            targets = targets.flatten()

        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        # 平均绝对百分比误差 (MAPE)
        mask = np.abs(targets) > 1e-6
        if mask.sum() > 0:
            mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
        else:
            mape = 0.0

        metrics = {
            f'{prefix}mse': float(mse),
            f'{prefix}rmse': float(rmse),
            f'{prefix}mae': float(mae),
            f'{prefix}r2': float(r2),
            f'{prefix}mape': float(mape),
        }

        return metrics

    @staticmethod
    def compute_for_sequence(predictions: np.ndarray,
                            targets: np.ndarray,
                            prefix: str = '') -> Dict[str, float]:
        """
        计算序列预测的指标

        参数:
            predictions: 预测序列 [n_samples, seq_len, n_dims]
            targets: 目标序列 [n_samples, seq_len, n_dims]
            prefix: 指标名称前缀

        返回:
            包含指标名称和值的字典
        """
        # 为每个维度计算指标
        n_dims = predictions.shape[-1]
        all_metrics = {}

        for dim in range(n_dims):
            dim_metrics = RegressionMetrics.compute(
                predictions[:, :, dim],
                targets[:, :, dim],
                prefix=f'{prefix}dim{dim}_'
            )
            all_metrics.update(dim_metrics)

        # 计算所有维度的平均指标
        avg_metrics = RegressionMetrics.compute(
                predictions.flatten(),
                targets.flatten(),
                prefix=f'{prefix}avg_'
            )
        all_metrics.update(avg_metrics)

        return all_metrics


class ClassificationMetrics:
    """
    用于故障检测的分类指标
    """

    @staticmethod
    def compute(predictions: np.ndarray,
                targets: np.ndarray,
                num_classes: int = 4,
                prefix: str = '') -> Dict[str, float]:
        """
        计算分类指标

        参数:
            predictions: 预测类别标签 [n_samples]
            targets: 真实类别标签 [n_samples]
            num_classes: 类别数量
            prefix: 指标名称前缀

        返回:
            包含指标名称和值的字典
        """
        # 准确率
        accuracy = accuracy_score(targets, predictions)

        # 精确率、召回率、F1
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )

        # 每类指标
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )

        metrics = {
            f'{prefix}accuracy': float(accuracy),
            f'{prefix}precision': float(precision),
            f'{prefix}recall': float(recall),
            f'{prefix}f1': float(f1),
        }

        # 添加每类指标
        for i in range(min(num_classes, len(precision_per_class))):
            metrics[f'{prefix}class{i}_precision'] = float(precision_per_class[i])
            metrics[f'{prefix}class{i}_recall'] = float(recall_per_class[i])
            metrics[f'{prefix}class{i}_f1'] = float(f1_per_class[i])

        return metrics

    @staticmethod
    def compute_confusion_matrix(predictions: np.ndarray,
                                targets: np.ndarray,
                                num_classes: int = 4) -> np.ndarray:
        """
        计算混淆矩阵

        参数:
            predictions: 预测类别标签 [n_samples]
            targets: 真实类别标签 [n_samples]
            num_classes: 类别数量

        返回:
            混淆矩阵 [num_classes, num_classes]
        """
        cm = confusion_matrix(targets, predictions, labels=list(range(num_classes)))
        return cm

    @staticmethod
    def compute_with_probabilities(predictions_prob: np.ndarray,
                                  targets: np.ndarray,
                                  prefix: str = '') -> Dict[str, float]:
        """
        使用概率预测计算分类指标

        参数:
            predictions_prob: 预测类别概率 [n_samples, num_classes]
            targets: 真实类别标签 [n_samples]
            prefix: 指标名称前缀

        返回:
            包含指标名称和值的字典
        """
        # 获取预测类别
        predictions = np.argmax(predictions_prob, axis=1)

        # 计算标准指标
        metrics = ClassificationMetrics.compute(predictions, targets, prefix=prefix)

        # 计算AUC (一对多)
        try:
            num_classes = predictions_prob.shape[1]
            if num_classes == 2:
                auc = roc_auc_score(targets, predictions_prob[:, 1])
                metrics[f'{prefix}auc'] = float(auc)
            else:
                # 多类别AUC (一对多)
                auc = roc_auc_score(targets, predictions_prob, multi_class='ovr', average='weighted')
                metrics[f'{prefix}auc'] = float(auc)
        except Exception as e:
            # 如果只存在一个类别，AUC可能会失败
            pass

        return metrics


class TrajectoryMetrics:
    """
    专门用于轨迹校正的指标
    """

    @staticmethod
    def compute_displacement_error(predicted_displacement: np.ndarray,
                                  target_displacement: np.ndarray,
                                  prefix: str = '') -> Dict[str, float]:
        """
        计算位移误差指标

        参数:
            predicted_displacement: 预定位移 [n_samples, 3] (dx, dy, dz)
            target_displacement: 目标位移 [n_samples, 3]
            prefix: 指标名称前缀

        返回:
            包含指标名称和值的字典
        """
        # 计算每个轴的误差
        errors = predicted_displacement - target_displacement

        # 误差大小
        error_magnitude = np.linalg.norm(errors, axis=1)

        metrics = {
            f'{prefix}error_magnitude_mean': float(np.mean(error_magnitude)),
            f'{prefix}error_magnitude_std': float(np.std(error_magnitude)),
            f'{prefix}error_magnitude_max': float(np.max(error_magnitude)),
            f'{prefix}error_magnitude_median': float(np.median(error_magnitude)),
        }

        # 每轴指标
        for i, axis in enumerate(['x', 'y', 'z']):
            axis_error = errors[:, i]
            metrics[f'{prefix}error_{axis}_mean'] = float(np.mean(np.abs(axis_error)))
            metrics[f'{prefix}error_{axis}_rmse'] = float(np.sqrt(np.mean(axis_error ** 2)))

        return metrics

    @staticmethod
    def compute_improvement_ratio(original_error: np.ndarray,
                                 corrected_error: np.ndarray) -> Dict[str, float]:
        """
        计算轨迹校正后的改进比率

        参数:
            original_error: 校正前的误差 [n_samples]
            corrected_error: 校正后的误差 [n_samples]

        返回:
            包含改进指标的字典
        """
        original_magnitude = np.linalg.norm(original_error, axis=1) if original_error.ndim > 1 else np.abs(original_error)
        corrected_magnitude = np.linalg.norm(corrected_error, axis=1) if corrected_error.ndim > 1 else np.abs(corrected_error)

        # 改进比率
        improvement = (original_magnitude - corrected_magnitude) / (original_magnitude + 1e-6)

        # 百分比改进
        improvement_percentage = improvement * 100

        # 改进的样本
        improved_samples = (improvement > 0).sum()
        total_samples = len(improvement)

        metrics = {
            'improvement_ratio_mean': float(np.mean(improvement)),
            'improvement_ratio_median': float(np.median(improvement)),
            'improvement_percentage_mean': float(np.mean(improvement_percentage)),
            'improved_samples': int(improved_samples),
            'total_samples': int(total_samples),
            'improvement_rate': float(improved_samples / total_samples),
        }

        return metrics


class QualityMetrics:
    """
    专门用于质量预测的指标
    """

    @staticmethod
    def compute_rul_metrics(predictions: np.ndarray,
                           targets: np.ndarray,
                           threshold: float = 100.0,
                           prefix: str = 'rul_') -> Dict[str, float]:
        """
        计算RUL专用指标

        参数:
            predictions: 预测的RUL值 [n_samples]
            targets: 真实的RUL值 [n_samples]
            threshold: 早期警告阈值（秒）
            prefix: 指标名称前缀

        返回:
            包含RUL指标的字典
        """
        # 标准回归指标
        metrics = RegressionMetrics.compute(predictions, targets, prefix=prefix)

        # 早期警告准确率
        pred_warning = predictions < threshold
        true_warning = targets < threshold

        warning_accuracy = accuracy_score(true_warning, pred_warning)

        metrics[f'{prefix}warning_accuracy'] = float(warning_accuracy)

        return metrics

    @staticmethod
    def compute_quality_score_metrics(predictions: np.ndarray,
                                     targets: np.ndarray,
                                     threshold: float = 0.5,
                                     prefix: str = 'quality_') -> Dict[str, float]:
        """
        计算质量分数专用指标

        参数:
            predictions: 预测质量分数 [n_samples]
            targets: 真实质量分数 [n_samples]
            threshold: 良好质量阈值
            prefix: 指标名称前缀

        返回:
            包含质量分数指标的字典
        """
        # 标准回归指标
        metrics = RegressionMetrics.compute(predictions, targets, prefix=prefix)

        # 基于阈值的二分类
        pred_good = predictions > threshold
        true_good = targets > threshold

        # 准确率
        accuracy = accuracy_score(true_good, pred_good)

        # 精确率、召回率、F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_good, pred_good, average='binary', zero_division=0
        )

        metrics[f'{prefix}binary_accuracy'] = float(accuracy)
        metrics[f'{prefix}binary_precision'] = float(precision)
        metrics[f'{prefix}binary_recall'] = float(recall)
        metrics[f'{prefix}binary_f1'] = float(f1)

        return metrics


class UnifiedMetrics:
    """
    用于所有任务的统一指标计算
    """

    @staticmethod
    def compute_all(predictions: Dict[str, np.ndarray],
                   targets: Dict[str, np.ndarray],
                   num_fault_classes: int = 4) -> Dict[str, float]:
        """
        计算所有任务的指标

        参数:
            predictions: 预测字典
            targets: 目标字典
            num_fault_classes: 故障类别数量

        返回:
            包含所有指标的字典
        """
        all_metrics = {}

        # 质量预测指标
        if 'rul' in predictions and 'rul' in targets:
            rul_metrics = QualityMetrics.compute_rul_metrics(
                predictions['rul'], targets['rul']
            )
            all_metrics.update(rul_metrics)

        if 'temperature' in predictions and 'temperature' in targets:
            temp_metrics = RegressionMetrics.compute(
                predictions['temperature'], targets['temperature'],
                prefix='temperature_'
            )
            all_metrics.update(temp_metrics)

        if 'vibration_x' in predictions and 'vibration_x' in targets:
            vib_x_metrics = RegressionMetrics.compute(
                predictions['vibration_x'], targets['vibration_x'],
                prefix='vibration_x_'
            )
            all_metrics.update(vib_x_metrics)

        if 'vibration_y' in predictions and 'vibration_y' in targets:
            vib_y_metrics = RegressionMetrics.compute(
                predictions['vibration_y'], targets['vibration_y'],
                prefix='vibration_y_'
            )
            all_metrics.update(vib_y_metrics)

        if 'quality_score' in predictions and 'quality_score' in targets:
            quality_metrics = QualityMetrics.compute_quality_score_metrics(
                predictions['quality_score'], targets['quality_score']
            )
            all_metrics.update(quality_metrics)

        # 故障分类指标
        if 'fault_pred' in predictions and 'fault_label' in targets:
            fault_metrics = ClassificationMetrics.compute(
                predictions['fault_pred'],
                targets['fault_label'],
                num_classes=num_fault_classes,
                prefix='fault_'
            )
            all_metrics.update(fault_metrics)

        # 轨迹校正指标
        if ('displacement_x' in predictions and 'displacement_x' in targets and
            'displacement_y' in predictions and 'displacement_y' in targets and
            'displacement_z' in predictions and 'displacement_z' in targets):

            pred_disp = np.stack([
                predictions['displacement_x'].flatten(),
                predictions['displacement_y'].flatten(),
                predictions['displacement_z'].flatten()
            ], axis=1)

            target_disp = np.stack([
                targets['displacement_x'].flatten(),
                targets['displacement_y'].flatten(),
                targets['displacement_z'].flatten()
            ], axis=1)

            traj_metrics = TrajectoryMetrics.compute_displacement_error(
                pred_disp, target_disp, prefix='trajectory_'
            )
            all_metrics.update(traj_metrics)

        return all_metrics

    @staticmethod
    def format_metrics(metrics: Dict[str, float],
                      precision: int = 4) -> str:
        """
        格式化指标以便打印

        参数:
            metrics: 指标字典
            precision: 小数位数

        返回:
            格式化字符串
        """
        lines = ["=" * 80, "评估指标", "=" * 80]

        # 按前缀分组指标
        groups = {}
        for key, value in metrics.items():
            prefix = key.split('_')[0]
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append((key, value))

        # 打印每个组
        for group_name in sorted(groups.keys()):
            lines.append(f"\n{group_name.upper()}:")
            for key, value in sorted(groups[group_name]):
                lines.append(f"  {key}: {value:.{precision}f}")

        lines.append("=" * 80)

        return "\n".join(lines)


def compute_model_metrics(model: torch.nn.Module,
                         data_loader: torch.utils.data.DataLoader,
                         device: str = 'cpu',
                         num_fault_classes: int = 4) -> Dict[str, float]:
    """
    在数据集上计算模型的指标

    参数:
        model: PyTorch模型
        data_loader: 数据加载器
        device: 运行设备
        num_fault_classes: 故障类别数量

    返回:
        包含所有指标的字典
    """
    model.eval()
    model.to(device)

    all_predictions = {}
    all_targets = {}

    with torch.no_grad():
        for batch in data_loader:
            # 移至设备
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # 前向传播
            outputs = model(inputs['features'])

            # 收集预测
            for key, value in outputs.items():
                if key not in all_predictions:
                    all_predictions[key] = []
                all_predictions[key].append(value.cpu().numpy())

            # 收集目标
            for key in ['rul', 'temperature', 'vibration_x', 'vibration_y',
                      'quality_score', 'fault_label', 'displacement_x',
                      'displacement_y', 'displacement_z']:
                if key in batch:
                    if key not in all_targets:
                        all_targets[key] = []
                    all_targets[key].append(batch[key].cpu().numpy() if
                                          isinstance(batch[key], torch.Tensor) else batch[key])

    # 连接所有批次
    for key in all_predictions:
        all_predictions[key] = np.concatenate(all_predictions[key], axis=0)

    for key in all_targets:
        all_targets[key] = np.concatenate(all_targets[key], axis=0)

    # 计算指标
    metrics = UnifiedMetrics.compute_all(all_predictions, all_targets, num_fault_classes)

    return metrics