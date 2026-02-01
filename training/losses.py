"""
Loss functions for multi-task learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining quality prediction, fault classification,
    trajectory correction, and physics constraints
    """

    def __init__(self,
                 lambda_quality: float = 1.0,
                 lambda_fault: float = 1.0,
                 lambda_trajectory: float = 1.0,
                 lambda_physics: float = 0.1):
        """
        Initialize multi-task loss

        Args:
            lambda_quality: Weight for quality prediction loss
            lambda_fault: Weight for fault classification loss
            lambda_trajectory: Weight for trajectory correction loss
            lambda_physics: Weight for physics constraint loss
        """
        super().__init__()

        self.lambda_quality = lambda_quality
        self.lambda_fault = lambda_fault
        self.lambda_trajectory = lambda_trajectory
        self.lambda_physics = lambda_physics

        # Individual loss functions
        self.mse_loss = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

    def quality_loss(self, predictions: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute quality prediction loss

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Quality loss
        """
        losses = []

        # RUL loss (MSE)
        if 'rul' in predictions and 'rul' in targets:
            rul_loss = self.mse_loss(predictions['rul'], targets['rul'])
            losses.append(rul_loss)

        # Temperature loss (MSE)
        if 'temperature' in predictions and 'temperature' in targets:
            temp_loss = self.mse_loss(predictions['temperature'], targets['temperature'])
            losses.append(temp_loss)

        # Vibration losses (MSE)
        if 'vibration_x' in predictions and 'vibration_x' in targets:
            vib_x_loss = self.mse_loss(predictions['vibration_x'], targets['vibration_x'])
            losses.append(vib_x_loss)

        if 'vibration_y' in predictions and 'vibration_y' in targets:
            vib_y_loss = self.mse_loss(predictions['vibration_y'], targets['vibration_y'])
            losses.append(vib_y_loss)

        # Quality score loss (Binary Cross Entropy)
        if 'quality_score' in predictions and 'quality_score' in targets:
            score_loss = F.binary_cross_entropy(
                predictions['quality_score'],
                targets['quality_score']
            )
            losses.append(score_loss)

        # Adhesion strength (MSE)
        if 'adhesion_strength' in predictions and 'adhesion_strength' in targets:
            adhesion_loss = self.mse_loss(predictions['adhesion_strength'], targets['adhesion_strength'])
            losses.append(adhesion_loss)

        # Internal stress (MSE)
        if 'internal_stress' in predictions and 'internal_stress' in targets:
            stress_loss = self.mse_loss(predictions['internal_stress'], targets['internal_stress'])
            losses.append(stress_loss)

        # Porosity (MSE)
        if 'porosity' in predictions and 'porosity' in targets:
            porosity_loss = self.mse_loss(predictions['porosity'], targets['porosity'])
            losses.append(porosity_loss)

        # Dimensional accuracy (MSE)
        if 'dimensional_accuracy' in predictions and 'dimensional_accuracy' in targets:
            accuracy_loss = self.mse_loss(predictions['dimensional_accuracy'], targets['dimensional_accuracy'])
            losses.append(accuracy_loss)

        if losses:
            return sum(losses) / len(losses)
        # 如果没有找到任何预期的损失项，则返回0.0，使用targets字典中的任意tensor获取设备信息
        device = next(iter(targets.values())).device if targets else torch.device('cpu')
        return torch.tensor(0.0, device=device)

    def fault_loss(self, predictions: Dict[str, torch.Tensor],
                  targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute fault classification loss

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Fault classification loss
        """
        if 'fault_logits' not in predictions or 'fault_label' not in targets:
            # 使用targets字典中的任意tensor获取设备信息，保证一致性
            device = next(iter(targets.values())).device if targets else torch.device('cpu')
            return torch.tensor(0.0, device=device, requires_grad=True)

        return self.ce_loss(predictions['fault_logits'], targets['fault_label'])

    def trajectory_loss(self, predictions: Dict[str, torch.Tensor],
                       targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute trajectory correction loss using MSE (not Smooth L1)

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Trajectory correction loss
        """
        losses = []

        # 先处理序列轨迹（如果存在）
        if 'displacement_x_seq' in predictions and 'displacement_x_seq' in targets:
            pred_x_seq = predictions['displacement_x_seq']
            target_x_seq = targets['displacement_x_seq']
            # 对齐序列长度（取末尾pred_len）
            if pred_x_seq.dim() == 3 and target_x_seq.dim() == 3:
                target_len = target_x_seq.size(1)
                if pred_x_seq.size(1) != target_len:
                    pred_x_seq = pred_x_seq[:, -target_len:, :]
            # 使用MSE而不是Smooth L1
            dx_seq_loss = self.mse_loss(pred_x_seq, target_x_seq)
            losses.append(dx_seq_loss)

        if 'displacement_y_seq' in predictions and 'displacement_y_seq' in targets:
            pred_y_seq = predictions['displacement_y_seq']
            target_y_seq = targets['displacement_y_seq']
            if pred_y_seq.dim() == 3 and target_y_seq.dim() == 3:
                target_len = target_y_seq.size(1)
                if pred_y_seq.size(1) != target_len:
                    pred_y_seq = pred_y_seq[:, -target_len:, :]
            # 使用MSE而不是Smooth L1
            dy_seq_loss = self.mse_loss(pred_y_seq, target_y_seq)
            losses.append(dy_seq_loss)

        if 'displacement_z_seq' in predictions and 'displacement_z_seq' in targets:
            pred_z_seq = predictions['displacement_z_seq']
            target_z_seq = targets['displacement_z_seq']
            if pred_z_seq.dim() == 3 and target_z_seq.dim() == 3:
                target_len = target_z_seq.size(1)
                if pred_z_seq.size(1) != target_len:
                    pred_z_seq = pred_z_seq[:, -target_len:, :]
            if target_z_seq.abs().max().item() > 1e-6:
                dz_seq_loss = self.mse_loss(pred_z_seq, target_z_seq)
                losses.append(dz_seq_loss)

        # 使用MSE而不是Smooth L1
        if 'displacement_x' in predictions and 'displacement_x' in targets:
            pred_x = predictions['displacement_x']
            target_x = targets['displacement_x']
            # Flatten if needed to match dimensions
            if pred_x.dim() > 2:
                pred_x = pred_x.reshape(pred_x.shape[0], -1)
            if target_x.dim() > 2:
                target_x = target_x.reshape(target_x.shape[0], -1)
            # 使用MSE
            dx_loss = self.mse_loss(pred_x, target_x)
            losses.append(dx_loss)

        if 'displacement_y' in predictions and 'displacement_y' in targets:
            pred_y = predictions['displacement_y']
            target_y = targets['displacement_y']
            # Flatten if needed to match dimensions
            if pred_y.dim() > 2:
                pred_y = pred_y.reshape(pred_y.shape[0], -1)
            if target_y.dim() > 2:
                target_y = target_y.reshape(target_y.shape[0], -1)
            # 使用MSE
            dy_loss = self.mse_loss(pred_y, target_y)
            losses.append(dy_loss)

        if 'displacement_z' in predictions and 'displacement_z' in targets:
            pred_z = predictions['displacement_z']
            target_z = targets['displacement_z']
            # Flatten if needed to match dimensions
            if pred_z.dim() > 2:
                pred_z = pred_z.reshape(pred_z.shape[0], -1)
            if target_z.dim() > 2:
                target_z = target_z.reshape(target_z.shape[0], -1)
            # 检查是否全为 0
            if target_z.abs().max().item() > 1e-6:
                # 使用MSE
                dz_loss = self.mse_loss(pred_z, target_z)
                losses.append(dz_loss)

        if losses:
            return sum(losses) / len(losses)
        # 如果没有找到任何预期的损失项，则返回0.0，使用targets字典中的任意tensor获取设备信息
        device = next(iter(targets.values())).device if targets else torch.device('cpu')
        return torch.tensor(0.0, device=device)
    def physics_loss(self, predictions: Dict[str, torch.Tensor],
                    inputs: Dict[str, torch.Tensor],
                    physics_config: Optional[Dict] = None) -> torch.Tensor:
        """
        基于真实二阶动力学系统计算物理约束损失

        完整动力学方程（来自MATLAB仿真）：
            m·x'' + c·x' + k·x = -m·a_ref

        其中误差定义：error = x_actual - x_reference

        物理参数（来自physics_parameters.m）：
        - X轴：m=0.35kg, k=8000N/m, c=15.0N·s/m
        - Y轴：m=0.45kg, k=8000N/m, c=15.0N·s/m

        稳态近似（忽略速度项）：
            k·error ≈ -m·a_ref
            error ≈ -(m/k)·a_ref

        Args:
            predictions: 模型预测的误差
            inputs: 输入特征（包含惯性力）
            physics_config: 物理参数配置

        Returns:
            物理约束损失
        """
        if physics_config is None:
            physics_config = {}

        # 真实物理参数（来自physics_parameters.m line 96-110）
        # 注意：这些是仿真实际使用的值，不是config中的理论值
        m_x = physics_config.get('mass_x', 0.35)         # kg - X轴质量
        m_y = physics_config.get('mass_y', 0.45)         # kg - Y轴质量
        k_x = physics_config.get('stiffness_x', 8000)    # N/m - X轴刚度（关键！）
        k_y = physics_config.get('stiffness_y', 8000)    # N/m - Y轴刚度
        c_x = physics_config.get('damping_x', 15.0)      # N·s/m - X轴阻尼
        c_y = physics_config.get('damping_y', 15.0)      # N·s/m - Y轴阻尼

        total_physics_loss = 0.0
        loss_count = 0

        # 物理约束 1: X轴动力学
        # 预测误差应该与惯性力成正比（稳态）
        if 'displacement_x' in predictions and 'F_inertia_x' in inputs:
            error_x = predictions['displacement_x']
            F_inertia_x = inputs.get('F_inertia_x')

            # 处理维度
            if error_x.dim() > 2:
                error_x = error_x.reshape(error_x.size(0), -1).mean(dim=1, keepdim=True)
            if error_x.dim() == 1:
                error_x = error_x.unsqueeze(-1)

            # 处理惯性力（可能是序列 [batch, seq_len] 或标量 [batch]）
            if F_inertia_x is not None and isinstance(F_inertia_x, torch.Tensor):
                if F_inertia_x.dim() > 1:
                    F_inertia_x = F_inertia_x.mean(dim=1, keepdim=True)
                if F_inertia_x.dim() == 1:
                    F_inertia_x = F_inertia_x.unsqueeze(-1)

                # 稳态物理约束：k*error ≈ -F_inertia
                # 来自：m·x'' + c·x' + k·x = -m·a_ref
                # 稳态时（x'=x''=0）：k*error = -m*a_ref = -F_inertia
                error_theory_x = -F_inertia_x / k_x

                # 使用Smooth L1 Loss（比MSE更鲁棒）
                physics_loss_x = F.smooth_l1_loss(error_x, error_theory_x)
                total_physics_loss += physics_loss_x
                loss_count += 1

        # 物理约束 2: Y轴动力学
        if 'displacement_y' in predictions and 'F_inertia_y' in inputs:
            error_y = predictions['displacement_y']
            F_inertia_y = inputs.get('F_inertia_y')

            # 处理维度
            if error_y.dim() > 2:
                error_y = error_y.reshape(error_y.size(0), -1).mean(dim=1, keepdim=True)
            if error_y.dim() == 1:
                error_y = error_y.unsqueeze(-1)

            # 处理惯性力
            if F_inertia_y is not None and isinstance(F_inertia_y, torch.Tensor):
                if F_inertia_y.dim() > 1:
                    F_inertia_y = F_inertia_y.mean(dim=1, keepdim=True)
                if F_inertia_y.dim() == 1:
                    F_inertia_y = F_inertia_y.unsqueeze(-1)

                # 稳态物理约束
                error_theory_y = -F_inertia_y / k_y

                physics_loss_y = F.smooth_l1_loss(error_y, error_theory_y)
                total_physics_loss += physics_loss_y
                loss_count += 1

        # 如果没有计算任何损失，返回零
        if loss_count == 0:
            device = next(iter(predictions.values())).device if predictions else torch.device('cpu')
            return torch.tensor(0.0, device=device, requires_grad=True)

        return total_physics_loss / max(loss_count, 1)

    def forward(self, predictions: Dict[str, torch.Tensor],
               targets: Dict[str, torch.Tensor],
               physics_config: Optional[Dict] = None,
               inputs: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute total multi-task loss

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            physics_config: Physics configuration
            inputs: Input features (for physics loss)

        Returns:
            Dictionary with individual losses and total loss
        """
        # Individual task losses
        L_quality = self.quality_loss(predictions, targets)
        L_fault = self.fault_loss(predictions, targets)
        L_trajectory = self.trajectory_loss(predictions, targets)

        # Physics constraint loss
        if inputs is not None and self.lambda_physics > 0:
            L_physics = self.physics_loss(predictions, inputs, physics_config)
        else:
            L_physics = torch.tensor(0.0, device=L_quality.device)

        # Total weighted loss
        total_loss = (
            self.lambda_quality * L_quality +
            self.lambda_fault * L_fault +
            self.lambda_trajectory * L_trajectory +
            self.lambda_physics * L_physics
        )

        # Return detailed loss information
        return {
            'total': total_loss,
            'quality': L_quality,
            'fault': L_fault,
            'trajectory': L_trajectory,
            'physics': L_physics,
        }

    def update_weights(self,
                      lambda_quality: Optional[float] = None,
                      lambda_fault: Optional[float] = None,
                      lambda_trajectory: Optional[float] = None,
                      lambda_physics: Optional[float] = None):
        """
        Update loss weights

        Args:
            lambda_quality: New quality loss weight
            lambda_fault: New fault loss weight
            lambda_trajectory: New trajectory loss weight
            lambda_physics: New physics loss weight
        """
        if lambda_quality is not None:
            self.lambda_quality = lambda_quality
        if lambda_fault is not None:
            self.lambda_fault = lambda_fault
        if lambda_trajectory is not None:
            self.lambda_trajectory = lambda_trajectory
        if lambda_physics is not None:
            self.lambda_physics = lambda_physics


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize focal loss

        Args:
            alpha: Weighting factor in [0, 1]
            gamma: Focusing parameter
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            inputs: Predictions (logits) [batch, num_classes]
            targets: Ground truth labels [batch]

        Returns:
            Focal loss
        """
        p = torch.softmax(inputs, dim=-1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        loss = -alpha_t * focal_weight * torch.log(p_t + 1e-8)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
