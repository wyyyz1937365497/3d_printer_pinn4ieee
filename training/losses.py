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

        if losses:
            return sum(losses) / len(losses)
        return torch.tensor(0.0, device=predictions['rul'].device)

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
            return torch.tensor(0.0, device=list(predictions.values())[0].device)

        return self.ce_loss(predictions['fault_logits'], targets['fault_label'])

    def trajectory_loss(self, predictions: Dict[str, torch.Tensor],
                       targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute trajectory correction loss

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Trajectory correction loss
        """
        losses = []

        # Use Smooth L1 loss for displacement (more robust to outliers)
        if 'displacement_x' in predictions and 'displacement_x' in targets:
            dx_loss = self.smooth_l1_loss(predictions['displacement_x'], targets['displacement_x'])
            losses.append(dx_loss)

        if 'displacement_y' in predictions and 'displacement_y' in targets:
            dy_loss = self.smooth_l1_loss(predictions['displacement_y'], targets['displacement_y'])
            losses.append(dy_loss)

        if 'displacement_z' in predictions and 'displacement_z' in targets:
            dz_loss = self.smooth_l1_loss(predictions['displacement_z'], targets['displacement_z'])
            losses.append(dz_loss)

        if losses:
            return sum(losses) / len(losses)
        return torch.tensor(0.0, device=list(predictions.values())[0].device)

    def physics_loss(self, predictions: Dict[str, torch.Tensor],
                    inputs: Dict[str, torch.Tensor],
                    physics_config: Optional[Dict] = None) -> torch.Tensor:
        """
        Compute physics constraint loss

        Args:
            predictions: Model predictions
            inputs: Input features
            physics_config: Physics configuration parameters

        Returns:
            Physics constraint loss
        """
        from ..utils.physics_utils import compute_physics_loss

        if physics_config is None:
            physics_config = {}

        return compute_physics_loss(predictions, {}, inputs, physics_config)

    def forward(self, predictions: Dict[str, torch.Tensor],
               targets: Dict[str, torch.Tensor],
               inputs: Optional[Dict[str, torch.Tensor]] = None,
               physics_config: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Compute total multi-task loss

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            inputs: Input features (for physics loss)
            physics_config: Physics configuration

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
