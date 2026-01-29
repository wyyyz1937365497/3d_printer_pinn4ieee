"""
Enhanced Implicit State Inference using Temporal Convolutional Network (TCN)

Improvements:
1. LayerNorm for input normalization
2. Physics-constrained activation functions
3. Attention-weighted pooling
4. Skip connections for multi-scale features
5. PINN-enhanced training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class TemporalBlock(nn.Module):
    """Dilated residual convolution block with LayerNorm"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        # Conv1 -> LayerNorm -> ReLU -> Dropout
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.norm1 = nn.LayerNorm(out_channels)

        # Conv2 -> LayerNorm -> ReLU -> Dropout
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.norm2 = nn.LayerNorm(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')

    def forward(self, x):
        # x: [B, C, T]
        out = self.conv1(x)
        out = out[..., :x.size(-1)]  # trim to original length

        # LayerNorm expects [B, T, C]
        out = out.transpose(1, 2)
        out = self.norm1(out)
        out = out.transpose(1, 2)

        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[..., :x.size(-1)]

        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)

        out = self.relu(out)
        out = self.dropout(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        res = res[..., :out.size(-1)]
        return self.relu(out + res)


class AttentionPooling(nn.Module):
    """Attention-weighted pooling for sequence aggregation"""
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x):
        # x: [B, C, T]
        x = x.transpose(1, 2)  # [B, T, C]

        attn_weights = self.attention(x)  # [B, T, 1]
        attn_weights = F.softmax(attn_weights, dim=1)  # [B, T, 1]

        pooled = (x * attn_weights).sum(dim=1)  # [B, C]
        return pooled


class TCNEncoder(nn.Module):
    """TCN encoder with skip connections"""
    def __init__(self, in_channels, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.channels = channels

        layers = []
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else channels[i - 1]
            layers.append(TemporalBlock(in_ch, ch, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)

        # Skip connection: concatenate intermediate representations
        self.use_skip = True

    def forward(self, x):
        # x: [B, C, T]
        # Process through blocks
        out = x
        intermediate_outputs = []

        for block in self.network:
            out = block(out)
            intermediate_outputs.append(out)

        # Skip connection: concatenate all intermediate outputs
        if self.use_skip:
            # Interpolate to same length and concatenate
            min_len = min(o.size(-1) for o in intermediate_outputs)
            aligned_outputs = [o[..., -min_len:] for o in intermediate_outputs]
            out = torch.cat(aligned_outputs, dim=1)  # [B, sum(channels), T]

        return out


class ImplicitStateTCN(nn.Module):
    """
    Enhanced TCN-based implicit state inference model.

    Features:
    - LayerNorm for stable training
    - Physics-constrained outputs
    - Attention-weighted pooling
    - Skip connections for multi-scale features
    """

    def __init__(self, config):
        super().__init__()

        self.in_channels = config.data.num_features
        self.tcn_channels = [128, 128, 256]
        self.kernel_size = 3
        self.dropout = 0.1

        # 1. Input LayerNorm
        self.input_norm = nn.LayerNorm(self.in_channels)

        # 2. TCN Encoder with skip connections
        self.encoder = TCNEncoder(
            in_channels=self.in_channels,
            channels=self.tcn_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        )

        # Calculate encoder output dimension (sum of channels due to skip)
        self.encoder_out_dim = sum(self.tcn_channels)

        # 3. Attention-weighted pooling
        self.attention_pool = AttentionPooling(self.encoder_out_dim)

        # 4. Multi-task head with separate branches
        hidden = 256

        # Shared features
        self.shared_fc = nn.Sequential(
            nn.Linear(self.encoder_out_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        # Task-specific branches
        self.adhesion_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.stress_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.porosity_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.accuracy_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.quality_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for all heads"""
        for head in [self.adhesion_head, self.stress_head, self.porosity_head,
                     self.accuracy_head, self.quality_head]:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with physics-constrained outputs

        Args:
            x: Input tensor [B, T, F]

        Returns:
            Dictionary with physics-constrained predictions
        """
        # 1. Input normalization
        x = self.input_norm(x)  # [B, T, F]

        # 2. Transpose for conv1d: [B, F, T]
        x = x.transpose(1, 2)

        # 3. TCN encoding with skip connections
        feats = self.encoder(x)  # [B, sum(channels), T]

        # 4. Attention-weighted pooling
        pooled = self.attention_pool(feats)  # [B, sum(channels)]

        # 5. Shared features
        shared = self.shared_fc(pooled)  # [B, hidden]

        # 6. Task-specific predictions with physics constraints
        adhesion_raw = self.adhesion_head(shared)
        stress_raw = self.stress_head(shared)
        porosity_raw = self.porosity_head(shared)
        accuracy_raw = self.accuracy_head(shared)
        quality_raw = self.quality_head(shared)

        # Apply physics constraints
        outputs = {
            # Adhesion: [0, 1] ratio (Wool-O'Connor healing model)
            'adhesion_strength': torch.sigmoid(adhesion_raw),

            # Stress: >= 0 MPa (non-negative stress)
            'internal_stress': F.relu(stress_raw) + 10.0,  # baseline + offset

            # Porosity: [0, 100] %
            'porosity': torch.sigmoid(porosity_raw) * 100.0,

            # Dimensional accuracy: unconstrained (could be positive/negative error)
            'dimensional_accuracy': accuracy_raw,

            # Quality score: [0, 1]
            'quality_score': torch.sigmoid(quality_raw),
        }

        return outputs

    def get_model_info(self) -> Dict[str, any]:
        return {
            'model_type': 'EnhancedImplicitStateTCN',
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'tcn_channels': self.tcn_channels,
            'encoder_output_dim': self.encoder_out_dim,
            'features': ['LayerNorm', 'SkipConnections', 'AttentionPooling', 'PhysicsConstraints']
        }


class AdaptiveMultiTaskLoss(nn.Module):
    """
    Adaptive multi-task loss with physics-informed constraints

    Features:
    - Automatic loss weighting based on target variance
    - Physics-informed regularization
    - Task uncertainty weighting
    """

    def __init__(self,
                 lambda_adhesion: float = 1.0,
                 lambda_stress: float = 1.0,
                 lambda_porosity: float = 1.0,
                 lambda_accuracy: float = 1.0,
                 lambda_quality: float = 1.0,
                 lambda_physics: float = 0.1,
                 use_adaptive_weights: bool = True):
        """
        Initialize adaptive multi-task loss

        Args:
            lambda_*: Base weights for each task
            lambda_physics: Weight for physics-informed loss
            use_adaptive_weights: Whether to use adaptive weighting based on variance
        """
        super().__init__()

        self.use_adaptive = use_adaptive_weights
        self.lambda_physics = lambda_physics

        # Learnable loss weights (homoscedastic uncertainty)
        if use_adaptive_weights:
            self.log_vars = nn.Parameter(torch.zeros(5))  # One per task

        # Base weights
        self.register_buffer('lambda_adhesion', torch.tensor(lambda_adhesion))
        self.register_buffer('lambda_stress', torch.tensor(lambda_stress))
        self.register_buffer('lambda_porosity', torch.tensor(lambda_porosity))
        self.register_buffer('lambda_accuracy', torch.tensor(lambda_accuracy))
        self.register_buffer('lambda_quality', torch.tensor(lambda_quality))

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def physics_loss(self, predictions: Dict[str, torch.Tensor],
                     inputs: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Physics-informed regularization

        Enforces:
        1. Output bounds (soft constraints)
        2. Inter-task correlations
        3. Minimal variance regularization (prevent collapse)
        """
        physics_losses = {}
        device = next(iter(predictions.values())).device

        # Extract predictions
        adhesion = predictions['adhesion_strength']
        stress = predictions['internal_stress']
        porosity = predictions['porosity']
        quality = predictions['quality_score']

        # 1. Soft bounds penalties (ONLY if predictions go outside reasonable ranges)
        # Adhesion: [0, 1] - sigmoid already enforces this, so minimal penalty
        adhesion_lower = torch.mean(F.relu(0.0 - adhesion)) * 0.01
        adhesion_upper = torch.mean(F.relu(adhesion - 1.0)) * 0.01
        physics_losses['adhesion_bounds'] = adhesion_lower + adhesion_upper

        # Stress: [10, 30] MPa - soft penalty if outside this range
        stress_lower = torch.mean(F.relu(10.0 - stress)) * 0.001
        stress_upper = torch.mean(F.relu(stress - 30.0)) * 0.001
        physics_losses['stress_bounds'] = stress_lower + stress_upper

        # Porosity: [0, 30] % - soft penalty if outside this range
        porosity_lower = torch.mean(F.relu(0.0 - porosity)) * 0.001
        porosity_upper = torch.mean(F.relu(porosity - 30.0)) * 0.001
        physics_losses['porosity_bounds'] = porosity_lower + porosity_upper

        # 2. Minimal variance penalty - prevent collapse to single value (REDUCED WEIGHT)
        # Only penalize if variance is extremely low (< 0.01)
        adhesion_var = torch.var(adhesion)
        stress_var = torch.var(stress)
        porosity_var = torch.var(porosity)
        quality_var = torch.var(quality)

        min_var_threshold = 0.01
        var_penalty = 0.0
        if adhesion_var < min_var_threshold:
            var_penalty += (min_var_threshold - adhesion_var) * 0.001
        if stress_var < min_var_threshold:
            var_penalty += (min_var_threshold - stress_var) * 0.0001
        if porosity_var < min_var_threshold:
            var_penalty += (min_var_threshold - porosity_var) * 0.0001
        if quality_var < min_var_threshold:
            var_penalty += (min_var_threshold - quality_var) * 0.001

        physics_losses['variance_collapse'] = var_penalty

        # 3. Inter-task consistency (REDUCED WEIGHT)
        # Quality should correlate with adhesion
        quality_adhesion_correlation = -torch.mean(quality * adhesion) * 0.01
        physics_losses['quality_adhesion'] = quality_adhesion_correlation

        return physics_losses

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                inputs: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute weighted multi-task loss with physics constraints

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            inputs: Input features (for physics loss)

        Returns:
            Dictionary with total loss and individual components
        """
        # Extract predictions
        adhesion_pred = predictions['adhesion_strength']
        stress_pred = predictions['internal_stress']
        porosity_pred = predictions['porosity']
        accuracy_pred = predictions['dimensional_accuracy']
        quality_pred = predictions['quality_score']

        # Extract targets
        adhesion_target = targets['adhesion_strength']
        stress_target = targets['internal_stress']
        porosity_target = targets['porosity']
        accuracy_target = targets['dimensional_accuracy']
        quality_target = targets['quality_score']

        # Compute individual losses
        loss_adhesion = self.mse_loss(adhesion_pred, adhesion_target)
        loss_stress = self.mse_loss(stress_pred, stress_target)
        loss_porosity = self.mse_loss(porosity_pred, porosity_target)
        loss_accuracy = self.mse_loss(accuracy_pred, accuracy_target)
        loss_quality = self.mse_loss(quality_pred, quality_target)

        # Adaptive weighting (Kendall et al. 2018)
        if self.use_adaptive:
            # Precision-weighted loss
            loss_adhesion = torch.exp(-self.log_vars[0]) * loss_adhesion + self.log_vars[0]
            loss_stress = torch.exp(-self.log_vars[1]) * loss_stress + self.log_vars[1]
            loss_porosity = torch.exp(-self.log_vars[2]) * loss_porosity + self.log_vars[2]
            loss_accuracy = torch.exp(-self.log_vars[3]) * loss_accuracy + self.log_vars[3]
            loss_quality = torch.exp(-self.log_vars[4]) * loss_quality + self.log_vars[4]
        else:
            # Fixed weighting
            loss_adhesion = self.lambda_adhesion * loss_adhesion
            loss_stress = self.lambda_stress * loss_stress
            loss_porosity = self.lambda_porosity * loss_porosity
            loss_accuracy = self.lambda_accuracy * loss_accuracy
            loss_quality = self.lambda_quality * loss_quality

        # Data loss
        data_loss = (loss_adhesion + loss_stress + loss_porosity +
                    loss_accuracy + loss_quality)

        # Physics-informed loss
        physics_loss_total = torch.tensor(0.0, device=data_loss.device)
        physics_losses_dict = {}

        if self.lambda_physics > 0:
            physics_losses_dict = self.physics_loss(predictions, inputs)
            if physics_losses_dict:
                physics_loss_total = sum(physics_losses_dict.values())
                physics_loss_total = self.lambda_physics * physics_loss_total

        # Total loss
        total_loss = data_loss + physics_loss_total

        return {
            'total': total_loss,
            'data': data_loss,
            'adhesion': loss_adhesion,
            'stress': loss_stress,
            'porosity': loss_porosity,
            'accuracy': loss_accuracy,
            'quality': loss_quality,
            'physics': physics_loss_total,
            **{f'physics_{k}': v for k, v in physics_losses_dict.items()}
        }

    def get_effective_weights(self):
        """Get current effective loss weights (for logging)"""
        if self.use_adaptive:
            return {
                'adhesion': torch.exp(-self.log_vars[0]).item(),
                'stress': torch.exp(-self.log_vars[1]).item(),
                'porosity': torch.exp(-self.log_vars[2]).item(),
                'accuracy': torch.exp(-self.log_vars[3]).item(),
                'quality': torch.exp(-self.log_vars[4]).item(),
            }
        else:
            return {
                'adhesion': self.lambda_adhesion.item(),
                'stress': self.lambda_stress.item(),
                'porosity': self.lambda_porosity.item(),
                'accuracy': self.lambda_accuracy.item(),
                'quality': self.lambda_quality.item(),
            }
