"""
Unified PINN-Seq3D Model

Combines quality prediction and trajectory correction in a single model
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .base_model import BaseModel
from .encoders import PINNTransformerEncoder
from .decoders.quality_decoder import QualityPredictionHead, FaultClassificationHead
from .decoders.trajectory_decoder import TrajectoryCorrectionHead


class UnifiedPINNSeq3D(BaseModel):
    """
    Unified model for 3D printer quality prediction and trajectory correction

    Architecture:
        Input (sensor data) -> Shared PINN-Guided Transformer Encoder
                                 -> Quality Prediction Head (RUL, temp, vibration, quality score)
                                 -> Fault Classification Head (4 fault types)
                                 -> Trajectory Correction Head (dx, dy, dz)
    """

    def __init__(self, config):
        """
        Initialize unified model

        Args:
            config: Configuration object with model parameters
        """
        super().__init__(config)

        # Shared encoder
        self.encoder = PINNTransformerEncoder(
            num_features=config.data.num_features,
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            d_ff=config.model.dim_feedforward,
            dropout=config.model.dropout,
            activation=config.model.activation,
        )

        # Quality prediction head
        self.quality_head = QualityPredictionHead(
            d_model=config.model.d_model,
            hidden_dims=config.model.quality_hidden_dims,
            dropout=config.model.quality_dropout,
            num_outputs=config.data.num_quality_outputs,
        )

        # Fault classification head
        self.fault_head = FaultClassificationHead(
            d_model=config.model.d_model,
            hidden_dims=config.model.fault_hidden_dims,
            dropout=config.model.fault_dropout,
            num_classes=config.data.num_fault_classes,
        )

        # Trajectory correction head
        self.trajectory_head = TrajectoryCorrectionHead(
            d_model=config.model.d_model,
            lstm_hidden=config.model.trajectory_lstm_hidden,
            lstm_layers=config.model.trajectory_lstm_layers,
            bidirectional=config.model.trajectory_bidirectional,
            use_attention=config.model.trajectory_attention,
            num_outputs=config.data.num_trajectory_outputs,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor [batch, seq_len, num_features]
            mask: Optional attention mask [batch, seq_len, seq_len]

        Returns:
            Dictionary containing all model outputs
        """
        # Shared encoding
        encoded = self.encoder(x, mask)  # [batch, seq_len, d_model]

        # Quality prediction
        quality_outputs = self.quality_head(encoded)

        # Fault classification
        fault_outputs = self.fault_head(encoded)

        # Trajectory correction
        trajectory_outputs = self.trajectory_head(encoded, mask)

        # Combine all outputs
        outputs = {
            'encoded': encoded,
            **quality_outputs,
            **fault_outputs,
            **trajectory_outputs,
        }

        return outputs

    def predict_quality(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Predict quality metrics only

        Args:
            x: Input tensor
            mask: Optional attention mask

        Returns:
            Quality prediction outputs
        """
        encoded = self.encoder(x, mask)
        return self.quality_head(encoded)

    def predict_fault(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Predict fault classification only

        Args:
            x: Input tensor
            mask: Optional attention mask

        Returns:
            Fault classification outputs
        """
        encoded = self.encoder(x, mask)
        return self.fault_head(encoded)

    def predict_trajectory(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Predict trajectory correction only

        Args:
            x: Input tensor
            mask: Optional attention mask

        Returns:
            Trajectory correction outputs
        """
        encoded = self.encoder(x, mask)
        return self.trajectory_head(encoded, mask)

    def get_model_info(self) -> Dict[str, any]:
        """
        Get model information

        Returns:
            Dictionary with model details
        """
        return {
            'model_type': 'UnifiedPINNSeq3D',
            'num_parameters': self.get_num_params(),
            'num_trainable_parameters': self.get_num_trainable_params(),
            'encoder_output_dim': self.encoder.get_output_dim(),
            'quality_outputs': self.quality_head.get_output_dim(),
            'fault_classes': self.fault_head.get_output_dim(),
            'trajectory_outputs': self.trajectory_head.get_output_dim(),
        }


class QualityPredictionOnlyModel(BaseModel):
    """
    Quality prediction only model (no trajectory correction)
    """

    def __init__(self, config):
        """
        Initialize quality prediction model

        Args:
            config: Configuration object
        """
        super().__init__(config)

        # Encoder
        self.encoder = PINNTransformerEncoder(
            num_features=config.data.num_features,
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            d_ff=config.model.dim_feedforward,
            dropout=config.model.dropout,
            activation=config.model.activation,
        )

        # Quality prediction head
        self.quality_head = QualityPredictionHead(
            d_model=config.model.d_model,
            hidden_dims=config.model.quality_hidden_dims,
            dropout=config.model.quality_dropout,
            num_outputs=config.data.num_quality_outputs,
        )

        # Fault classification head
        self.fault_head = FaultClassificationHead(
            d_model=config.model.d_model,
            hidden_dims=config.model.fault_hidden_dims,
            dropout=config.model.fault_dropout,
            num_classes=config.data.num_fault_classes,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor
            mask: Optional attention mask

        Returns:
            Quality and fault outputs
        """
        encoded = self.encoder(x, mask)

        quality_outputs = self.quality_head(encoded)
        fault_outputs = self.fault_head(encoded)

        return {
            'encoded': encoded,
            **quality_outputs,
            **fault_outputs,
        }


class TrajectoryCorrectionOnlyModel(BaseModel):
    """
    Trajectory correction only model (no quality prediction)
    """

    def __init__(self, config):
        """
        Initialize trajectory correction model

        Args:
            config: Configuration object
        """
        super().__init__(config)

        # Encoder
        self.encoder = PINNTransformerEncoder(
            num_features=config.data.num_features,
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            d_ff=config.model.dim_feedforward,
            dropout=config.model.dropout,
            activation=config.model.activation,
        )

        # Trajectory correction head
        self.trajectory_head = TrajectoryCorrectionHead(
            d_model=config.model.d_model,
            lstm_hidden=config.model.trajectory_lstm_hidden,
            lstm_layers=config.model.trajectory_lstm_layers,
            bidirectional=config.model.trajectory_bidirectional,
            use_attention=config.model.trajectory_attention,
            num_outputs=config.data.num_trajectory_outputs,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor
            mask: Optional attention mask

        Returns:
            Trajectory correction outputs
        """
        encoded = self.encoder(x, mask)
        trajectory_outputs = self.trajectory_head(encoded, mask)

        return {
            'encoded': encoded,
            **trajectory_outputs,
        }
