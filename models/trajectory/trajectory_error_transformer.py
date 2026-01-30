"""
Trajectory Error Correction Transformer

Refactored from previous_code/3D_printer_loss_perdict transformer model
for paper-ready implementation.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from ..base_model import BaseModel
from ..encoders import PINNTransformerEncoder
from ..decoders.trajectory_decoder import TrajectorySequenceDecoder


class TrajectoryErrorTransformer(BaseModel):
    """
    Trajectory error correction model.

    Inputs: command/sensor sequence
    Outputs: error_x/error_y sequence
    """

    def __init__(self, config):
        super().__init__(config)

        self.encoder = PINNTransformerEncoder(
            num_features=config.data.num_features,
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            d_ff=config.model.dim_feedforward,
            dropout=config.model.dropout,
            activation=config.model.activation,
        )

        self.decoder = TrajectorySequenceDecoder(
            d_model=config.model.d_model,
            lstm_hidden=config.model.trajectory_lstm_hidden,
            lstm_layers=config.model.trajectory_lstm_layers,
            bidirectional=config.model.trajectory_bidirectional,
            use_attention=config.model.trajectory_attention,
            num_outputs=config.data.num_trajectory_outputs,
            dropout=config.model.dropout,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(x, mask)
        outputs = self.decoder(encoded, mask)
        return outputs

    def get_model_info(self) -> Dict[str, any]:
        return {
            'model_type': 'TrajectoryErrorTransformer',
            'num_parameters': self.get_num_params(),
            'num_trainable_parameters': self.get_num_trainable_params(),
        }
