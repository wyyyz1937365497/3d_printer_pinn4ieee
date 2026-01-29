"""
Implicit State Inference Transformer

Refactored from previous_code/3D_printer_pinn_transformer (PhysicalPredictor)
for paper-ready implementation.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from ..base_model import BaseModel
from ..encoders import PINNTransformerEncoder


class ImplicitStateTransformer(BaseModel):
    """
    Implicit state inference model.

    Inputs: observable sensor/command sequence
    Outputs: hard-to-measure physical/quality states
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

        hidden_dims = config.model.quality_hidden_dims
        dropout = config.model.quality_dropout

        layers = []
        input_dim = config.model.d_model
        for h in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = h

        layers.append(nn.Linear(input_dim, config.data.num_quality_outputs))
        self.mlp = nn.Sequential(*layers)

        nn.init.xavier_uniform_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(x, mask)
        pooled = encoded.mean(dim=1)
        preds = self.mlp(pooled)

        outputs = {
            'adhesion_strength': preds[:, 0:1],
            'internal_stress': preds[:, 1:2],
            'porosity': torch.sigmoid(preds[:, 2:3]) * 100.0,
            'dimensional_accuracy': preds[:, 3:4],
            'quality_score': torch.sigmoid(preds[:, 4:5]),
        }
        return outputs

    def get_model_info(self) -> Dict[str, any]:
        return {
            'model_type': 'ImplicitStateTransformer',
            'num_parameters': self.get_num_params(),
            'num_trainable_parameters': self.get_num_trainable_params(),
        }
