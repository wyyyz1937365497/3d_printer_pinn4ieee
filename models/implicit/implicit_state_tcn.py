"""
Implicit State Inference using Temporal Convolutional Network (TCN)
"""

import torch
import torch.nn as nn
from typing import Dict

from ..base_model import BaseModel


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
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
        out = self.conv1(x)
        out = out[..., :x.size(-1)]  # trim to original length
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[..., :x.size(-1)]
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        res = res[..., :out.size(-1)]
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else channels[i - 1]
            layers.append(TemporalBlock(in_ch, ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, T]
        return self.network(x)


class ImplicitStateTCN(BaseModel):
    """
    TCN-based implicit state inference model.
    Better for local temporal patterns and smoother regression targets.
    """

    def __init__(self, config):
        super().__init__(config)

        self.in_channels = config.data.num_features
        self.tcn_channels = [128, 128, 256]
        self.kernel_size = 3
        self.dropout = 0.1

        self.encoder = TCNEncoder(
            in_channels=self.in_channels,
            channels=self.tcn_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        )

        hidden = 256
        self.head = nn.Sequential(
            nn.Linear(self.tcn_channels[-1], hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, config.data.num_quality_outputs)
        )

        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)
        feats = self.encoder(x)  # [B, C, T]
        pooled = feats.mean(dim=-1)  # [B, C]
        preds = self.head(pooled)

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
            'model_type': 'ImplicitStateTCN',
            'num_parameters': self.get_num_params(),
            'num_trainable_parameters': self.get_num_trainable_params(),
        }
