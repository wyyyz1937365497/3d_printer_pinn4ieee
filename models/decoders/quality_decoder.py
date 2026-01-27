"""
Quality Prediction Decoder Head

Predicts IMPLICIT QUALITY PARAMETERS that cannot be directly measured during printing:
- Interlayer Adhesion Strength (层间粘合力)
- Internal Stress (内应力)
- Porosity (孔隙率)
- Dimensional Accuracy (尺寸精度)
- Overall Quality Score (综合质量评分)

These parameters are inferred from observable sensor data (temperature, vibration, etc.)
using physics-informed neural networks.
"""

import torch
import torch.nn as nn


class QualityPredictionHead(nn.Module):
    """
    Quality prediction head for inferring implicit quality parameters

    This module predicts CRITICAL QUALITY METRICS that cannot be directly measured
    during the printing process. Instead, it uses observable sensor data to infer
    these hidden parameters through physics-informed learning.

    Outputs (Implicit Quality Parameters):
    - Interlayer Adhesion Strength (MPa): Bond strength between layers
    - Internal Stress (MPa): Residual stress accumulated during printing
    - Porosity (%): Void fraction in the printed part
    - Dimensional Accuracy (mm): Deviation from intended dimensions
    - Overall Quality Score [0, 1]: Composite quality indicator

    Key Innovation:
    - Bridges the gap between observable (temperature, vibration) and unobservable
      (adhesion, stress) quality parameters
    - Uses PINN to embed physical constraints relating these quantities
    - Enables early stopping by predicting final quality from early-stage data
    """

    def __init__(self,
                 d_model: int,
                 hidden_dims: list = [256, 128],
                 dropout: float = 0.2,
                 num_outputs: int = 5):
        """
        Initialize quality prediction head

        Args:
            d_model: Input dimension (from encoder)
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            num_outputs: Number of quality outputs (default: 5)
        """
        super().__init__()

        self.d_model = d_model
        self.num_outputs = num_outputs

        # Build MLP layers
        layers = []
        input_dim = d_model

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, num_outputs))

        self.mlp = nn.Sequential(*layers)

        # Initialize output layer weights
        nn.init.xavier_uniform_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, encoder_output: torch.Tensor) -> dict:
        """
        Forward pass

        Args:
            encoder_output: Encoded tensor [batch, seq_len, d_model]

        Returns:
            Dictionary with predicted quality metrics
        """
        # Global average pooling over sequence length
        pooled = encoder_output.mean(dim=1)  # [batch, d_model]

        # Pass through MLP
        predictions = self.mlp(pooled)  # [batch, num_outputs]

        # Split into individual outputs (IMPLICIT QUALITY PARAMETERS)
        outputs = {
            'adhesion_strength': predictions[:, 0:1],  # Interlayer adhesion strength (MPa)
            'internal_stress': predictions[:, 1:2],    # Internal/residual stress (MPa)
            'porosity': torch.sigmoid(predictions[:, 2:3]) * 100,  # Porosity (%), [0, 100]
            'dimensional_accuracy': predictions[:, 3:4],  # Dimensional accuracy error (mm)
            'quality_score': torch.sigmoid(predictions[:, 4:5]),  # Overall quality score [0, 1]
        }

        return outputs

    def get_output_dim(self) -> int:
        """
        Get total output dimension

        Returns:
            Number of outputs
        """
        return self.num_outputs


class FaultClassificationHead(nn.Module):
    """
    Fault classification head

    Outputs:
    - Fault probabilities for 4 classes:
        0: Normal
        1: Nozzle Clog
        2: Mechanical Loose
        3: Motor Fault
    """

    def __init__(self,
                 d_model: int,
                 hidden_dims: list = [128],
                 dropout: float = 0.3,
                 num_classes: int = 4):
        """
        Initialize fault classification head

        Args:
            d_model: Input dimension (from encoder)
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            num_classes: Number of fault classes
        """
        super().__init__()

        self.d_model = d_model
        self.num_classes = num_classes

        # Build MLP layers
        layers = []
        input_dim = d_model

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim

        # Output layer (logits)
        layers.append(nn.Linear(input_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

        # Initialize output layer weights
        nn.init.xavier_uniform_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, encoder_output: torch.Tensor) -> dict:
        """
        Forward pass

        Args:
            encoder_output: Encoded tensor [batch, seq_len, d_model]

        Returns:
            Dictionary with fault probabilities
        """
        # Global average pooling over sequence length
        pooled = encoder_output.mean(dim=1)  # [batch, d_model]

        # Pass through MLP to get logits
        logits = self.mlp(pooled)  # [batch, num_classes]

        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)

        # Get predicted class
        preds = torch.argmax(probs, dim=-1)

        return {
            'fault_logits': logits,
            'fault_probs': probs,
            'fault_pred': preds,
        }

    def get_output_dim(self) -> int:
        """
        Get output dimension

        Returns:
            Number of classes
        """
        return self.num_classes
