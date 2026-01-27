"""
Quality Prediction Decoder Head

Predicts quality metrics including RUL, temperature, vibration, and quality score
"""

import torch
import torch.nn as nn


class QualityPredictionHead(nn.Module):
    """
    Quality prediction head for multi-task quality metrics

    Outputs:
    - RUL (Remaining Useful Life): Continuous value
    - Temperature: Continuous value
    - Vibration (X, Y): Continuous values
    - Quality Score: Continuous value [0, 1]
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

        # Split into individual outputs
        outputs = {
            'rul': predictions[:, 0:1],  # Remaining Useful Life
            'temperature': predictions[:, 1:2],  # Temperature
            'vibration_x': predictions[:, 2:3],  # X-axis vibration
            'vibration_y': predictions[:, 3:4],  # Y-axis vibration
            'quality_score': torch.sigmoid(predictions[:, 4:5]),  # Quality score [0, 1]
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
