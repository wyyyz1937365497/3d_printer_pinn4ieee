"""
PINN-Guided Transformer Encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PhysicsInformedFeedForward(nn.Module):
    """
    Physics-informed feed-forward network

    Incorporates physical constraints into the feed-forward transformation
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = 'gelu'):
        """
        Initialize physics-informed FFN

        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function type
        """
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        # Physics-aware skip connection
        self.physics_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

        # Activation function
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with physics-aware gating

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Transformed tensor
        """
        # Standard feed-forward transformation
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))

        # Physics-aware gating
        gate = self.physics_gate(x)
        output = x + gate * ff_out

        return output


class PINNTransformerEncoderLayer(nn.Module):
    """
    Single layer of PINN-Guided Transformer Encoder
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, activation: str = 'gelu'):
        """
        Initialize encoder layer

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Physics-informed feed-forward
        self.ffn = PhysicsInformedFeedForward(d_model, d_ff, dropout, activation)

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor [batch, seq_len, d_model]
            src_mask: Optional attention mask

        Returns:
            Encoded tensor [batch, seq_len, d_model]
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, attn_mask=src_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Physics-informed feed-forward with residual connection
        x = self.ffn(x)
        x = self.norm2(x)

        return x


class PINNTransformerEncoder(nn.Module):
    """
    PINN-Guided Transformer Encoder

    Shared encoder that learns physics-aware representations
    """

    def __init__(self,
                 num_features: int,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 1024,
                 dropout: float = 0.1,
                 activation: str = 'gelu',
                 max_seq_len: int = 5000):
        """
        Initialize encoder

        Args:
            num_features: Number of input features
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()

        self.d_model = d_model
        self.num_features = num_features

        # Input projection
        self.input_projection = nn.Linear(num_features, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            PINNTransformerEncoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor [batch, seq_len, num_features]
            mask: Optional attention mask

        Returns:
            Encoded tensor [batch, seq_len, d_model]
        """
        # Project input to d_model dimension
        x = self.input_projection(x)

        # Scale input (as in Transformer paper)
        x = x * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)

        return x

    def get_output_dim(self) -> int:
        """
        Get output dimension

        Returns:
            Output dimension (d_model)
        """
        return self.d_model
