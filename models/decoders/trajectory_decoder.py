"""
Trajectory Correction Decoder Head

Predicts displacement corrections for 3D printing trajectories
Combines BiLSTM with attention mechanism for sequence-based correction
"""

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for trajectory correction
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize multi-head attention

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass

        Args:
            query: Query tensor [batch, seq_len_q, d_model]
            key: Key tensor [batch, seq_len_k, d_model]
            value: Value tensor [batch, seq_len_v, d_model]
            mask: Optional attention mask

        Returns:
            Attention output [batch, seq_len_q, d_model]
        """
        batch_size = query.size(0)

        # Linear projections and reshape
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(attn_output)

        return output


class TrajectoryCorrectionHead(nn.Module):
    """
    Trajectory correction head with BiLSTM and attention

    Outputs:
    - X displacement correction (dx)
    - Y displacement correction (dy)
    - Z displacement correction (dz)
    """

    def __init__(self,
                 d_model: int,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 bidirectional: bool = True,
                 use_attention: bool = True,
                 num_attention_heads: int = 8,
                 dropout: float = 0.1,
                 num_outputs: int = 3):
        """
        Initialize trajectory correction head

        Args:
            d_model: Input dimension (from encoder)
            lstm_hidden: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            bidirectional: Use bidirectional LSTM
            use_attention: Use attention mechanism
            num_attention_heads: Number of attention heads
            dropout: Dropout rate
            num_outputs: Number of correction outputs (default: 3 for dx, dy, dz)
        """
        super().__init__()

        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_outputs = num_outputs

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # Compute LSTM output dimension
        lstm_output_dim = lstm_hidden * (2 if bidirectional else 1)

        # Optional attention layer
        if use_attention:
            self.attention = MultiHeadAttention(
                d_model=lstm_output_dim,
                num_heads=num_attention_heads,
                dropout=dropout
            )
            attention_dim = lstm_output_dim
        else:
            self.attention = None
            attention_dim = lstm_output_dim

        # Output projection layers
        self.fc_layers = nn.Sequential(
            nn.Linear(attention_dim, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_outputs)
        )

        # Initialize output layer weights
        nn.init.xavier_uniform_(self.fc_layers[-1].weight)
        nn.init.zeros_(self.fc_layers[-1].bias)

    def forward(self, encoder_output: torch.Tensor, mask: torch.Tensor = None) -> dict:
        """
        Forward pass

        Args:
            encoder_output: Encoded tensor [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Dictionary with trajectory corrections
        """
        # Pass through LSTM
        lstm_output, _ = self.lstm(encoder_output)  # [batch, seq_len, lstm_output_dim]

        # Apply attention if enabled
        if self.use_attention:
            # Self-attention on LSTM output
            attended_output = self.attention(lstm_output, lstm_output, lstm_output, mask)
        else:
            attended_output = lstm_output

        # Use last timestep output for prediction
        last_output = attended_output[:, -1, :]  # [batch, attention_dim]

        # Project to output dimension
        corrections = self.fc_layers(last_output)  # [batch, num_outputs]

        # Split into individual corrections
        outputs = {
            'displacement_x': corrections[:, 0:1],  # X correction
            'displacement_y': corrections[:, 1:2],  # Y correction
            'displacement_z': corrections[:, 2:3],  # Z correction
        }

        return outputs

    def get_output_dim(self) -> int:
        """
        Get output dimension

        Returns:
            Number of outputs
        """
        return self.num_outputs


class TrajectorySequenceDecoder(nn.Module):
    """
    Sequence-based trajectory decoder that predicts corrections for each time step

    Useful for detailed trajectory optimization
    """

    def __init__(self,
                 d_model: int,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 bidirectional: bool = True,
                 use_attention: bool = True,
                 num_attention_heads: int = 8,
                 dropout: float = 0.1,
                 num_outputs: int = 3):
        """
        Initialize sequence trajectory decoder

        Args:
            d_model: Input dimension (from encoder)
            lstm_hidden: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            bidirectional: Use bidirectional LSTM
            use_attention: Use attention mechanism
            num_attention_heads: Number of attention heads
            dropout: Dropout rate
            num_outputs: Number of correction outputs per time step
        """
        super().__init__()

        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_outputs = num_outputs

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # Compute LSTM output dimension
        lstm_output_dim = lstm_hidden * (2 if bidirectional else 1)

        # Optional attention layer
        if use_attention:
            self.attention = MultiHeadAttention(
                d_model=lstm_output_dim,
                num_heads=num_attention_heads,
                dropout=dropout
            )
            attention_dim = lstm_output_dim
        else:
            self.attention = None
            attention_dim = lstm_output_dim

        # Output projection (applied to each time step)
        self.output_projection = nn.Linear(attention_dim, num_outputs)

        # Initialize output layer weights
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, encoder_output: torch.Tensor, mask: torch.Tensor = None) -> dict:
        """
        Forward pass

        Args:
            encoder_output: Encoded tensor [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Dictionary with sequence trajectory corrections [batch, seq_len, num_outputs]
        """
        # Pass through LSTM
        lstm_output, _ = self.lstm(encoder_output)  # [batch, seq_len, lstm_output_dim]

        # Apply attention if enabled
        if self.use_attention:
            # Self-attention on LSTM output
            attended_output = self.attention(lstm_output, lstm_output, lstm_output, mask)
        else:
            attended_output = lstm_output

        # Project to output dimension for each time step
        corrections = self.output_projection(attended_output)  # [batch, seq_len, num_outputs]

        # Split into individual corrections
        outputs = {
            'displacement_x_seq': corrections[:, :, 0:1],  # X correction sequence
            'displacement_y_seq': corrections[:, :, 1:2],  # Y correction sequence
            'displacement_z_seq': corrections[:, :, 2:3],  # Z correction sequence
        }

        return outputs
