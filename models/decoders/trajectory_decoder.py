"""
Trajectory Error Correction Decoder Head

Predicts EXPLICIT TRAJECTORY ERRORS that can be corrected in real-time:
- X-axis deviation (X轴偏差)
- Y-axis deviation (Y轴偏差)

These corrections are applied to the commanded trajectory to compensate
for mechanical inaccuracies and dynamic effects during printing.
"""

import torch
import torch.nn as nn


class TrajectoryCorrectionHead(nn.Module):
    """
    Trajectory correction head for predicting spatial deviations
    
    This module predicts real-time trajectory corrections needed to compensate
    for mechanical inaccuracies, resonance effects, and dynamic disturbances
    during 3D printing. Unlike quality prediction (implicit parameters), 
    trajectory errors are explicit geometric deviations that can be directly 
    corrected by adjusting the commanded positions.
    
    Outputs (Trajectory Corrections):
    - error_x: Deviation in X direction (mm)
    - error_y: Deviation in Y direction (mm)
    
    Key Features:
    - Processes sequential trajectory data to capture temporal dependencies
    - Uses LSTM for modeling dynamic system response
    - Includes attention mechanism for focusing on critical time steps
    - Handles variable-length sequences for real-time applications
    """

    def __init__(self, 
                 d_model: int,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 bidirectional: bool = True,
                 use_attention: bool = True,
                 num_outputs: int = 2):
        """
        Initialize trajectory correction head
        
        Args:
            d_model: Input dimension (from encoder)
            lstm_hidden: LSTM hidden size
            lstm_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention mechanism
            num_outputs: Number of trajectory outputs (default: 2 for x, y)
        """
        super().__init__()
        
        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_outputs = num_outputs
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if lstm_layers > 1 else 0
        )
        
        # Attention mechanism if enabled
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_hidden * (2 if bidirectional else 1),
                num_heads=8,
                batch_first=True
            )
            self.attn_norm = nn.LayerNorm(lstm_hidden * (2 if bidirectional else 1))
        
        # Output projection
        lstm_output_dim = lstm_hidden * (2 if bidirectional else 1)
        self.output_proj = nn.Linear(lstm_output_dim, num_outputs)
        
        # Initialize output layer weights
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, encoder_output: torch.Tensor, mask=None) -> dict:
        """
        Forward pass
        
        Args:
            encoder_output: Encoded tensor [batch, seq_len, d_model]
            mask: Attention mask [batch, seq_len] or [batch, seq_len, seq_len]
            
        Returns:
            Dictionary with trajectory correction predictions
        """
        batch_size, seq_len, _ = encoder_output.shape
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(encoder_output)  # [batch, seq_len, lstm_hidden*directions]
        
        # Apply attention if enabled
        if self.use_attention:
            # Self-attention on LSTM outputs
            attn_out, attn_weights = self.attention(
                lstm_out, lstm_out, lstm_out,
                attn_mask=self._create_causal_mask(seq_len, encoder_output.device) if mask is None else mask
            )
            # Add & Norm
            lstm_out = self.attn_norm(attn_out + lstm_out)
        
        # Average over sequence dimension to get fixed-size representation
        # This allows handling variable-length sequences
        seq_axis = 1
        pooled_output = lstm_out.mean(dim=seq_axis)  # [batch, lstm_hidden*directions]
        
        # Project to output space
        corrections = self.output_proj(pooled_output)  # [batch, num_outputs]
        
        # Split into individual outputs (TRAJECTORY CORRECTIONS)
        outputs = {
            'error_x': corrections[:, 0:1],  # X-axis deviation (mm)
            'error_y': corrections[:, 1:2],  # Y-axis deviation (mm)
        }
        
        # Add attention weights if attention is used (for visualization)
        if self.use_attention:
            outputs['attn_weights'] = attn_weights
            
        return outputs

    def _create_causal_mask(self, seq_len, device):
        """Create causal mask for attention mechanism"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

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
