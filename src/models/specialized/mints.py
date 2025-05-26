"""MINTS (Multi-Input Neural Time Series) model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional

class MINTS(nn.Module):
    """Multi-Input Neural Time Series model for time series prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MINTS model.
        
        Args:
            config: Model configuration dictionary containing:
                - input_size: Size of input features
                - hidden_dim: Dimension of hidden layers
                - num_layers: Number of transformer layers
                - num_heads: Number of attention heads
                - dropout: Dropout rate
                - use_residual: Whether to use residual connections
                - use_layer_norm: Whether to use layer normalization
                - max_seq_length: Maximum sequence length
        """
        super().__init__()
        
        self.input_size = config['input_size']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']
        self.use_residual = config['use_residual']
        self.use_layer_norm = config['use_layer_norm']
        self.max_seq_length = config['max_seq_length']
        
        # Input projection
        self.input_proj = nn.Linear(self.input_size, self.hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.max_seq_length)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.input_size)
        
        # Layer normalization
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.hidden_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            mask: Optional mask tensor of shape (batch_size, seq_length)
            
        Returns:
            Output tensor of shape (batch_size, seq_length, input_size)
        """
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        if mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            x = self.transformer_encoder(x)
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            x = self.layer_norm(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
        """
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            
        Returns:
            Output tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1)] 