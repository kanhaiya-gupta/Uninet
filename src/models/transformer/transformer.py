import torch
import torch.nn as nn
import math
from typing import Dict, Any, List, Tuple, Optional
from ..base import BaseNeuralNetwork

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional encoding to input
        return x + self.pe[:x.size(1)]

class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = residual + self.dropout(x)
        
        # Feed-forward network
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        
        return x

class Transformer(BaseNeuralNetwork):
    """Transformer implementation with configurable architecture."""
    
    def _build_model(self) -> nn.Module:
        """Build the transformer architecture based on configuration."""
        # Get configuration parameters
        input_size = self.config.get('input_size', 1)
        d_model = self.config.get('d_model', 512)
        num_heads = self.config.get('num_heads', 8)
        num_layers = self.config.get('num_layers', 6)
        d_ff = self.config.get('d_ff', 2048)
        output_size = self.config.get('output_size', 1)
        dropout = self.config.get('dropout', 0.1)
        max_seq_length = self.config.get('max_seq_length', 5000)
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, output_size)
        
        # Add final activation based on task type
        if self.config['task_type'] == 'classification':
            self.final_activation = nn.Softmax(dim=-1)
        elif self.config['task_type'] == 'regression':
            self.final_activation = nn.Identity()
        
        return nn.Module()  # Placeholder, we'll use the layers directly
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the network."""
        # x shape: (batch_size, sequence_length, input_size)
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        
        # Output projection
        x = self.output_proj(x)
        x = self.final_activation(x)
        
        return x
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU()
        }
        return activations.get(name.lower(), nn.ReLU()) 