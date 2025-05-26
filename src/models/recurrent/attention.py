import torch
import torch.nn as nn
import math
from typing import Dict, Any, List, Tuple, Optional
from ..base import BaseNeuralNetwork

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear projections and reshape
        q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights
        output = torch.matmul(attention_weights, v)
        
        # Reshape and apply final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output

class Attention(BaseNeuralNetwork):
    """Attention mechanism implementation with configurable architecture."""
    
    def _build_model(self) -> nn.Module:
        """Build the attention architecture based on configuration."""
        # Get configuration parameters
        input_size = self.config.get('input_size', 1)
        d_model = self.config.get('d_model', 512)
        num_heads = self.config.get('num_heads', 8)
        num_layers = self.config.get('num_layers', 6)
        output_size = self.config.get('output_size', 1)
        dropout = self.config.get('dropout', 0.1)
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Position-wise feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
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
        
        # Self-attention
        residual = x
        x = self.layer_norm1(x)
        x = self.attention(x, x, x, mask)
        x = residual + x
        
        # Feed-forward network
        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = residual + x
        
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