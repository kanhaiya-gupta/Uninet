"""MINTS (Multi-Input Neural Time Series) experiment implementation."""

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
                - hidden_layers: List of hidden layer sizes
                - output_size: Size of output features
                - activation: Activation function to use
                - dropout: Dropout rate
                - use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_size = config['input_size']
        self.hidden_layers = config['hidden_layers']
        self.output_size = config['output_size']
        self.activation = getattr(F, config['activation'])
        self.dropout = config['dropout']
        self.use_batch_norm = config['use_batch_norm']
        
        # Build layers
        layers = []
        prev_size = self.input_size
        
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(self.dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, self.output_size))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if isinstance(layer, nn.Linear) and i < len(self.layers) - 1:
                x = self.activation(x)
        
        return x 