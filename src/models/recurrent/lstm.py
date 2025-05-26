import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from ..base import BaseNeuralNetwork

class LSTM(BaseNeuralNetwork):
    """Long Short-Term Memory (LSTM) implementation with configurable architecture."""
    
    def _build_model(self) -> nn.Module:
        """Build the LSTM architecture based on configuration."""
        # Get configuration parameters
        input_size = self.config.get('input_size', 1)
        hidden_size = self.config.get('hidden_size', 64)
        num_layers = self.config.get('num_layers', 2)
        output_size = self.config.get('output_size', 1)
        bidirectional = self.config.get('bidirectional', False)
        dropout = self.config.get('dropout', 0.2)
        use_attention = self.config.get('use_attention', False)
        
        # Build LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Calculate the size of the output features
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Build attention layer if enabled
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_size, lstm_output_size),
                nn.Tanh(),
                nn.Linear(lstm_output_size, 1)
            )
        
        # Build fully connected layer
        self.fc = nn.Linear(lstm_output_size, output_size)
        
        # Add final activation based on task type
        if self.config['task_type'] == 'classification':
            self.final_activation = nn.Softmax(dim=-1)
        elif self.config['task_type'] == 'regression':
            self.final_activation = nn.Identity()
        
        return nn.Module()  # Placeholder, we'll use lstm and fc directly
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(
            self.lstm.num_layers * (2 if self.lstm.bidirectional else 1),
            batch_size,
            self.lstm.hidden_size
        ).to(self.device)
        c0 = torch.zeros_like(h0)
        
        # LSTM forward pass
        output, (_, _) = self.lstm(x, (h0, c0))
        
        # Apply attention if enabled
        if hasattr(self, 'attention'):
            # Calculate attention weights
            attention_weights = self.attention(output)
            attention_weights = torch.softmax(attention_weights, dim=1)
            
            # Apply attention weights
            output = torch.sum(output * attention_weights, dim=1)
        else:
            # Get the last time step output
            output = output[:, -1, :]
        
        # Fully connected layer
        out = self.fc(output)
        out = self.final_activation(out)
        
        return out
    
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