import torch
import torch.nn as nn
from typing import Dict, Any, List
from ..base import BaseNeuralNetwork

class MLP(BaseNeuralNetwork):
    """Multi-Layer Perceptron implementation with advanced features."""
    
    def _build_model(self) -> nn.Module:
        """Build the MLP architecture based on configuration."""
        layers: List[nn.Module] = []
        
        # Get configuration parameters
        input_size = self.config.get('input_size', 784)
        hidden_sizes = self.config.get('hidden_sizes', [512, 256, 128])
        output_size = self.config.get('output_size', 10)
        activation = self.config.get('activation', 'relu')
        use_batch_norm = self.config.get('use_batch_norm', True)
        use_dropout = self.config.get('use_dropout', True)
        dropout_rate = self.config.get('dropout_rate', 0.2)
        
        # Input layer
        current_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(self._get_activation(activation))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
            
            current_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(current_size, output_size))
        
        # Add final activation based on task type
        if self.config['task_type'] == 'classification':
            layers.append(nn.Softmax(dim=1))
        elif self.config['task_type'] == 'regression':
            pass  # No activation for regression
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        return self.model(x)
    
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