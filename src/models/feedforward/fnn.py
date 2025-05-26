import torch
import torch.nn as nn
from typing import Dict, Any, List
from ..base import BaseNeuralNetwork

class FNN(BaseNeuralNetwork):
    """Feedforward Neural Network implementation."""
    
    def _build_model(self) -> nn.Module:
        """Build the FNN architecture based on configuration."""
        layers: List[nn.Module] = []
        
        # Input layer
        input_size = self.config.get('input_size', 784)  # Default for MNIST
        hidden_layers = self.config.get('hidden_layers', 2)
        neurons_per_layer = self.config.get('neurons_per_layer', 64)
        output_size = self.config.get('output_size', 10)  # Default for MNIST
        activation = self.config.get('activation', 'relu')
        
        # Add hidden layers
        current_size = input_size
        for _ in range(hidden_layers):
            layers.extend([
                nn.Linear(current_size, neurons_per_layer),
                self._get_activation(activation),
                nn.BatchNorm1d(neurons_per_layer),
                nn.Dropout(self.config.get('dropout', 0.2))
            ])
            current_size = neurons_per_layer
        
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
            'elu': nn.ELU()
        }
        return activations.get(name.lower(), nn.ReLU()) 