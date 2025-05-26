import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from ..base import BaseNeuralNetwork

class CNN(BaseNeuralNetwork):
    """Convolutional Neural Network implementation with configurable architecture."""
    
    def _build_model(self) -> nn.Module:
        """Build the CNN architecture based on configuration."""
        # Get configuration parameters
        input_channels = self.config.get('input_channels', 3)
        conv_layers = self.config.get('conv_layers', [
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ])
        fc_layers = self.config.get('fc_layers', [512, 256])
        output_size = self.config.get('output_size', 10)
        activation = self.config.get('activation', 'relu')
        use_batch_norm = self.config.get('use_batch_norm', True)
        use_dropout = self.config.get('use_dropout', True)
        dropout_rate = self.config.get('dropout_rate', 0.2)
        pool_type = self.config.get('pool_type', 'max')  # 'max' or 'avg'
        
        layers: List[nn.Module] = []
        current_channels = input_channels
        
        # Convolutional layers
        for conv_config in conv_layers:
            # Conv layer
            layers.append(nn.Conv2d(
                current_channels,
                conv_config['out_channels'],
                kernel_size=conv_config['kernel_size'],
                stride=conv_config['stride'],
                padding=conv_config['padding']
            ))
            
            # Activation
            layers.append(self._get_activation(activation))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(conv_config['out_channels']))
            
            # Pooling
            if pool_type == 'max':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            
            current_channels = conv_config['out_channels']
        
        # Flatten layer
        layers.append(nn.Flatten())
        
        # Calculate the size of the flattened features
        # This is a placeholder - actual size will be calculated during forward pass
        self._flatten_size = None
        
        # Fully connected layers
        current_size = self._flatten_size
        for fc_size in fc_layers:
            layers.append(nn.Linear(current_size, fc_size))
            layers.append(self._get_activation(activation))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(fc_size))
            
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
            
            current_size = fc_size
        
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
        # Calculate the size of flattened features if not already done
        if self._flatten_size is None:
            self._flatten_size = self._calculate_flatten_size(x)
            # Rebuild the model with the correct flatten size
            self.model = self._build_model()
            self.model = self.model.to(self.device)
        
        return self.model(x)
    
    def _calculate_flatten_size(self, x: torch.Tensor) -> int:
        """Calculate the size of flattened features after convolutional layers."""
        # Create a temporary model with just the convolutional layers
        temp_layers: List[nn.Module] = []
        current_channels = self.config.get('input_channels', 3)
        
        for conv_config in self.config.get('conv_layers', []):
            temp_layers.append(nn.Conv2d(
                current_channels,
                conv_config['out_channels'],
                kernel_size=conv_config['kernel_size'],
                stride=conv_config['stride'],
                padding=conv_config['padding']
            ))
            temp_layers.append(self._get_activation(self.config.get('activation', 'relu')))
            
            if self.config.get('use_batch_norm', True):
                temp_layers.append(nn.BatchNorm2d(conv_config['out_channels']))
            
            if self.config.get('pool_type', 'max') == 'max':
                temp_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                temp_layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            
            current_channels = conv_config['out_channels']
        
        temp_model = nn.Sequential(*temp_layers)
        temp_model = temp_model.to(self.device)
        
        # Forward pass to get the output size
        with torch.no_grad():
            output = temp_model(x)
            return output.numel() // output.size(0)
    
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