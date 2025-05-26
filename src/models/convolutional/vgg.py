import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from ..base import BaseNeuralNetwork

class VGG(BaseNeuralNetwork):
    """VGG implementation with configurable architecture."""
    
    def _build_model(self) -> nn.Module:
        """Build the VGG architecture based on configuration."""
        # Get configuration parameters
        input_channels = self.config.get('input_channels', 3)
        vgg_type = self.config.get('vgg_type', 'vgg16')  # 'vgg11', 'vgg13', 'vgg16', 'vgg19'
        num_classes = self.config.get('output_size', 10)
        activation = self.config.get('activation', 'relu')
        use_batch_norm = self.config.get('use_batch_norm', True)
        use_dropout = self.config.get('use_dropout', True)
        dropout_rate = self.config.get('dropout_rate', 0.5)
        
        # VGG configurations
        vgg_configs = {
            'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        }
        
        config = vgg_configs.get(vgg_type, vgg_configs['vgg16'])
        
        # Build features
        features: List[nn.Module] = []
        in_channels = input_channels
        
        for v in config:
            if v == 'M':
                features.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                features.append(conv)
                if use_batch_norm:
                    features.append(nn.BatchNorm2d(v))
                features.append(self._get_activation(activation))
                in_channels = v
        
        # Build classifier
        classifier: List[nn.Module] = []
        classifier.append(nn.AdaptiveAvgPool2d((7, 7)))
        classifier.append(nn.Flatten())
        
        # Calculate the size of the flattened features
        # This is a placeholder - actual size will be calculated during forward pass
        self._flatten_size = None
        
        # Fully connected layers
        if self._flatten_size is None:
            self._flatten_size = 512 * 7 * 7  # Default size for VGG
        
        classifier.extend([
            nn.Linear(self._flatten_size, 4096),
            self._get_activation(activation),
            nn.Dropout(dropout_rate) if use_dropout else nn.Identity(),
            nn.Linear(4096, 4096),
            self._get_activation(activation),
            nn.Dropout(dropout_rate) if use_dropout else nn.Identity(),
            nn.Linear(4096, num_classes)
        ])
        
        # Add final activation based on task type
        if self.config['task_type'] == 'classification':
            classifier.append(nn.Softmax(dim=1))
        elif self.config['task_type'] == 'regression':
            pass  # No activation for regression
        
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(*classifier)
        
        return nn.Module()  # Placeholder, we'll use features and classifier directly
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU()
        }
        return activations.get(name.lower(), nn.ReLU()) 