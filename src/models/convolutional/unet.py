import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from ..base import BaseNeuralNetwork

class DoubleConv(nn.Module):
    """Double convolution block for UNet."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = 'relu',
        use_batch_norm: bool = True
    ):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
            self._get_activation(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
            self._get_activation(activation)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)
    
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

class UNet(BaseNeuralNetwork):
    """UNet implementation with configurable architecture."""
    
    def _build_model(self) -> nn.Module:
        """Build the UNet architecture based on configuration."""
        # Get configuration parameters
        input_channels = self.config.get('input_channels', 3)
        output_channels = self.config.get('output_channels', 1)
        base_channels = self.config.get('base_channels', 64)
        depth = self.config.get('depth', 4)
        activation = self.config.get('activation', 'relu')
        use_batch_norm = self.config.get('use_batch_norm', True)
        bilinear = self.config.get('bilinear', True)
        
        # Calculate channel sizes for each level
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]
        
        # Downsampling path
        self.inc = DoubleConv(input_channels, channels[0], activation, use_batch_norm)
        self.down_path = nn.ModuleList()
        
        for i in range(depth):
            self.down_path.append(nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(channels[i], channels[i + 1], activation, use_batch_norm)
            ))
        
        # Upsampling path
        self.up_path = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            up = nn.Sequential(
                nn.ConvTranspose2d(
                    channels[i + 1],
                    channels[i],
                    kernel_size=2,
                    stride=2
                ) if not bilinear else nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(channels[i + 1], channels[i], kernel_size=1)
                ),
                DoubleConv(channels[i] * 2, channels[i], activation, use_batch_norm)
            )
            self.up_path.append(up)
        
        # Output layer
        self.outc = nn.Conv2d(channels[0], output_channels, kernel_size=1)
        
        # Add final activation based on task type
        if self.config['task_type'] == 'segmentation':
            self.final_activation = nn.Sigmoid()
        elif self.config['task_type'] == 'regression':
            self.final_activation = nn.Identity()
        
        return nn.Module()  # Placeholder, we'll use the layers directly
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        # Downsampling path
        features = []
        x = self.inc(x)
        features.append(x)
        
        for down in self.down_path:
            x = down(x)
            features.append(x)
        
        # Remove the last feature as it's not needed for upsampling
        features = features[:-1]
        
        # Upsampling path
        for i, up in enumerate(self.up_path):
            x = up(x)
            x = torch.cat([x, features[-(i + 1)]], dim=1)
        
        # Output layer
        x = self.outc(x)
        x = self.final_activation(x)
        
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