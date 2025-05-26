import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional, Callable
from ..base import BaseNeuralNetwork

class BasicBlock(nn.Module):
    """Basic block for ResNet."""
    
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        activation: str = 'relu',
        use_batch_norm: bool = True
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.activation = self._get_activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.activation(out)
        
        return out
    
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

class ResNet(BaseNeuralNetwork):
    """ResNet implementation with configurable architecture."""
    
    def _build_model(self) -> nn.Module:
        """Build the ResNet architecture based on configuration."""
        # Get configuration parameters
        input_channels = self.config.get('input_channels', 3)
        block = BasicBlock
        layers = self.config.get('layers', [2, 2, 2, 2])  # ResNet-18 default
        num_classes = self.config.get('output_size', 10)
        activation = self.config.get('activation', 'relu')
        use_batch_norm = self.config.get('use_batch_norm', True)
        
        self.in_channels = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        self.activation = self._get_activation(activation)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], activation=activation, use_batch_norm=use_batch_norm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, activation=activation, use_batch_norm=use_batch_norm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, activation=activation, use_batch_norm=use_batch_norm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, activation=activation, use_batch_norm=use_batch_norm)
        
        # Global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Add final activation based on task type
        if self.config['task_type'] == 'classification':
            self.final_activation = nn.Softmax(dim=1)
        elif self.config['task_type'] == 'regression':
            self.final_activation = nn.Identity()
        
        return nn.Module()  # Placeholder, we'll use the layers directly
    
    def _make_layer(
        self,
        block: nn.Module,
        out_channels: int,
        blocks: int,
        stride: int = 1,
        activation: str = 'relu',
        use_batch_norm: bool = True
    ) -> nn.Sequential:
        """Create a layer of residual blocks."""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion) if use_batch_norm else nn.Identity()
            )
        
        layers = []
        layers.append(block(
            self.in_channels,
            out_channels,
            stride,
            downsample,
            activation=activation,
            use_batch_norm=use_batch_norm
        ))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                activation=activation,
                use_batch_norm=use_batch_norm
            ))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
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