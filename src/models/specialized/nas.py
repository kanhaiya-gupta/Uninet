import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Union
from ..base import BaseNeuralNetwork

class NASCell(nn.Module):
    """Neural Architecture Search cell."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        operations: List[str] = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Default operations if none provided
        if operations is None:
            operations = [
                'max_pool_3x3',
                'avg_pool_3x3',
                'skip_connect',
                'sep_conv_3x3',
                'sep_conv_5x5',
                'dil_conv_3x3',
                'dil_conv_5x5'
            ]
        
        # Create operation modules
        self.ops = nn.ModuleDict()
        for op_name in operations:
            self.ops[op_name] = self._create_operation(op_name)
        
        # Architecture parameters (softmax temperature)
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Initialize architecture weights
        self.arch_weights = nn.Parameter(torch.ones(len(operations)) / len(operations))
    
    def _create_operation(self, op_name: str) -> nn.Module:
        """Create operation module based on name."""
        if op_name == 'max_pool_3x3':
            return nn.Sequential(
                nn.MaxPool2d(3, stride=self.stride, padding=1),
                nn.BatchNorm2d(self.in_channels)
            )
        elif op_name == 'avg_pool_3x3':
            return nn.Sequential(
                nn.AvgPool2d(3, stride=self.stride, padding=1),
                nn.BatchNorm2d(self.in_channels)
            )
        elif op_name == 'skip_connect':
            return nn.Identity() if self.stride == 1 else nn.Sequential(
                nn.AvgPool2d(self.stride, stride=self.stride),
                nn.BatchNorm2d(self.in_channels)
            )
        elif op_name == 'sep_conv_3x3':
            return nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, 3, stride=self.stride, padding=1, groups=self.in_channels),
                nn.Conv2d(self.in_channels, self.out_channels, 1),
                nn.BatchNorm2d(self.out_channels)
            )
        elif op_name == 'sep_conv_5x5':
            return nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, 5, stride=self.stride, padding=2, groups=self.in_channels),
                nn.Conv2d(self.in_channels, self.out_channels, 1),
                nn.BatchNorm2d(self.out_channels)
            )
        elif op_name == 'dil_conv_3x3':
            return nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, 3, stride=self.stride, padding=2, dilation=2, groups=self.in_channels),
                nn.Conv2d(self.in_channels, self.out_channels, 1),
                nn.BatchNorm2d(self.out_channels)
            )
        elif op_name == 'dil_conv_5x5':
            return nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, 5, stride=self.stride, padding=4, dilation=2, groups=self.in_channels),
                nn.Conv2d(self.in_channels, self.out_channels, 1),
                nn.BatchNorm2d(self.out_channels)
            )
        else:
            raise ValueError(f"Unknown operation: {op_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute operation weights using softmax
        weights = F.softmax(self.arch_weights / self.temperature, dim=0)
        
        # Apply each operation and sum the results
        output = 0
        for i, (op_name, op) in enumerate(self.ops.items()):
            output += weights[i] * op(x)
        
        return output

class NAS(BaseNeuralNetwork):
    """Neural Architecture Search implementation with configurable architecture."""
    
    def _build_model(self) -> nn.Module:
        """Build the NAS architecture based on configuration."""
        # Get configuration parameters
        input_channels = self.config.get('input_channels', 3)
        num_cells = self.config.get('num_cells', 8)
        num_operations = self.config.get('num_operations', 4)
        stem_channels = self.config.get('stem_channels', 64)
        operations = self.config.get('operations', None)
        
        # Stem layers
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, stem_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(stem_channels)
        )
        
        # Build cells
        self.cells = nn.ModuleList()
        current_channels = stem_channels
        
        for i in range(num_cells):
            # Determine stride for this cell
            stride = 2 if i in [num_cells//3, 2*num_cells//3] else 1
            
            # Double channels at stride=2
            out_channels = current_channels * 2 if stride == 2 else current_channels
            
            # Create cell
            cell = NASCell(
                current_channels,
                out_channels,
                stride,
                operations
            )
            self.cells.append(cell)
            
            current_channels = out_channels
        
        # Final layers
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(current_channels, self.config.get('output_size', 10))
        )
        
        return nn.Module()  # Placeholder, we'll use components directly
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        x = self.stem(x)
        
        for cell in self.cells:
            x = cell(x)
        
        return self.final(x)
    
    def get_architecture(self) -> Dict[str, List[str]]:
        """Get the current architecture based on learned weights."""
        architecture = {}
        
        for i, cell in enumerate(self.cells):
            # Get operation weights
            weights = F.softmax(cell.arch_weights / cell.temperature, dim=0)
            
            # Get top operations
            top_ops = []
            for j, (op_name, _) in enumerate(cell.ops.items()):
                if weights[j] > 0.1:  # Threshold for considering an operation
                    top_ops.append((op_name, weights[j].item()))
            
            # Sort by weight
            top_ops.sort(key=lambda x: x[1], reverse=True)
            architecture[f'cell_{i}'] = [op for op, _ in top_ops]
        
        return architecture
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(name.lower(), nn.ReLU()) 