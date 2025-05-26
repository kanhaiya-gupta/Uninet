import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Union
from ..base import BaseNeuralNetwork

class NTKLayer(nn.Module):
    """Neural Tangent Kernel layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        activation: str = 'relu'
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights with proper scaling
        self.weight = nn.Parameter(
            torch.randn(in_features, out_features) / math.sqrt(in_features)
        )
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.activation = self._get_activation(activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear transformation
        output = F.linear(x, self.weight, self.bias)
        
        # Apply activation
        if self.activation is not None:
            output = self.activation(output)
        
        return output
    
    def _get_activation(self, name: str) -> Optional[nn.Module]:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'none': None
        }
        return activations.get(name.lower(), nn.ReLU())

class NTK(BaseNeuralNetwork):
    """Neural Tangent Kernel implementation with configurable architecture."""
    
    def _build_model(self) -> nn.Module:
        """Build the NTK architecture based on configuration."""
        # Get configuration parameters
        input_size = self.config.get('input_size', 1)
        hidden_sizes = self.config.get('hidden_sizes', [64, 32])
        output_size = self.config.get('output_size', 1)
        activation = self.config.get('activation', 'relu')
        use_bias = self.config.get('use_bias', True)
        width = self.config.get('width', 1000)  # Width of the network
        
        # Build layers
        layers = []
        current_size = input_size
        
        # First layer
        layers.append(NTKLayer(current_size, width, use_bias, activation))
        current_size = width
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(NTKLayer(current_size, hidden_size, use_bias, activation))
            current_size = hidden_size
        
        # Output layer
        layers.append(NTKLayer(current_size, output_size, use_bias, 'none'))
        
        self.network = nn.Sequential(*layers)
        
        return nn.Module()  # Placeholder, we'll use network directly
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        return self.network(x)
    
    def compute_ntk(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the Neural Tangent Kernel.
        
        Args:
            x1: First input tensor [batch_size1, input_size]
            x2: Second input tensor [batch_size2, input_size]. If None, x2 = x1.
        
        Returns:
            NTK matrix [batch_size1, batch_size2]
        """
        if x2 is None:
            x2 = x1
        
        # Initialize kernel matrix
        batch_size1 = x1.size(0)
        batch_size2 = x2.size(0)
        kernel = torch.zeros(batch_size1, batch_size2, device=x1.device)
        
        # Compute kernel for each layer
        for layer in self.network:
            if isinstance(layer, NTKLayer):
                # Forward pass
                x1_out = layer(x1)
                x2_out = layer(x2)
                
                # Compute kernel contribution
                if layer.activation is None:
                    # Linear layer
                    kernel += torch.matmul(x1_out, x2_out.t())
                else:
                    # Non-linear layer
                    kernel += torch.matmul(x1_out, x2_out.t()) * (
                        torch.matmul(x1, x2.t()) > 0
                    ).float()
                
                # Update inputs for next layer
                x1 = x1_out
                x2 = x2_out
        
        return kernel
    
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