import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, List, Tuple, Optional, Union
from ..base import BaseNeuralNetwork

class PositionalEncoding(nn.Module):
    """Positional encoding for input coordinates."""
    
    def __init__(self, num_frequencies: int = 10):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.freq_bands = 2.0 ** torch.linspace(0.0, num_frequencies - 1, num_frequencies)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., 3] -> [..., 3 * (2 * num_frequencies + 1)]
        x_expanded = x.unsqueeze(-2)  # [..., 1, 3]
        proj = (2.0 * math.pi * x_expanded) * self.freq_bands.view(-1, 1)  # [..., num_frequencies, 3]
        
        # Compute sin and cos
        sin_proj = torch.sin(proj)  # [..., num_frequencies, 3]
        cos_proj = torch.cos(proj)  # [..., num_frequencies, 3]
        
        # Concatenate
        output = torch.cat([sin_proj, cos_proj], dim=-1)  # [..., num_frequencies, 6]
        output = output.reshape(*x.shape[:-1], -1)  # [..., num_frequencies * 6]
        
        # Concatenate with original input
        return torch.cat([x, output], dim=-1)

class NeRFBlock(nn.Module):
    """NeRF network block."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        use_skip: bool = True
    ):
        super().__init__()
        self.use_skip = use_skip
        
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features)
        )
        
        if use_skip:
            self.skip = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_skip:
            return self.net(x) + self.skip(x)
        return self.net(x)

class NeRF(BaseNeuralNetwork):
    """Neural Radiance Fields implementation with configurable architecture."""
    
    def _build_model(self) -> nn.Module:
        """Build the NeRF architecture based on configuration."""
        # Get configuration parameters
        num_frequencies = self.config.get('num_frequencies', 10)
        hidden_features = self.config.get('hidden_features', 256)
        use_skip = self.config.get('use_skip', True)
        num_samples = self.config.get('num_samples', 64)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(num_frequencies)
        
        # Input features after positional encoding
        input_features = 3 * (2 * num_frequencies + 1)  # 3D coordinates
        view_features = 3 * (2 * num_frequencies + 1)  # 3D view direction
        
        # NeRF blocks
        self.xyz_encoding = NeRFBlock(
            input_features,
            hidden_features,
            hidden_features,
            use_skip
        )
        
        self.dir_encoding = NeRFBlock(
            view_features + hidden_features,
            hidden_features // 2,
            4,  # RGB + density
            use_skip
        )
        
        self.num_samples = num_samples
        
        return nn.Module()  # Placeholder, we'll use components directly
    
    def forward(
        self,
        xyz: torch.Tensor,
        view_dir: torch.Tensor,
        return_density: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the network.
        
        Args:
            xyz: 3D coordinates [batch_size, num_samples, 3]
            view_dir: View direction [batch_size, 3]
            return_density: Whether to return density values
        
        Returns:
            RGB values [batch_size, num_samples, 3] and optionally density [batch_size, num_samples]
        """
        # Encode inputs
        xyz_encoded = self.pos_encoding(xyz)  # [batch_size, num_samples, input_features]
        view_dir_encoded = self.pos_encoding(view_dir)  # [batch_size, view_features]
        
        # Expand view direction to match xyz samples
        view_dir_encoded = view_dir_encoded.unsqueeze(1).expand(-1, self.num_samples, -1)
        
        # Process through network
        xyz_features = self.xyz_encoding(xyz_encoded)
        xyz_view = torch.cat([xyz_features, view_dir_encoded], dim=-1)
        output = self.dir_encoding(xyz_view)
        
        # Split into RGB and density
        rgb = torch.sigmoid(output[..., :3])
        density = F.softplus(output[..., 3])
        
        if return_density:
            return rgb, density
        return rgb
    
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