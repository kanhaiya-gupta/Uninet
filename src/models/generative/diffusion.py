import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from ..base import BaseNeuralNetwork

class TimeEmbedding(nn.Module):
    """Time embedding for diffusion models."""
    
    def __init__(self, time_dim: int, hidden_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(t)

class ResidualBlock(nn.Module):
    """Residual block with time embedding."""
    
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, use_batch_norm: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.act = nn.SiLU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # First convolution
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act(h)
        
        # Time embedding
        t = self.time_mlp(t)
        t = t.view(t.size(0), -1, 1, 1)
        h = h + t
        
        # Second convolution
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.act(h)
        
        # Shortcut connection
        return h + self.shortcut(x)

class UNet(nn.Module):
    """U-Net architecture for diffusion models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # Get configuration parameters
        self.input_channels = config.get('input_channels', 3)
        self.hidden_dims = config.get('hidden_dims', [64, 128, 256, 512])
        self.time_dim = config.get('time_dim', 256)
        self.use_batch_norm = config.get('use_batch_norm', True)
        
        # Time embedding
        self.time_embedding = TimeEmbedding(1, self.time_dim)
        
        # Initial convolution
        self.init_conv = nn.Conv2d(self.input_channels, self.hidden_dims[0], kernel_size=3, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1):
            self.down_blocks.append(
                ResidualBlock(
                    self.hidden_dims[i],
                    self.hidden_dims[i + 1],
                    self.time_dim,
                    self.use_batch_norm
                )
            )
        
        # Middle block
        self.middle_block = ResidualBlock(
            self.hidden_dims[-1],
            self.hidden_dims[-1],
            self.time_dim,
            self.use_batch_norm
        )
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1, 0, -1):
            self.up_blocks.append(
                ResidualBlock(
                    self.hidden_dims[i] * 2,  # *2 for skip connection
                    self.hidden_dims[i - 1],
                    self.time_dim,
                    self.use_batch_norm
                )
            )
        
        # Final convolution
        self.final_conv = nn.Conv2d(self.hidden_dims[0], self.input_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t = self.time_embedding(t)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Downsampling path
        skip_connections = []
        for down_block in self.down_blocks:
            x = down_block(x, t)
            skip_connections.append(x)
            x = F.avg_pool2d(x, 2)
        
        # Middle block
        x = self.middle_block(x, t)
        
        # Upsampling path
        for i, up_block in enumerate(self.up_blocks):
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = torch.cat([x, skip_connections[-(i + 1)]], dim=1)
            x = up_block(x, t)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x

class DiffusionModel(BaseNeuralNetwork):
    """Diffusion Model implementation with configurable architecture."""
    
    def _build_model(self) -> nn.Module:
        """Build the diffusion model architecture based on configuration."""
        # Get configuration parameters
        self.timesteps = self.config.get('timesteps', 1000)
        self.beta_start = self.config.get('beta_start', 1e-4)
        self.beta_end = self.config.get('beta_end', 0.02)
        
        # Create noise schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Create U-Net
        self.unet = UNet(self.config)
        
        return nn.Module()  # Placeholder, we'll use unet directly
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass of the diffusion model."""
        return self.unet(x, t)
    
    def sample(self, num_samples: int, image_size: int) -> torch.Tensor:
        """Generate samples using the diffusion model."""
        # Start from random noise
        x = torch.randn(num_samples, self.config.get('input_channels', 3), image_size, image_size, device=self.device)
        
        # Gradually denoise
        for t in range(self.timesteps - 1, -1, -1):
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.unet(x, t_batch)
            
            # Get noise schedule parameters
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            
            # Update sample
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise) + torch.sqrt(beta) * noise
        
        return x
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }
        return activations.get(name.lower(), nn.SiLU()) 