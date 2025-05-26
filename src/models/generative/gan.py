import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from ..base import BaseNeuralNetwork

class Generator(BaseNeuralNetwork):
    """Generator network for GAN."""
    
    def _build_model(self) -> nn.Module:
        """Build the generator architecture based on configuration."""
        # Get configuration parameters
        latent_dim = self.config.get('latent_dim', 100)
        output_channels = self.config.get('output_channels', 3)
        output_size = self.config.get('output_size', 64)
        hidden_dims = self.config.get('hidden_dims', [512, 256, 128, 64])
        activation = self.config.get('activation', 'relu')
        use_batch_norm = self.config.get('use_batch_norm', True)
        
        # Calculate initial size
        initial_size = output_size // (2 ** len(hidden_dims))
        
        # Input layer
        self.input = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0] * initial_size * initial_size),
            nn.BatchNorm1d(hidden_dims[0] * initial_size * initial_size) if use_batch_norm else nn.Identity(),
            self._get_activation(activation)
        )
        
        # Transposed convolution layers
        layers = []
        in_channels = hidden_dims[0]
        
        for i, out_channels in enumerate(hidden_dims[1:], 1):
            layers.extend([
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=not use_batch_norm
                ),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
                self._get_activation(activation)
            ])
            in_channels = out_channels
        
        # Output layer
        layers.extend([
            nn.ConvTranspose2d(
                in_channels,
                output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True
            ),
            nn.Tanh()  # Output in range [-1, 1]
        ])
        
        self.main = nn.Sequential(*layers)
        
        return nn.Module()  # Placeholder, we'll use input and main directly
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass of the generator."""
        # Input layer
        x = self.input(z)
        
        # Reshape for transposed convolution
        x = x.view(x.size(0), -1, self.config.get('output_size', 64) // (2 ** len(self.config.get('hidden_dims', [512, 256, 128, 64]))), -1)
        
        # Main layers
        x = self.main(x)
        
        return x
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU()
        }
        return activations.get(name.lower(), nn.ReLU())

class Discriminator(BaseNeuralNetwork):
    """Discriminator network for GAN."""
    
    def _build_model(self) -> nn.Module:
        """Build the discriminator architecture based on configuration."""
        # Get configuration parameters
        input_channels = self.config.get('input_channels', 3)
        hidden_dims = self.config.get('hidden_dims', [64, 128, 256, 512])
        activation = self.config.get('activation', 'leaky_relu')
        use_batch_norm = self.config.get('use_batch_norm', True)
        dropout = self.config.get('dropout', 0.3)
        
        # Convolution layers
        layers = []
        in_channels = input_channels
        
        for i, out_channels in enumerate(hidden_dims):
            layers.extend([
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=not use_batch_norm
                ),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
                self._get_activation(activation),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
            ])
            in_channels = out_channels
        
        # Output layer
        layers.extend([
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()  # Output in range [0, 1]
        ])
        
        self.main = nn.Sequential(*layers)
        
        return nn.Module()  # Placeholder, we'll use main directly
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the discriminator."""
        return self.main(x)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU()
        }
        return activations.get(name.lower(), nn.LeakyReLU(0.2))

class GAN(BaseNeuralNetwork):
    """Generative Adversarial Network implementation with configurable architecture."""
    
    def _build_model(self) -> nn.Module:
        """Build the GAN architecture based on configuration."""
        # Create generator and discriminator
        self.generator = Generator(self.config)
        self.discriminator = Discriminator(self.config)
        
        return nn.Module()  # Placeholder, we'll use generator and discriminator directly
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the GAN."""
        # Generate fake images
        fake_images = self.generator(z)
        
        # Get discriminator predictions
        real_pred = self.discriminator(fake_images)
        
        return fake_images, real_pred
    
    def generate(self, num_samples: int) -> torch.Tensor:
        """Generate samples using the generator."""
        # Generate random latent vectors
        z = torch.randn(num_samples, self.config.get('latent_dim', 100), device=self.device)
        
        # Generate images
        with torch.no_grad():
            fake_images = self.generator(z)
        
        return fake_images
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU()
        }
        return activations.get(name.lower(), nn.ReLU()) 