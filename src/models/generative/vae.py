import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from ..base import BaseNeuralNetwork

class Encoder(BaseNeuralNetwork):
    """Encoder network for VAE."""
    
    def _build_model(self) -> nn.Module:
        """Build the encoder architecture based on configuration."""
        # Get configuration parameters
        input_channels = self.config.get('input_channels', 3)
        hidden_dims = self.config.get('hidden_dims', [32, 64, 128, 256])
        latent_dim = self.config.get('latent_dim', 100)
        activation = self.config.get('activation', 'relu')
        use_batch_norm = self.config.get('use_batch_norm', True)
        
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
                self._get_activation(activation)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate the size of the flattened features
        self.flatten_size = hidden_dims[-1] * (self.config.get('input_size', 64) // (2 ** len(hidden_dims))) ** 2
        
        # Latent space projection
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, latent_dim)
        
        return nn.Module()  # Placeholder, we'll use conv_layers and fc layers directly
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the encoder."""
        # Convolution layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Latent space projection
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        
        return mu, log_var
    
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

class Decoder(BaseNeuralNetwork):
    """Decoder network for VAE."""
    
    def _build_model(self) -> nn.Module:
        """Build the decoder architecture based on configuration."""
        # Get configuration parameters
        latent_dim = self.config.get('latent_dim', 100)
        output_channels = self.config.get('output_channels', 3)
        hidden_dims = self.config.get('hidden_dims', [256, 128, 64, 32])
        activation = self.config.get('activation', 'relu')
        use_batch_norm = self.config.get('use_batch_norm', True)
        
        # Calculate initial size
        initial_size = self.config.get('input_size', 64) // (2 ** len(hidden_dims))
        
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
            nn.Sigmoid()  # Output in range [0, 1]
        ])
        
        self.main = nn.Sequential(*layers)
        
        return nn.Module()  # Placeholder, we'll use input and main directly
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder."""
        # Input layer
        x = self.input(z)
        
        # Reshape for transposed convolution
        x = x.view(x.size(0), -1, self.config.get('input_size', 64) // (2 ** len(self.config.get('hidden_dims', [256, 128, 64, 32]))), -1)
        
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

class VAE(BaseNeuralNetwork):
    """Variational Autoencoder implementation with configurable architecture."""
    
    def _build_model(self) -> nn.Module:
        """Build the VAE architecture based on configuration."""
        # Create encoder and decoder
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        
        return nn.Module()  # Placeholder, we'll use encoder and decoder directly
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the VAE."""
        # Encode
        mu, log_var = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x_recon = self.decoder(z)
        
        return x_recon, mu, log_var
    
    def loss_function(self, x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the VAE loss."""
        # Reconstruction loss (binary cross entropy)
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        total_loss = recon_loss + kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def generate(self, num_samples: int) -> torch.Tensor:
        """Generate samples using the decoder."""
        # Generate random latent vectors
        z = torch.randn(num_samples, self.config.get('latent_dim', 100), device=self.device)
        
        # Generate images
        with torch.no_grad():
            x_recon = self.decoder(z)
        
        return x_recon
    
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