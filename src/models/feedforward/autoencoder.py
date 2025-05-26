import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from ..base import BaseNeuralNetwork

class Autoencoder(BaseNeuralNetwork):
    """Autoencoder implementation with encoder and decoder."""
    
    def _build_model(self) -> nn.Module:
        """Build the Autoencoder architecture based on configuration."""
        # Get configuration parameters
        input_size = self.config.get('input_size', 784)
        hidden_sizes = self.config.get('hidden_sizes', [512, 256, 128])
        latent_size = self.config.get('latent_size', 64)
        activation = self.config.get('activation', 'relu')
        use_batch_norm = self.config.get('use_batch_norm', True)
        use_dropout = self.config.get('use_dropout', True)
        dropout_rate = self.config.get('dropout_rate', 0.2)
        
        # Build encoder
        encoder_layers: List[nn.Module] = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            encoder_layers.extend([
                nn.Linear(current_size, hidden_size),
                self._get_activation(activation)
            ])
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_size))
            if use_dropout:
                encoder_layers.append(nn.Dropout(dropout_rate))
            current_size = hidden_size
        
        # Latent layer
        encoder_layers.append(nn.Linear(current_size, latent_size))
        
        # Build decoder
        decoder_layers: List[nn.Module] = []
        current_size = latent_size
        
        for hidden_size in reversed(hidden_sizes):
            decoder_layers.extend([
                nn.Linear(current_size, hidden_size),
                self._get_activation(activation)
            ])
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden_size))
            if use_dropout:
                decoder_layers.append(nn.Dropout(dropout_rate))
            current_size = hidden_size
        
        # Output layer
        decoder_layers.append(nn.Linear(current_size, input_size))
        decoder_layers.append(nn.Sigmoid())  # For reconstruction
        
        # Create encoder and decoder
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        
        return nn.Module()  # Placeholder, we'll use encoder and decoder directly
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the network."""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent space to reconstruction."""
        return self.decoder(z)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        return activations.get(name.lower(), nn.ReLU()) 