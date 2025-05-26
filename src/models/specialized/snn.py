import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Union
from ..base import BaseNeuralNetwork

class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron."""
    
    def __init__(
        self,
        threshold: float = 1.0,
        decay: float = 0.9,
        reset_voltage: float = 0.0
    ):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.reset_voltage = reset_voltage
    
    def forward(
        self,
        x: torch.Tensor,
        membrane_potential: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update membrane potential
        membrane_potential = self.decay * membrane_potential + x
        
        # Generate spikes
        spikes = (membrane_potential >= self.threshold).float()
        
        # Reset membrane potential
        membrane_potential = torch.where(
            spikes > 0,
            torch.full_like(membrane_potential, self.reset_voltage),
            membrane_potential
        )
        
        return spikes, membrane_potential

class SNNLayer(nn.Module):
    """Spiking Neural Network layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        threshold: float = 1.0,
        decay: float = 0.9,
        reset_voltage: float = 0.0,
        use_bias: bool = True
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        self.neuron = LIFNeuron(threshold, decay, reset_voltage)
    
    def forward(
        self,
        x: torch.Tensor,
        membrane_potential: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Linear transformation
        x = self.linear(x)
        
        # Apply neuron dynamics
        spikes, membrane_potential = self.neuron(x, membrane_potential)
        
        return spikes, membrane_potential

class SNN(BaseNeuralNetwork):
    """Spiking Neural Network implementation with configurable architecture."""
    
    def _build_model(self) -> nn.Module:
        """Build the SNN architecture based on configuration."""
        # Get configuration parameters
        input_size = self.config.get('input_size', 1)
        hidden_sizes = self.config.get('hidden_sizes', [64, 32])
        output_size = self.config.get('output_size', 1)
        threshold = self.config.get('threshold', 1.0)
        decay = self.config.get('decay', 0.9)
        reset_voltage = self.config.get('reset_voltage', 0.0)
        use_bias = self.config.get('use_bias', True)
        time_steps = self.config.get('time_steps', 10)
        
        # Build layers
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(SNNLayer(
                current_size,
                hidden_size,
                threshold,
                decay,
                reset_voltage,
                use_bias
            ))
            current_size = hidden_size
        
        # Output layer
        layers.append(SNNLayer(
            current_size,
            output_size,
            threshold,
            decay,
            reset_voltage,
            use_bias
        ))
        
        self.layers = nn.ModuleList(layers)
        self.time_steps = time_steps
        
        return nn.Module()  # Placeholder, we'll use layers directly
    
    def forward(
        self,
        x: torch.Tensor,
        return_membrane_potentials: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass of the network."""
        batch_size = x.size(0)
        
        # Initialize membrane potentials
        membrane_potentials = [
            torch.zeros(batch_size, layer.linear.out_features, device=x.device)
            for layer in self.layers
        ]
        
        # Initialize output spikes
        output_spikes = torch.zeros(batch_size, self.layers[-1].linear.out_features, device=x.device)
        
        # Simulate for multiple time steps
        for _ in range(self.time_steps):
            current_input = x
            
            # Process through layers
            for i, layer in enumerate(self.layers):
                spikes, membrane_potentials[i] = layer(current_input, membrane_potentials[i])
                current_input = spikes
            
            # Accumulate output spikes
            output_spikes += current_input
        
        # Average output spikes over time steps
        output = output_spikes / self.time_steps
        
        if return_membrane_potentials:
            return output, membrane_potentials
        return output
    
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