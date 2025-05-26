import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional, Callable
from ..base import BaseNeuralNetwork

class PINN(BaseNeuralNetwork):
    """Physics-Informed Neural Network implementation with configurable architecture."""
    
    def _build_model(self) -> nn.Module:
        """Build the PINN architecture based on configuration."""
        # Get configuration parameters
        input_size = self.config.get('input_size', 2)  # Default for 2D problems
        hidden_sizes = self.config.get('hidden_sizes', [64, 64, 64])
        output_size = self.config.get('output_size', 1)
        activation = self.config.get('activation', 'tanh')  # tanh is often used in PINNs
        use_batch_norm = self.config.get('use_batch_norm', False)  # Usually not used in PINNs
        
        # Build neural network layers
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                self._get_activation(activation)
            ])
            current_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(current_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        return nn.Module()  # Placeholder, we'll use network directly
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        return self.network(x)
    
    def compute_derivatives(self, x: torch.Tensor, order: int = 1) -> List[torch.Tensor]:
        """Compute derivatives of the network output with respect to inputs."""
        x.requires_grad_(True)
        y = self.forward(x)
        
        derivatives = []
        current_derivative = y
        
        for _ in range(order):
            # Compute gradients
            grad = torch.autograd.grad(
                current_derivative.sum(),
                x,
                create_graph=True,
                retain_graph=True
            )[0]
            
            derivatives.append(grad)
            current_derivative = grad
        
        return derivatives
    
    def compute_pde_residual(
        self,
        x: torch.Tensor,
        pde_fn: Callable[[torch.Tensor, List[torch.Tensor]], torch.Tensor]
    ) -> torch.Tensor:
        """Compute the PDE residual using automatic differentiation."""
        # Get network output and its derivatives
        y = self.forward(x)
        derivatives = self.compute_derivatives(x)
        
        # Compute PDE residual
        residual = pde_fn(y, derivatives)
        
        return residual
    
    def compute_ic_bc_residual(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        residual_type: str = 'mse'
    ) -> torch.Tensor:
        """Compute initial condition or boundary condition residual."""
        y = self.forward(x)
        
        if residual_type == 'mse':
            return torch.mean((y - target) ** 2)
        elif residual_type == 'mae':
            return torch.mean(torch.abs(y - target))
        else:
            raise ValueError(f"Unknown residual type: {residual_type}")
    
    def loss_function(
        self,
        x_pde: torch.Tensor,
        pde_fn: Callable[[torch.Tensor, List[torch.Tensor]], torch.Tensor],
        x_ic: Optional[torch.Tensor] = None,
        y_ic: Optional[torch.Tensor] = None,
        x_bc: Optional[torch.Tensor] = None,
        y_bc: Optional[torch.Tensor] = None,
        pde_weight: float = 1.0,
        ic_weight: float = 1.0,
        bc_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Compute the total loss combining PDE, IC, and BC residuals."""
        losses = {}
        
        # PDE residual
        pde_residual = self.compute_pde_residual(x_pde, pde_fn)
        losses['pde_loss'] = pde_weight * torch.mean(pde_residual ** 2)
        
        # Initial condition residual
        if x_ic is not None and y_ic is not None:
            ic_residual = self.compute_ic_bc_residual(x_ic, y_ic)
            losses['ic_loss'] = ic_weight * ic_residual
        
        # Boundary condition residual
        if x_bc is not None and y_bc is not None:
            bc_residual = self.compute_ic_bc_residual(x_bc, y_bc)
            losses['bc_loss'] = bc_weight * bc_residual
        
        # Total loss
        losses['total_loss'] = sum(losses.values())
        
        return losses
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'tanh': nn.Tanh(),  # Default for PINNs
            'sin': torch.sin,  # Periodic activation
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU()
        }
        return activations.get(name.lower(), nn.Tanh()) 