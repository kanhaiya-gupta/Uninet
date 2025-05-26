import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional, Callable
from ..base import BaseNeuralNetwork

class ODEFunc(nn.Module):
    """Neural ODE function."""
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        activation: str = 'relu'
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        
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
        layers.append(nn.Linear(current_size, input_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ODE function."""
        return self.network(x)
    
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

class NeuralODE(BaseNeuralNetwork):
    """Neural ODE implementation with configurable architecture."""
    
    def _build_model(self) -> nn.Module:
        """Build the Neural ODE architecture based on configuration."""
        # Get configuration parameters
        input_size = self.config.get('input_size', 1)
        hidden_sizes = self.config.get('hidden_sizes', [64, 64])
        output_size = self.config.get('output_size', 1)
        activation = self.config.get('activation', 'relu')
        solver = self.config.get('solver', 'dopri5')  # 'euler', 'rk4', 'dopri5'
        rtol = self.config.get('rtol', 1e-3)
        atol = self.config.get('atol', 1e-4)
        
        # Create ODE function
        self.ode_func = ODEFunc(input_size, hidden_sizes, activation)
        
        # Create output projection
        self.output_proj = nn.Linear(input_size, output_size)
        
        # Store solver parameters
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        
        return nn.Module()  # Placeholder, we'll use ode_func and output_proj directly
    
    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        return_trajectory: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the network."""
        if t is None:
            t = torch.tensor([0.0, 1.0], device=x.device)
        
        # Solve ODE
        if self.solver == 'euler':
            trajectory = self._euler_solver(x, t)
        elif self.solver == 'rk4':
            trajectory = self._rk4_solver(x, t)
        else:  # dopri5
            trajectory = self._dopri5_solver(x, t)
        
        # Get final state
        final_state = trajectory[-1]
        
        # Project to output space
        output = self.output_proj(final_state)
        
        if return_trajectory:
            return output, trajectory
        return output
    
    def _euler_solver(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Euler method solver."""
        trajectory = [x]
        dt = (t[1] - t[0]) / 100  # Fixed number of steps
        
        for i in range(100):
            t_i = t[0] + i * dt
            dx = self.ode_func(t_i, trajectory[-1])
            x_next = trajectory[-1] + dt * dx
            trajectory.append(x_next)
        
        return torch.stack(trajectory)
    
    def _rk4_solver(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Runge-Kutta 4th order solver."""
        trajectory = [x]
        dt = (t[1] - t[0]) / 100  # Fixed number of steps
        
        for i in range(100):
            t_i = t[0] + i * dt
            x_i = trajectory[-1]
            
            # RK4 steps
            k1 = self.ode_func(t_i, x_i)
            k2 = self.ode_func(t_i + dt/2, x_i + dt/2 * k1)
            k3 = self.ode_func(t_i + dt/2, x_i + dt/2 * k2)
            k4 = self.ode_func(t_i + dt, x_i + dt * k3)
            
            x_next = x_i + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            trajectory.append(x_next)
        
        return torch.stack(trajectory)
    
    def _dopri5_solver(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Dormand-Prince 5th order solver."""
        # This is a simplified version. In practice, you would use torchdiffeq
        # or implement the full DOPRI5 algorithm
        return self._rk4_solver(x, t)  # Fallback to RK4
    
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