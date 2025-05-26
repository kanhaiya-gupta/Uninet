"""Optimizer implementations for neural networks."""

import torch
import torch.optim as optim
from typing import Dict, Any, List, Optional

class BaseOptimizer:
    """Base class for optimizers."""
    
    def __init__(self, model_params: List[torch.Tensor], config: Dict[str, Any]):
        """Initialize optimizer.
        
        Args:
            model_params: List of model parameters to optimize
            config: Optimizer configuration dictionary
        """
        self.model_params = model_params
        self.config = config
        self.optimizer = self._create_optimizer()
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer instance."""
        raise NotImplementedError
    
    def step(self):
        """Perform a single optimization step."""
        self.optimizer.step()
    
    def zero_grad(self):
        """Zero the gradients."""
        self.optimizer.zero_grad()

class AdamOptimizer(BaseOptimizer):
    """Adam optimizer implementation."""
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create Adam optimizer."""
        return optim.Adam(
            self.model_params,
            lr=self.config.get('learning_rate', 0.001),
            betas=(
                self.config.get('beta1', 0.9),
                self.config.get('beta2', 0.999)
            ),
            eps=self.config.get('eps', 1e-8),
            weight_decay=self.config.get('weight_decay', 0)
        )

class SGDOptimizer(BaseOptimizer):
    """Stochastic Gradient Descent optimizer implementation."""
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create SGD optimizer."""
        return optim.SGD(
            self.model_params,
            lr=self.config.get('learning_rate', 0.01),
            momentum=self.config.get('momentum', 0),
            weight_decay=self.config.get('weight_decay', 0),
            nesterov=self.config.get('nesterov', False)
        )

class RMSpropOptimizer(BaseOptimizer):
    """RMSprop optimizer implementation."""
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create RMSprop optimizer."""
        return optim.RMSprop(
            self.model_params,
            lr=self.config.get('learning_rate', 0.01),
            alpha=self.config.get('alpha', 0.99),
            eps=self.config.get('eps', 1e-8),
            weight_decay=self.config.get('weight_decay', 0),
            momentum=self.config.get('momentum', 0),
            centered=self.config.get('centered', False)
        )

def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> BaseOptimizer:
    """Create an optimizer based on configuration.
    
    Args:
        model: PyTorch model
        config: Optimizer configuration dictionary containing:
            - type: Optimizer type ('adam', 'sgd', 'rmsprop')
            - learning_rate: Learning rate
            - Other optimizer-specific parameters
    
    Returns:
        Optimizer instance
    """
    optimizer_type = config.get('type', 'adam').lower()
    optimizers = {
        'adam': AdamOptimizer,
        'sgd': SGDOptimizer,
        'rmsprop': RMSpropOptimizer
    }
    
    if optimizer_type not in optimizers:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizers[optimizer_type](model.parameters(), config) 