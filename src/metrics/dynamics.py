import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List
import numpy as np
from .base import BaseMetric

class TrajectoryError(BaseMetric):
    """Metric for evaluating trajectory accuracy in Neural ODEs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.time_steps = self.config.get('time_steps', None)
    
    def reset(self):
        self.total_error = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: Predicted trajectory
        # targets: Ground truth trajectory
        
        if self.time_steps is None:
            raise ValueError("Time steps must be provided for trajectory error calculation")
        
        # Calculate error at each time step
        errors = []
        for t in range(len(self.time_steps)):
            error = F.mse_loss(predictions[t], targets[t])
            errors.append(error)
        
        # Average error over trajectory
        trajectory_error = torch.mean(torch.stack(errors))
        self.total_error += trajectory_error.item()
        self.count += 1
    
    def compute(self) -> float:
        return self.total_error / self.count if self.count > 0 else 0.0

class PhaseSpaceError(BaseMetric):
    """Metric for evaluating phase space accuracy in Neural ODEs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.dimensions = self.config.get('dimensions', None)
    
    def reset(self):
        self.total_error = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: Predicted phase space points
        # targets: Ground truth phase space points
        
        if self.dimensions is None:
            raise ValueError("Phase space dimensions must be provided")
        
        # Calculate error in phase space
        error = F.mse_loss(predictions, targets)
        self.total_error += error.item()
        self.count += 1
    
    def compute(self) -> float:
        return self.total_error / self.count if self.count > 0 else 0.0

class ConservationError(BaseMetric):
    """Metric for evaluating conservation laws in Neural ODEs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.conserved_quantities = self.config.get('conserved_quantities', [])
    
    def reset(self):
        self.total_error = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: Predicted states
        # targets: Ground truth states
        
        error = 0.0
        for quantity in self.conserved_quantities:
            # Calculate conserved quantity for predictions and targets
            pred_quantity = quantity(predictions)
            target_quantity = quantity(targets)
            
            # Calculate error in conservation
            error += F.mse_loss(pred_quantity, target_quantity)
        
        self.total_error += error.item()
        self.count += 1
    
    def compute(self) -> float:
        return self.total_error / self.count if self.count > 0 else 0.0

class StabilityError(BaseMetric):
    """Metric for evaluating stability properties in Neural ODEs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.perturbation_size = self.config.get('perturbation_size', 0.01)
    
    def reset(self):
        self.total_error = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: Predicted states
        # targets: Ground truth states
        
        # Add small perturbation to initial state
        perturbed_state = predictions[0] + self.perturbation_size * torch.randn_like(predictions[0])
        
        # Calculate error between perturbed and unperturbed trajectories
        error = F.mse_loss(predictions, perturbed_state)
        self.total_error += error.item()
        self.count += 1
    
    def compute(self) -> float:
        return self.total_error / self.count if self.count > 0 else 0.0 