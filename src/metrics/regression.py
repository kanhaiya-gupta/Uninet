import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List
import numpy as np
from .base import BaseMetric

class MSE(BaseMetric):
    """Mean Squared Error metric for regression tasks."""
    
    def reset(self):
        self.sum_squared_error = 0.0
        self.total = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        squared_error = F.mse_loss(predictions, targets, reduction='none')
        self.sum_squared_error += squared_error.sum().item()
        self.total += targets.numel()
    
    def compute(self) -> float:
        return self.sum_squared_error / self.total if self.total > 0 else 0.0

class RMSE(BaseMetric):
    """Root Mean Squared Error metric for regression tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.mse = MSE()
    
    def reset(self):
        self.mse.reset()
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        self.mse.update(predictions, targets)
    
    def compute(self) -> float:
        return np.sqrt(self.mse.compute())

class MAE(BaseMetric):
    """Mean Absolute Error metric for regression tasks."""
    
    def reset(self):
        self.sum_absolute_error = 0.0
        self.total = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        absolute_error = F.l1_loss(predictions, targets, reduction='none')
        self.sum_absolute_error += absolute_error.sum().item()
        self.total += targets.numel()
    
    def compute(self) -> float:
        return self.sum_absolute_error / self.total if self.total > 0 else 0.0

class R2Score(BaseMetric):
    """R-squared score metric for regression tasks."""
    
    def reset(self):
        self.sum_squared_error = 0.0
        self.sum_squared_total = 0.0
        self.total = 0
        self.mean_target = 0.0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # Update mean target
        self.mean_target = (self.mean_target * self.total + targets.mean().item() * targets.numel()) / (self.total + targets.numel())
        
        # Update squared error
        squared_error = F.mse_loss(predictions, targets, reduction='none')
        self.sum_squared_error += squared_error.sum().item()
        
        # Update squared total
        squared_total = (targets - self.mean_target) ** 2
        self.sum_squared_total += squared_total.sum().item()
        
        self.total += targets.numel()
    
    def compute(self) -> float:
        if self.total == 0 or self.sum_squared_total == 0:
            return 0.0
        return 1 - (self.sum_squared_error / self.sum_squared_total)

class ExplainedVariance(BaseMetric):
    """Explained Variance metric for regression tasks."""
    
    def reset(self):
        self.sum_squared_error = 0.0
        self.sum_squared_total = 0.0
        self.total = 0
        self.mean_target = 0.0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # Update mean target
        self.mean_target = (self.mean_target * self.total + targets.mean().item() * targets.numel()) / (self.total + targets.numel())
        
        # Update squared error
        squared_error = F.mse_loss(predictions, targets, reduction='none')
        self.sum_squared_error += squared_error.sum().item()
        
        # Update squared total
        squared_total = (targets - self.mean_target) ** 2
        self.sum_squared_total += squared_total.sum().item()
        
        self.total += targets.numel()
    
    def compute(self) -> float:
        if self.total == 0 or self.sum_squared_total == 0:
            return 0.0
        return 1 - (self.sum_squared_error / self.sum_squared_total) 