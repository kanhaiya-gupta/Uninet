import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Tuple, List
from .base import BaseLoss

class PINNLoss(BaseLoss):
    """Loss function for Physics-Informed Neural Networks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.pde_weight = self.config.get('pde_weight', 1.0)
        self.ic_weight = self.config.get('ic_weight', 1.0)
        self.bc_weight = self.config.get('bc_weight', 1.0)
        self.data_weight = self.config.get('data_weight', 1.0)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pde_residuals: Optional[List[torch.Tensor]] = None,
        ic_residuals: Optional[List[torch.Tensor]] = None,
        bc_residuals: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute PINN loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets (if available)
            pde_residuals: List of PDE residuals
            ic_residuals: List of initial condition residuals
            bc_residuals: List of boundary condition residuals
            **kwargs: Additional arguments
        
        Returns:
            PINN loss and optional metrics
        """
        metrics = {}
        total_loss = 0.0
        
        # Data loss (if targets are provided)
        if targets is not None:
            data_loss = F.mse_loss(predictions, targets)
            total_loss += self.data_weight * data_loss
            metrics['data_loss'] = data_loss
        
        # PDE residuals
        if pde_residuals is not None:
            pde_loss = sum(torch.mean(residual ** 2) for residual in pde_residuals)
            total_loss += self.pde_weight * pde_loss
            metrics['pde_loss'] = pde_loss
        
        # Initial condition residuals
        if ic_residuals is not None:
            ic_loss = sum(torch.mean(residual ** 2) for residual in ic_residuals)
            total_loss += self.ic_weight * ic_loss
            metrics['ic_loss'] = ic_loss
        
        # Boundary condition residuals
        if bc_residuals is not None:
            bc_loss = sum(torch.mean(residual ** 2) for residual in bc_residuals)
            total_loss += self.bc_weight * bc_loss
            metrics['bc_loss'] = bc_loss
        
        return total_loss, metrics 