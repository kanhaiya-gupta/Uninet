import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Tuple
from .base import BaseLoss

class MSELoss(BaseLoss):
    """Mean Squared Error loss for regression tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.reduction = self.config.get('reduction', 'mean')
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute MSE loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional arguments
        
        Returns:
            MSE loss
        """
        return F.mse_loss(predictions, targets, reduction=self.reduction)

class CrossEntropyLoss(BaseLoss):
    """Cross Entropy loss for classification tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.reduction = self.config.get('reduction', 'mean')
        self.label_smoothing = self.config.get('label_smoothing', 0.0)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute cross entropy loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional arguments
        
        Returns:
            Cross entropy loss
        """
        return F.cross_entropy(
            predictions,
            targets,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
        ) 