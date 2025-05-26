import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Tuple
from .base import BaseLoss

class FocalLoss(BaseLoss):
    """Focal Loss for handling class imbalance in classification tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.alpha = self.config.get('alpha', 0.25)
        self.gamma = self.config.get('gamma', 2.0)
        self.reduction = self.config.get('reduction', 'mean')
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional arguments
        
        Returns:
            Focal loss
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class DiceLoss(BaseLoss):
    """Dice Loss for image segmentation tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.smooth = self.config.get('smooth', 1.0)
        self.eps = self.config.get('eps', 1e-7)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute dice loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional arguments
        
        Returns:
            Dice loss
        """
        predictions = torch.sigmoid(predictions)
        
        # Flatten predictions and targets
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth + self.eps
        )
        
        return 1 - dice 