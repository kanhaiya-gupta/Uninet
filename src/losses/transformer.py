import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Tuple
from .base import BaseLoss

class TransformerLoss(BaseLoss):
    """Loss function for Transformer networks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.label_smoothing = self.config.get('label_smoothing', 0.1)
        self.attention_weight = self.config.get('attention_weight', 0.01)
        self.auxiliary_weight = self.config.get('auxiliary_weight', 0.1)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        auxiliary_predictions: Optional[torch.Tensor] = None,
        auxiliary_targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute transformer loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            attention_weights: Attention weights for regularization
            auxiliary_predictions: Auxiliary task predictions
            auxiliary_targets: Auxiliary task targets
            **kwargs: Additional arguments
        
        Returns:
            Transformer loss and optional metrics
        """
        # Main task loss with label smoothing
        if self.label_smoothing > 0:
            n_classes = predictions.size(-1)
            one_hot = torch.zeros_like(predictions).scatter(1, targets.unsqueeze(1), 1)
            smooth_one_hot = one_hot * (1 - self.label_smoothing) + (self.label_smoothing / n_classes)
            loss = (-smooth_one_hot * F.log_softmax(predictions, dim=1)).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(predictions, targets)
        
        metrics = {'main_loss': loss}
        
        # Add attention regularization if provided
        if attention_weights is not None:
            # Encourage sparse attention
            attention_loss = torch.mean(torch.abs(attention_weights))
            loss += self.attention_weight * attention_loss
            metrics['attention_loss'] = attention_loss
        
        # Add auxiliary task loss if provided
        if auxiliary_predictions is not None and auxiliary_targets is not None:
            auxiliary_loss = F.cross_entropy(auxiliary_predictions, auxiliary_targets)
            loss += self.auxiliary_weight * auxiliary_loss
            metrics['auxiliary_loss'] = auxiliary_loss
        
        return loss, metrics 