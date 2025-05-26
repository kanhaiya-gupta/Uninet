import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Tuple
from .base import BaseLoss

class CTCLoss(BaseLoss):
    """Connectionist Temporal Classification loss for sequence prediction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.blank = self.config.get('blank', 0)
        self.zero_infinity = self.config.get('zero_infinity', True)
        self.reduction = self.config.get('reduction', 'mean')
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute CTC loss.
        
        Args:
            predictions: Model predictions (log probabilities)
            targets: Target sequences
            input_lengths: Lengths of input sequences
            target_lengths: Lengths of target sequences
            **kwargs: Additional arguments
        
        Returns:
            CTC loss
        """
        return F.ctc_loss(
            predictions,
            targets,
            input_lengths,
            target_lengths,
            blank=self.blank,
            zero_infinity=self.zero_infinity,
            reduction=self.reduction
        )

class SequenceLoss(BaseLoss):
    """Loss function for sequence prediction tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.task_type = self.config.get('task_type', 'classification')
        self.label_smoothing = self.config.get('label_smoothing', 0.0)
        self.mask_padding = self.config.get('mask_padding', True)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute sequence loss.
        
        Args:
            predictions: Model predictions
            targets: Target sequences
            mask: Padding mask (1 for valid tokens, 0 for padding)
            **kwargs: Additional arguments
        
        Returns:
            Sequence loss and optional metrics
        """
        if self.task_type == 'classification':
            if self.mask_padding and mask is not None:
                # Apply padding mask
                predictions = predictions.view(-1, predictions.size(-1))
                targets = targets.view(-1)
                mask = mask.view(-1)
                
                # Compute loss only on non-padded tokens
                loss = F.cross_entropy(
                    predictions[mask],
                    targets[mask],
                    label_smoothing=self.label_smoothing
                )
            else:
                loss = F.cross_entropy(
                    predictions.view(-1, predictions.size(-1)),
                    targets.view(-1),
                    label_smoothing=self.label_smoothing
                )
                
        elif self.task_type == 'regression':
            if self.mask_padding and mask is not None:
                # Apply padding mask
                predictions = predictions.view(-1)
                targets = targets.view(-1)
                mask = mask.view(-1)
                
                # Compute loss only on non-padded tokens
                loss = F.mse_loss(predictions[mask], targets[mask])
            else:
                loss = F.mse_loss(
                    predictions.view(-1),
                    targets.view(-1)
                )
        
        metrics = {'sequence_loss': loss}
        return loss, metrics 