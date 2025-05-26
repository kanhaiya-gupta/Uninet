import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Tuple
from .base import BaseLoss

class AutoencoderLoss(BaseLoss):
    """Loss function for Autoencoders."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.reconstruction_weight = self.config.get('reconstruction_weight', 1.0)
        self.regularization_weight = self.config.get('regularization_weight', 0.01)
        self.loss_type = self.config.get('loss_type', 'mse')
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        latent: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute autoencoder loss.
        
        Args:
            predictions: Reconstructed samples
            targets: Original samples
            latent: Latent representations
            **kwargs: Additional arguments
        
        Returns:
            Autoencoder loss and optional metrics
        """
        metrics = {}
        
        # Reconstruction loss
        if self.loss_type == 'mse':
            reconstruction_loss = F.mse_loss(predictions, targets)
        elif self.loss_type == 'l1':
            reconstruction_loss = F.l1_loss(predictions, targets)
        elif self.loss_type == 'bce':
            reconstruction_loss = F.binary_cross_entropy(predictions, targets)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        total_loss = self.reconstruction_weight * reconstruction_loss
        metrics['reconstruction_loss'] = reconstruction_loss
        
        # Add regularization if latent representations are provided
        if latent is not None:
            # L2 regularization on latent space
            regularization_loss = torch.mean(torch.norm(latent, dim=1))
            total_loss += self.regularization_weight * regularization_loss
            metrics['regularization_loss'] = regularization_loss
        
        return total_loss, metrics 