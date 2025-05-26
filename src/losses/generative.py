import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Tuple
from .base import BaseLoss

class GANLoss(BaseLoss):
    """Loss function for Generative Adversarial Networks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.gan_mode = self.config.get('gan_mode', 'vanilla')  # 'vanilla' or 'lsgan'
        self.real_label = self.config.get('real_label', 1.0)
        self.fake_label = self.config.get('fake_label', 0.0)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        is_real: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """Compute GAN loss.
        
        Args:
            predictions: Discriminator predictions
            targets: Ground truth labels (not used in GAN)
            is_real: Whether the predictions are for real or fake samples
            **kwargs: Additional arguments
        
        Returns:
            GAN loss
        """
        target = torch.full_like(predictions, self.real_label if is_real else self.fake_label)
        
        if self.gan_mode == 'vanilla':
            loss = F.binary_cross_entropy_with_logits(predictions, target)
        else:  # lsgan
            loss = F.mse_loss(predictions, target)
        
        return loss

class VAELoss(BaseLoss):
    """Loss function for Variational Autoencoders."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.reconstruction_weight = self.config.get('reconstruction_weight', 1.0)
        self.kl_weight = self.config.get('kl_weight', 1.0)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute VAE loss.
        
        Args:
            predictions: Reconstructed samples
            targets: Original samples
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            **kwargs: Additional arguments
        
        Returns:
            VAE loss and optional metrics
        """
        # Reconstruction loss
        reconstruction_loss = F.binary_cross_entropy(predictions, targets, reduction='sum')
        
        # Initialize total loss
        loss = self.reconstruction_weight * reconstruction_loss
        
        # Add KL divergence if provided
        if mu is not None and logvar is not None:
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss += self.kl_weight * kl_loss
            
            return loss, {
                'reconstruction_loss': reconstruction_loss,
                'kl_loss': kl_loss
            }
        
        return loss

class DiffusionLoss(BaseLoss):
    """Loss function for Diffusion Models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.noise_weight = self.config.get('noise_weight', 1.0)
        self.consistency_weight = self.config.get('consistency_weight', 0.1)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute diffusion loss.
        
        Args:
            predictions: Predicted noise
            targets: Target noise
            noise: Original noise added to the sample
            timesteps: Diffusion timesteps
            **kwargs: Additional arguments
        
        Returns:
            Diffusion loss and optional metrics
        """
        # Noise prediction loss
        noise_loss = F.mse_loss(predictions, targets)
        
        # Initialize total loss
        loss = self.noise_weight * noise_loss
        
        # Add consistency loss if provided
        if noise is not None and timesteps is not None:
            # Consistency loss (encourage smooth transitions)
            consistency_loss = torch.mean(
                torch.abs(predictions[1:] - predictions[:-1])
            )
            loss += self.consistency_weight * consistency_loss
            
            return loss, {
                'noise_loss': noise_loss,
                'consistency_loss': consistency_loss
            }
        
        return loss 