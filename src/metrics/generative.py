import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List
import numpy as np
from .base import BaseMetric

class FID(BaseMetric):
    """Fréchet Inception Distance metric for GAN evaluation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.feature_extractor = self.config.get('feature_extractor', None)
        if self.feature_extractor is None:
            raise ValueError("Feature extractor must be provided for FID calculation")
    
    def reset(self):
        self.real_features = []
        self.fake_features = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: Generated images
        # targets: Real images
        with torch.no_grad():
            real_features = self.feature_extractor(targets)
            fake_features = self.feature_extractor(predictions)
            
            self.real_features.append(real_features.cpu())
            self.fake_features.append(fake_features.cpu())
    
    def compute(self) -> float:
        real_features = torch.cat(self.real_features, dim=0)
        fake_features = torch.cat(self.fake_features, dim=0)
        
        # Calculate mean and covariance
        mu_real = real_features.mean(dim=0)
        mu_fake = fake_features.mean(dim=0)
        sigma_real = torch.cov(real_features.t())
        sigma_fake = torch.cov(fake_features.t())
        
        # Calculate FID
        ssdiff = torch.sum((mu_real - mu_fake) ** 2)
        covmean = torch.matrix_power(sigma_real @ sigma_fake, 0.5)
        
        fid = ssdiff + torch.trace(sigma_real + sigma_fake - 2 * covmean)
        return fid.item()

class InceptionScore(BaseMetric):
    """Inception Score metric for GAN evaluation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.classifier = self.config.get('classifier', None)
        if self.classifier is None:
            raise ValueError("Classifier must be provided for Inception Score calculation")
        self.splits = self.config.get('splits', 10)
    
    def reset(self):
        self.predictions = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: Generated images
        with torch.no_grad():
            logits = self.classifier(predictions)
            probs = F.softmax(logits, dim=1)
            self.predictions.append(probs.cpu())
    
    def compute(self) -> float:
        probs = torch.cat(self.predictions, dim=0)
        
        # Split into groups
        split_size = len(probs) // self.splits
        scores = []
        
        for i in range(self.splits):
            part = probs[i * split_size:(i + 1) * split_size]
            kl = part * (torch.log(part) - torch.log(part.mean(dim=0, keepdim=True)))
            kl = kl.sum(dim=1).mean()
            scores.append(torch.exp(kl))
        
        return torch.stack(scores).mean().item()

class LPIPS(BaseMetric):
    """Learned Perceptual Image Patch Similarity metric."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = self.config.get('model', None)
        if self.model is None:
            raise ValueError("LPIPS model must be provided")
    
    def reset(self):
        self.total_distance = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: Generated images
        # targets: Real images
        with torch.no_grad():
            distance = self.model(predictions, targets)
            self.total_distance += distance.sum().item()
            self.count += distance.numel()
    
    def compute(self) -> float:
        return self.total_distance / self.count if self.count > 0 else 0.0

class PSNR(BaseMetric):
    """Peak Signal-to-Noise Ratio metric for image quality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_val = self.config.get('max_val', 1.0)
    
    def reset(self):
        self.total_mse = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        mse = F.mse_loss(predictions, targets, reduction='none')
        self.total_mse += mse.sum().item()
        self.count += mse.numel()
    
    def compute(self) -> float:
        if self.count == 0:
            return 0.0
        mse = self.total_mse / self.count
        return 20 * torch.log10(torch.tensor(self.max_val)) - 10 * torch.log10(torch.tensor(mse))

class SSIM(BaseMetric):
    """Structural Similarity Index Measure for image quality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.window_size = self.config.get('window_size', 11)
        self.sigma = self.config.get('sigma', 1.5)
        self.k1 = self.config.get('k1', 0.01)
        self.k2 = self.config.get('k2', 0.03)
    
    def reset(self):
        self.total_ssim = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # Create Gaussian window
        window = torch.exp(-(torch.arange(self.window_size) - self.window_size//2)**2 / (2*self.sigma**2))
        window = window / window.sum()
        window = window.unsqueeze(0)
        
        # Calculate means
        mu_x = F.conv2d(predictions, window.unsqueeze(0).unsqueeze(0))
        mu_y = F.conv2d(targets, window.unsqueeze(0).unsqueeze(0))
        
        # Calculate variances and covariance
        sigma_x = F.conv2d(predictions**2, window.unsqueeze(0).unsqueeze(0)) - mu_x**2
        sigma_y = F.conv2d(targets**2, window.unsqueeze(0).unsqueeze(0)) - mu_y**2
        sigma_xy = F.conv2d(predictions*targets, window.unsqueeze(0).unsqueeze(0)) - mu_x*mu_y
        
        # Calculate SSIM
        c1 = (self.k1 * self.max_val)**2
        c2 = (self.k2 * self.max_val)**2
        
        ssim = ((2*mu_x*mu_y + c1)*(2*sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1)*(sigma_x + sigma_y + c2))
        
        self.total_ssim += ssim.sum().item()
        self.count += ssim.numel()
    
    def compute(self) -> float:
        return self.total_ssim / self.count if self.count > 0 else 0.0 