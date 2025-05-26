import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union, List, Tuple
from .base import BaseMetric
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class LossPlotter(BaseMetric):
    """Metric for plotting training and validation losses."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
    
    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        epoch = kwargs.get('epoch', len(self.epochs))
        train_loss = kwargs.get('train_loss', None)
        val_loss = kwargs.get('val_loss', None)
        
        if train_loss is not None:
            self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if epoch not in self.epochs:
            self.epochs.append(epoch)
    
    def compute(self) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.epochs, self.train_losses, label='Training Loss')
        if self.val_losses:
            ax.plot(self.epochs, self.val_losses, label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Losses')
        ax.legend()
        ax.grid(True)
        return fig, ax

class ROCPlotter(BaseMetric):
    """Metric for plotting ROC curves."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.predictions = []
        self.targets = []
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self) -> Tuple[plt.Figure, plt.Axes]:
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(self.targets, self.predictions)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        return fig, ax

class ConfusionMatrixPlotter(BaseMetric):
    """Metric for plotting confusion matrices."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.predictions = []
        self.targets = []
        self.n_classes = self.config.get('n_classes', 2)
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        self.predictions.extend(predictions.argmax(dim=1).cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self) -> Tuple[plt.Figure, plt.Axes]:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(self.targets, self.predictions)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        return fig, ax

class LearningCurvePlotter(BaseMetric):
    """Metric for plotting learning curves."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.train_scores = []
        self.val_scores = []
        self.train_sizes = []
    
    def reset(self):
        self.train_scores = []
        self.val_scores = []
        self.train_sizes = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        train_size = kwargs.get('train_size', None)
        train_score = kwargs.get('train_score', None)
        val_score = kwargs.get('val_score', None)
        
        if all(x is not None for x in [train_size, train_score, val_score]):
            self.train_sizes.append(train_size)
            self.train_scores.append(train_score)
            self.val_scores.append(val_score)
    
    def compute(self) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.train_sizes, self.train_scores, label='Training Score')
        ax.plot(self.train_sizes, self.val_scores, label='Validation Score')
        ax.set_xlabel('Training Examples')
        ax.set_ylabel('Score')
        ax.set_title('Learning Curve')
        ax.legend()
        ax.grid(True)
        return fig, ax

class PSNR(BaseMetric):
    """Peak Signal-to-Noise Ratio metric."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_value = self.config.get('max_value', 1.0)
    
    def reset(self):
        self.total_psnr = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # Convert to numpy for PSNR calculation
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Calculate PSNR for each image in the batch
        for pred, target in zip(pred_np, target_np):
            psnr_value = psnr(target, pred, data_range=self.max_value)
            self.total_psnr += psnr_value
            self.count += 1
    
    def compute(self) -> float:
        return self.total_psnr / self.count if self.count > 0 else 0.0

class SSIM(BaseMetric):
    """Structural Similarity Index Measure metric."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.channel_axis = self.config.get('channel_axis', 0)
    
    def reset(self):
        self.total_ssim = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # Convert to numpy for SSIM calculation
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Calculate SSIM for each image in the batch
        for pred, target in zip(pred_np, target_np):
            ssim_value = ssim(
                target,
                pred,
                channel_axis=self.channel_axis,
                data_range=1.0
            )
            self.total_ssim += ssim_value
            self.count += 1
    
    def compute(self) -> float:
        return self.total_ssim / self.count if self.count > 0 else 0.0

class LPIPS(BaseMetric):
    """Learned Perceptual Image Patch Similarity metric."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='alex')
        except ImportError:
            raise ImportError("LPIPS metric requires the lpips package. Install with: pip install lpips")
    
    def reset(self):
        self.total_lpips = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # Calculate LPIPS for each image in the batch
        with torch.no_grad():
            lpips_value = self.lpips_fn(predictions, targets)
            self.total_lpips += lpips_value.mean().item()
            self.count += 1
    
    def compute(self) -> float:
        return self.total_lpips / self.count if self.count > 0 else 0.0 