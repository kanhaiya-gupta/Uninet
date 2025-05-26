import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List
import numpy as np
from .base import BaseMetric

class IoU(BaseMetric):
    """Intersection over Union metric for segmentation tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.num_classes = self.config.get('num_classes', 2)
        self.ignore_index = self.config.get('ignore_index', -100)
    
    def reset(self):
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        predictions = predictions.argmax(dim=1)
        
        # Create masks for valid pixels
        mask = (targets != self.ignore_index)
        predictions = predictions[mask]
        targets = targets[mask]
        
        # Calculate intersection and union for each class
        for cls in range(self.num_classes):
            pred_mask = (predictions == cls)
            target_mask = (targets == cls)
            
            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()
            
            self.intersection[cls] += intersection
            self.union[cls] += union
    
    def compute(self) -> Union[float, Dict[str, float]]:
        iou_per_class = self.intersection / (self.union + 1e-8)
        mean_iou = iou_per_class.mean().item()
        
        return {
            'mean_iou': mean_iou,
            'iou_per_class': {f'class_{i}': iou.item() for i, iou in enumerate(iou_per_class)}
        }

class DiceScore(BaseMetric):
    """Dice coefficient metric for segmentation tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.num_classes = self.config.get('num_classes', 2)
        self.smooth = self.config.get('smooth', 1.0)
    
    def reset(self):
        self.intersection = torch.zeros(self.num_classes)
        self.sum_pred = torch.zeros(self.num_classes)
        self.sum_target = torch.zeros(self.num_classes)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        predictions = F.softmax(predictions, dim=1)
        
        for cls in range(self.num_classes):
            pred = predictions[:, cls]
            target = (targets == cls).float()
            
            self.intersection[cls] += (pred * target).sum().item()
            self.sum_pred[cls] += pred.sum().item()
            self.sum_target[cls] += target.sum().item()
    
    def compute(self) -> Union[float, Dict[str, float]]:
        dice_per_class = (2. * self.intersection + self.smooth) / (
            self.sum_pred + self.sum_target + self.smooth
        )
        mean_dice = dice_per_class.mean().item()
        
        return {
            'mean_dice': mean_dice,
            'dice_per_class': {f'class_{i}': dice.item() for i, dice in enumerate(dice_per_class)}
        }

class PixelAccuracy(BaseMetric):
    """Pixel-wise accuracy metric for segmentation tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.ignore_index = self.config.get('ignore_index', -100)
    
    def reset(self):
        self.correct = 0
        self.total = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        predictions = predictions.argmax(dim=1)
        
        # Create mask for valid pixels
        mask = (targets != self.ignore_index)
        predictions = predictions[mask]
        targets = targets[mask]
        
        self.correct += (predictions == targets).sum().item()
        self.total += mask.sum().item()
    
    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0 