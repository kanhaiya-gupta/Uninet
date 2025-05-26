"""Graph-specific metrics for neural networks."""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List, Tuple
from .base import BaseMetric
import numpy as np

class NodeClassificationAccuracy(BaseMetric):
    """Accuracy metric for node classification tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.ignore_index = self.config.get('ignore_index', -100)
    
    def reset(self):
        self.correct = 0
        self.total = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs):
        # Get predicted classes
        pred_classes = predictions.argmax(dim=-1)
        
        # Apply mask if provided
        if mask is not None:
            pred_classes = pred_classes[mask]
            targets = targets[mask]
        
        # Count correct predictions
        correct = (pred_classes == targets).sum().item()
        total = targets.numel()
        
        self.correct += correct
        self.total += total
    
    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

class GraphClassificationAccuracy(BaseMetric):
    """Accuracy metric for graph classification tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
    
    def reset(self):
        self.correct = 0
        self.total = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # Get predicted classes
        pred_classes = predictions.argmax(dim=-1)
        
        # Count correct predictions
        correct = (pred_classes == targets).sum().item()
        total = targets.numel()
        
        self.correct += correct
        self.total += total
    
    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

class EdgePredictionMetrics(BaseMetric):
    """Metrics for edge prediction tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.threshold = self.config.get('threshold', 0.5)
    
    def reset(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.total = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs):
        # Apply sigmoid and threshold
        pred_probs = torch.sigmoid(predictions)
        pred_classes = (pred_probs > self.threshold).float()
        
        # Apply mask if provided
        if mask is not None:
            pred_classes = pred_classes[mask]
            targets = targets[mask]
        
        # Calculate metrics
        tp = ((pred_classes == 1) & (targets == 1)).sum().item()
        fp = ((pred_classes == 1) & (targets == 0)).sum().item()
        fn = ((pred_classes == 0) & (targets == 1)).sum().item()
        total = targets.numel()
        
        self.true_positives += tp
        self.false_positives += fp
        self.false_negatives += fn
        self.total += total
    
    def compute(self) -> Dict[str, float]:
        precision = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0.0
        recall = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

class GraphSimilarityMetrics(BaseMetric):
    """Metrics for measuring graph similarity."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
    
    def reset(self):
        self.total_similarity = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # Calculate cosine similarity between graph embeddings
        similarity = F.cosine_similarity(predictions, targets, dim=-1)
        self.total_similarity += similarity.sum().item()
        self.count += similarity.numel()
    
    def compute(self) -> float:
        return self.total_similarity / self.count if self.count > 0 else 0.0 