import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from .base import BaseMetric

class Accuracy(BaseMetric):
    """Accuracy metric for classification tasks."""
    
    def reset(self):
        self.correct = 0
        self.total = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        predictions = predictions.argmax(dim=1)
        self.correct += (predictions == targets).sum().item()
        self.total += targets.size(0)
    
    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

class Precision(BaseMetric):
    """Precision metric for classification tasks."""
    
    def reset(self):
        self.true_positives = 0
        self.false_positives = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        predictions = predictions.argmax(dim=1)
        self.true_positives += ((predictions == 1) & (targets == 1)).sum().item()
        self.false_positives += ((predictions == 1) & (targets == 0)).sum().item()
    
    def compute(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else 0.0

class Recall(BaseMetric):
    """Recall metric for classification tasks."""
    
    def reset(self):
        self.true_positives = 0
        self.false_negatives = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        predictions = predictions.argmax(dim=1)
        self.true_positives += ((predictions == 1) & (targets == 1)).sum().item()
        self.false_negatives += ((predictions == 0) & (targets == 1)).sum().item()
    
    def compute(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator > 0 else 0.0

class F1Score(BaseMetric):
    """F1 Score metric for classification tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.precision = Precision()
        self.recall = Recall()
    
    def reset(self):
        self.precision.reset()
        self.recall.reset()
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        self.precision.update(predictions, targets)
        self.recall.update(predictions, targets)
    
    def compute(self) -> float:
        p = self.precision.compute()
        r = self.recall.compute()
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

class ROCAUC(BaseMetric):
    """ROC AUC metric for classification tasks."""
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        self.predictions.extend(F.softmax(predictions, dim=1)[:, 1].cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self) -> float:
        return roc_auc_score(self.targets, self.predictions)

class AveragePrecision(BaseMetric):
    """Average Precision metric for classification tasks."""
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        self.predictions.extend(F.softmax(predictions, dim=1)[:, 1].cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self) -> float:
        return average_precision_score(self.targets, self.predictions) 