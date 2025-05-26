import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List
import numpy as np
from .base import BaseMetric

class FeatureReconstructionError(BaseMetric):
    """Metric for evaluating feature reconstruction in DBN."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.layer_index = self.config.get('layer_index', -1)  # Which layer to evaluate
    
    def reset(self):
        self.total_error = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: Reconstructed features
        # targets: Original features
        
        error = F.mse_loss(predictions, targets)
        self.total_error += error.item()
        self.count += 1
    
    def compute(self) -> float:
        return self.total_error / self.count if self.count > 0 else 0.0

class FeatureSparsity(BaseMetric):
    """Metric for evaluating feature sparsity in DBN."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.target_sparsity = self.config.get('target_sparsity', 0.1)
    
    def reset(self):
        self.total_sparsity = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: Feature activations
        
        # Calculate sparsity as the proportion of active units
        sparsity = torch.mean((predictions > 0).float())
        self.total_sparsity += sparsity.item()
        self.count += 1
    
    def compute(self) -> float:
        return self.total_sparsity / self.count if self.count > 0 else 0.0

class FeatureDiversity(BaseMetric):
    """Metric for evaluating feature diversity in DBN."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
    
    def reset(self):
        self.total_diversity = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: Feature activations
        
        # Calculate pairwise cosine similarity between features
        normalized_features = F.normalize(predictions, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_features, normalized_features.t())
        
        # Calculate diversity as 1 - average similarity
        diversity = 1.0 - torch.mean(similarity_matrix)
        self.total_diversity += diversity.item()
        self.count += 1
    
    def compute(self) -> float:
        return self.total_diversity / self.count if self.count > 0 else 0.0

class FeatureStability(BaseMetric):
    """Metric for evaluating feature stability in DBN."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.window_size = self.config.get('window_size', 10)
    
    def reset(self):
        self.feature_history = []
        self.total_stability = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: Feature activations
        
        # Store feature activations in history
        self.feature_history.append(predictions.detach())
        if len(self.feature_history) > self.window_size:
            self.feature_history.pop(0)
        
        # Calculate stability if we have enough history
        if len(self.feature_history) == self.window_size:
            stability = self._calculate_stability()
            self.total_stability += stability
            self.count += 1
    
    def _calculate_stability(self) -> float:
        # Calculate stability as the average correlation between consecutive feature activations
        stability = 0.0
        for i in range(len(self.feature_history) - 1):
            corr = torch.corrcoef(torch.stack([
                self.feature_history[i].flatten(),
                self.feature_history[i + 1].flatten()
            ]))
            stability += corr[0, 1].item()
        return stability / (len(self.feature_history) - 1)
    
    def compute(self) -> float:
        return self.total_stability / self.count if self.count > 0 else 0.0 