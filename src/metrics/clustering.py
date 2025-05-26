import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List
import numpy as np
from .base import BaseMetric

class TopologyPreservationError(BaseMetric):
    """Metric for evaluating topology preservation in SOM."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.neighborhood_size = self.config.get('neighborhood_size', 1)
    
    def reset(self):
        self.total_error = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: SOM weights
        # targets: Input data
        
        # Compute distances in input space
        input_distances = self._compute_distances(targets)
        
        # Compute distances in output space (SOM grid)
        output_distances = self._compute_distances(predictions)
        
        # Calculate topology preservation error
        error = F.mse_loss(input_distances, output_distances)
        self.total_error += error.item()
        self.count += 1
    
    def _compute_distances(self, data: torch.Tensor) -> torch.Tensor:
        # Compute pairwise distances
        return torch.cdist(data, data)
    
    def compute(self) -> float:
        return self.total_error / self.count if self.count > 0 else 0.0

class QuantizationError(BaseMetric):
    """Metric for evaluating quantization error in SOM."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
    
    def reset(self):
        self.total_error = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: SOM weights
        # targets: Input data
        
        # Find best matching units
        bmu_indices = self._find_bmu(predictions, targets)
        
        # Calculate quantization error
        error = torch.mean(torch.norm(targets - predictions[bmu_indices], dim=1))
        self.total_error += error.item()
        self.count += 1
    
    def _find_bmu(self, weights: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        # Find best matching unit for each input
        distances = torch.cdist(data, weights)
        return torch.argmin(distances, dim=1)
    
    def compute(self) -> float:
        return self.total_error / self.count if self.count > 0 else 0.0

class ClusteringQuality(BaseMetric):
    """Metric for evaluating clustering quality in SOM."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.metric = self.config.get('metric', 'silhouette')  # 'silhouette' or 'calinski_harabasz'
    
    def reset(self):
        self.total_score = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: SOM weights
        # targets: Input data
        
        # Assign clusters
        clusters = self._assign_clusters(predictions, targets)
        
        # Calculate clustering quality score
        if self.metric == 'silhouette':
            score = self._silhouette_score(targets, clusters)
        else:  # calinski_harabasz
            score = self._calinski_harabasz_score(targets, clusters)
        
        self.total_score += score
        self.count += 1
    
    def _assign_clusters(self, weights: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        # Assign each data point to nearest weight vector
        distances = torch.cdist(data, weights)
        return torch.argmin(distances, dim=1)
    
    def _silhouette_score(self, data: torch.Tensor, clusters: torch.Tensor) -> float:
        # Calculate silhouette score
        # This is a simplified version; full implementation would be more complex
        return 0.0  # Placeholder
    
    def _calinski_harabasz_score(self, data: torch.Tensor, clusters: torch.Tensor) -> float:
        # Calculate Calinski-Harabasz score
        # This is a simplified version; full implementation would be more complex
        return 0.0  # Placeholder
    
    def compute(self) -> float:
        return self.total_score / self.count if self.count > 0 else 0.0 