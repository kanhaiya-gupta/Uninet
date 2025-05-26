import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List
import numpy as np
from .base import BaseMetric

class HitRate(BaseMetric):
    """Hit Rate metric for recommendation systems."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.k = self.config.get('k', 10)
    
    def reset(self):
        self.hits = 0
        self.total = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: User-item scores
        # targets: Ground truth items
        _, top_k_indices = torch.topk(predictions, self.k, dim=1)
        
        for pred_items, target_items in zip(top_k_indices, targets):
            hits = torch.isin(pred_items, target_items).any().item()
            self.hits += hits
            self.total += 1
    
    def compute(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0

class NDCG(BaseMetric):
    """Normalized Discounted Cumulative Gain metric for recommendation systems."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.k = self.config.get('k', 10)
    
    def reset(self):
        self.total_ndcg = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: User-item scores
        # targets: Ground truth items
        _, top_k_indices = torch.topk(predictions, self.k, dim=1)
        
        for pred_items, target_items in zip(top_k_indices, targets):
            # Calculate DCG
            dcg = 0.0
            for i, item in enumerate(pred_items):
                if item in target_items:
                    dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0
            
            # Calculate IDCG (ideal DCG)
            idcg = 0.0
            for i in range(min(len(target_items), self.k)):
                idcg += 1.0 / np.log2(i + 2)
            
            # Calculate NDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            self.total_ndcg += ndcg
            self.count += 1
    
    def compute(self) -> float:
        return self.total_ndcg / self.count if self.count > 0 else 0.0

class MRR(BaseMetric):
    """Mean Reciprocal Rank metric for recommendation systems."""
    
    def reset(self):
        self.total_reciprocal_rank = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: User-item scores
        # targets: Ground truth items
        _, indices = torch.sort(predictions, descending=True)
        
        for pred_items, target_items in zip(indices, targets):
            for rank, item in enumerate(pred_items, 1):
                if item in target_items:
                    self.total_reciprocal_rank += 1.0 / rank
                    break
            self.count += 1
    
    def compute(self) -> float:
        return self.total_reciprocal_rank / self.count if self.count > 0 else 0.0

class Coverage(BaseMetric):
    """Coverage metric for recommendation systems."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.k = self.config.get('k', 10)
        self.total_items = self.config.get('total_items', None)
        if self.total_items is None:
            raise ValueError("Total number of items must be provided for coverage calculation")
    
    def reset(self):
        self.recommended_items = set()
        self.total_recommendations = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: User-item scores
        _, top_k_indices = torch.topk(predictions, self.k, dim=1)
        
        for pred_items in top_k_indices:
            self.recommended_items.update(pred_items.tolist())
            self.total_recommendations += self.k
    
    def compute(self) -> float:
        return len(self.recommended_items) / self.total_items

class Diversity(BaseMetric):
    """Diversity metric for recommendation systems."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.k = self.config.get('k', 10)
        self.item_embeddings = self.config.get('item_embeddings', None)
        if self.item_embeddings is None:
            raise ValueError("Item embeddings must be provided for diversity calculation")
    
    def reset(self):
        self.total_diversity = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: User-item scores
        _, top_k_indices = torch.topk(predictions, self.k, dim=1)
        
        for pred_items in top_k_indices:
            # Get embeddings for recommended items
            item_embeddings = self.item_embeddings[pred_items]
            
            # Calculate average pairwise cosine similarity
            similarity = F.cosine_similarity(
                item_embeddings.unsqueeze(1),
                item_embeddings.unsqueeze(0),
                dim=2
            )
            
            # Exclude self-similarities
            similarity = similarity - torch.eye(self.k, device=similarity.device)
            
            # Calculate diversity as 1 - average similarity
            diversity = 1 - similarity.sum() / (self.k * (self.k - 1))
            
            self.total_diversity += diversity.item()
            self.count += 1
    
    def compute(self) -> float:
        return self.total_diversity / self.count if self.count > 0 else 0.0 