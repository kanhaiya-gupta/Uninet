import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Tuple
from .base import BaseLoss

class GNNLoss(BaseLoss):
    """Loss function for Graph Neural Networks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.task_type = self.config.get('task_type', 'node_classification')
        self.edge_weight = self.config.get('edge_weight', 0.1)
        self.structure_weight = self.config.get('structure_weight', 0.01)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute GNN loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            edge_index: Graph connectivity
            edge_attr: Edge features
            **kwargs: Additional arguments
        
        Returns:
            GNN loss and optional metrics
        """
        metrics = {}
        
        if self.task_type == 'node_classification':
            # Node classification loss
            loss = F.cross_entropy(predictions, targets)
            metrics['classification_loss'] = loss
            
        elif self.task_type == 'graph_classification':
            # Graph classification loss
            loss = F.cross_entropy(predictions, targets)
            metrics['classification_loss'] = loss
            
        elif self.task_type == 'link_prediction':
            # Link prediction loss
            loss = F.binary_cross_entropy_with_logits(predictions, targets)
            metrics['link_prediction_loss'] = loss
            
        elif self.task_type == 'node_regression':
            # Node regression loss
            loss = F.mse_loss(predictions, targets)
            metrics['regression_loss'] = loss
        
        # Add edge prediction loss if edge information is provided
        if edge_index is not None and edge_attr is not None:
            edge_loss = F.mse_loss(edge_attr, predictions[edge_index[0]] - predictions[edge_index[1]])
            loss += self.edge_weight * edge_loss
            metrics['edge_loss'] = edge_loss
        
        # Add structure preservation loss
        if edge_index is not None:
            # Encourage similar nodes to have similar embeddings
            structure_loss = torch.mean(
                torch.abs(
                    predictions[edge_index[0]] - predictions[edge_index[1]]
                )
            )
            loss += self.structure_weight * structure_loss
            metrics['structure_loss'] = structure_loss
        
        return loss, metrics 