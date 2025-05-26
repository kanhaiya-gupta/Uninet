import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Tuple, List
from .base import BaseLoss

class SNNLoss(BaseLoss):
    """Loss function for Spiking Neural Networks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.spike_weight = self.config.get('spike_weight', 0.1)
        self.membrane_weight = self.config.get('membrane_weight', 0.01)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        membrane_potentials: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Classification loss
        loss = F.cross_entropy(predictions, targets)
        
        # Add membrane potential regularization if provided
        if membrane_potentials is not None:
            membrane_loss = sum(
                torch.mean(torch.abs(potential))
                for potential in membrane_potentials
            )
            loss += self.membrane_weight * membrane_loss
        
        return loss

class QNNLoss(BaseLoss):
    """Loss function for Quantum Neural Networks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.phase_weight = self.config.get('phase_weight', 0.01)
        self.entanglement_weight = self.config.get('entanglement_weight', 0.01)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        phase: Optional[torch.Tensor] = None,
        entanglement: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Main task loss
        loss = F.cross_entropy(predictions, targets)
        
        # Add quantum-specific regularization
        if phase is not None:
            phase_loss = torch.mean(torch.abs(phase))
            loss += self.phase_weight * phase_loss
        
        if entanglement is not None:
            entanglement_loss = torch.mean(torch.abs(entanglement))
            loss += self.entanglement_weight * entanglement_loss
        
        return loss

class NeRFLoss(BaseLoss):
    """Loss function for Neural Radiance Fields."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.rgb_weight = self.config.get('rgb_weight', 1.0)
        self.depth_weight = self.config.get('depth_weight', 0.1)
        self.sparsity_weight = self.config.get('sparsity_weight', 0.01)
        self.smoothness_weight = self.config.get('smoothness_weight', 0.01)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        density: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # RGB reconstruction loss
        rgb_loss = F.mse_loss(predictions, targets)
        total_loss = self.rgb_weight * rgb_loss
        metrics = {'rgb_loss': rgb_loss}
        
        # Add depth supervision if provided
        if depth is not None:
            depth_loss = F.mse_loss(depth, kwargs.get('target_depth', depth))
            total_loss += self.depth_weight * depth_loss
            metrics['depth_loss'] = depth_loss
        
        # Add density regularization if provided
        if density is not None:
            # Density sparsity loss
            sparsity_loss = torch.mean(torch.abs(density))
            total_loss += self.sparsity_weight * sparsity_loss
            metrics['sparsity_loss'] = sparsity_loss
            
            # Density smoothness loss
            smoothness_loss = torch.mean(torch.abs(density[..., 1:] - density[..., :-1]))
            total_loss += self.smoothness_weight * smoothness_loss
            metrics['smoothness_loss'] = smoothness_loss
        
        return total_loss, metrics

class NTKLoss(BaseLoss):
    """Loss function for Neural Tangent Kernel networks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.kernel_weight = self.config.get('kernel_weight', 0.1)
        self.regularization_weight = self.config.get('regularization_weight', 0.01)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        kernel: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Main task loss
        loss = F.cross_entropy(predictions, targets)
        
        # Add kernel regularization if provided
        if kernel is not None:
            # Kernel alignment loss
            kernel_loss = torch.mean(torch.abs(kernel - torch.eye(kernel.size(0), device=kernel.device)))
            loss += self.kernel_weight * kernel_loss
            
            # Kernel smoothness loss
            smoothness_loss = torch.mean(torch.abs(kernel[1:] - kernel[:-1]))
            loss += self.regularization_weight * smoothness_loss
        
        return loss

class NASLoss(BaseLoss):
    """Loss function for Neural Architecture Search."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.arch_weight = self.config.get('arch_weight', 0.1)
        self.entropy_weight = self.config.get('entropy_weight', 0.01)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        arch_weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Main task loss
        loss = F.cross_entropy(predictions, targets)
        
        # Add architecture-specific regularization if provided
        if arch_weights is not None:
            # Architecture entropy loss (encourage exploration)
            entropy = -torch.sum(arch_weights * torch.log(arch_weights + 1e-8))
            loss += self.entropy_weight * entropy
            
            # Architecture sparsity loss (encourage selection)
            sparsity_loss = torch.mean(torch.abs(arch_weights))
            loss += self.arch_weight * sparsity_loss
        
        return loss

class RBFNLoss(BaseLoss):
    """Loss function for Radial Basis Function Networks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.approximation_weight = self.config.get('approximation_weight', 1.0)
        self.smoothness_weight = self.config.get('smoothness_weight', 0.1)
        self.center_weight = self.config.get('center_weight', 0.01)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        centers: Optional[torch.Tensor] = None,
        widths: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Approximation error
        approximation_loss = F.mse_loss(predictions, targets)
        total_loss = self.approximation_weight * approximation_loss
        metrics = {'approximation_loss': approximation_loss}
        
        # Add smoothness regularization if widths are provided
        if widths is not None:
            smoothness_loss = torch.mean(torch.abs(widths[1:] - widths[:-1]))
            total_loss += self.smoothness_weight * smoothness_loss
            metrics['smoothness_loss'] = smoothness_loss
        
        # Add center regularization if centers are provided
        if centers is not None:
            center_loss = torch.mean(torch.abs(centers[1:] - centers[:-1]))
            total_loss += self.center_weight * center_loss
            metrics['center_loss'] = center_loss
        
        return total_loss, metrics

class SOMLoss(BaseLoss):
    """Loss function for Self-Organizing Maps."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.quantization_weight = self.config.get('quantization_weight', 1.0)
        self.topology_weight = self.config.get('topology_weight', 0.1)
        self.neighborhood_size = self.config.get('neighborhood_size', 1)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Quantization error
        quantization_loss = F.mse_loss(predictions, targets)
        total_loss = self.quantization_weight * quantization_loss
        metrics = {'quantization_loss': quantization_loss}
        
        # Add topology preservation loss if weights are provided
        if weights is not None:
            # Calculate distances in input space
            input_distances = torch.cdist(targets, targets)
            
            # Calculate distances in output space (SOM grid)
            output_distances = torch.cdist(weights, weights)
            
            # Calculate topology preservation error
            topology_loss = F.mse_loss(input_distances, output_distances)
            total_loss += self.topology_weight * topology_loss
            metrics['topology_loss'] = topology_loss
        
        return total_loss, metrics

class NeuralODELoss(BaseLoss):
    """Loss function for Neural ODEs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.trajectory_weight = self.config.get('trajectory_weight', 1.0)
        self.conservation_weight = self.config.get('conservation_weight', 0.1)
        self.stability_weight = self.config.get('stability_weight', 0.01)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        conserved_quantities: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Trajectory error
        trajectory_loss = F.mse_loss(predictions, targets)
        total_loss = self.trajectory_weight * trajectory_loss
        metrics = {'trajectory_loss': trajectory_loss}
        
        # Add conservation loss if conserved quantities are provided
        if conserved_quantities is not None:
            conservation_loss = 0.0
            for quantity in conserved_quantities:
                # Calculate conserved quantity for predictions and targets
                pred_quantity = quantity(predictions)
                target_quantity = quantity(targets)
                
                # Calculate error in conservation
                conservation_loss += F.mse_loss(pred_quantity, target_quantity)
            
            total_loss += self.conservation_weight * conservation_loss
            metrics['conservation_loss'] = conservation_loss
        
        # Add stability loss
        # Add small perturbation to initial state
        perturbed_state = predictions[0] + 0.01 * torch.randn_like(predictions[0])
        stability_loss = F.mse_loss(predictions, perturbed_state)
        total_loss += self.stability_weight * stability_loss
        metrics['stability_loss'] = stability_loss
        
        return total_loss, metrics

class GNNLoss(BaseLoss):
    """Loss function for Graph Neural Networks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.task_weight = self.config.get('task_weight', 1.0)
        self.edge_weight = self.config.get('edge_weight', 0.1)
        self.regularization_weight = self.config.get('regularization_weight', 0.01)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Main task loss (node/graph classification/regression)
        task_loss = F.cross_entropy(predictions, targets)
        total_loss = self.task_weight * task_loss
        metrics = {'task_loss': task_loss}
        
        # Add edge prediction loss if provided
        if edge_index is not None and edge_attr is not None:
            edge_loss = F.binary_cross_entropy_with_logits(
                edge_attr,
                kwargs.get('target_edge_attr', edge_attr)
            )
            total_loss += self.edge_weight * edge_loss
            metrics['edge_loss'] = edge_loss
        
        # Add regularization if node features are provided
        if 'node_features' in kwargs:
            # L2 regularization on node features
            reg_loss = torch.mean(torch.norm(kwargs['node_features'], dim=1))
            total_loss += self.regularization_weight * reg_loss
            metrics['regularization_loss'] = reg_loss
        
        return total_loss, metrics 