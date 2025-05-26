import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List
import numpy as np
from .base import BaseMetric

class PhysicsResidual(BaseMetric):
    """Physics residual metric for PINNs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.pde_terms = self.config.get('pde_terms', [])  # List of PDE terms to evaluate
        self.boundary_terms = self.config.get('boundary_terms', [])  # List of boundary conditions
        self.initial_terms = self.config.get('initial_terms', [])  # List of initial conditions
    
    def reset(self):
        self.total_residual = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: Model outputs
        # targets: Ground truth values
        # kwargs should contain:
        #   - pde_points: Points for PDE evaluation
        #   - boundary_points: Points for boundary conditions
        #   - initial_points: Points for initial conditions
        
        residual = 0.0
        
        # Evaluate PDE residuals
        if self.pde_terms and 'pde_points' in kwargs:
            pde_residual = self._evaluate_pde_residual(predictions, kwargs['pde_points'])
            residual += pde_residual
        
        # Evaluate boundary condition residuals
        if self.boundary_terms and 'boundary_points' in kwargs:
            boundary_residual = self._evaluate_boundary_residual(predictions, kwargs['boundary_points'])
            residual += boundary_residual
        
        # Evaluate initial condition residuals
        if self.initial_terms and 'initial_points' in kwargs:
            initial_residual = self._evaluate_initial_residual(predictions, kwargs['initial_points'])
            residual += initial_residual
        
        self.total_residual += residual
        self.count += 1
    
    def _evaluate_pde_residual(self, predictions: torch.Tensor, points: torch.Tensor) -> float:
        residual = 0.0
        for term in self.pde_terms:
            residual += term(predictions, points)
        return residual
    
    def _evaluate_boundary_residual(self, predictions: torch.Tensor, points: torch.Tensor) -> float:
        residual = 0.0
        for term in self.boundary_terms:
            residual += term(predictions, points)
        return residual
    
    def _evaluate_initial_residual(self, predictions: torch.Tensor, points: torch.Tensor) -> float:
        residual = 0.0
        for term in self.initial_terms:
            residual += term(predictions, points)
        return residual
    
    def compute(self) -> float:
        return self.total_residual / self.count if self.count > 0 else 0.0

class ConservationError(BaseMetric):
    """Conservation error metric for PINNs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.conservation_laws = self.config.get('conservation_laws', [])
        self.domain = self.config.get('domain', None)
    
    def reset(self):
        self.total_error = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: Model outputs
        # targets: Ground truth values
        # kwargs should contain integration points
        
        if 'integration_points' not in kwargs:
            raise ValueError("Integration points must be provided for conservation error calculation")
        
        points = kwargs['integration_points']
        error = 0.0
        
        for law in self.conservation_laws:
            error += law(predictions, points)
        
        self.total_error += error
        self.count += 1
    
    def compute(self) -> float:
        return self.total_error / self.count if self.count > 0 else 0.0

class EnergyError(BaseMetric):
    """Energy error metric for PINNs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.energy_function = self.config.get('energy_function', None)
        if self.energy_function is None:
            raise ValueError("Energy function must be provided for energy error calculation")
    
    def reset(self):
        self.total_error = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: Model outputs
        # targets: Ground truth values
        
        predicted_energy = self.energy_function(predictions)
        target_energy = self.energy_function(targets)
        
        error = F.mse_loss(predicted_energy, target_energy)
        self.total_error += error.item()
        self.count += 1
    
    def compute(self) -> float:
        return self.total_error / self.count if self.count > 0 else 0.0

class SymmetryError(BaseMetric):
    """Symmetry error metric for PINNs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.symmetry_operations = self.config.get('symmetry_operations', [])
    
    def reset(self):
        self.total_error = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: Model outputs
        # targets: Ground truth values
        
        error = 0.0
        for operation in self.symmetry_operations:
            transformed = operation(predictions)
            error += F.mse_loss(transformed, predictions)
        
        self.total_error += error.item()
        self.count += 1
    
    def compute(self) -> float:
        return self.total_error / self.count if self.count > 0 else 0.0 