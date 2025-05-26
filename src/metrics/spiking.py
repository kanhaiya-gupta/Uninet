import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List
import numpy as np
from .base import BaseMetric

class SpikeTimingError(BaseMetric):
    """Metric for evaluating spike timing accuracy in SNNs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.time_window = self.config.get('time_window', 1.0)  # Time window for spike matching
        self.tolerance = self.config.get('tolerance', 0.1)  # Tolerance for spike timing
    
    def reset(self):
        self.total_error = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: Predicted spike times
        # targets: Ground truth spike times
        
        # Calculate timing error for each spike
        errors = []
        for pred_spike, target_spike in zip(predictions, targets):
            if abs(pred_spike - target_spike) <= self.tolerance:
                errors.append(0.0)
            else:
                errors.append(abs(pred_spike - target_spike))
        
        # Average error over spikes
        spike_error = torch.mean(torch.tensor(errors))
        self.total_error += spike_error.item()
        self.count += 1
    
    def compute(self) -> float:
        return self.total_error / self.count if self.count > 0 else 0.0

class SpikeCountError(BaseMetric):
    """Metric for evaluating spike count accuracy in SNNs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
    
    def reset(self):
        self.total_error = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: Predicted spike counts
        # targets: Ground truth spike counts
        
        error = F.mse_loss(predictions.float(), targets.float())
        self.total_error += error.item()
        self.count += 1
    
    def compute(self) -> float:
        return self.total_error / self.count if self.count > 0 else 0.0

class SpikePatternError(BaseMetric):
    """Metric for evaluating spike pattern accuracy in SNNs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.pattern_length = self.config.get('pattern_length', 10)
    
    def reset(self):
        self.total_error = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: Predicted spike patterns
        # targets: Ground truth spike patterns
        
        # Calculate pattern similarity using cross-correlation
        similarity = F.conv1d(
            predictions.unsqueeze(0).float(),
            targets.unsqueeze(0).float(),
            padding=self.pattern_length - 1
        )
        
        # Convert similarity to error
        error = 1.0 - torch.max(similarity) / (self.pattern_length * torch.sqrt(
            torch.sum(predictions.float() ** 2) * torch.sum(targets.float() ** 2)
        ))
        
        self.total_error += error.item()
        self.count += 1
    
    def compute(self) -> float:
        return self.total_error / self.count if self.count > 0 else 0.0

class EnergyEfficiency(BaseMetric):
    """Metric for evaluating energy efficiency in SNNs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.energy_per_spike = self.config.get('energy_per_spike', 1.0)
    
    def reset(self):
        self.total_energy = 0.0
        self.count = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: Predicted spike counts
        # targets: Ground truth spike counts
        
        # Calculate energy consumption
        energy = torch.sum(predictions.float()) * self.energy_per_spike
        self.total_energy += energy.item()
        self.count += 1
    
    def compute(self) -> float:
        return self.total_energy / self.count if self.count > 0 else 0.0 