import torch
from typing import Dict, Any, Optional, Union, List
import numpy as np

class BaseMetric:
    """Base class for all metrics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the metric.
        
        Args:
            config: Configuration dictionary for the metric
        """
        self.config = config or {}
        self.reset()
    
    def reset(self):
        """Reset the metric state."""
        raise NotImplementedError("Subclasses must implement reset method")
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        """Update the metric state.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional arguments
        """
        raise NotImplementedError("Subclasses must implement update method")
    
    def compute(self) -> Union[float, Dict[str, float]]:
        """Compute the metric value.
        
        Returns:
            Metric value(s)
        """
        raise NotImplementedError("Subclasses must implement compute method")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the metric.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy() 