import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Tuple

class BaseLoss(nn.Module):
    """Base class for all loss functions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the loss function.
        
        Args:
            config: Configuration dictionary for the loss function
        """
        super().__init__()
        self.config = config or {}
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute the loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional arguments specific to the loss function
        
        Returns:
            Either the loss tensor or a tuple of (loss tensor, additional metrics)
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the loss function.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy() 