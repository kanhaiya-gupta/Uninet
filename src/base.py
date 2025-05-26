import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseNeuralNetwork(nn.Module, ABC):
    """Base class for all neural networks in the project."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the neural network with configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing network parameters
        """
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model()
        self.model.to(self.device)
        
    @abstractmethod
    def _build_model(self) -> nn.Module:
        """
        Build the neural network architecture.
        Must be implemented by each network type.
        
        Returns:
            nn.Module: The PyTorch model
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        Must be implemented by each network type.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        pass
    
    def save(self, path: str):
        """
        Save the model state.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """
        Load the model state.
        
        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config'] 