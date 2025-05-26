import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Union
from ..base import BaseNeuralNetwork

class GraphConvolution(nn.Module):
    """Graph Convolution layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Optional[nn.Module] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.activation = activation
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output += self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

class GraphAttention(nn.Module):
    """Graph Attention layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.6,
        alpha: float = 0.2
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W)
        nn.init.kaiming_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Linear transformation
        Wh = torch.matmul(x, self.W)
        
        # Compute attention coefficients
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        
        # Mask attention coefficients
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention
        h_prime = torch.matmul(attention, Wh)
        
        return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        N = Wh.size(1)  # Number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        return all_combinations_matrix.view(Wh.size(0), N, N, 2 * self.out_features)

class GNN(BaseNeuralNetwork):
    """Graph Neural Network implementation with configurable architecture."""
    
    def _build_model(self) -> nn.Module:
        """Build the GNN architecture based on configuration."""
        # Get configuration parameters
        input_features = self.config.get('input_features', 1)
        hidden_features = self.config.get('hidden_features', [64, 32])
        output_features = self.config.get('output_features', 1)
        layer_type = self.config.get('layer_type', 'gcn')  # 'gcn' or 'gat'
        dropout = self.config.get('dropout', 0.5)
        activation = self.config.get('activation', 'relu')
        use_batch_norm = self.config.get('use_batch_norm', True)
        
        # Build layers
        layers = []
        current_features = input_features
        
        for hidden_size in hidden_features:
            if layer_type == 'gcn':
                layers.append(GraphConvolution(
                    current_features,
                    hidden_size,
                    activation=self._get_activation(activation)
                ))
            else:  # GAT
                layers.append(GraphAttention(
                    current_features,
                    hidden_size,
                    dropout=dropout
                ))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.append(nn.Dropout(dropout))
            current_features = hidden_size
        
        # Output layer
        if layer_type == 'gcn':
            layers.append(GraphConvolution(
                current_features,
                output_features
            ))
        else:  # GAT
            layers.append(GraphAttention(
                current_features,
                output_features,
                dropout=dropout
            ))
        
        self.layers = nn.ModuleList(layers)
        
        return nn.Module()  # Placeholder, we'll use layers directly
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        for layer in self.layers:
            if isinstance(layer, (GraphConvolution, GraphAttention)):
                x = layer(x, adj)
            else:
                x = layer(x)
        return x
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(name.lower(), nn.ReLU())