import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from ..base import BaseNeuralNetwork

class RNN(BaseNeuralNetwork):
    """Recurrent Neural Network implementation with configurable architecture."""
    
    def _build_model(self) -> nn.Module:
        """Build the RNN architecture based on configuration."""
        # Get configuration parameters
        input_size = self.config.get('input_size', 1)
        hidden_size = self.config.get('hidden_size', 64)
        num_layers = self.config.get('num_layers', 2)
        output_size = self.config.get('output_size', 1)
        bidirectional = self.config.get('bidirectional', False)
        dropout = self.config.get('dropout', 0.2)
        rnn_type = self.config.get('rnn_type', 'lstm')  # 'rnn', 'lstm', 'gru'
        
        # Build RNN layer
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        else:  # vanilla RNN
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        
        # Calculate the size of the output features
        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Build fully connected layer
        self.fc = nn.Linear(rnn_output_size, output_size)
        
        # Add final activation based on task type
        if self.config['task_type'] == 'classification':
            self.final_activation = nn.Softmax(dim=-1)
        elif self.config['task_type'] == 'regression':
            self.final_activation = nn.Identity()
        
        return nn.Module()  # Placeholder, we'll use rnn and fc directly
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state
        if isinstance(self.rnn, nn.LSTM):
            h0 = torch.zeros(
                self.rnn.num_layers * (2 if self.rnn.bidirectional else 1),
                batch_size,
                self.rnn.hidden_size
            ).to(self.device)
            c0 = torch.zeros_like(h0)
            hidden = (h0, c0)
        else:
            hidden = torch.zeros(
                self.rnn.num_layers * (2 if self.rnn.bidirectional else 1),
                batch_size,
                self.rnn.hidden_size
            ).to(self.device)
        
        # RNN forward pass
        output, _ = self.rnn(x, hidden)
        
        # Get the last time step output
        last_output = output[:, -1, :]
        
        # Fully connected layer
        out = self.fc(last_output)
        out = self.final_activation(out)
        
        return out
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU()
        }
        return activations.get(name.lower(), nn.ReLU()) 