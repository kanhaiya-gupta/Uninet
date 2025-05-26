import torch
import torch.nn as nn
import math
from typing import Dict, Any, List, Tuple, Optional
from ..base import BaseNeuralNetwork

class BertEmbedding(nn.Module):
    """BERT embedding layer."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_length: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Position embedding
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Segment embedding
        self.segment_embedding = nn.Embedding(2, d_model)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Get sequence length
        seq_length = input_ids.size(1)
        
        # Create position indices
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Create token type ids if not provided
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Get embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        segment_embeddings = self.segment_embedding(token_type_ids)
        
        # Combine embeddings
        embeddings = token_embeddings + position_embeddings + segment_embeddings
        
        # Apply layer normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BertLayer(nn.Module):
    """BERT layer."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = residual + self.dropout(x)
        
        # Feed-forward network
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        
        return x

class BERT(BaseNeuralNetwork):
    """BERT implementation with configurable architecture."""
    
    def _build_model(self) -> nn.Module:
        """Build the BERT architecture based on configuration."""
        # Get configuration parameters
        vocab_size = self.config.get('vocab_size', 30522)  # BERT base vocab size
        d_model = self.config.get('d_model', 768)
        num_heads = self.config.get('num_heads', 12)
        num_layers = self.config.get('num_layers', 12)
        d_ff = self.config.get('d_ff', 3072)
        max_seq_length = self.config.get('max_seq_length', 512)
        dropout = self.config.get('dropout', 0.1)
        output_size = self.config.get('output_size', 2)  # Default for binary classification
        
        # Embedding layer
        self.embedding = BertEmbedding(vocab_size, d_model, max_seq_length, dropout)
        
        # BERT layers
        self.layers = nn.ModuleList([
            BertLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Pooler
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, output_size)
        
        # Add final activation based on task type
        if self.config['task_type'] == 'classification':
            self.final_activation = nn.Softmax(dim=-1)
        elif self.config['task_type'] == 'regression':
            self.final_activation = nn.Identity()
        
        return nn.Module()  # Placeholder, we'll use the layers directly
    
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the network."""
        # Get embeddings
        x = self.embedding(input_ids, token_type_ids)
        
        # Create attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # BERT layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Pooling
        x = self.pooler(x[:, 0])  # Use [CLS] token
        
        # Output projection
        x = self.output_proj(x)
        x = self.final_activation(x)
        
        return x
    
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