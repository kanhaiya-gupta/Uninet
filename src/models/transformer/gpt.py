import torch
import torch.nn as nn
import math
from typing import Dict, Any, List, Tuple, Optional
from ..base import BaseNeuralNetwork

class GPTEmbedding(nn.Module):
    """GPT embedding layer."""
    
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
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Get sequence length
        seq_length = input_ids.size(1)
        
        # Create position indices
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Get embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        
        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        
        # Apply layer normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class GPTLayer(nn.Module):
    """GPT layer."""
    
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

class GPT(BaseNeuralNetwork):
    """GPT implementation with configurable architecture."""
    
    def _build_model(self) -> nn.Module:
        """Build the GPT architecture based on configuration."""
        # Get configuration parameters
        vocab_size = self.config.get('vocab_size', 50257)  # GPT-2 vocab size
        d_model = self.config.get('d_model', 768)
        num_heads = self.config.get('num_heads', 12)
        num_layers = self.config.get('num_layers', 12)
        d_ff = self.config.get('d_ff', 3072)
        max_seq_length = self.config.get('max_seq_length', 1024)
        dropout = self.config.get('dropout', 0.1)
        
        # Embedding layer
        self.embedding = GPTEmbedding(vocab_size, d_model, max_seq_length, dropout)
        
        # GPT layers
        self.layers = nn.ModuleList([
            GPTLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights with embedding
        self.output.weight = self.embedding.token_embedding.weight
        
        return nn.Module()  # Placeholder, we'll use the layers directly
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the network."""
        # Get embeddings
        x = self.embedding(input_ids)
        
        # Create causal mask
        seq_length = input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=input_ids.device),
            diagonal=1
        ).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Combine with attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask & ~causal_mask
            attention_mask = (1.0 - attention_mask.float()) * -10000.0
        else:
            attention_mask = causal_mask.float() * -10000.0
        
        # GPT layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Output layer
        logits = self.output(x)
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_return_sequences: int = 1
    ) -> torch.Tensor:
        """Generate text using the model."""
        batch_size = input_ids.size(0)
        generated = input_ids
        
        for _ in range(max_length - input_ids.size(1)):
            # Get model predictions
            with torch.no_grad():
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=num_return_sequences)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_tokens], dim=1)
        
        return generated
    
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