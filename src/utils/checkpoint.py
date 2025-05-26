"""Checkpoint utilities for saving and loading model states."""

import torch
import os
import json
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime

class CheckpointManager:
    """Manager for saving and loading model checkpoints."""
    
    def __init__(
        self,
        save_dir: str,
        max_to_keep: int = 5,
        save_best_only: bool = True
    ):
        """Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            max_to_keep: Maximum number of checkpoints to keep
            save_best_only: If True, only save the best model
        """
        self.save_dir = save_dir
        self.max_to_keep = max_to_keep
        self.save_best_only = save_best_only
        self.best_metric = float('inf')
        self.checkpoints = []
        
        os.makedirs(save_dir, exist_ok=True)
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        is_best: bool = False
    ) -> str:
        """Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Dictionary of metrics
            config: Model configuration
            is_best: Whether this is the best model so far
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config
        }
        
        # Create checkpoint filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'checkpoint_epoch_{epoch}_{timestamp}.pt'
        if is_best:
            filename = 'best_model.pt'
        
        # Save checkpoint
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)
        
        # Save metadata
        metadata = {
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': timestamp,
            'is_best': is_best
        }
        metadata_path = os.path.join(self.save_dir, f'{filename}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Update checkpoint list
        self.checkpoints.append(path)
        
        # Remove old checkpoints if exceeding max_to_keep
        if len(self.checkpoints) > self.max_to_keep:
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                os.remove(f'{old_checkpoint}.json')
        
        logging.info(f'Saved checkpoint: {path}')
        return path
    
    def load(
        self,
        path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Tuple[int, Dict[str, float], Dict[str, Any]]:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
        
        Returns:
            Tuple of (epoch, metrics, config)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f'Checkpoint not found: {path}')
        
        checkpoint = torch.load(path)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logging.info(f'Loaded checkpoint: {path}')
        return checkpoint['epoch'], checkpoint['metrics'], checkpoint['config']
    
    def load_best(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[int, Dict[str, float], Dict[str, Any]]:
        """Load the best model checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
        
        Returns:
            Tuple of (epoch, metrics, config)
        """
        best_path = os.path.join(self.save_dir, 'best_model.pt')
        return self.load(best_path, model, optimizer)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the latest checkpoint."""
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]
    
    def get_checkpoint_metrics(self, path: str) -> Dict[str, float]:
        """Get metrics from a checkpoint."""
        metadata_path = f'{path}.json'
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f'Checkpoint metadata not found: {metadata_path}')
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata['metrics']

def save_model_summary(
    model: torch.nn.Module,
    save_dir: str,
    input_size: Tuple[int, ...],
    filename: str = 'model_summary.txt'
) -> None:
    """Save model architecture summary.
    
    Args:
        model: Model to summarize
        save_dir: Directory to save summary
        input_size: Input tensor size
        filename: Output filename
    """
    from torchsummary import summary
    
    # Create summary
    summary_str = []
    summary_str.append('Model Architecture Summary')
    summary_str.append('=' * 50)
    summary_str.append(str(model))
    summary_str.append('=' * 50)
    
    # Get parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary_str.append(f'Total Parameters: {total_params:,}')
    summary_str.append(f'Trainable Parameters: {trainable_params:,}')
    summary_str.append(f'Non-trainable Parameters: {total_params - trainable_params:,}')
    
    # Save summary
    path = os.path.join(save_dir, filename)
    with open(path, 'w') as f:
        f.write('\n'.join(summary_str))
    
    logging.info(f'Saved model summary to: {path}')

def save_training_config(
    config: Dict[str, Any],
    save_dir: str,
    filename: str = 'training_config.json'
) -> None:
    """Save training configuration.
    
    Args:
        config: Configuration dictionary
        save_dir: Directory to save config
        filename: Output filename
    """
    path = os.path.join(save_dir, filename)
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
    
    logging.info(f'Saved training config to: {path}') 