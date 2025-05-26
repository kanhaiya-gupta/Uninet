"""Data utilities for neural network training and evaluation."""

import torch
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

class DataPreprocessor:
    """Base class for data preprocessing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize preprocessor with configuration."""
        self.config = config
        self.scaler = None
    
    def fit(self, data: np.ndarray) -> None:
        """Fit the preprocessor to the data."""
        raise NotImplementedError
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform the data."""
        raise NotImplementedError
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform the data."""
        raise NotImplementedError

class StandardScalerPreprocessor(DataPreprocessor):
    """Standard scaler for numerical data."""
    
    def fit(self, data: np.ndarray) -> None:
        """Fit the standard scaler."""
        self.scaler = StandardScaler()
        self.scaler.fit(data)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using standard scaling."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return self.scaler.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform the scaled data."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return self.scaler.inverse_transform(data)

class MinMaxScalerPreprocessor(DataPreprocessor):
    """Min-max scaler for numerical data."""
    
    def fit(self, data: np.ndarray) -> None:
        """Fit the min-max scaler."""
        self.scaler = MinMaxScaler()
        self.scaler.fit(data)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using min-max scaling."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return self.scaler.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform the scaled data."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return self.scaler.inverse_transform(data)

def create_data_loaders(
    dataset: Dataset,
    batch_size: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def get_image_transforms(
    train: bool = True,
    resize: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    augment: bool = True
) -> transforms.Compose:
    """Create image transforms for training or evaluation."""
    transform_list = []
    
    if resize:
        transform_list.append(transforms.Resize(resize))
    
    if train and augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])
    
    transform_list.append(transforms.ToTensor())
    
    if normalize:
        transform_list.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))
    
    return transforms.Compose(transform_list)

def create_sequence_dataset(
    data: np.ndarray,
    sequence_length: int,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for time series data."""
    sequences = []
    targets = []
    
    for i in range(0, len(data) - sequence_length, stride):
        sequences.append(data[i:i + sequence_length])
        targets.append(data[i + sequence_length])
    
    return np.array(sequences), np.array(targets)

def create_graph_dataset(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    edge_weights: Optional[np.ndarray] = None
) -> Dict[str, torch.Tensor]:
    """Create a graph dataset for GNNs."""
    dataset = {
        'node_features': torch.FloatTensor(node_features),
        'edge_index': torch.LongTensor(edge_index)
    }
    
    if edge_weights is not None:
        dataset['edge_weights'] = torch.FloatTensor(edge_weights)
    
    return dataset 