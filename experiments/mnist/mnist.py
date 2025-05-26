"""MNIST classification experiment."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from typing import Dict, Any, Tuple
import os
import logging

from src.base import BaseExperiment
from src.models.feedforward.fnn import FNN
from src.metrics import Accuracy, Precision, Recall, F1Score
from src.losses import CrossEntropyLoss
from src.optimizers import create_optimizer

class MNISTExperiment(BaseExperiment):
    """MNIST classification experiment using Feedforward Neural Network."""
    
    def create_model(self) -> nn.Module:
        """Create the FNN model."""
        return FNN(self.config['model'])
    
    def create_loss(self) -> nn.Module:
        """Create the loss function."""
        return CrossEntropyLoss()
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create the optimizer."""
        return create_optimizer(model, self.config['training'])
    
    def create_metrics(self) -> Dict[str, Any]:
        """Create the metrics."""
        return {
            'accuracy': Accuracy(),
            'precision': Precision(),
            'recall': Recall(),
            'f1': F1Score()
        }
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and preprocess MNIST dataset."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load training data
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        
        # Split into train and validation
        train_size = int(len(train_dataset) * self.config['data']['train_size'])
        val_size = int(len(train_dataset) * self.config['data']['val_size'])
        test_size = len(train_dataset) - train_size - val_size
        
        train_dataset, val_dataset, _ = random_split(
            train_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.config['data']['random_seed'])
        )
        
        # Load test data
        test_dataset = datasets.MNIST('data', train=False, transform=transform)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=self.config['data']['shuffle']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   criterion: nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0
        metrics = self.create_metrics()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            data = data.view(data.size(0), -1)  # Flatten the images
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update metrics
            for metric in metrics.values():
                metric.update(output, target)
            
            if batch_idx % self.config['logging']['log_interval'] == 0:
                logging.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                           f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Log metrics
        avg_loss = total_loss / len(train_loader)
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        
        for name, metric in metrics.items():
            value = metric.compute()
            self.writer.add_scalar(f'{name}/train', value, epoch)
            logging.info(f'Train {name}: {value:.4f}')
        
        return avg_loss
    
    def validate(self, model: nn.Module, val_loader: DataLoader,
                criterion: nn.Module, epoch: int) -> float:
        """Validate the model."""
        model.eval()
        val_loss = 0
        metrics = self.create_metrics()
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.view(data.size(0), -1)  # Flatten the images
                
                output = model(data)
                val_loss += criterion(output, target).item()
                
                # Update metrics
                for metric in metrics.values():
                    metric.update(output, target)
        
        val_loss /= len(val_loader)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        
        for name, metric in metrics.items():
            value = metric.compute()
            self.writer.add_scalar(f'{name}/val', value, epoch)
            logging.info(f'Validation {name}: {value:.4f}')
        
        return val_loss
    
    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on test data."""
        model.eval()
        all_predictions = []
        all_targets = []
        metrics = self.create_metrics()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.view(data.size(0), -1)  # Flatten the images
                
                output = model(data)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # Update metrics
                for metric in metrics.values():
                    metric.update(output, target)
        
        # Compute final metrics
        results = {}
        for name, metric in metrics.items():
            value = metric.compute()
            results[name] = value
            logging.info(f'Test {name}: {value:.4f}')
        
        # Plot confusion matrix
        self.plot_confusion_matrix(all_predictions, all_targets)
        
        # Plot sample predictions
        self.plot_sample_predictions(model, test_loader)
        
        # Save results
        self.save_results(results)
        
        return results
    
    def plot_confusion_matrix(self, predictions: list, targets: list):
        """Plot confusion matrix."""
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(self.config['logging']['save_dir'], 'confusion_matrix.png'))
        plt.close()
    
    def plot_sample_predictions(self, model: nn.Module, test_loader: DataLoader, num_samples: int = 10):
        """Plot sample predictions."""
        model.eval()
        with torch.no_grad():
            data, target = next(iter(test_loader))
            data, target = data.to(self.device), target.to(self.device)
            data = data.view(data.size(0), -1)
            
            output = model(data)
            predictions = output.argmax(dim=1)
            
            # Plot samples
            fig, axes = plt.subplots(2, num_samples, figsize=(20, 4))
            for i in range(num_samples):
                # Original image
                img = data[i].view(28, 28).cpu().numpy()
                axes[0, i].imshow(img, cmap='gray')
                axes[0, i].set_title(f'True: {target[i].item()}')
                axes[0, i].axis('off')
                
                # Prediction probabilities
                probs = output[i].softmax(dim=0).cpu().numpy()
                axes[1, i].bar(range(10), probs)
                axes[1, i].set_title(f'Pred: {predictions[i].item()}')
                axes[1, i].set_xticks(range(10))
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config['logging']['save_dir'], 'sample_predictions.png'))
            plt.close()
    
    def save_results(self, results: Dict[str, float]):
        """Save evaluation results."""
        with open(os.path.join(self.config['logging']['save_dir'], 'evaluation_results.txt'), 'w') as f:
            for metric, value in results.items():
                f.write(f'{metric}: {value:.4f}\n') 