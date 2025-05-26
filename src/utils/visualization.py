"""Visualization utilities for neural network training and evaluation."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot training history including loss and metrics."""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    for metric in history.keys():
        if metric not in ['train_loss', 'val_loss']:
            plt.plot(history[metric], label=metric)
    plt.title('Training Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if labels:
        plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45)
        plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_roc_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot ROC curves for multi-class classification."""
    plt.figure(figsize=(10, 8))
    
    for i in range(y_scores.shape[1]):
        fpr, tpr, _ = roc_curve(y_true == i, y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        label = labels[i] if labels else f'Class {i}'
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_feature_importance(
    importance: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot feature importance."""
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importance)
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance')
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_attention_weights(
    attention_weights: np.ndarray,
    input_tokens: List[str],
    output_tokens: List[str],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot attention weights."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(attention_weights, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.xticks(np.arange(len(output_tokens)) + 0.5, output_tokens, rotation=45)
    plt.yticks(np.arange(len(input_tokens)) + 0.5, input_tokens, rotation=0)
    plt.title('Attention Weights')
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_3d_trajectory(
    trajectory: np.ndarray,
    title: str = '3D Trajectory',
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot 3D trajectory using plotly."""
    fig = go.Figure(data=[go.Scatter3d(
        x=trajectory[:, 0],
        y=trajectory[:, 1],
        z=trajectory[:, 2],
        mode='lines+markers',
        marker=dict(
            size=4,
            color=np.arange(len(trajectory)),
            colorscale='Viridis',
        ),
        line=dict(
            color='darkblue',
            width=2
        )
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    
    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()

def plot_phase_space(
    states: np.ndarray,
    title: str = 'Phase Space',
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot phase space for dynamical systems."""
    plt.figure(figsize=(10, 8))
    plt.plot(states[:, 0], states[:, 1], 'b-', label='Trajectory')
    plt.scatter(states[0, 0], states[0, 1], c='r', label='Start')
    plt.scatter(states[-1, 0], states[-1, 1], c='g', label='End')
    plt.xlabel('State 1')
    plt.ylabel('State 2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_learning_curves(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    title: str = 'Learning Curves',
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot learning curves."""
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, val_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close() 