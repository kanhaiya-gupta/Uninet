"""API endpoints for managing neural network hyperparameters."""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional, Union, Literal
from pydantic import BaseModel, Field, validator
import json
import os

router = APIRouter(prefix="/hyperparameters", tags=["hyperparameters"])

# Loss function configuration
class LossConfig(BaseModel):
    type: Literal["cross_entropy", "mse", "mae", "bce", "bce_with_logits", "kl_div", "huber", "smooth_l1"] = Field(..., description="Loss function type")
    
    # Cross entropy specific
    reduction: Optional[Literal["mean", "sum", "none"]] = Field("mean", description="Reduction method")
    label_smoothing: Optional[float] = Field(0.0, description="Label smoothing factor")
    
    # MSE/MAE specific
    reduction: Optional[Literal["mean", "sum", "none"]] = Field("mean", description="Reduction method")
    
    # BCE specific
    pos_weight: Optional[float] = Field(None, description="Positive weight for BCE loss")
    
    # Huber/Smooth L1 specific
    beta: Optional[float] = Field(1.0, description="Beta parameter for Huber/Smooth L1 loss")

# Optimizer configuration
class OptimizerConfig(BaseModel):
    type: Literal["adam", "sgd", "rmsprop"] = Field(..., description="Optimizer type")
    learning_rate: float = Field(..., description="Learning rate")
    weight_decay: float = Field(0.0, description="Weight decay (L2 penalty)")
    
    # Adam specific
    beta1: Optional[float] = Field(0.9, description="Adam beta1 parameter")
    beta2: Optional[float] = Field(0.999, description="Adam beta2 parameter")
    eps: Optional[float] = Field(1e-8, description="Adam epsilon parameter")
    
    # SGD specific
    momentum: Optional[float] = Field(0.0, description="SGD momentum")
    nesterov: Optional[bool] = Field(False, description="Whether to use Nesterov momentum")
    
    # RMSprop specific
    alpha: Optional[float] = Field(0.99, description="RMSprop alpha parameter")
    centered: Optional[bool] = Field(False, description="Whether to use centered RMSprop")

# Base configuration models
class BaseConfig(BaseModel):
    # Training configuration
    batch_size: int = Field(..., description="Batch size for training")
    epochs: int = Field(..., description="Number of training epochs")
    optimizer: OptimizerConfig = Field(..., description="Optimizer configuration")
    loss: LossConfig = Field(..., description="Loss function configuration")
    
    # Data configuration
    train_size: float = Field(0.8, description="Training set size ratio")
    val_size: float = Field(0.1, description="Validation set size ratio")
    test_size: float = Field(0.1, description="Test set size ratio")
    shuffle: bool = Field(True, description="Whether to shuffle the data")
    random_seed: int = Field(42, description="Random seed for reproducibility")
    
    # Logging configuration
    save_dir: str = Field("checkpoints", description="Directory to save model checkpoints")
    log_interval: int = Field(100, description="Interval for logging training progress")
    save_interval: int = Field(1, description="Interval for saving model checkpoints")
    tensorboard: bool = Field(True, description="Whether to use TensorBoard logging")

# FNN specific configuration
class FNNConfig(BaseConfig):
    network_type: Literal["fnn"] = "fnn"
    task_type: Literal["classification", "regression"] = Field(..., description="Task type")
    input_size: int = Field(..., description="Size of input features")
    hidden_layers: List[int] = Field(..., description="List of hidden layer sizes")
    output_size: int = Field(..., description="Size of output features")
    activation: Literal["relu", "sigmoid", "tanh", "leaky_relu", "elu"] = Field("relu", description="Activation function")
    dropout: float = Field(0.2, description="Dropout rate")
    use_batch_norm: bool = Field(True, description="Whether to use batch normalization")
    
    @validator('loss')
    def validate_loss_for_task(cls, v, values):
        task_type = values.get('task_type')
        if task_type == 'classification':
            if v.type not in ['cross_entropy', 'bce', 'bce_with_logits']:
                raise ValueError(f"Classification tasks require cross entropy or BCE loss, got {v.type}")
        elif task_type == 'regression':
            if v.type not in ['mse', 'mae', 'huber', 'smooth_l1']:
                raise ValueError(f"Regression tasks require MSE, MAE, Huber, or Smooth L1 loss, got {v.type}")
        return v

# CNN specific configuration
class CNNConfig(BaseConfig):
    network_type: Literal["cnn"] = "cnn"
    task_type: Literal["classification", "regression", "segmentation", "detection"] = Field(..., description="Task type")
    input_channels: int = Field(..., description="Number of input channels")
    conv_layers: List[Dict[str, Any]] = Field(..., description="List of convolutional layer configurations")
    dense_layers: List[int] = Field(..., description="List of dense layer sizes")
    output_size: int = Field(..., description="Size of output features")
    activation: Literal["relu", "sigmoid", "tanh", "leaky_relu", "elu"] = Field("relu", description="Activation function")
    dropout: float = Field(0.2, description="Dropout rate")
    use_batch_norm: bool = Field(True, description="Whether to use batch normalization")
    
    @validator('loss')
    def validate_loss_for_task(cls, v, values):
        task_type = values.get('task_type')
        if task_type in ['classification', 'detection']:
            if v.type not in ['cross_entropy', 'bce', 'bce_with_logits']:
                raise ValueError(f"Classification/detection tasks require cross entropy or BCE loss, got {v.type}")
        elif task_type in ['regression', 'segmentation']:
            if v.type not in ['mse', 'mae', 'huber', 'smooth_l1']:
                raise ValueError(f"Regression/segmentation tasks require MSE, MAE, Huber, or Smooth L1 loss, got {v.type}")
        return v

# RNN specific configuration
class RNNConfig(BaseConfig):
    network_type: Literal["rnn"] = "rnn"
    task_type: Literal["sequence", "timeseries", "nlp"] = Field(..., description="Task type")
    input_size: int = Field(..., description="Size of input features")
    hidden_size: int = Field(..., description="Size of hidden state")
    num_layers: int = Field(..., description="Number of RNN layers")
    rnn_type: Literal["lstm", "gru", "simple"] = Field(..., description="RNN type")
    bidirectional: bool = Field(False, description="Whether to use bidirectional RNN")
    dropout: float = Field(0.2, description="Dropout rate")
    output_size: int = Field(..., description="Size of output features")
    
    @validator('loss')
    def validate_loss_for_task(cls, v, values):
        task_type = values.get('task_type')
        if task_type == 'nlp':
            if v.type not in ['cross_entropy', 'kl_div']:
                raise ValueError(f"NLP tasks require cross entropy or KL divergence loss, got {v.type}")
        elif task_type in ['sequence', 'timeseries']:
            if v.type not in ['mse', 'mae', 'huber', 'smooth_l1']:
                raise ValueError(f"Sequence/timeseries tasks require MSE, MAE, Huber, or Smooth L1 loss, got {v.type}")
        return v

# Transformer specific configuration
class TransformerConfig(BaseConfig):
    network_type: Literal["transformer"] = "transformer"
    task_type: Literal["nlp", "seq2seq", "timeseries"] = Field(..., description="Task type")
    input_size: int = Field(..., description="Size of input features")
    d_model: int = Field(..., description="Dimension of the model")
    num_heads: int = Field(..., description="Number of attention heads")
    num_layers: int = Field(..., description="Number of transformer layers")
    d_ff: int = Field(..., description="Dimension of feed-forward network")
    dropout: float = Field(0.2, description="Dropout rate")
    max_seq_length: int = Field(..., description="Maximum sequence length")
    output_size: int = Field(..., description="Size of output features")
    
    @validator('loss')
    def validate_loss_for_task(cls, v, values):
        task_type = values.get('task_type')
        if task_type == 'nlp':
            if v.type not in ['cross_entropy', 'kl_div']:
                raise ValueError(f"NLP tasks require cross entropy or KL divergence loss, got {v.type}")
        elif task_type in ['seq2seq', 'timeseries']:
            if v.type not in ['mse', 'mae', 'huber', 'smooth_l1']:
                raise ValueError(f"Seq2seq/timeseries tasks require MSE, MAE, Huber, or Smooth L1 loss, got {v.type}")
        return v

# Union of all possible configurations
NetworkConfig = Union[FNNConfig, CNNConfig, RNNConfig, TransformerConfig]

def get_network_schema() -> Dict[str, Any]:
    """Generate hyperparameters schema from Pydantic models."""
    schema = {}
    
    # Get schema for each network type
    for config_class in [FNNConfig, CNNConfig, RNNConfig, TransformerConfig]:
        network_type = config_class.__annotations__['network_type'].__args__[0]
        task_type = config_class.__annotations__['task_type'].__args__
        
        schema[network_type] = {
            task: config_class.schema()['properties']
            for task in task_type
        }
    
    return schema

# Get available networks and their tasks
@router.get("/networks", response_model=Dict[str, List[str]])
async def get_networks():
    """Get all available neural network types and their supported tasks."""
    schema = get_network_schema()
    networks = {}
    for network, tasks in schema.items():
        networks[network] = list(tasks.keys())
    return networks

# Get hyperparameters schema for a specific network and task
@router.get("/schema/{network}/{task}", response_model=Dict[str, Any])
async def get_hyperparameters_schema(network: str, task: str):
    """Get the hyperparameters schema for a specific network and task."""
    schema = get_network_schema()
    if network in schema and task in schema[network]:
        return schema[network][task]
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Network '{network}' or task '{task}' not found"
    )

# Save network configuration
@router.post("/save", response_model=Dict[str, Any])
async def save_network_config(config: NetworkConfig):
    """Save a neural network configuration for training."""
    try:
        # Convert config to dict
        config_dict = config.dict()
        
        # Create experiment directory if it doesn't exist
        experiment_dir = os.path.join("experiments", config_dict["network_type"])
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save configuration to YAML file
        config_path = os.path.join(experiment_dir, "config.yaml")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        return {
            "message": "Network configuration saved successfully",
            "config": config_dict,
            "path": config_path
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 