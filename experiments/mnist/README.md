# MNIST Classification using Feedforward Neural Network

This experiment demonstrates the use of a Feedforward Neural Network (FNN) for classifying handwritten digits from the MNIST dataset.

## Dataset

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels. The dataset is split into:
- 60,000 training images
- 10,000 test images

## Model Architecture

The FNN model consists of:
- Input layer: 784 neurons (28x28 flattened images)
- Hidden layers: 2 layers with 128 neurons each
- Output layer: 10 neurons (one for each digit)
- Activation: ReLU
- Dropout: 0.2
- Batch Normalization: Yes

## Training

The model is trained with the following configuration:
- Batch size: 64
- Epochs: 10
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: Cross Entropy
- Early stopping: Patience of 5 epochs

## Evaluation Metrics

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Results

The evaluation results are saved in:
- `evaluation_results.txt`: Numerical metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `sample_predictions.png`: Sample predictions with probabilities

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train.py
```

3. Evaluate the model:
```bash
python evaluate.py
```

## Directory Structure

```
mnist/
├── config.py           # Configuration parameters
├── train.py           # Training script
├── evaluate.py        # Evaluation script
├── README.md          # This file
└── checkpoints/       # Saved models and results
    ├── best_model.pth
    ├── evaluation_results.txt
    ├── confusion_matrix.png
    └── sample_predictions.png
```

## Visualization

The training process can be monitored using TensorBoard:
```bash
tensorboard --logdir checkpoints/tensorboard
```

## Customization

You can modify the model architecture and training parameters in `config.py`:
- Change the number of hidden layers
- Adjust the number of neurons per layer
- Modify the activation function
- Change the dropout rate
- Adjust the training parameters 