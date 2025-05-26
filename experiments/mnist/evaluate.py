"""Evaluation script for MNIST classification using Feedforward Neural Network."""

import os
import sys
import yaml

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from mnist import MNISTExperiment

def main():
    """Main evaluation function."""
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create experiment
    experiment = MNISTExperiment(config)
    
    # Load best model
    model = experiment.create_model().to(experiment.device)
    experiment.load_model(model, os.path.join(config['logging']['save_dir'], 'best_model.pth'))
    
    # Load test data
    _, _, test_loader = experiment.load_data()
    
    # Evaluate
    experiment.evaluate(model, test_loader)

if __name__ == '__main__':
    main() 