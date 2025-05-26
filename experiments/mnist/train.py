"""Training script for MNIST classification using Feedforward Neural Network."""

import os
import sys
import yaml

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from mnist import MNISTExperiment

def main():
    """Main training function."""
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create and run experiment
    experiment = MNISTExperiment(config)
    experiment.run()

if __name__ == '__main__':
    main() 