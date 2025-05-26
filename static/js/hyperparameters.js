// Define hyperparameters object
const hyperparameters = {
    "fnn": {
        "classification": [
            {
                "name": "hidden_layers",
                "label": "Number of Hidden Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 2
            },
            {
                "name": "neurons_per_layer",
                "label": "Neurons per Layer",
                "type": "number",
                "min": 16,
                "max": 512,
                "default": 64
            },
            {
                "name": "activation",
                "label": "Activation Function",
                "type": "select",
                "options": [
                    { value: "relu", label: "ReLU" },
                    { value: "sigmoid", label: "Sigmoid" },
                    { value: "tanh", label: "Tanh" }
                ],
                "default": "relu"
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "number",
                "min": 0.0001,
                "max": 0.1,
                "step": 0.0001,
                "default": 0.001
            },
            {
                "name": "batch_size",
                "label": "Batch Size",
                "type": "number",
                "min": 8,
                "max": 256,
                "default": 32
            },
            {
                "name": "epochs",
                "label": "Number of Epochs",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 100
            }
        ],
        "regression": [
            {
                "name": "hidden_layers",
                "label": "Number of Hidden Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 2
            },
            {
                "name": "neurons_per_layer",
                "label": "Neurons per Layer",
                "type": "number",
                "min": 16,
                "max": 512,
                "default": 64
            },
            {
                "name": "activation",
                "label": "Activation Function",
                "type": "select",
                "options": [
                    { value: "relu", label: "ReLU" },
                    { value: "linear", label: "Linear" }
                ],
                "default": "relu"
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "number",
                "min": 0.0001,
                "max": 0.1,
                "step": 0.0001,
                "default": 0.001
            },
            {
                "name": "batch_size",
                "label": "Batch Size",
                "type": "number",
                "min": 8,
                "max": 256,
                "default": 32
            },
            {
                "name": "epochs",
                "label": "Number of Epochs",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 100
            }
        ]
    },
    "cnn": {
        "classification": [
            {
                "name": "conv_layers",
                "label": "Number of Convolutional Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 3
            },
            {
                "name": "filters",
                "label": "Number of Filters",
                "type": "number",
                "min": 16,
                "max": 512,
                "default": 64
            },
            {
                "name": "kernel_size",
                "label": "Kernel Size",
                "type": "number",
                "min": 1,
                "max": 7,
                "default": 3
            },
            {
                "name": "pooling",
                "label": "Pooling Type",
                "type": "select",
                "options": [
                    { value: "max", label: "Max Pooling" },
                    { value: "avg", label: "Average Pooling" }
                ],
                "default": "max"
            },
            {
                "name": "dense_layers",
                "label": "Number of Dense Layers",
                "type": "number",
                "min": 1,
                "max": 5,
                "default": 2
            },
            {
                "name": "dense_units",
                "label": "Dense Layer Units",
                "type": "number",
                "min": 32,
                "max": 512,
                "default": 128
            },
            {
                "name": "dropout",
                "label": "Dropout Rate",
                "type": "number",
                "min": 0,
                "max": 0.5,
                "step": 0.1,
                "default": 0.2
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "number",
                "min": 0.0001,
                "max": 0.1,
                "step": 0.0001,
                "default": 0.001
            },
            {
                "name": "batch_size",
                "label": "Batch Size",
                "type": "number",
                "min": 8,
                "max": 256,
                "default": 32
            },
            {
                "name": "epochs",
                "label": "Number of Epochs",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 100
            }
        ],
        "regression": [
            {
                "name": "conv_layers",
                "label": "Number of Convolutional Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 3
            },
            {
                "name": "filters",
                "label": "Number of Filters",
                "type": "number",
                "min": 16,
                "max": 512,
                "default": 64
            },
            {
                "name": "kernel_size",
                "label": "Kernel Size",
                "type": "number",
                "min": 1,
                "max": 7,
                "default": 3
            },
            {
                "name": "pooling",
                "label": "Pooling Type",
                "type": "select",
                "options": [
                    { value: "max", label: "Max Pooling" },
                    { value: "avg", label: "Average Pooling" }
                ],
                "default": "max"
            },
            {
                "name": "dense_layers",
                "label": "Number of Dense Layers",
                "type": "number",
                "min": 1,
                "max": 5,
                "default": 2
            },
            {
                "name": "dense_units",
                "label": "Dense Layer Units",
                "type": "number",
                "min": 32,
                "max": 512,
                "default": 128
            },
            {
                "name": "dropout",
                "label": "Dropout Rate",
                "type": "number",
                "min": 0,
                "max": 0.5,
                "step": 0.1,
                "default": 0.2
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "number",
                "min": 0.0001,
                "max": 0.1,
                "step": 0.0001,
                "default": 0.001
            },
            {
                "name": "batch_size",
                "label": "Batch Size",
                "type": "number",
                "min": 8,
                "max": 256,
                "default": 32
            },
            {
                "name": "epochs",
                "label": "Number of Epochs",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 100
            }
        ]
    },
    "cnn": {
        "segmentation": [
            {
                "name": "encoder_layers",
                "label": "Number of Encoder Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 4
            },
            {
                "name": "decoder_layers",
                "label": "Number of Decoder Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 4
            },
            {
                "name": "filters",
                "label": "Base Number of Filters",
                "type": "number",
                "min": 16,
                "max": 512,
                "default": 64
            },
            {
                "name": "kernel_size",
                "label": "Kernel Size",
                "type": "number",
                "min": 1,
                "max": 7,
                "default": 3
            },
            {
                "name": "skip_connections",
                "label": "Use Skip Connections",
                "type": "select",
                "options": ["true", "false"],
                "default": "true"
            },
            {
                "name": "dropout",
                "label": "Dropout Rate",
                "type": "number",
                "min": 0,
                "max": 0.9,
                "step": 0.1,
                "default": 0.5
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "number",
                "min": 0.0001,
                "max": 0.1,
                "step": 0.0001,
                "default": 0.001
            },
            {
                "name": "batch_size",
                "label": "Batch Size",
                "type": "number",
                "min": 8,
                "max": 256,
                "default": 32
            },
            {
                "name": "epochs",
                "label": "Number of Epochs",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 100
            }
        ],
        "detection": [
            {
                "name": "backbone",
                "label": "Backbone Network",
                "type": "select",
                "options": ["resnet50", "vgg16", "mobilenet"],
                "default": "resnet50"
            },
            {
                "name": "anchor_scales",
                "label": "Anchor Scales",
                "type": "text",
                "default": "8,16,32"
            },
            {
                "name": "anchor_ratios",
                "label": "Anchor Ratios",
                "type": "text",
                "default": "0.5,1,2"
            },
            {
                "name": "nms_threshold",
                "label": "NMS Threshold",
                "type": "number",
                "min": 0,
                "max": 1,
                "step": 0.1,
                "default": 0.5
            },
            {
                "name": "confidence_threshold",
                "label": "Confidence Threshold",
                "type": "number",
                "min": 0,
                "max": 1,
                "step": 0.1,
                "default": 0.7
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "number",
                "min": 0.0001,
                "max": 0.1,
                "step": 0.0001,
                "default": 0.001
            },
            {
                "name": "batch_size",
                "label": "Batch Size",
                "type": "number",
                "min": 8,
                "max": 256,
                "default": 32
            },
            {
                "name": "epochs",
                "label": "Number of Epochs",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 100
            }
        ]
    },
    "rnn": {
        "sequence": [
            {
                "name": "rnn_type",
                "label": "RNN Type",
                "type": "select",
                "options": [
                    { value: "lstm", label: "LSTM" },
                    { value: "gru", label: "GRU" },
                    { value: "simple", label: "Simple RNN" }
                ],
                "default": "lstm"
            },
            {
                "name": "num_layers",
                "label": "Number of Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 2
            },
            {
                "name": "hidden_size",
                "label": "Hidden Size",
                "type": "number",
                "min": 16,
                "max": 512,
                "default": 128
            },
            {
                "name": "bidirectional",
                "label": "Bidirectional",
                "type": "select",
                "options": [
                    { value: "true", label: "Yes" },
                    { value: "false", label: "No" }
                ],
                "default": "false"
            },
            {
                "name": "dropout",
                "label": "Dropout Rate",
                "type": "number",
                "min": 0,
                "max": 0.5,
                "step": 0.1,
                "default": 0.2
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "number",
                "min": 0.0001,
                "max": 0.1,
                "step": 0.0001,
                "default": 0.001
            },
            {
                "name": "batch_size",
                "label": "Batch Size",
                "type": "number",
                "min": 8,
                "max": 256,
                "default": 32
            },
            {
                "name": "epochs",
                "label": "Number of Epochs",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 100
            }
        ],
        "timeseries": [
            {
                "name": "rnn_type",
                "label": "RNN Type",
                "type": "select",
                "options": ["lstm", "gru", "simple"],
                "default": "lstm"
            },
            {
                "name": "num_layers",
                "label": "Number of Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 2
            },
            {
                "name": "hidden_size",
                "label": "Hidden Size",
                "type": "number",
                "min": 16,
                "max": 1024,
                "default": 128
            },
            {
                "name": "sequence_length",
                "label": "Sequence Length",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 24
            },
            {
                "name": "prediction_horizon",
                "label": "Prediction Horizon",
                "type": "number",
                "min": 1,
                "max": 100,
                "default": 12
            },
            {
                "name": "dropout",
                "label": "Dropout Rate",
                "type": "number",
                "min": 0,
                "max": 0.9,
                "step": 0.1,
                "default": 0.5
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "number",
                "min": 0.0001,
                "max": 0.1,
                "step": 0.0001,
                "default": 0.001
            },
            {
                "name": "batch_size",
                "label": "Batch Size",
                "type": "number",
                "min": 8,
                "max": 256,
                "default": 32
            },
            {
                "name": "epochs",
                "label": "Number of Epochs",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 100
            }
        ],
        "nlp": [
            {
                "name": "rnn_type",
                "label": "RNN Type",
                "type": "select",
                "options": ["lstm", "gru", "simple"],
                "default": "lstm"
            },
            {
                "name": "num_layers",
                "label": "Number of Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 2
            },
            {
                "name": "hidden_size",
                "label": "Hidden Size",
                "type": "number",
                "min": 16,
                "max": 1024,
                "default": 256
            },
            {
                "name": "embedding_dim",
                "label": "Embedding Dimension",
                "type": "number",
                "min": 16,
                "max": 512,
                "default": 100
            },
            {
                "name": "bidirectional",
                "label": "Bidirectional",
                "type": "select",
                "options": ["true", "false"],
                "default": "true"
            },
            {
                "name": "dropout",
                "label": "Dropout Rate",
                "type": "number",
                "min": 0,
                "max": 0.9,
                "step": 0.1,
                "default": 0.5
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "number",
                "min": 0.0001,
                "max": 0.1,
                "step": 0.0001,
                "default": 0.001
            },
            {
                "name": "batch_size",
                "label": "Batch Size",
                "type": "number",
                "min": 8,
                "max": 256,
                "default": 32
            },
            {
                "name": "epochs",
                "label": "Number of Epochs",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 100
            }
        ]
    },
    "transformer": {
        "nlp": [
            {
                "name": "num_layers",
                "label": "Number of Layers",
                "type": "number",
                "min": 1,
                "max": 24,
                "default": 6
            },
            {
                "name": "d_model",
                "label": "Model Dimension",
                "type": "number",
                "min": 64,
                "max": 1024,
                "default": 512
            },
            {
                "name": "num_heads",
                "label": "Number of Attention Heads",
                "type": "number",
                "min": 1,
                "max": 16,
                "default": 8
            },
            {
                "name": "d_ff",
                "label": "Feed-forward Dimension",
                "type": "number",
                "min": 128,
                "max": 4096,
                "default": 2048
            },
            {
                "name": "dropout",
                "label": "Dropout Rate",
                "type": "number",
                "min": 0,
                "max": 0.5,
                "step": 0.1,
                "default": 0.1
            },
            {
                "name": "max_seq_length",
                "label": "Maximum Sequence Length",
                "type": "number",
                "min": 32,
                "max": 512,
                "default": 128
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "number",
                "min": 0.0001,
                "max": 0.1,
                "step": 0.0001,
                "default": 0.001
            },
            {
                "name": "batch_size",
                "label": "Batch Size",
                "type": "number",
                "min": 8,
                "max": 256,
                "default": 32
            },
            {
                "name": "epochs",
                "label": "Number of Epochs",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 100
            }
        ],
        "seq2seq": [
            {
                "name": "num_layers",
                "label": "Number of Layers",
                "type": "number",
                "min": 1,
                "max": 24,
                "default": 6
            },
            {
                "name": "d_model",
                "label": "Model Dimension",
                "type": "number",
                "min": 64,
                "max": 4096,
                "default": 512
            },
            {
                "name": "num_heads",
                "label": "Number of Attention Heads",
                "type": "number",
                "min": 1,
                "max": 64,
                "default": 8
            },
            {
                "name": "d_ff",
                "label": "Feed Forward Dimension",
                "type": "number",
                "min": 128,
                "max": 16384,
                "default": 2048
            },
            {
                "name": "dropout",
                "label": "Dropout Rate",
                "type": "number",
                "min": 0,
                "max": 0.9,
                "step": 0.1,
                "default": 0.1
            },
            {
                "name": "max_seq_length",
                "label": "Maximum Sequence Length",
                "type": "number",
                "min": 32,
                "max": 4096,
                "default": 512
            },
            {
                "name": "beam_size",
                "label": "Beam Size for Decoding",
                "type": "number",
                "min": 1,
                "max": 16,
                "default": 4
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "number",
                "min": 0.0001,
                "max": 0.1,
                "step": 0.0001,
                "default": 0.0000
            },
            {
                "name": "batch_size",
                "label": "Batch Size",
                "type": "number",
                "min": 8,
                "max": 256,
                "default": 32
            },
            {
                "name": "epochs",
                "label": "Number of Epochs",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 100
            }
        ],
        "timeseries": [
            {
                "name": "num_layers",
                "label": "Number of Layers",
                "type": "number",
                "min": 1,
                "max": 12,
                "default": 3
            },
            {
                "name": "d_model",
                "label": "Model Dimension",
                "type": "number",
                "min": 32,
                "max": 512,
                "default": 128
            },
            {
                "name": "num_heads",
                "label": "Number of Attention Heads",
                "type": "number",
                "min": 1,
                "max": 16,
                "default": 4
            },
            {
                "name": "d_ff",
                "label": "Feed Forward Dimension",
                "type": "number",
                "min": 64,
                "max": 2048,
                "default": 512
            },
            {
                "name": "dropout",
                "label": "Dropout Rate",
                "type": "number",
                "min": 0,
                "max": 0.9,
                "step": 0.1,
                "default": 0.1
            },
            {
                "name": "sequence_length",
                "label": "Input Sequence Length",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 24
            },
            {
                "name": "prediction_horizon",
                "label": "Prediction Horizon",
                "type": "number",
                "min": 1,
                "max": 100,
                "default": 12
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "number",
                "min": 0.0001,
                "max": 0.1,
                "step": 0.0001,
                "default": 0.0001
            },
            {
                "name": "batch_size",
                "label": "Batch Size",
                "type": "number",
                "min": 8,
                "max": 256,
                "default": 32
            },
            {
                "name": "epochs",
                "label": "Number of Epochs",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 100
            }
        ]
    },
    "autoencoder": {
        "compression": [
            {
                "name": "encoder_layers",
                "label": "Number of Encoder Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 3
            },
            {
                "name": "decoder_layers",
                "label": "Number of Decoder Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 3
            },
            {
                "name": "latent_dim",
                "label": "Latent Dimension",
                "type": "number",
                "min": 2,
                "max": 512,
                "default": 32
            },
            {
                "name": "activation",
                "label": "Activation Function",
                "type": "select",
                "options": ["relu", "tanh", "sigmoid"],
                "default": "relu"
            },
            {
                "name": "dropout",
                "label": "Dropout Rate",
                "type": "number",
                "min": 0,
                "max": 0.9,
                "step": 0.1,
                "default": 0.2
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "number",
                "min": 0.0001,
                "max": 0.1,
                "step": 0.0001,
                "default": 0.001
            },
            {
                "name": "batch_size",
                "label": "Batch Size",
                "type": "number",
                "min": 8,
                "max": 256,
                "default": 32
            },
            {
                "name": "epochs",
                "label": "Number of Epochs",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 100
            }
        ],
        "anomaly": [
            {
                "name": "encoder_layers",
                "label": "Number of Encoder Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 3
            },
            {
                "name": "decoder_layers",
                "label": "Number of Decoder Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 3
            },
            {
                "name": "latent_dim",
                "label": "Latent Dimension",
                "type": "number",
                "min": 2,
                "max": 512,
                "default": 32
            },
            {
                "name": "activation",
                "label": "Activation Function",
                "type": "select",
                "options": ["relu", "tanh", "sigmoid"],
                "default": "relu"
            },
            {
                "name": "reconstruction_threshold",
                "label": "Reconstruction Threshold",
                "type": "number",
                "min": 0,
                "max": 1,
                "step": 0.01,
                "default": 0.5
            },
            {
                "name": "dropout",
                "label": "Dropout Rate",
                "type": "number",
                "min": 0,
                "max": 0.9,
                "step": 0.1,
                "default": 0.2
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "number",
                "min": 0.0001,
                "max": 0.1,
                "step": 0.0001,
                "default": 0.001
            },
            {
                "name": "batch_size",
                "label": "Batch Size",
                "type": "number",
                "min": 8,
                "max": 256,
                "default": 32
            },
            {
                "name": "epochs",
                "label": "Number of Epochs",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 100
            }
        ]
    },
    "vae": {
        "generation": [
            {
                "name": "encoder_layers",
                "label": "Number of Encoder Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 3
            },
            {
                "name": "decoder_layers",
                "label": "Number of Decoder Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 3
            },
            {
                "name": "latent_dim",
                "label": "Latent Dimension",
                "type": "number",
                "min": 2,
                "max": 512,
                "default": 32
            },
            {
                "name": "activation",
                "label": "Activation Function",
                "type": "select",
                "options": ["relu", "tanh", "sigmoid"],
                "default": "relu"
            },
            {
                "name": "kl_weight",
                "label": "KL Divergence Weight",
                "type": "number",
                "min": 0,
                "max": 1,
                "step": 0.01,
                "default": 0.1
            },
            {
                "name": "dropout",
                "label": "Dropout Rate",
                "type": "number",
                "min": 0,
                "max": 0.9,
                "step": 0.1,
                "default": 0.2
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "number",
                "min": 0.0001,
                "max": 0.1,
                "step": 0.0001,
                "default": 0.001
            },
            {
                "name": "batch_size",
                "label": "Batch Size",
                "type": "number",
                "min": 8,
                "max": 256,
                "default": 32
            },
            {
                "name": "epochs",
                "label": "Number of Epochs",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 100
            }
        ],
        "feature": [
            {
                "name": "encoder_layers",
                "label": "Number of Encoder Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 3
            },
            {
                "name": "decoder_layers",
                "label": "Number of Decoder Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 3
            },
            {
                "name": "latent_dim",
                "label": "Latent Dimension",
                "type": "number",
                "min": 2,
                "max": 512,
                "default": 32
            },
            {
                "name": "activation",
                "label": "Activation Function",
                "type": "select",
                "options": ["relu", "tanh", "sigmoid"],
                "default": "relu"
            },
            {
                "name": "kl_weight",
                "label": "KL Divergence Weight",
                "type": "number",
                "min": 0,
                "max": 1,
                "step": 0.01,
                "default": 0.1
            },
            {
                "name": "feature_dim",
                "label": "Feature Dimension",
                "type": "number",
                "min": 2,
                "max": 512,
                "default": 64
            },
            {
                "name": "dropout",
                "label": "Dropout Rate",
                "type": "number",
                "min": 0,
                "max": 0.9,
                "step": 0.1,
                "default": 0.2
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "number",
                "min": 0.0001,
                "max": 0.1,
                "step": 0.0001,
                "default": 0.001
            },
            {
                "name": "batch_size",
                "label": "Batch Size",
                "type": "number",
                "min": 8,
                "max": 256,
                "default": 32
            },
            {
                "name": "epochs",
                "label": "Number of Epochs",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 100
            }
        ]
    },
    "gan": {
        "generation": [
            {
                "name": "latent_dim",
                "label": "Latent Dimension",
                "type": "number",
                "min": 2,
                "max": 512,
                "default": 100
            },
            {
                "name": "generator_layers",
                "label": "Number of Generator Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 4
            },
            {
                "name": "discriminator_layers",
                "label": "Number of Discriminator Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 4
            },
            {
                "name": "generator_units",
                "label": "Generator Units per Layer",
                "type": "number",
                "min": 32,
                "max": 1024,
                "default": 256
            },
            {
                "name": "discriminator_units",
                "label": "Discriminator Units per Layer",
                "type": "number",
                "min": 32,
                "max": 1024,
                "default": 256
            },
            {
                "name": "activation",
                "label": "Activation Function",
                "type": "select",
                "options": ["relu", "leaky_relu", "tanh"],
                "default": "leaky_relu"
            },
            {
                "name": "dropout",
                "label": "Dropout Rate",
                "type": "number",
                "min": 0,
                "max": 0.9,
                "step": 0.1,
                "default": 0.3
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "number",
                "min": 0.0001,
                "max": 0.1,
                "step": 0.0001,
                "default": 0.0002
            },
            {
                "name": "batch_size",
                "label": "Batch Size",
                "type": "number",
                "min": 8,
                "max": 256,
                "default": 32
            },
            {
                "name": "epochs",
                "label": "Number of Epochs",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 100
            }
        ],
        "style": [
            {
                "name": "latent_dim",
                "label": "Latent Dimension",
                "type": "number",
                "min": 2,
                "max": 512,
                "default": 100
            },
            {
                "name": "generator_layers",
                "label": "Number of Generator Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 4
            },
            {
                "name": "discriminator_layers",
                "label": "Number of Discriminator Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 4
            },
            {
                "name": "style_dim",
                "label": "Style Dimension",
                "type": "number",
                "min": 2,
                "max": 512,
                "default": 64
            },
            {
                "name": "content_weight",
                "label": "Content Loss Weight",
                "type": "number",
                "min": 0,
                "max": 1,
                "step": 0.1,
                "default": 0.5
            },
            {
                "name": "style_weight",
                "label": "Style Loss Weight",
                "type": "number",
                "min": 0,
                "max": 1,
                "step": 0.1,
                "default": 0.5
            },
            {
                "name": "activation",
                "label": "Activation Function",
                "type": "select",
                "options": ["relu", "leaky_relu", "tanh"],
                "default": "leaky_relu"
            },
            {
                "name": "dropout",
                "label": "Dropout Rate",
                "type": "number",
                "min": 0,
                "max": 0.9,
                "step": 0.1,
                "default": 0.3
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "number",
                "min": 0.0001,
                "max": 0.1,
                "step": 0.0001,
                "default": 0.0002
            },
            {
                "name": "batch_size",
                "label": "Batch Size",
                "type": "number",
                "min": 8,
                "max": 256,
                "default": 32
            },
            {
                "name": "epochs",
                "label": "Number of Epochs",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 100
            }
        ]
    },
    "pinn": {
        "differential": [
            {
                "name": "hidden_layers",
                "label": "Number of Hidden Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 4
            },
            {
                "name": "neurons_per_layer",
                "label": "Neurons per Layer",
                "type": "number",
                "min": 16,
                "max": 512,
                "default": 128
            },
            {
                "name": "activation",
                "label": "Activation Function",
                "type": "select",
                "options": ["tanh", "sin", "relu"],
                "default": "tanh"
            },
            {
                "name": "physics_weight",
                "label": "Physics Loss Weight",
                "type": "number",
                "min": 0,
                "max": 1,
                "step": 0.1,
                "default": 0.5
            },
            {
                "name": "boundary_weight",
                "label": "Boundary Condition Weight",
                "type": "number",
                "min": 0,
                "max": 1,
                "step": 0.1,
                "default": 0.5
            },
            {
                "name": "collocation_points",
                "label": "Number of Collocation Points",
                "type": "number",
                "min": 100,
                "max": 10000,
                "default": 1000
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "number",
                "min": 0.0001,
                "max": 0.1,
                "step": 0.0001,
                "default": 0.001
            },
            {
                "name": "batch_size",
                "label": "Batch Size",
                "type": "number",
                "min": 8,
                "max": 256,
                "default": 32
            },
            {
                "name": "epochs",
                "label": "Number of Epochs",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 100
            }
        ],
        "simulation": [
            {
                "name": "hidden_layers",
                "label": "Number of Hidden Layers",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 4
            },
            {
                "name": "neurons_per_layer",
                "label": "Neurons per Layer",
                "type": "number",
                "min": 16,
                "max": 512,
                "default": 128
            },
            {
                "name": "activation",
                "label": "Activation Function",
                "type": "select",
                "options": ["tanh", "sin", "relu"],
                "default": "tanh"
            },
            {
                "name": "physics_weight",
                "label": "Physics Loss Weight",
                "type": "number",
                "min": 0,
                "max": 1,
                "step": 0.1,
                "default": 0.5
            },
            {
                "name": "initial_weight",
                "label": "Initial Condition Weight",
                "type": "number",
                "min": 0,
                "max": 1,
                "step": 0.1,
                "default": 0.5
            },
            {
                "name": "time_steps",
                "label": "Number of Time Steps",
                "type": "number",
                "min": 10,
                "max": 1000,
                "default": 100
            },
            {
                "name": "spatial_points",
                "label": "Number of Spatial Points",
                "type": "number",
                "min": 10,
                "max": 1000,
                "default": 100
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "number",
                "min": 0.0001,
                "max": 0.1,
                "step": 0.0001,
                "default": 0.001
            },
            {
                "name": "batch_size",
                "label": "Batch Size",
                "type": "number",
                "min": 8,
                "max": 256,
                "default": 32
            },
            {
                "name": "epochs",
                "label": "Number of Epochs",
                "type": "number",
                "min": 1,
                "max": 1000,
                "default": 100
            }
        ]
    }
};

// Make hyperparameters available globally
window.hyperparameters = hyperparameters;

// Export for module usage
export default hyperparameters;