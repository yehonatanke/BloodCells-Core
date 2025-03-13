<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#bloodcells-core">BloodCells-Core</a></li>
    <li><a href="#supported-models">Supported Models</a>
      <ul>
        <li><a href="#convolutional-neural-networks-cnns">Convolutional Neural Networks</a></li>
        <li><a href="#custom-models">Custom Models</a></li>
        <li><a href="#transfer-learning">Transfer Learning</a></li>
      </ul>
    </li>
    <li><a href="#frameworks-and-implementations">Frameworks and Implementations</a>
      <ul>
        <li><a href="#frameworks-and-model-implementations">Frameworks and Model Implementations</a></li>
        <li><a href="#methods">Methods</a></li>
        <li><a href="#training-techniques">Training Techniques</a></li>
      </ul>
    </li>
    <li><a href="#features">Features</a></li>
    <li><a href="#repo-structure">Repo Structure</a>
      <ul>
        <li><a href="#data-module-datapy">Data Module (data.py)</a></li>
        <li><a href="#model-module-modelpy">Model Module (model.py)</a></li>
        <li><a href="#training-module-trainpy">Training Module (train.py)</a></li>
        <li><a href="#visualization-module-visualizepy">Visualization Module (visualize.py)</a></li>
        <li><a href="#utilities-module-utilspy">Utilities Module (utils.py)</a></li>
      </ul>
    </li>
  </ol>
</details>

# BloodCells-Core
A core codebase for training and predicting blood cell classifications using machine learning. This repository contains the essential scripts extracted from Jupyter notebooks for streamlined usage and development.
The goal is to implement a simple and small codebase for blood cell analysis, making it easy to understand, modify, and extend for further research or development.

## Supported Models

The repo supports various deep learning architectures:

### Convolutional Neural Networks 
- **ResNet50**: Deep residual network with skip connections that helps mitigate the vanishing gradient problem. Excellent for complex image classification tasks with 50 layers.
- **DenseNet121**: Features dense connections between layers, where each layer receives inputs from all preceding layers. Efficient parameter usage with 121 layers.
- **EfficientNetB0**: Optimized architecture that balances network depth, width, and resolution using compound scaling. Provides high accuracy with fewer parameters.
- **VGG16**: Classic architecture with 16 layers featuring small convolutional filters. Simple but effective design for feature extraction.
- **MobileNetV2**: Lightweight architecture designed for mobile and edge devices. Uses depthwise separable convolutions to reduce computational cost.
- **InceptionV3**: Employs inception modules with multiple filter sizes operating on the same level. Efficient at capturing features at different scales.

### Custom Models
- **Custom CNN**: A tailored convolutional neural network specifically designed for blood cell morphology. Features a balanced architecture optimized for the unique characteristics of blood cell images.

### Transfer Learning
All supported models can be used with transfer learning, leveraging pre-trained weights from ImageNet or other datasets to improve performance and reduce training time, especially when working with limited data.

## Frameworks and Implementations

### Frameworks and Model Implementations
- [ ] PyTorch
- [ ] Lightning.AI (PyTorch Lightning)
- [ ] TensorFlow/Keras
- [ ] YOLO 
- [ ] UNet
- [ ] Vision Transformer 
- [ ] FastAI

### Methods
* Transfer Learning
* Ensemble Learning
* Hyperparameter Optimization
* Cross-Validation
* Data Augmentation

### Training Techniques
- **Progressive Resizing**: Training on increasing image resolutions for faster convergence
- **Learning Rate Scheduling**: Cyclical learning rates and warm-up strategies
- **Mixed Precision Training**: Using FP16 computation for faster training on compatible hardware
- **Gradient Accumulation**: Enabling larger effective batch sizes on memory-constrained systems
- **Knowledge Distillation**: Compressing larger models into smaller, deployment-friendly versions

## Features
- Data preprocessing and augmentation for blood cell images
- Training and evaluation of blood cell classification models
- Model deployment for inference
- Performance monitoring and analysis
- Support for multiple model architectures
- Visualization tools for model interpretation

## Repo Structure

### Data Module (`data.py`)
- `load_dataset()`: Loads blood cell images from specified directories
- `preprocess_images()`: Applies standard preprocessing to blood cell images
- `augment_data()`: Performs data augmentation to increase training set diversity
- `create_data_generators()`: Creates data generators for training and validation
- `split_dataset()`: Splits dataset into training, validation, and test sets

### Model Module (`model.py`)
- `create_model()`: Creates a new model with specified architecture
  - Supported models: ResNet50, DenseNet121, EfficientNetB0, VGG16, MobileNetV2, InceptionV3
- `load_model()`: Loads a pre-trained model from disk
- `save_model()`: Saves a trained model to disk
- `compile_model()`: Configures the model for training
- `get_model_summary()`: Returns a summary of the model architecture
- `transfer_learning()`: Applies transfer learning using pre-trained weights
- `custom_cnn()`: Creates a custom CNN architecture for blood cell classification

### Training Module (`train.py`)
- `train_model()`: Trains the model on the provided dataset
- `evaluate_model()`: Evaluates model performance on test data
- `fine_tune_model()`: Fine-tunes a pre-trained model
- `early_stopping()`: Implements early stopping to prevent overfitting
- `learning_rate_scheduler()`: Adjusts learning rate during training

### Visualization Module (`visualize.py`)
- `plot_training_history()`: Plots training and validation metrics over epochs
- `visualize_layer_activations()`: Visualizes activations of specific layers
- `generate_gradcam()`: Generates Grad-CAM visualizations for model interpretability
- `plot_feature_maps()`: Plots feature maps from convolutional layers
- `visualize_data_distribution()`: Visualizes the distribution of classes in the dataset

### Utilities Module (`utils.py`)
- `setup_logging()`: Sets up logging configuration
- `set_random_seed()`: Sets random seeds for reproducibility
- `get_available_devices()`: Detects available CPU/GPU devices
- `memory_usage()`: Monitors memory usage during training
- `timing_decorator()`: Decorator for timing function execution

