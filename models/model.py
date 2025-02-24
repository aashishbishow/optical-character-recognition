import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import mnist
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import time
import psutil
import random
import os
import pickle
import json
import logging
from logging.handlers import RotatingFileHandler
import gc
import traceback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, # Capture all logs (INFO, DEBUG, ERROR, etc.)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("model_training.log", maxBytes=10485760, backupCount=5),  # 10 MB logs, keep last 5 files
        logging.StreamHandler()  # Also output logs to console
    ]
)
logger = logging.getLogger("SelfImprovingCRNN")

# Ensure reproducibility in Python hash functions
os.environ['PYTHONHASHSEED'] = '42'

# Set random seeds for reproducibility
np.random.seed(42)            # NumPy randomness (e.g., dataset shuffling)
tf.random.set_seed(42)        # TensorFlow randomness (weights, datasets)
random.seed(42)               # Python's built-in random module

# Ensure TensorFlow uses deterministic operations (for GPU reproducibility)
tf.config.enable_op_determinism()



class SelfImprovingCRNN:
    def __init__(self, input_shape=(28, 28, 1), initial_training=True,
                 lr_factor=0.2, lr_patience=5, min_lr=1e-7, early_stopping_patience=15, epochs=30):
        """
        Initialize the Self-Improving CRNN model with failsafe mechanisms.

        Args:
            input_shape (tuple): Input image dimensions
            initial_training (bool): Whether to train the model initially
            lr_factor (float): Factor by which to reduce learning rate
            lr_patience (int): Patience for learning rate reduction
            min_lr (float): Minimum learning rate
            early_stopping_patience (int): Patience for early stopping
            epochs (int): Number of training epochs
        """
        try:
            # Set params with validation
            self.input_shape = input_shape

            self.lr_factor = max(0.01, min(0.5, lr_factor)) # Constrain between 0.01-0.5
            self.lr_patience = max(1,lr_patience) # Must be at least 1
            self.min_lr = max(1e-10, min(1e-3, min_lr)) # Constrain between 1e-10 and 1e-3
            self.early_stopping_patience = max(3, early_stopping_patience) # At least 3
            self.epochs = max(5, epochs) # At least 5 epoch

            # Initialize model components
            self.model = None
            self.history = None
            self.version = 0
            self.performance_history = []                   # Track performance over iterations
            self.uncertainty_threshold = 0.3                # Threshold for active learning

            # Initial threshold for active learning
            self.learning_pool = []                         # Data pool for uncertain samples
            self.learning_pool_labels = []                  # Corresponding labels
            self.model_improvements = []                    # Track model changes
            self.data_loaded = False                        # Flag for data readiness

            # Set checkpoint path            
            self.checkpoint_path = os.path.join("checkpoints", "model_checkpoint.keras")

            # Ensure checkpoints directory exists
            os.makedirs("checkpoints", exist_ok=True)

            # Initialize TF GPU memory growth to prevent OOM errors
            self._setup_gpu()

            # Load data safely and validate data
            if not self.load_data():
                logger.error("Failed to load data. Initialization aborted.")
                return

            # Build initial CRNN model with error handling
            if not self.build_model():
                logger.error("Failed to build model. Initialization aborted.")
                return

            # Initial training if requested
            if initial_training:
                self.train()

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _setup_gpu(self):
        """Configure GPU memory growth to prevent OOM errors"""
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                for device in physical_devices:
                    # Use the updated memory growth API
                    tf.config.set_logical_device_configuration(device, [tf.config.LogicalDeviceConfiguration(memory_limit = None)])
                    tf.config.experimental.set_memory_growth(device, True)
                logger.info(f"Found {len(physical_devices)} GPU(s), memory growth enabled")
            else:
                logger.info("No GPU found, using CPU")
        except Exception as e:
            logger.warning(f"GPU setup failed: {str(e)}")

    def load_data(self):
        """Load MNIST dataset with error handling and data validation"""
        try:
            logger.info("Loading MNIST dataset...")

            # Validate input shape is set
            if not hasattr(self, 'input_shape'):
                logger.error("Input shape not defined")
                return False

            # Attempt to load MNIST dataset with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    (x_train, y_train), (x_test, y_test) = mnist.load_data()
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to load MNIST dataset after {max_retries} attempts: {str(e)}")
                        return False
                    logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            # Validate data shapes and content
            expected_shape = self.input_shape[:2]  # Only compare height and width
            if x_train.shape[1:] != expected_shape or x_test.shape[1:] != expected_shape:
                logger.error(f"Data shape mismatch. Expected {expected_shape}, got {x_train.shape[1:]}")
                return False
            
            # Validate data range and type before normalization
            if x_train.min() < 0 or x_train.max() > 255 or x_test.min() < 0 or x_test.max() > 255:
                logger.error("Invalid pixel values detected")
                return False

            # Check for NaN or infinite values
            if np.isnan(x_train).any() or np.isnan(x_test).any():
                logger.error("NaN values detected in data")
                return False
            
            # Normalize pixel values (0 to 1) with error checking
            try:
                x_train = x_train.astype('float32') / 255.0
                x_test = x_test.astype('float32') / 255.0
                
                # Verify normalization
                if x_train.max() > 1.0 or x_train.min() < 0.0:
                    logger.error("Normalization failed: values outside [0,1] range")
                    return False
            except Exception as e:
                logger.error(f"Error during normalization: {str(e)}")
                return False

            # Add channel dimension if needed
            if len(self.input_shape) == 3 and self.input_shape[-1] == 1:
                x_train = np.expand_dims(x_train, axis=-1)
                x_test = np.expand_dims(x_test, axis=-1)

            # Validate labels before conversion
            num_classes = 10
            if not (np.all(np.unique(y_train) == np.arange(num_classes)) and 
                    np.all(np.unique(y_test) == np.arange(num_classes))):
                logger.error("Missing classes in dataset")
                return False

            # Convert labels to one-hot encoding with validation
            try:
                y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
                y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes)
                
                # Verify one-hot encoding
                if not (y_train_onehot.shape[1] == num_classes and y_test_onehot.shape[1] == num_classes):
                    logger.error("One-hot encoding failed")
                    return False
            except Exception as e:
                logger.error(f"Error during one-hot encoding: {str(e)}")
                return False

            # Create stratified splits with validation
            try:
                # Validation split
                x_train, x_val, y_train, y_val = train_test_split(
                    x_train, y_train, 
                    test_size=0.1, 
                    random_state=42, 
                    stratify=y_train
                )
                
                # Verify stratification
                train_dist = np.bincount(y_train) / len(y_train)
                val_dist = np.bincount(y_val) / len(y_val)
                if not np.allclose(train_dist, val_dist, atol=0.01):
                    logger.warning("Possible stratification imbalance in validation split")

                # Convert split labels to one-hot
                y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
                y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes)

                # Active learning pool split
                x_test, x_active_pool, y_test, y_active_pool = train_test_split(
                    x_test, y_test,
                    test_size=0.3,
                    random_state=42,
                    stratify=y_test
                )
                
                # Verify active learning pool stratification
                test_dist = np.bincount(y_test) / len(y_test)
                active_dist = np.bincount(y_active_pool) / len(y_active_pool)
                if not np.allclose(test_dist, active_dist, atol=0.01):
                    logger.warning("Possible stratification imbalance in active learning pool")

                y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes)
                y_active_pool_onehot = tf.keras.utils.to_categorical(y_active_pool, num_classes)

            except Exception as e:
                logger.error(f"Error during data splitting: {str(e)}")
                return False

            # Store data with memory usage tracking
            try:
                self.x_train, self.y_train, self.y_train_onehot = x_train, y_train, y_train_onehot
                self.x_val, self.y_val, self.y_val_onehot = x_val, y_val, y_val_onehot
                self.x_test, self.y_test, self.y_test_onehot = x_test, y_test, y_test_onehot
                self.x_active, self.y_active, self.y_active_onehot = (
                    x_active_pool, y_active_pool, y_active_pool_onehot
                )

                # Calculate and log memory usage
                total_memory_mb = (
                    sum(arr.nbytes for arr in [
                        x_train, y_train, y_train_onehot,
                        x_val, y_val, y_val_onehot,
                        x_test, y_test, y_test_onehot,
                        x_active_pool, y_active_pool, y_active_pool_onehot
                    ]) / (1024 * 1024)
                )
                logger.info(f"Total data memory usage: {total_memory_mb:.2f} MB")

            except Exception as e:
                logger.error(f"Error storing dataset: {str(e)}")
                return False

            # Log dataset statistics
            logger.info("Dataset statistics:")
            logger.info(f"Train set: {x_train.shape[0]} samples (min: {x_train.min():.3f}, max: {x_train.max():.3f})")
            logger.info(f"Validation set: {x_val.shape[0]} samples (min: {x_val.min():.3f}, max: {x_val.max():.3f})")
            logger.info(f"Test set: {x_test.shape[0]} samples (min: {x_test.min():.3f}, max: {x_test.max():.3f})")
            logger.info(f"Active learning pool: {x_active_pool.shape[0]} samples (min: {x_active_pool.min():.3f}, max: {x_active_pool.max():.3f})")

            self.data_loaded = True
            return True
                
        except Exception as e:
            logger.error(f"Error in load_data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    def build_model(self, custom_architecture=None):
        """Build CRNN model with error handling and dynamic architecture flexibility"""
        try:
            logger.info(f"Building model version {self.version}...")

            if not hasattr(self, 'data_loaded') or not self.data_loaded:
                logger.error("Cannot build model: Data not loaded")
                return False

            # Free memory from previous model if it exists
            if hasattr(self, 'model') and self.model is not None:
                K.clear_session()
                del self.model
                gc.collect()
            
            # Input layer - validate and create input shape
            if not isinstance(self.input_shape, tuple) or len(self.input_shape) < 2:
                logger.error("Invalid input shape provided.")
                return False
                
            inputs = layers.Input(shape=self.input_shape)

            if custom_architecture:
                try:
                    # Use custom architecture if provided
                    x = custom_architecture(inputs)

                    # Validate custom output
                    if not isinstance(x, tf.Tensor):
                        logger.error("Custom architecture must return a valid tensor")
                        return False
                    
                    if x.shape.rank == 0:
                        logger.error("Custom architecture returned a scalar tensor")
                        return False
                        
                except Exception as e:
                    logger.error(f"Error in custom architecture: {str(e)}")
                    return False

            else:
                # Default architecture
                # Reshape for CNN - add channel dimension if needed
                if len(self.input_shape) == 2:
                    x = layers.Reshape((*self.input_shape, 1))(inputs)
                elif len(self.input_shape) == 3:
                    x = inputs
                else:
                    logger.error(f"Unsupported input shape: {self.input_shape}")
                    return False

                # CNN Feature Extraction
                # First convolutional block with error checking
                try:
                    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
                    x = layers.BatchNormalization()(x)
                    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
                    x = layers.MaxPooling2D((2, 2))(x)
                    x = layers.Dropout(0.25)(x)

                    # Second convolutional block
                    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
                    x = layers.BatchNormalization()(x)
                    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
                    x = layers.MaxPooling2D((2, 2))(x)
                    x = layers.Dropout(0.25)(x)

                except ValueError as ve:
                    logger.error(f"Invalid tensor dimensions in CNN layers: {str(ve)}")
                    return False

                # Calculate feature dimensions for RNN
                try:
                    # Use proper reshaping for RNN input
                    conv_output = x.shape
                    if conv_output[1] is None or conv_output[2] is None:
                        logger.warning("Dynamic shape detected, using GlobalAveragePooling2D")
                        x = layers.GlobalAveragePooling2D()(x)
                    else:
                        # Reshape maintaining spatial information
                        new_shape = (conv_output[1], conv_output[2] * conv_output[3])
                        x = layers.Reshape(new_shape)(x)

                    # RNN layers with proper sequence handling
                    if len(x.shape) == 2:
                        x = layers.RepeatVector(1)(x)
                    
                    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
                    x = layers.Dropout(0.25)(x)
                    x = layers.Bidirectional(layers.LSTM(64))(x)
                    x = layers.Dropout(0.25)(x)

                except ValueError as ve:
                    logger.error(f"Error in RNN reshape/processing: {str(ve)}")
                    return False

                # Dense layers for classification with gradient clipping
                x = layers.Dense(
                    128, 
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                    kernel_constraint=tf.keras.constraints.MaxNorm(3)
                )(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.5)(x)

            # Output layer with shape validation
            try:
                outputs = layers.Dense(10, activation='softmax')(x)
                if outputs.shape[-1] != 10:
                    logger.error(f"Output shape mismatch. Expected 10, got {outputs.shape[-1]}")
                    return False
            except Exception as e:
                logger.error(f"Error in output layer: {str(e)}")
                return False

            # Create and compile model with error handling
            try:
                self.model = models.Model(inputs=inputs, outputs=outputs)
                
                # Learning rate schedule with warmup
                initial_learning_rate = 0.001
                warmup_steps = 1000
                decay_steps = 10000
                
                lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                    boundaries=[warmup_steps],
                    values=[
                        initial_learning_rate * tf.cast(step, tf.float32) / tf.cast(warmup_steps, tf.float32)
                        for step in range(1, warmup_steps + 1)
                    ] + [
                        initial_learning_rate * tf.math.exp(-0.1 * (step - warmup_steps) / decay_steps)
                        for step in range(warmup_steps + 1, decay_steps + 1)
                    ]
                )

                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=lr_schedule,
                    clipnorm=1.0  # Gradient clipping
                )
                
                self.model.compile(
                    optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=[
                        'accuracy',
                        tf.keras.metrics.Precision(),
                        tf.keras.metrics.Recall(),
                        tf.keras.metrics.AUC()
                    ]
                )

            except Exception as e:
                logger.error(f"Failed to create/compile model: {str(e)}")
                return False

            # Log model summary with memory estimation
            try:
                model_summary_lines = []
                self.model.summary(print_fn=lambda line: model_summary_lines.append(line))
                logger.info("Model architecture:\n" + "\n".join(model_summary_lines))
                
                # Estimate model memory usage
                model_memory = sum(
                    [tf.keras.backend.count_params(w) * 4 / (1024 ** 2) for w in self.model.trainable_weights]
                )
                logger.info(f"Estimated model memory usage: {model_memory:.2f} MB")

            except Exception as e:
                logger.warning(f"Failed to log model summary: {str(e)}")
                # Don't return False here as this is non-critical

            return True

        except Exception as e:
            logger.error(f"Error in build_model: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    

    def create_data_augmentation(self):
        """Create data augmentation pipeline with comprehensive error handling and validation
        
        Returns:
            tf.keras.Sequential: Data augmentation pipeline or identity layer as fallback
        """
        try:
            # Validate TensorFlow version for augmentation compatibility
            tf_version = tf.__version__.split('.')
            if int(tf_version[0]) < 2 or (int(tf_version[0]) == 2 and int(tf_version[1]) < 3):
                logger.warning("TensorFlow version < 2.3 may have limited augmentation support")
                
            # Configure augmentation parameters based on input shape
            if not hasattr(self, 'input_shape'):
                logger.error("Input shape not defined")
                return tf.keras.Sequential()
                
            # Validate input shape has correct dimensions
            if len(self.input_shape) not in [2, 3]:
                logger.error(f"Invalid input shape for augmentation: {self.input_shape}")
                return tf.keras.Sequential()

            # Create augmentation pipeline with parameter validation
            try:
                augmentation_pipeline = tf.keras.Sequential([
                    # Spatial augmentations
                    layers.RandomRotation(
                        factor=0.1,
                        fill_mode='nearest',
                        interpolation='bilinear',
                        seed=42
                    ),
                    layers.RandomTranslation(
                        height_factor=(-0.1, 0.1),
                        width_factor=(-0.1, 0.1),
                        fill_mode='nearest',
                        interpolation='bilinear',
                        seed=43
                    ),
                    layers.RandomZoom(
                        height_factor=(-0.1, 0.1),
                        width_factor=(-0.1, 0.1),
                        fill_mode='nearest',
                        interpolation='bilinear',
                        seed=44
                    ),
                    
                    # Intensity augmentations
                    layers.RandomBrightness(
                        factor=0.1,
                        value_range=(0, 1),
                        seed=45
                    ),
                    layers.RandomContrast(
                        factor=0.1,
                        seed=46
                    ),
                    
                    # Ensure output values stay in valid range
                    layers.experimental.preprocessing.Rescaling(
                        scale=1.0,
                        offset=0.0
                    )
                ])
                
                # Verify augmentation pipeline
                if len(augmentation_pipeline.layers) == 0:
                    raise ValueError("Empty augmentation pipeline created")
                    
                # Test augmentation pipeline with dummy data
                try:
                    test_shape = (1,) + self.input_shape
                    test_input = tf.random.uniform(test_shape)
                    _ = augmentation_pipeline(test_input)
                    logger.info("Augmentation pipeline validated successfully")
                except Exception as e:
                    logger.error(f"Augmentation pipeline validation failed: {str(e)}")
                    return tf.keras.Sequential()

                return augmentation_pipeline

            except Exception as e:
                logger.error(f"Failed to create augmentation layers: {str(e)}")
                return tf.keras.Sequential()
                
        except Exception as e:
            logger.error(f"Error in create_data_augmentation: {str(e)}")
            logger.error(traceback.format_exc())
            return tf.keras.Sequential()

        finally:
            # Log augmentation configuration
            try:
                if 'augmentation_pipeline' in locals():
                    aug_config = {
                        'num_layers': len(augmentation_pipeline.layers),
                        'layer_names': [layer.__class__.__name__ for layer in augmentation_pipeline.layers]
                    }
                    logger.info(f"Augmentation configuration: {aug_config}")
            except Exception as e:
                logger.warning(f"Failed to log augmentation configuration: {str(e)}")

    def augment_data(self, x, y, augmentation_factor=0.5, batch_size=32):
        """Augment data with comprehensive error handling and validation
        
        Args:
            x (np.ndarray): Input data to augment
            y (np.ndarray): Labels corresponding to input data
            augmentation_factor (float): Fraction of data to augment (0.1 to 1.0)
            batch_size (int): Batch size for processing augmentations
            
        Returns:
            tuple: (augmented_data, augmented_labels) or (empty_array, empty_array) on failure
        """
        try:
            # Input validation
            if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
                logger.error("Input data must be numpy arrays")
                return np.array([]), np.array([])
                
            if x.shape[0] != y.shape[0]:
                logger.error(f"Data shape mismatch: x={x.shape[0]}, y={y.shape[0]}")
                return np.array([]), np.array([])
                
            if np.isnan(x).any() or (np.isnan(y).any() if len(y.shape) > 1 else False):
                logger.error("Input data contains NaN values")
                return np.array([]), np.array([])

            # Create and validate augmentation pipeline
            data_augmentation = self.create_data_augmentation()
            if len(data_augmentation.layers) == 0:
                logger.warning("Empty augmentation pipeline, skipping augmentation")
                return np.array([]), np.array([])

            # Validate and adjust augmentation factor
            augmentation_factor = np.clip(augmentation_factor, 0.1, 1.0)
            aug_size = int(len(x) * augmentation_factor)
            
            if aug_size <= 0:
                logger.warning("Augmentation size too small, skipping augmentation")
                return np.array([]), np.array([])

            # Validate batch size
            batch_size = min(batch_size, aug_size)
            if batch_size <= 0:
                logger.error("Invalid batch size")
                return np.array([]), np.array([])

            # Select indices to augment with stratification if possible
            try:
                if len(y.shape) > 1:  # One-hot encoded
                    labels = np.argmax(y, axis=1)
                else:
                    labels = y
                    
                aug_idx = []
                for class_label in np.unique(labels):
                    class_indices = np.where(labels == class_label)[0]
                    class_size = int(len(class_indices) * augmentation_factor)
                    if class_size > 0:
                        class_aug_idx = np.random.choice(
                            class_indices, 
                            size=class_size, 
                            replace=False
                        )
                        aug_idx.extend(class_aug_idx)
                
                aug_idx = np.array(aug_idx)
                np.random.shuffle(aug_idx)
                
            except Exception as e:
                logger.warning(f"Stratified sampling failed, falling back to random: {str(e)}")
                aug_idx = np.random.choice(range(len(x)), size=aug_size, replace=False)

            # Initialize output arrays
            x_aug = np.zeros((len(aug_idx),) + x.shape[1:], dtype=x.dtype)
            augmentation_stats = {'success': 0, 'failed': 0}

            # Process augmentations in batches
            for batch_start in range(0, len(aug_idx), batch_size):
                batch_end = min(batch_start + batch_size, len(aug_idx))
                batch_idx = aug_idx[batch_start:batch_end]
                
                try:
                    # Handle different input dimensions
                    if len(x.shape) == 3:  # (samples, height, width)
                        batch_data = x[batch_idx].reshape((-1,) + x.shape[1:] + (1,))
                    else:  # Already has channel dimension
                        batch_data = x[batch_idx]

                    # Apply augmentation with memory cleanup
                    try:
                        with tf.device('/cpu:0'):  # Ensure CPU processing for better memory handling
                            aug_batch = data_augmentation(batch_data)
                            if len(x.shape) == 3:
                                aug_batch = tf.squeeze(aug_batch, axis=-1)
                            x_aug[batch_start:batch_end] = aug_batch.numpy()
                        augmentation_stats['success'] += len(batch_idx)
                        
                    except Exception as e:
                        logger.warning(f"Batch augmentation failed, using originals: {str(e)}")
                        x_aug[batch_start:batch_end] = x[batch_idx]
                        augmentation_stats['failed'] += len(batch_idx)

                    # Clear any cached tensors
                    tf.keras.backend.clear_session()
                    
                except Exception as e:
                    logger.warning(f"Failed to process batch {batch_start}-{batch_end}: {str(e)}")
                    x_aug[batch_start:batch_end] = x[batch_idx]
                    augmentation_stats['failed'] += len(batch_idx)

            # Get corresponding labels
            if len(y.shape) > 1:  # One-hot encoded
                y_aug = y[aug_idx]
            else:  # Integer labels
                y_aug = y[aug_idx]

            # Validate output
            if np.isnan(x_aug).any():
                logger.error("Generated augmented data contains NaN values")
                return np.array([]), np.array([])

            # Log augmentation statistics
            success_rate = (augmentation_stats['success'] / len(aug_idx)) * 100
            logger.info(f"Augmentation complete - Success rate: {success_rate:.1f}%")
            logger.info(f"Generated {len(x_aug)} augmented samples")
            logger.info(f"Original data range: [{x.min():.3f}, {x.max():.3f}]")
            logger.info(f"Augmented data range: [{x_aug.min():.3f}, {x_aug.max():.3f}]")

            return x_aug, y_aug

        except Exception as e:
            logger.error(f"Error in augment_data: {str(e)}")
            logger.error(traceback.format_exc())
            return np.array([]), np.array([])
        
    def train(self, epochs=None, batch_size=128, recovery_mode=False):
        """Train the model with comprehensive error handling and monitoring
        
        Args:
            epochs (int, optional): Number of training epochs
            batch_size (int): Training batch size
            recovery_mode (bool): Whether to attempt recovery from checkpoint
            
        Returns:
            bool: True if training succeeded, False otherwise
        """
        try:
            # Validate prerequisites
            if not hasattr(self, 'data_loaded') or not self.data_loaded:
                logger.error("Cannot train: Data not loaded")
                return False
                
            if not hasattr(self, 'model') or self.model is None:
                logger.error("Cannot train: Model not built")
                return False

            # Validate and configure training parameters
            epochs = max(1, epochs if epochs is not None else self.epochs)
            original_batch_size = batch_size
            batch_size = max(16, min(1024, batch_size))
            
            if batch_size != original_batch_size:
                logger.warning(f"Adjusted batch size from {original_batch_size} to {batch_size}")

            logger.info(f"\nTraining model version {self.version}")
            logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
            
            # Load checkpoint if recovery mode
            if recovery_mode:
                if not hasattr(self, 'checkpoint_path'):
                    logger.error("Checkpoint path not defined")
                    return False
                    
                if os.path.exists(self.checkpoint_path):
                    try:
                        self.model.load_weights(self.checkpoint_path)
                        logger.info("Successfully loaded checkpoint weights")
                    except Exception as e:
                        logger.error(f"Failed to load checkpoint: {str(e)}")
                        return False
                else:
                    logger.warning("No checkpoint found for recovery")

            # Prepare training data
            try:
                # Create augmented data
                x_aug, y_aug_onehot = self.augment_data(
                    self.x_train, 
                    self.y_train_onehot,
                    batch_size=batch_size
                )

                # Combine data sources
                data_sources = [(self.x_train, self.y_train_onehot, "original")]
                if len(x_aug) > 0:
                    data_sources.append((x_aug, y_aug_onehot, "augmented"))
                    
                if hasattr(self, 'learning_pool') and len(self.learning_pool) > 0:
                    data_sources.append((
                        np.array(self.learning_pool),
                        np.array(self.learning_pool_labels),
                        "learning pool"
                    ))

                # Combine all data sources
                try:
                    x_train_combined = np.concatenate([src[0] for src in data_sources])
                    y_train_onehot_combined = np.concatenate([src[1] for src in data_sources])
                    
                    # Log data composition
                    total_samples = len(x_train_combined)
                    for _, _, source_name in data_sources:
                        source_ratio = len(_) / total_samples * 100
                        logger.info(f"{source_name} data: {len(_)} samples ({source_ratio:.1f}%)")
                        
                except Exception as e:
                    logger.error(f"Failed to combine training data: {str(e)}")
                    return False

                # Validate combined data
                if len(x_train_combined) == 0 or len(y_train_onehot_combined) == 0:
                    logger.error("Empty training data after combination")
                    return False
                    
                if x_train_combined.shape[0] != y_train_onehot_combined.shape[0]:
                    logger.error("Data shape mismatch after combination")
                    return False

            except Exception as e:
                logger.error(f"Failed to prepare training data: {str(e)}")
                return False

            # Setup training infrastructure
            try:
                # Create checkpoint directory
                if hasattr(self, 'checkpoint_path'):
                    os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
                
                # Configure callbacks with memory management
                callbacks_list = [
                    callbacks.ModelCheckpoint(
                        self.checkpoint_path,
                        save_best_only=True,
                        monitor='val_accuracy',
                        save_weights_only=True  # Reduce memory usage
                    ),
                    callbacks.EarlyStopping(
                        monitor='val_accuracy',
                        patience=self.early_stopping_patience,
                        restore_best_weights=True,
                        verbose=1
                    ),
                    callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=self.lr_factor,
                        patience=self.lr_patience,
                        verbose=1,
                        min_lr=self.min_lr
                    ),
                    # Custom callback for progress tracking
                    callbacks.BaseLogger(
                        stateful_metrics=['loss', 'accuracy', 'val_loss', 'val_accuracy']
                    ),
                    # Memory management callbacks
                    callbacks.TerminateOnNaN(),
                    callbacks.LambdaCallback(
                        on_epoch_end=lambda epoch, logs: gc.collect()
                    )
                ]
                
                # Add TensorBoard if log directory is defined
                if hasattr(self, 'log_dir'):
                    callbacks_list.append(
                        callbacks.TensorBoard(
                            log_dir=f'{self.log_dir}/version_{self.version}',
                            histogram_freq=1,
                            profile_batch=0  # Disable profiling to save memory
                        )
                    )

            except Exception as e:
                logger.error(f"Failed to setup training infrastructure: {str(e)}")
                return False

            # Execute training with error recovery
            try:
                # Configure training parameters
                train_params = {
                    'x': x_train_combined,
                    'y': y_train_onehot_combined,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'validation_data': (self.x_val, self.y_val_onehot),
                    'callbacks': callbacks_list,
                    'verbose': 1,
                    'shuffle': True
                }

                # Execute training with memory monitoring
                start_time = time.time()
                initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                self.history = self.model.fit(**train_params)
                
                # Log training statistics
                end_time = time.time()
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                logger.info(f"\nTraining completed in {(end_time - start_time) / 60:.1f} minutes")
                logger.info(f"Memory usage: {final_memory - initial_memory:.1f}MB")
                
                # Force cleanup
                gc.collect()
                tf.keras.backend.clear_session()

                # Evaluate model
                if hasattr(self, 'evaluate'):
                    evaluation_success = self.evaluate()
                    if not evaluation_success:
                        logger.warning("Model evaluation failed after training")

                return True

            except tf.errors.ResourceExhaustedError:
                logger.error("Out of memory during training")
                
                # Attempt recovery with reduced batch size
                if batch_size > 32:
                    reduced_batch = batch_size // 2
                    logger.info(f"Attempting recovery with batch size: {reduced_batch}")
                    return self.train(
                        epochs=epochs,
                        batch_size=reduced_batch,
                        recovery_mode=True
                    )
                return False

            except Exception as e:
                logger.error(f"Training failed: {str(e)}")
                logger.error(traceback.format_exc())
                return False

        except Exception as e:
            logger.error(f"Error in train method: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        

    def evaluate(self):
        """Evaluate model with comprehensive metrics and error handling
        
        Returns:
            dict: Performance metrics or None on failure
        """
        try:
            # Validate prerequisites
            if not hasattr(self, 'data_loaded') or not self.data_loaded:
                logger.error("Cannot evaluate: Data not loaded")
                return None
                
            if not hasattr(self, 'model') or self.model is None:
                logger.error("Cannot evaluate: Model not built")
                return None
                
            if not hasattr(self, 'x_test') or len(self.x_test) == 0:
                logger.error("Cannot evaluate: Test data not available")
                return None

            logger.info(f"\nEvaluating model version {self.version}...")

            # Record evaluation start time and memory
            start_time = time.time()
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

            try:
                # Evaluate model
                test_results = self.model.evaluate(
                    self.x_test, 
                    self.y_test_onehot,
                    batch_size=min(len(self.x_test), 256),  # Prevent OOM
                    verbose=1
                )
                
                # Get predictions for additional metrics
                y_pred = self.model.predict(
                    self.x_test,
                    batch_size=min(len(self.x_test), 256),
                    verbose=0
                )
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = np.argmax(self.y_test_onehot, axis=1)

                # Calculate additional metrics
                try:
                    from sklearn.metrics import (
                        confusion_matrix,
                        classification_report,
                        roc_auc_score,
                        f1_score
                    )
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_true_classes, y_pred_classes)
                    
                    # Per-class metrics
                    class_report = classification_report(
                        y_true_classes,
                        y_pred_classes,
                        output_dict=True
                    )
                    
                    # ROC AUC (multi-class)
                    roc_auc = roc_auc_score(
                        self.y_test_onehot,
                        y_pred,
                        multi_class='ovr',
                        average='macro'
                    )
                    
                    # F1 Score
                    f1 = f1_score(
                        y_true_classes,
                        y_pred_classes,
                        average='macro'
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate additional metrics: {str(e)}")
                    cm, class_report, roc_auc, f1 = None, None, None, None

            except Exception as e:
                logger.error(f"Model evaluation failed: {str(e)}")
                return None

            # Calculate evaluation time and memory usage
            end_time = time.time()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            eval_time = end_time - start_time
            memory_used = final_memory - initial_memory

            # Compile performance metrics
            performance = {
                'version': self.version,
                'timestamp': time.time(),
                'eval_time_seconds': eval_time,
                'memory_usage_mb': memory_used,
                
                # Basic metrics
                'test_loss': float(test_results[0]),
                'test_accuracy': float(test_results[1]),
                'test_precision': float(test_results[2]),
                'test_recall': float(test_results[3]),
                
                # Additional metrics
                'f1_score': float(f1) if f1 is not None else None,
                'roc_auc': float(roc_auc) if roc_auc is not None else None,
                
                # Model size info
                'total_params': self.model.count_params(),
                'trainable_params': sum([K.count_params(w) for w in self.model.trainable_weights]),
                
                # Batch statistics
                'samples_evaluated': len(self.x_test),
                'evaluation_batch_size': min(len(self.x_test), 256)
            }

            # Add per-class metrics if available
            if class_report is not None:
                for class_id, metrics in class_report.items():
                    if isinstance(metrics, dict):
                        performance[f'class_{class_id}_precision'] = metrics['precision']
                        performance[f'class_{class_id}_recall'] = metrics['recall']
                        performance[f'class_{class_id}_f1'] = metrics['f1-score']

            # Store performance history
            if not hasattr(self, 'performance_history'):
                self.performance_history = []
            self.performance_history.append(performance)

            # Log evaluation results
            logger.info("\nEvaluation Results:")
            logger.info(f"Test Loss: {performance['test_loss']:.4f}")
            logger.info(f"Test Accuracy: {performance['test_accuracy']:.4f}")
            logger.info(f"Test Precision: {performance['test_precision']:.4f}")
            logger.info(f"Test Recall: {performance['test_recall']:.4f}")
            logger.info(f"F1 Score: {performance['f1_score']:.4f}" if performance['f1_score'] is not None else "F1 Score: N/A")
            logger.info(f"ROC AUC: {performance['roc_auc']:.4f}" if performance['roc_auc'] is not None else "ROC AUC: N/A")
            logger.info(f"Evaluation Time: {eval_time:.2f} seconds")
            logger.info(f"Memory Usage: {memory_used:.1f} MB")

            # Log confusion matrix if available
            if cm is not None:
                logger.info("\nConfusion Matrix:")
                logger.info("\n" + str(cm))

            # Save performance history
            try:
                if hasattr(self, 'save_performance_history'):
                    self.save_performance_history()
            except Exception as e:
                logger.warning(f"Failed to save performance history: {str(e)}")

            # Cleanup
            gc.collect()
            tf.keras.backend.clear_session()

            return performance

        except Exception as e:
            logger.error(f"Error in evaluate method: {str(e)}")
            logger.error(traceback.format_exc())
            return None