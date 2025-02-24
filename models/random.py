import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import time
import os
import pickle
import json
import logging
import gc
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SelfImprovingCRNN")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class SelfImprovingCRNN:
    def __init__(self, input_shape=(28, 28), initial_training=True,
                 lr_factor=0.2, lr_patience=5, min_lr=1e-7,
                 early_stopping_patience=15, epochs=30):
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
            self.lr_factor = max(0.01, min(0.5, lr_factor))  # Constrain between 0.01-0.5
            self.lr_patience = max(1, lr_patience)  # Must be at least 1
            self.min_lr = max(1e-10, min(1e-3, min_lr))  # Constrain between 1e-10 and 1e-3
            self.early_stopping_patience = max(3, early_stopping_patience)  # At least 3
            self.epochs = max(5, epochs)  # At least 5 epochs

            # Initialize model components
            self.model = None
            self.history = None
            self.version = 0
            self.performance_history = []
            self.uncertainty_threshold = 0.3  # Initial threshold for active learning
            self.learning_pool = []
            self.learning_pool_labels = []
            self.model_improvements = []
            self.data_loaded = False
            self.checkpoint_path = os.path.join("checkpoints", "model_checkpoint.keras")

            # Create checkpoint directory
            os.makedirs("checkpoints", exist_ok=True)

            # Initialize TF GPU memory growth to prevent OOM errors
            self._setup_gpu()

            # Load data safely
            if not self.load_data():
                logger.error("Failed to load data. Initialization aborted.")
                return

            # Build initial model with error handling
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
                    tf.config.experimental.set_memory_growth(device, True)
                logger.info(f"Found {len(physical_devices)} GPU(s), memory growth enabled")
            else:
                logger.info("No GPU found, using CPU")
        except Exception as e:
            logger.warning(f"GPU setup failed: {str(e)}")

    def load_data(self):
        """Load MNIST dataset with error handling"""
        try:
            logger.info("Loading MNIST dataset...")

            # Attempt to load MNIST dataset
            try:
                (x_train, y_train), (x_test, y_test) = mnist.load_data()
            except Exception as e:
                logger.error(f"Failed to load MNIST dataset: {str(e)}")
                return False

            # Validate data shapes
            if x_train.shape[1:] != self.input_shape or x_test.shape[1:] != self.input_shape:
                logger.error(f"Data shape mismatch. Expected {self.input_shape}, got {x_train.shape[1:]}")
                return False

            # Normalize pixel values (0 to 1)
            x_train = x_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0

            # Convert labels to one-hot encoding
            y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
            y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)

            # Create separate validation set for active learning (avoid test set leakage)
            x_train, x_val, y_train_onehot, y_val_onehot = train_test_split(
                x_train, y_train_onehot, test_size=0.1, random_state=42
            )

            # Get integer labels for stratification and other uses
            y_train = np.argmax(y_train_onehot, axis=1)
            y_val = np.argmax(y_val_onehot, axis=1)

            # Create separate active learning pool (portion of test data)
            # This is to avoid leaking test data into training
            x_test, x_active_pool, y_test_onehot, y_active_pool_onehot = train_test_split(
                x_test, y_test_onehot, test_size=0.3, random_state=42
            )
            y_active_pool = np.argmax(y_active_pool_onehot, axis=1)
            y_test = np.argmax(y_test_onehot, axis=1)

            # Store data
            # self.x_train = x_train
            self.y_train = y_train
            # self.y_train_onehot = y_train_onehot
            # self.x_val = x_val
            self.y_val = y_val
            # self.y_val_onehot = y_val_onehot
            # self.x_test = x_test
            self.y_test = y_test
            # self.y_test_onehot = y_test_onehot
            # self.x_active = x_active_pool
            self.y_active = y_active_pool
            self.y_active_onehot = y_active_pool_onehot

            logger.info(f"Train set: {x_train.shape[0]} samples")
            logger.info(f"Validation set: {x_val.shape[0]} samples")
            logger.info(f"Test set: {x_test.shape[0]} samples")
            logger.info(f"Active learning pool: {x_active_pool.shape[0]} samples")

            self.data_loaded = True
            return True

        except Exception as e:
            logger.error(f"Error in load_data: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def build_model(self, custom_architecture=None):
        """Build CRNN model with error handling and architecture flexibility"""
        try:
            logger.info(f"Building model version {self.version}...")

            if not self.data_loaded:
                logger.error("Cannot build model: Data not loaded")
                return False

            # Free memory from previous model if it exists
            if self.model is not None:
                del self.model
                gc.collect()

            # Input layer - adapt to any input shape
            inputs = layers.Input(shape=self.input_shape)

            if custom_architecture:
                # Use custom architecture if provided
                x = custom_architecture(inputs)
                if not isinstance(x, tf.Tensor):
                    logger.error("Custom architecture must return a tensor")
                    return False
            else:
                # Default architecture
                # Reshape for CNN - add channel dimension
                x = layers.Reshape((*self.input_shape, 1))(inputs)

                # CNN Feature Extraction
                # First convolutional block
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

                # Prepare feature maps for RNN - calculate dimensions dynamically
                new_shape = ((self.input_shape[0] // 4), (self.input_shape[1] // 4) * 64)
                x = layers.Reshape(new_shape)(x)

                # RNN layers
                x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
                x = layers.Dropout(0.25)(x)
                x = layers.Bidirectional(layers.LSTM(64))(x)
                x = layers.Dropout(0.25)(x)

                # Dense layers for classification
                x = layers.Dense(128, activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.5)(x)

            # Output layer (always the same)
            outputs = layers.Dense(10, activation='softmax')(x)

            # Create model with error handling
            try:
                self.model = models.Model(inputs=inputs, outputs=outputs)
            except Exception as e:
                logger.error(f"Failed to create model: {str(e)}")
                return False

            # Compile model with validated parameters
            try:
                # Compile model
                initial_learning_rate = 0.001

                # Define the exponential decay learning rate schedule
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=initial_learning_rate,
                    decay_steps=1000,
                    decay_rate=0.96
                )

                # Define optimizer with learning rate schedule
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
                
                self.model.compile(
                    optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
                )
            except Exception as e:
                logger.error(f"Failed to compile model: {str(e)}")
                return False

            # Log model summary to file
            model_summary_lines = []
            self.model.summary(print_fn=lambda line: model_summary_lines.append(line))
            logger.info("Model architecture:\n" + "\n".join(model_summary_lines))

            return True

        except Exception as e:
            logger.error(f"Error in build_model: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def create_data_augmentation(self):
        """Create data augmentation pipeline with error handling"""
        try:
            return tf.keras.Sequential([
                layers.RandomRotation(0.1),
                layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                layers.RandomZoom(0.1),
            ])
        except Exception as e:
            logger.warning(f"Failed to create data augmentation: {str(e)}")
            # Return identity layer as fallback
            return tf.keras.Sequential()

    def augment_data(self, x, y, augmentation_factor=0.5):
        """Augment data with error handling"""
        try:
            data_augmentation = self.create_data_augmentation()

            # Validate augmentation factor
            augmentation_factor = max(0.1, min(1.0, augmentation_factor))

            # Select indices to augment
            aug_size = int(len(x) * augmentation_factor)
            if aug_size <= 0:
                logger.warning("Augmentation size too small, skipping augmentation")
                return np.array([]), np.array([])

            aug_idx = np.random.choice(range(len(x)), size=aug_size, replace=False)

            # Create augmented samples
            x_aug = np.zeros((aug_size,) + x.shape[1:])
            for i, idx in enumerate(aug_idx):
                try:
                    # Handle different input dimensions
                    if len(x.shape) == 3:  # (samples, height, width)
                        aug_sample = data_augmentation(
                            x[idx].reshape((*x.shape[1:], 1))
                        ).numpy().reshape(x.shape[1:])
                    else:  # Already has channel dimension
                        aug_sample = data_augmentation(x[idx]).numpy()
                    x_aug[i] = aug_sample
                except Exception as e:
                    logger.warning(f"Failed to augment sample {idx}: {str(e)}")
                    # Use original sample as fallback
                    x_aug[i] = x[idx]

            # Get corresponding labels
            if len(y.shape) > 1:  # One-hot encoded
                y_aug = y[aug_idx]
            else:  # Integer labels
                y_aug = y[aug_idx]

            return x_aug, y_aug

        except Exception as e:
            logger.error(f"Error in augment_data: {str(e)}")
            logger.error(traceback.format_exc())
            return np.array([]), np.array([])

    def train(self, epochs=None, batch_size=128, recovery_mode=False):
        """Train the model with current dataset using validated parameters"""
        try:
            if not self.data_loaded or self.model is None:
                logger.error("Cannot train: Data not loaded or model not built")
                return False

            # Use instance epochs if not specified
            if epochs is None:
                epochs = self.epochs

            # Validate epochs and batch size
            epochs = max(1, epochs)
            batch_size = max(16, min(1024, batch_size))  # Reasonable range

            logger.info(f"\nTraining model version {self.version} for {epochs} epochs...")

            # Attempt to recover from previous checkpoint if requested
            if recovery_mode and os.path.exists(self.checkpoint_path):
                try:
                    logger.info(f"Attempting to recover from checkpoint: {self.checkpoint_path}")
                    self.model.load_weights(self.checkpoint_path)
                    logger.info("Successfully loaded weights from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint: {str(e)}")

            # Create augmented data with error handling
            x_aug, y_aug_onehot = self.augment_data(self.x_train, self.y_train_onehot)

            # Check if augmentation succeeded
            if len(x_aug) > 0:
                # Combine original and augmented data
                x_train_combined = np.concatenate([self.x_train, x_aug])
                y_train_onehot_combined = np.concatenate([self.y_train_onehot, y_aug_onehot])
                logger.info(f"Added {len(x_aug)} augmented samples")
            else:
                # Use original data if augmentation failed
                x_train_combined = self.x_train
                y_train_onehot_combined = self.y_train_onehot
                logger.warning("Using original data without augmentation")

            # Include learning pool data if available
            if len(self.learning_pool) > 0:
                try:
                    x_train_combined = np.concatenate([x_train_combined, np.array(self.learning_pool)])
                    y_train_onehot_combined = np.concatenate([y_train_onehot_combined, np.array(self.learning_pool_labels)])
                    logger.info(f"Including {len(self.learning_pool)} samples from learning pool")
                except Exception as e:
                    logger.warning(f"Failed to include learning pool: {str(e)}")

            # Create directories for checkpoints
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

            # Define callbacks with validated parameters
            callbacks_list = [
                callbacks.ModelCheckpoint(
                    self.checkpoint_path,
                    save_best_only=True,
                    monitor='val_accuracy'
                ),
                callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=self.early_stopping_patience,
                    restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=self.lr_factor,
                    patience=self.lr_patience,
                    verbose=1,
                    min_lr=self.min_lr
                ),
                callbacks.TensorBoard(
                    log_dir=f'./logs/version_{self.version}',
                    histogram_freq=1
                ),
                # Custom callback for OOM prevention
                callbacks.TerminateOnNaN()
            ]

            # Train the model with error handling
            try:
                self.history = self.model.fit(
                    x_train_combined, y_train_onehot_combined,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(self.x_val, self.y_val_onehot),
                    callbacks=callbacks_list,
                    verbose=1,
                    shuffle=True
                )

                # Force garbage collection after training
                gc.collect()

                # Evaluate after training
                self.evaluate()
                return True

            except tf.errors.ResourceExhaustedError:
                logger.error("OOM error during training. Try reducing batch size.")
                # Attempt to reduce batch size and recover
                if batch_size > 32:
                    reduced_batch = batch_size // 2
                    logger.info(f"Attempting recovery with reduced batch size: {reduced_batch}")
                    return self.train(epochs=epochs, batch_size=reduced_batch, recovery_mode=True)
                return False

            except Exception as e:
                logger.error(f"Training failed: {str(e)}")
                return False

        except Exception as e:
            logger.error(f"Error in train method: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def evaluate(self):
        """Evaluate model on test set and record performance with error handling"""
        try:
            if not self.data_loaded or self.model is None:
                logger.error("Cannot evaluate: Data not loaded or model not built")
                return None

            logger.info(f"\nEvaluating model version {self.version}...")

            try:
                test_results = self.model.evaluate(self.x_test, self.y_test_onehot, verbose=1)
            except Exception as e:
                logger.error(f"Evaluation failed: {str(e)}")
                return None

            # Record performance metrics
            performance = {
                'version': self.version,
                'test_loss': float(test_results[0]),
                'test_accuracy': float(test_results[1]),
                'test_precision': float(test_results[2]),
                'test_recall': float(test_results[3]),
                'timestamp': time.time()
            }
            self.performance_history.append(performance)

            logger.info(f"Test loss: {test_results[0]:.4f}")
            logger.info(f"Test accuracy: {test_results[1]:.4f}")
            logger.info(f"Test precision: {test_results[2]:.4f}")
            logger.info(f"Test recall: {test_results[3]:.4f}")

            # Save performance history
            self.save_performance_history()

            return performance

        except Exception as e:
            logger.error(f"Error in evaluate method: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def identify_improvement_areas(self):
        """Identify areas where the model needs improvement with error handling"""
        try:
            if not self.data_loaded or self.model is None:
                logger.error("Cannot identify improvement areas: Data not loaded or model not built")
                return None, None, None

            logger.info("\nIdentifying improvement areas...")

            try:
                # Generate predictions on test set
                y_pred = self.model.predict(self.x_test, batch_size=128)
                y_pred_classes = np.argmax(y_pred, axis=1)
            except Exception as e:
                logger.error(f"Prediction failed: {str(e)}")
                return None, None, None

            # Find misclassified samples and their uncertainties
            misclassified_indices = np.where(y_pred_classes != self.y_test)[0]
            logger.info(f"Found {len(misclassified_indices)} misclassified samples")

            # Calculate prediction certainty (max probability)
            prediction_certainty = np.max(y_pred, axis=1)

            # Low certainty predictions (even if correct)
            # Validate uncertainty threshold
            valid_threshold = max(0.1, min(0.9, self.uncertainty_threshold))
            uncertain_indices = np.where(prediction_certainty < valid_threshold)[0]
            logger.info(f"Found {len(uncertain_indices)} uncertain predictions")

            # Find error patterns by looking at the confusion matrix
            try:
                cm = confusion_matrix(self.y_test, y_pred_classes)
                # Create a copy to avoid modifying original
                cm_no_diagonal = cm.copy()
                np.fill_diagonal(cm_no_diagonal, 0)
                max_confusion = np.unravel_index(np.argmax(cm_no_diagonal), cm_no_diagonal.shape)
                logger.info(f"Highest confusion: True {max_confusion[0]} predicted as {max_confusion[1]} ({cm_no_diagonal[max_confusion]} instances)")
            except Exception as e:
                logger.warning(f"Could not create confusion matrix: {str(e)}")
                max_confusion = (0, 0)

            # Collect these areas for improvement
            improvement_areas = {
                'misclassified_count': int(len(misclassified_indices)),
                'uncertain_count': int(len(uncertain_indices)),
                'max_confusion_pair': max_confusion,
                'max_confusion_count': int(cm_no_diagonal[max_confusion]) if 'cm_no_diagonal' in locals() else 0,
                'uncertain_threshold': float(valid_threshold),
                'timestamp': time.time()
            }
            self.model_improvements.append(improvement_areas)

            # Save improvements data
            try:
                with open(f'model_improvements_v{self.version}.json', 'w') as f:
                    json.dump(improvement_areas, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save improvements data: {str(e)}")

            return misclassified_indices, uncertain_indices, max_confusion

        except Exception as e:
            logger.error(f"Error in identify_improvement_areas: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None, None

    def active_learning_sample_selection(self):
        """Select samples for active learning from dedicated active learning pool"""
        try:
            if not self.data_loaded or self.model is None:
                logger.error("Cannot perform active learning: Data not loaded or model not built")
                return False

            # Generate predictions on active learning pool (not test set)
            try:
                y_pred = self.model.predict(self.x_active, batch_size=128)
                prediction_certainty = np.max(y_pred, axis=1)
            except Exception as e:
                logger.error(f"Prediction failed for active learning: {str(e)}")
                return False

            # Find samples with high uncertainty (low confidence)
            # Validate uncertainty threshold
            valid_threshold = max(0.1, min(0.9, self.uncertainty_threshold))
            uncertain_indices = np.where(prediction_certainty < valid_threshold)[0]

            if len(uncertain_indices) == 0:
                logger.info("No uncertain samples found for active learning")
                return False

            # Select a random subset to add to learning pool (to avoid too many similar samples)
            selection_size = min(100, len(uncertain_indices))
            if selection_size > 0:
                selected_indices = np.random.choice(uncertain_indices, selection_size, replace=False)
            else:
                logger.info("No samples selected for active learning")
                return False

            # Add selected samples to learning pool with deduplication
            added_count = 0
            for idx in selected_indices:
                # Skip if already in pool using efficient numpy comparison
                if self.learning_pool:
                    # Convert to numpy array for efficient comparison if not already
                    learning_pool_array = np.array(self.learning_pool)
                    # Check if sample is already in pool
                    if any(np.allclose(learning_pool_array[i], self.x_active[idx])
                            for i in range(len(learning_pool_array))):
                        continue

                # Add if not already in pool
                self.learning_pool.append(self.x_active[idx])
                self.learning_pool_labels.append(self.y_active_onehot[idx])
                added_count += 1

            logger.info(f"Added {added_count} new samples to learning pool")
            logger.info(f"Learning pool now contains {len(self.learning_pool)} samples")

            # Save learning pool periodically
            if len(self.learning_pool) > 0 and len(self.learning_pool) % 100 == 0:
                self.save_learning_pool()

            return added_count > 0

        except Exception as e:
            logger.error(f"Error in active_learning_sample_selection: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def save_learning_pool(self):
        """Save learning pool to file"""
        try:
            if len(self.learning_pool) > 0:
                with open(f"learning_pool_v{self.version}.pkl", 'wb') as f:
                    pickle.dump({
                        'samples': self.learning_pool,
                        'labels': self.learning_pool_labels
                    }, f)
                logger.info(f"Saved learning pool with {len(self.learning_pool)} samples")
        except Exception as e:
            logger.warning(f"Failed to save learning pool: {str(e)}")

    def self_improve(self):
        """Main self-improvement loop with error handling"""
        try:
            logger.info("\nStarting self-improvement cycle...")

            if not self.data_loaded or self.model is None:
                logger.error("Cannot self-improve: Data not loaded or model not built")
                return False

            # 1. Identify improvement areas
            misclassified, uncertain, confusion_pair = self.identify_improvement_areas()
            if misclassified is None:
                logger.warning("Could not identify improvement areas, using default strategy")

            # 2. Perform active learning sample selection
            al_success = self.active_learning_sample_selection()
            if not al_success:
                logger.warning("Active learning did not add new samples")

            # 3. Update model architecture or hyperparameters if needed
            architecture_changed = self.evolve_model()

            # 4. Increment version and train the new model
            self.version += 1
            training_success = self.train()
            if not training_success:
                logger.error("Training failed during self-improvement")
                self.version -= 1  # Revert version increment
                return False

            # 5. Update uncertainty threshold based on model performance
            self.update_uncertainty_threshold()

            # 6. Save evolved model
            self.save_model()

            return True

        except Exception as e:
            logger.error(f"Error in self_improve method: {str(e)}")
            logger.error(traceback.format_exc())
            # Revert version increment if error occurred
            if hasattr(self, 'version') and self.version > 0:
                self.version -= 1
            return False

    def update_uncertainty_threshold(self):
        """Update uncertainty threshold based on model performance"""
        try:
            if len(self.performance_history) >= 2:
                current_acc = self.performance_history[-1]['test_accuracy']
                previous_acc = self.performance_history[-2]['test_accuracy']

                # Only adjust if we have valid accuracy values
                if 0 <= current_acc <= 1 and 0 <= previous_acc <= 1:
                    if current_acc > previous_acc:
                        # Model improved, reduce threshold (be more selective)
                        new_threshold = max(0.1, self.uncertainty_threshold * 0.9)
                    else:
                        # Model didn't improve, increase threshold (be more inclusive)
                        new_threshold = min(0.9, self.uncertainty_threshold * 1.1)

                    logger.info(f"Adjusted uncertainty threshold: {self.uncertainty_threshold:.3f} -> {new_threshold:.3f}")
                    self.uncertainty_threshold = new_threshold
        except Exception as e:
            logger.warning(f"Failed to update uncertainty threshold: {str(e)}")

    def evolve_model(self):
        """Evolve model architecture based on performance with error handling"""
        try:
            if not hasattr(self, 'history') or self.history is None:
                logger.warning("No training history available, skipping model evolution")
                return False

            # Check if we have enough history to detect trends
            if not hasattr(self.history, 'history') or len(self.history.history.get('val_loss', [])) < 3:
                logger.info("Not enough training history to evolve model")
                return False

            # Get training and validation metrics
            train_loss_end = self.history.history['loss'][-1]
            val_loss_end = self.history.history['val_loss'][-1]

            # Check for overfitting or underfitting
            if val_loss_end != 0 and train_loss_end != 0:  # Avoid division by zero
                overfitting_ratio = train_loss_end / val_loss_end

            
            # Save current weights before rebuilding
                try:
                    weights_path = f"temp_weights_v{self.version}.h5"
                    self.model.save_weights(weights_path)
                    logger.info(f"Saved temporary weights to {weights_path}")
                except Exception as e:
                    logger.warning(f"Failed to save temporary weights: {str(e)}")
                    weights_path = None

                # Determine architecture adjustments based on overfitting/underfitting
                if overfitting_ratio < 0.6:  # Significant overfitting
                    logger.info(f"Detected overfitting (ratio: {overfitting_ratio:.2f}), simplifying model")
                    success = self.build_model(custom_architecture=self._create_simpler_architecture())
                elif overfitting_ratio > 0.95:  # Underfitting
                    logger.info(f"Detected underfitting (ratio: {overfitting_ratio:.2f}), increasing model capacity")
                    success = self.build_model(custom_architecture=self._create_complex_architecture())
                else:
                    logger.info(f"Model complexity seems appropriate (ratio: {overfitting_ratio:.2f}), making minor adjustments")
                    success = self.build_model(custom_architecture=self._create_refined_architecture())

                # Restore weights if possible and applicable
                if success and weights_path and os.path.exists(weights_path):
                    try:
                        # Try to load compatible weights
                        self.model.load_weights(weights_path, by_name=True, skip_mismatch=True)
                        logger.info("Restored compatible weights from previous version")
                    except Exception as e:
                        logger.warning(f"Could not restore weights: {str(e)}")

                    # Clean up temporary weights file
                    try:
                        os.remove(weights_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary weights file: {str(e)}")

                return success
            else:
                logger.warning("Invalid loss values detected, skipping model evolution")
                return False

        except Exception as e:
            logger.error(f"Error in evolve_model: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _create_simpler_architecture(self):
        """Create a simpler architecture to combat overfitting"""
        def architecture(inputs):
            try:
                # Reshape for CNN - add channel dimension
                x = layers.Reshape((*self.input_shape, 1))(inputs)

                # Simpler architecture with stronger regularization
                x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.3)(x)
                x = layers.MaxPooling2D((2, 2))(x)

                x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.4)(x)
                x = layers.MaxPooling2D((2, 2))(x)

                # Calculate dimensions dynamically
                new_shape = ((self.input_shape[0] // 4), (self.input_shape[1] // 4) * 32)
                x = layers.Reshape(new_shape)(x)

                # Single RNN layer
                x = layers.Bidirectional(layers.LSTM(64))(x)
                x = layers.Dropout(0.4)(x)

                # Smaller fully connected layer
                x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.5)(x)

                return x
            except Exception as e:
                logger.error(f"Error in simpler architecture: {str(e)}")
                # Return default architecture if this fails
                return self._create_default_architecture()(inputs)

        return architecture

    def _create_complex_architecture(self):
        """Create a more complex architecture to combat underfitting"""
        def architecture(inputs):
            try:
                # Reshape for CNN - add channel dimension
                x = layers.Reshape((*self.input_shape, 1))(inputs)

                # More complex CNN with residual connections
                # First block with residual connection
                input_layer = x
                x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
                x = layers.BatchNormalization()(x)
                # Add residual connection if shapes match
                if input_layer.shape[-1] == 32:
                    x = layers.add([x, input_layer])
                x = layers.MaxPooling2D((2, 2))(x)
                x = layers.Dropout(0.2)(x)

                # Second block with residual connection
                input_layer = x
                x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
                x = layers.BatchNormalization()(x)
                # Add 1x1 conv to match shapes for residual if needed
                if input_layer.shape[-1] != 64:
                    input_layer = layers.Conv2D(64, (1, 1), padding='same')(input_layer)
                x = layers.add([x, input_layer])
                x = layers.MaxPooling2D((2, 2))(x)
                x = layers.Dropout(0.2)(x)

                # Third block
                x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
                x = layers.BatchNormalization()(x)
                x = layers.MaxPooling2D((2, 2))(x)
                x = layers.Dropout(0.2)(x)

                # Calculate dimensions dynamically based on pooling operations
                pool_factor = 8  # 2^3 because of three MaxPooling2D layers
                new_height = self.input_shape[0] // pool_factor
                new_width = self.input_shape[1] // pool_factor
                new_shape = (new_height, new_width * 128)
                x = layers.Reshape(new_shape)(x)

                # More complex RNN stack
                x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
                x = layers.LayerNormalization()(x)
                x = layers.Dropout(0.2)(x)
                x = layers.Bidirectional(layers.LSTM(64))(x)
                x = layers.LayerNormalization()(x)
                x = layers.Dropout(0.2)(x)

                # Wider fully connected layers
                x = layers.Dense(256, activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.3)(x)
                x = layers.Dense(128, activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.3)(x)

                return x
            except Exception as e:
                logger.error(f"Error in complex architecture: {str(e)}")
                # Return default architecture if this fails
                return self._create_default_architecture()(inputs)

        return architecture

    def _create_refined_architecture(self):
        """Create a refined architecture with minor adjustments"""
        def architecture(inputs):
            try:
                # Reshape for CNN - add channel dimension
                x = layers.Reshape((*self.input_shape, 1))(inputs)

                # Enhanced CNN feature extraction
                x = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
                x = layers.MaxPooling2D((2, 2))(x)
                x = layers.Dropout(0.25)(x)

                x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
                x = layers.MaxPooling2D((2, 2))(x)
                x = layers.Dropout(0.25)(x)

                # Calculate dimensions dynamically
                new_shape = ((self.input_shape[0] // 4), (self.input_shape[1] // 4) * 64)
                x = layers.Reshape(new_shape)(x)

                # Refined RNN with attention
                rnn = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
                attention = layers.Dense(1, activation='tanh')(rnn)
                attention = layers.Flatten()(attention)
                attention = layers.Activation('softmax')(attention)
                attention = layers.RepeatVector(256)(attention)
                attention = layers.Permute([2, 1])(attention)
                attention_mul = layers.Multiply()([rnn, attention])
                attention_sum = layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(attention_mul)

                # Combine with another LSTM path
                x = layers.Bidirectional(layers.LSTM(64))(x)
                x = layers.Concatenate()([x, attention_sum])
                x = layers.Dropout(0.25)(x)

                # Dense layers with batch normalization
                x = layers.Dense(128, activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.4)(x)

                return x
            except Exception as e:
                logger.error(f"Error in refined architecture: {str(e)}")
                # Return default architecture if this fails
                return self._create_default_architecture()(inputs)

        return architecture

    def _create_default_architecture(self):
        """Create the default architecture as a fallback"""
        def architecture(inputs):
            # Reshape for CNN - add channel dimension
            x = layers.Reshape((*self.input_shape, 1))(inputs)

            # CNN Feature Extraction
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)

            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)

            # Prepare feature maps for RNN
            new_shape = ((self.input_shape[0] // 4), (self.input_shape[1] // 4) * 64)
            x = layers.Reshape(new_shape)(x)

            # RNN layers
            x = layers.Bidirectional(layers.LSTM(64))(x)
            x = layers.Dropout(0.25)(x)

            # Dense layer
            x = layers.Dense(128, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)

            return x

        return architecture

    def save_model(self):
        """Save model with error handling"""
        try:
            if self.model is None:
                logger.error("Cannot save: Model not built")
                return False

            model_path = f"model_v{self.version}.keras"
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")

            # Save model config in JSON format
            model_config = self.model.get_config()
            try:
                with open(f"{model_path}_config.json", 'w') as f:
                    json.dump(model_config, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save model config: {str(e)}")

            return True

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            logger.error(traceback.format_exc())

            # Try saving weights only as fallback
            try:
                weights_path = f"model_weights_v{self.version}.h5"
                self.model.save_weights(weights_path)
                logger.info(f"Saved weights to {weights_path} as fallback")
                return True
            except Exception as e2:
                logger.error(f"Failed to save weights as fallback: {str(e2)}")
                return False

    def load_model(self, version=None):
        """Load a saved model with error handling"""
        try:
            # Determine which version to load
            if version is None:
                # Find the latest version saved
                model_files = [f for f in os.listdir() if f.startswith("model_v") and os.path.isdir(f)]
                if not model_files:
                    logger.error("No saved models found")
                    return False

                # Extract version numbers and find highest
                versions = [int(f.split('_v')[1]) for f in model_files]
                version = max(versions)

            model_path = f"model_v{version}.keras"

            if not os.path.exists(model_path):
                logger.error(f"Model path {model_path} does not exist")
                return False

            # Free memory from previous model
            if self.model is not None:
                del self.model
                gc.collect()

            # Load the model
            try:
                self.model = tf.keras.models.load_model(model_path)
                self.version = version
                logger.info(f"Loaded model version {version}")

                # Load performance history if exists
                self.load_performance_history()

                return True
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")

                # Try loading weights as fallback
                try:
                    weights_path = f"model_weights_v{version}.h5"
                    if os.path.exists(weights_path):
                        # Need to build the model first
                        self.build_model()
                        self.model.load_weights(weights_path)
                        self.version = version
                        logger.info(f"Loaded model weights for version {version} as fallback")
                        return True
                    else:
                        logger.error(f"Weights file {weights_path} not found")
                        return False
                except Exception as e2:
                    logger.error(f"Failed to load weights as fallback: {str(e2)}")
                    return False

        except Exception as e:
            logger.error(f"Error in load_model: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def save_performance_history(self):
        """Save performance history to file"""
        try:
            if self.performance_history:
                with open('performance_history.json', 'w') as f:
                    json.dump(self.performance_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save performance history: {str(e)}")

    def load_performance_history(self):
        """Load performance history from file"""
        try:
            if os.path.exists('performance_history.json'):
                with open('performance_history.json', 'r') as f:
                    self.performance_history = json.load(f)
                logger.info(f"Loaded performance history with {len(self.performance_history)} entries")
        except Exception as e:
            logger.warning(f"Failed to load performance history: {str(e)}")

    def get_model_summary(self):
        """Get a comprehensive model summary with error handling"""
        if self.model is None:
            return "Model not built"

        try:
            # Collect model structure
            summary_lines = []
            self.model.summary(print_fn=lambda line: summary_lines.append(line))
            model_summary = "\n".join(summary_lines)

            # Performance metrics if available
            if self.performance_history and len(self.performance_history) > 0:
                latest_perf = self.performance_history[-1]
                perf_summary = (
                    f"\nLatest Performance Metrics (Version {latest_perf['version']}):\n"
                    f"Test Accuracy: {latest_perf['test_accuracy']:.4f}\n"
                    f"Test Precision: {latest_perf['test_precision']:.4f}\n"
                    f"Test Recall: {latest_perf['test_recall']:.4f}\n"
                    f"Test Loss: {latest_perf['test_loss']:.4f}\n"
                )
            else:
                perf_summary = "\nNo performance metrics available yet."

            # Learning pool status
            learning_pool_info = f"\nActive Learning Pool Size: {len(self.learning_pool)} samples"

            # Training history if available
            if hasattr(self, 'history') and self.history is not None:
                epoch_count = len(self.history.history.get('accuracy', []))
                final_acc = self.history.history.get('accuracy', [0])[-1]
                final_val_acc = self.history.history.get('val_accuracy', [0])[-1]
                train_summary = (
                    f"\nTraining Summary:\n"
                    f"Epochs Trained: {epoch_count}\n"
                    f"Final Training Accuracy: {final_acc:.4f}\n"
                    f"Final Validation Accuracy: {final_val_acc:.4f}\n"
                )
            else:
                train_summary = "\nNo training history available yet."

            return model_summary + perf_summary + learning_pool_info + train_summary

        except Exception as e:
            logger.error(f"Error generating model summary: {str(e)}")
            return f"Error generating model summary: {str(e)}"

    def visualize_performance_history(self, save_plot=True):
        """Visualize model performance history across versions"""
        try:
            if not self.performance_history:
                logger.warning("No performance history available to visualize")
                return None

            plt.figure(figsize=(12, 8))

            # Extract data
            versions = [p['version'] for p in self.performance_history]
            accuracies = [p['test_accuracy'] for p in self.performance_history]
            precisions = [p['test_precision'] for p in self.performance_history]
            recalls = [p['test_recall'] for p in self.performance_history]
            losses = [p['test_loss'] for p in self.performance_history]

            # Create subplot for metrics
            plt.subplot(2, 1, 1)
            plt.plot(versions, accuracies, 'o-', label='Accuracy')
            plt.plot(versions, precisions, 's-', label='Precision')
            plt.plot(versions, recalls, '^-', label='Recall')
            plt.title('Model Performance Metrics Across Versions')
            plt.xlabel('Model Version')
            plt.ylabel('Score')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

            # Create subplot for loss
            plt.subplot(2, 1, 2)
            plt.plot(versions, losses, 'o-', color='red')
            plt.title('Model Loss Across Versions')
            plt.xlabel('Model Version')
            plt.ylabel('Test Loss')
            plt.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()

            # Save the plot if requested
            if save_plot:
                plt.savefig('performance_history.png')
                logger.info("Performance history visualization saved to performance_history.png")

            return plt

        except Exception as e:
            logger.error(f"Error visualizing performance history: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def predict_with_reliability(self, x, reliability_threshold=0.8):
        """Make predictions with reliability assessment"""
        try:
            if self.model is None:
                logger.error("Cannot predict: Model not built")
                return None, None, None

            # Convert input to correct format if needed
            if not isinstance(x, np.ndarray):
                x = np.array(x)

            # Add batch dimension if needed
            if len(x.shape) == len(self.input_shape):
                x = np.expand_dims(x, 0)

            # Normalize input if needed
            if x.max() > 1.0:
                x = x.astype('float32') / 255.0

            # Make prediction
            predictions = self.model.predict(x)

            # Get class with highest probability
            predicted_classes = np.argmax(predictions, axis=1)

            # Get confidence scores (max probability)
            confidence_scores = np.max(predictions, axis=1)

            # Assess reliability based on confidence threshold
            reliability = np.where(confidence_scores >= reliability_threshold, True, False)

            return predicted_classes, confidence_scores, reliability

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None, None

    def generate_confusion_matrix(self, save_plot=True):
        """Generate and visualize confusion matrix"""
        try:
            if not self.data_loaded or self.model is None:
                logger.error("Cannot generate confusion matrix: Data not loaded or model not built")
                return None

            # Generate predictions
            y_pred = self.model.predict(self.x_test)
            y_pred_classes = np.argmax(y_pred, axis=1)

            # Calculate confusion matrix
            cm = confusion_matrix(self.y_test, y_pred_classes)

            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - Model Version {self.version}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            if save_plot:
                plt.savefig(f'confusion_matrix_v{self.version}.png')
                logger.info(f"Confusion matrix saved to confusion_matrix_v{self.version}.png")

            return cm, plt

        except Exception as e:
            logger.error(f"Error generating confusion matrix: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None

    def cleanup(self):
        """Clean up resources"""
        try:
            # Free memory
            if self.model is not None:
                del self.model

            # Clear TF session
            tf.keras.backend.clear_session()

            # Collect garbage
            gc.collect()

            logger.info("Resources cleaned up")
            return True

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    try:
        # Initialize the self-improving model
        model = SelfImprovingCRNN(
            input_shape=(28, 28),
            lr_factor=0.5,
            lr_patience=3,
            early_stopping_patience=10,
            epochs=20
        )

        # Train initial model
        model.train()

        # Evaluate performance
        model.evaluate()

        # Identify areas for improvement
        model.identify_improvement_areas()

        # Perform active learning
        model.active_learning_sample_selection()

        # Self-improve for several cycles
        for cycle in range(3):
            logger.info(f"\n=== Starting Improvement Cycle {cycle+1} ===")
            model.self_improve()

        # Generate final performance visualization
        model.visualize_performance_history()

        # Generate confusion matrix
        model.generate_confusion_matrix()

        # Print final model summary
        print(model.get_model_summary())

        # Clean up resources
        model.cleanup()

    except Exception as e:
        logger.critical(f"Critical error in main execution: {str(e)}")
        logger.critical(traceback.format_exc())