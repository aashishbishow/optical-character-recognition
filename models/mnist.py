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

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class SelfImprovingCRNN:
    def __init__(self, input_shape=(28, 28), initial_training=True):
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.version = 0
        self.performance_history = []
        self.uncertainty_threshold = 0.3  # Initial threshold for active learning
        self.learning_pool = []
        self.learning_pool_labels = []
        self.model_improvements = []
        
        # Load data
        self.load_data()
        
        # Build initial model
        self.build_model()
        
        # Initial training if requested
        if initial_training:
            self.train()
    
    def load_data(self):
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Normalize pixel values (0 to 1)
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Convert labels to one-hot encoding
        y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
        y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)
        
        # Create train/validation split from the training data - FIXED
        x_train, x_val, y_train_onehot, y_val_onehot = train_test_split(
            x_train, y_train_onehot, test_size=0.1, random_state=42
        )
        # Get integer labels for stratification and other uses
        y_train = np.argmax(y_train_onehot, axis=1)
        y_val = np.argmax(y_val_onehot, axis=1)
        
        # Store data
        self.x_train = x_train
        self.y_train = y_train
        self.y_train_onehot = y_train_onehot
        self.x_val = x_val
        self.y_val = y_val
        self.y_val_onehot = y_val_onehot
        self.x_test = x_test
        self.y_test = y_test
        self.y_test_onehot = y_test_onehot
        
        print(f"Train set: {x_train.shape[0]} samples")
        print(f"Validation set: {x_val.shape[0]} samples")
        print(f"Test set: {x_test.shape[0]} samples")
    
    def build_model(self):
        """Build CRNN model with current best architecture"""
        print(f"Building model version {self.version}...")
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Reshape for CNN - add channel dimension
        x = layers.Reshape((28, 28, 1))(inputs)
        
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
        
        # Prepare feature maps for RNN
        # (batch_size, 7, 7, 64) -> (batch_size, 7, 7*64)
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
        outputs = layers.Dense(10, activation='softmax')(x)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        initial_learning_rate = 0.001

        # Define optimizer with a fixed learning rate (will be adjusted dynamically)
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        print(self.model.summary())
    
    def create_data_augmentation(self):
        """Create data augmentation pipeline"""
        return tf.keras.Sequential([
            layers.RandomRotation(0.1),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomZoom(0.1),
        ])
    
    def augment_data(self, x, y, augmentation_factor=0.5):
        """Augment a portion of the data"""
        data_augmentation = self.create_data_augmentation()
        
        # Select indices to augment
        aug_size = int(len(x) * augmentation_factor)
        aug_idx = np.random.choice(range(len(x)), size=aug_size, replace=False)
        
        # Create augmented samples
        x_aug = np.zeros((aug_size,) + x.shape[1:])
        for i, idx in enumerate(aug_idx):
            x_aug[i] = data_augmentation(x[idx].reshape(28, 28, 1)).numpy().reshape(28, 28)
        
        # Get corresponding labels
        if len(y.shape) > 1:  # One-hot encoded
            y_aug = y[aug_idx]
        else:  # Integer labels
            y_aug = y[aug_idx]
        
        return x_aug, y_aug
    
    def train(self, epochs=30, batch_size=128):
        """Train the model with current dataset"""
        print(f"\nTraining model version {self.version}...")
        
        # Augment training data
        x_aug, y_aug_onehot = self.augment_data(self.x_train, self.y_train_onehot)
        
        # Combine original and augmented data
        x_train_combined = np.concatenate([self.x_train, x_aug])
        y_train_onehot_combined = np.concatenate([self.y_train_onehot, y_aug_onehot])
        
        # Include learning pool data if available
        if len(self.learning_pool) > 0:
            x_train_combined = np.concatenate([x_train_combined, np.array(self.learning_pool)])
            y_train_onehot_combined = np.concatenate([y_train_onehot_combined, np.array(self.learning_pool_labels)])
            print(f"Including {len(self.learning_pool)} samples from learning pool")
        
        # Define callbacks with fixed learning rate reduction
        checkpoint_cb = callbacks.ModelCheckpoint(
            f'model_v{self.version}_best.h5', 
            save_best_only=True, 
            monitor='val_accuracy'
        )
        early_stopping_cb = callbacks.EarlyStopping(
            monitor='val_accuracy', 
            patience=15, 
            restore_best_weights=True
        )
        # FIXED: Using fixed reduction values instead of LearningRateSchedule
        reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss",   # Reduce LR when validation loss stops improving
        factor=0.2,           # Reduce LR by 50% (multiplication factor)
        patience=5,           # Wait for 3 epochs before reducing LR
        verbose=1,            # Print LR changes
        min_lr=1e-7           # Minimum possible LR
        )
        
        # Train the model
        self.history = self.model.fit(
            x_train_combined, y_train_onehot_combined,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.x_val, self.y_val_onehot),
            callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
            verbose=1,
            shuffle=True
        )
        
        # Evaluate after training
        self.evaluate()
    
    def evaluate(self):
        """Evaluate model on test set and record performance"""
        print(f"\nEvaluating model version {self.version}...")
        test_results = self.model.evaluate(self.x_test, self.y_test_onehot, verbose=1)
        
        # Record performance metrics
        performance = {
            'version': self.version,
            'test_loss': test_results[0],
            'test_accuracy': test_results[1],
            'test_precision': test_results[2],
            'test_recall': test_results[3],
            'timestamp': time.time()
        }
        self.performance_history.append(performance)
        
        print(f"Test loss: {test_results[0]:.4f}")
        print(f"Test accuracy: {test_results[1]:.4f}")
        print(f"Test precision: {test_results[2]:.4f}")
        print(f"Test recall: {test_results[3]:.4f}")
        
        # Save performance history
        self.save_performance_history()
    
    def identify_improvement_areas(self):
        """Identify areas where the model needs improvement"""
        print("\nIdentifying improvement areas...")
        
        # Generate predictions
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Find misclassified samples and their uncertainties
        misclassified_indices = np.where(y_pred_classes != self.y_test)[0]
        print(f"Found {len(misclassified_indices)} misclassified samples")
        
        # Calculate prediction certainty (max probability)
        prediction_certainty = np.max(y_pred, axis=1)
        
        # Low certainty predictions (even if correct)
        uncertain_indices = np.where(prediction_certainty < self.uncertainty_threshold)[0]
        print(f"Found {len(uncertain_indices)} uncertain predictions")
        
        # Find error patterns by looking at the confusion matrix
        cm = confusion_matrix(self.y_test, y_pred_classes)
        
        # Find the top confusions (excluding the diagonal)
        np.fill_diagonal(cm, 0)
        max_confusion = np.unravel_index(np.argmax(cm), cm.shape)
        print(f"Highest confusion: True {max_confusion[0]} predicted as {max_confusion[1]} ({cm[max_confusion]} instances)")
        
        # Collect these areas for improvement
        improvement_areas = {
            'misclassified_count': len(misclassified_indices),
            'uncertain_count': len(uncertain_indices),
            'max_confusion_pair': max_confusion,
            'max_confusion_count': cm[max_confusion],
            'uncertain_threshold': self.uncertainty_threshold,
            'timestamp': time.time()
        }
        self.model_improvements.append(improvement_areas)
        
        return misclassified_indices, uncertain_indices, max_confusion
    
    def active_learning_sample_selection(self):
        """Select samples for active learning using uncertainty sampling"""
        # Generate predictions on test set
        y_pred = self.model.predict(self.x_test)
        prediction_certainty = np.max(y_pred, axis=1)
        
        # Find samples with high uncertainty (low confidence)
        uncertain_indices = np.where(prediction_certainty < self.uncertainty_threshold)[0]
        
        # Select a random subset to add to learning pool (to avoid too many similar samples)
        if len(uncertain_indices) > 100:
            selected_indices = np.random.choice(uncertain_indices, 100, replace=False)
        else:
            selected_indices = uncertain_indices
        
        # Add selected samples to learning pool
        for idx in selected_indices:
            # Only add if not already in pool
            if not any(np.array_equal(self.x_test[idx], sample) for sample in self.learning_pool):
                self.learning_pool.append(self.x_test[idx])
                self.learning_pool_labels.append(self.y_test_onehot[idx])
        
        print(f"Added {len(selected_indices)} new samples to learning pool")
        print(f"Learning pool now contains {len(self.learning_pool)} samples")
    
    def self_improve(self):
        """Main self-improvement loop"""
        print("\nStarting self-improvement cycle...")
        
        # 1. Identify improvement areas
        misclassified, uncertain, confusion_pair = self.identify_improvement_areas()
        
        # 2. Perform active learning sample selection
        self.active_learning_sample_selection()
        
        # 3. Update model architecture or hyperparameters if needed
        self.evolve_model()
        
        # 4. Increment version and train the new model
        self.version += 1
        self.train()
        
        # 5. Update uncertainty threshold based on model performance
        # Lower threshold as model improves (becomes more certain)
        if len(self.performance_history) >= 2:
            current_acc = self.performance_history[-1]['test_accuracy']
            previous_acc = self.performance_history[-2]['test_accuracy']
            if current_acc > previous_acc:
                self.uncertainty_threshold = max(0.1, self.uncertainty_threshold * 0.9)
                print(f"Model improved, reducing uncertainty threshold to {self.uncertainty_threshold:.3f}")
    
    def evolve_model(self):
        """Evolve model architecture based on performance"""
        # This is where we would implement architecture search or hyperparameter tuning
        # For simplicity, we'll just modify the dropout rates based on overfitting indication
        
        if len(self.history.history['val_loss']) > 3:
            train_loss_end = self.history.history['loss'][-1]
            val_loss_end = self.history.history['val_loss'][-1]
            
            # Check for overfitting
            if train_loss_end * 1.2 < val_loss_end:
                print("Detected potential overfitting, increasing regularization...")
                # We'll rebuild the model with increased dropout
                # Save weights before rebuilding
                weights = self.model.get_weights()
                
                # Rebuild model with more regularization
                inputs = layers.Input(shape=self.input_shape)
                x = layers.Reshape((28, 28, 1))(inputs)
                
                # Increased dropout in CNN layers
                x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
                x = layers.MaxPooling2D((2, 2))(x)
                x = layers.Dropout(0.35)(x)  # Increased from 0.25
                
                x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
                x = layers.MaxPooling2D((2, 2))(x)
                x = layers.Dropout(0.35)(x)  # Increased from 0.25
                
                new_shape = ((self.input_shape[0] // 4), (self.input_shape[1] // 4) * 64)
                x = layers.Reshape(new_shape)(x)
                
                x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
                x = layers.Dropout(0.35)(x)  # Increased from 0.25
                x = layers.Bidirectional(layers.LSTM(64))(x)
                x = layers.Dropout(0.35)(x)  # Increased from 0.25
                
                x = layers.Dense(128, activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.6)(x)  # Increased from 0.5
                outputs = layers.Dense(10, activation='softmax')(x)
                
                # Create new model
                self.model = models.Model(inputs=inputs, outputs=outputs)
                
                # Compile with fixed learning rate
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
                )
                
                # Try to restore weights (may not be compatible if architecture changed significantly)
                try:
                    self.model.set_weights(weights)
                    print("Successfully transferred weights to new model")
                except:
                    print("Could not transfer weights, starting with fresh weights")
    
    def visualize_learning_progress(self):
        """Visualize the model's learning progress over versions"""
        if len(self.performance_history) < 2:
            print("Not enough history to visualize progress")
            return
        
        # Extract metrics for plotting
        versions = [p['version'] for p in self.performance_history]
        accuracy = [p['test_accuracy'] for p in self.performance_history]
        precision = [p['test_precision'] for p in self.performance_history]
        recall = [p['test_recall'] for p in self.performance_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(versions, accuracy, 'o-', label='Accuracy')
        plt.plot(versions, precision, 's-', label='Precision')
        plt.plot(versions, recall, '^-', label='Recall')
        plt.xlabel('Model Version')
        plt.ylabel('Score')
        plt.title('Model Performance Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('learning_progress.png')
        plt.show()
        
        # Plot learning pool growth
        if hasattr(self, 'model_improvements') and len(self.model_improvements) > 0:
            misclassified = [area['misclassified_count'] for area in self.model_improvements]
            uncertain = [area['uncertain_count'] for area in self.model_improvements]
            
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(misclassified)), misclassified, 'r-', label='Misclassified Samples')
            plt.plot(range(len(uncertain)), uncertain, 'b-', label='Uncertain Predictions')
            plt.xlabel('Improvement Cycle')
            plt.ylabel('Count')
            plt.title('Model Error Analysis Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('error_analysis.png')
            plt.show()
    
    def save_performance_history(self):
        """Save performance history to file"""
        history_file = 'model_performance_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
        print(f"Saved performance history to {history_file}")
    
    def save_model(self, model_name=None):
        """Save the current model and its metadata"""
        if model_name is None:
            model_name = f"mnist_crnn_v{self.version}"
        
        # Save model
        self.model.save(f"{model_name}_model.h5")
        
        # Save learning pool
        if len(self.learning_pool) > 0:
            with open(f"{model_name}_learning_pool.pkl", 'wb') as f:
                pickle.dump({
                    'samples': self.learning_pool,
                    'labels': self.learning_pool_labels
                }, f)
        
        # Save metadata
        metadata = {
            'version': self.version,
            'uncertainty_threshold': self.uncertainty_threshold,
            'performance_history': self.performance_history,
            'learning_pool_size': len(self.learning_pool),
            'timestamp': time.time()
        }
        with open(f"{model_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved as {model_name}")
    
    def load_model(self, model_path, metadata_path=None, learning_pool_path=None):
        """Load a previously saved model and its metadata"""
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")
        
        # Load metadata if provided
        if metadata_path:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.version = metadata['version']
                self.uncertainty_threshold = metadata['uncertainty_threshold']
                self.performance_history = metadata['performance_history']
            print(f"Loaded metadata from {metadata_path}")
        
        # Load learning pool if provided
        if learning_pool_path:
            with open(learning_pool_path, 'rb') as f:
                pool_data = pickle.load(f)
                self.learning_pool = pool_data['samples']
                self.learning_pool_labels = pool_data['labels']
            print(f"Loaded learning pool with {len(self.learning_pool)} samples")

# Main execution function
def main():
    # Create results directory
    os.makedirs("mnist_self_improvement_results", exist_ok=True)
    os.chdir("mnist_self_improvement_results")
    
    start_time = time.time()
    
    # Initialize self-improving model
    print("Initializing self-improving CRNN model...")
    model = SelfImprovingCRNN(initial_training=True)
    
    # Run multiple improvement cycles
    improvement_cycles = 3
    for cycle in range(improvement_cycles):
        print(f"\n===== IMPROVEMENT CYCLE {cycle+1}/{improvement_cycles} =====")
        model.self_improve()
    
    # Visualize learning progress
    model.visualize_learning_progress()
    
    # Final evaluation and model saving
    model.save_model("final_self_improved_model")
    
    # Plot confusion matrix for final model
    y_pred = model.model.predict(model.x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(model.y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Final Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('final_confusion_matrix.png')
    plt.show()
    
    # Print classification report
    print("\nFinal Classification Report:")
    print(classification_report(model.y_test, y_pred_classes))
    
    # Single prediction demo
    index = np.random.randint(0, len(model.x_test))
    image = model.x_test[index]
    true_label = model.y_test[index]
    
    # Show the image
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(f"Sample Test Image (True Label: {true_label})")
    plt.colorbar()
    plt.grid(False)
    plt.savefig('final_test_sample.png')
    plt.show()
    
    # Prediction
    prediction = model.model.predict(np.expand_dims(image, axis=0))[0]
    predicted_label = np.argmax(prediction)
    
    # Print prediction results
    print(f"\nFinal Prediction for test sample:")
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Confidence: {prediction[predicted_label]:.4f}")
    
    # Plot confidence
    plt.figure(figsize=(10, 4))
    plt.bar(range(10), prediction)
    plt.xticks(range(10))
    plt.xlabel('Digit Class')
    plt.ylabel('Prediction Confidence')
    plt.title('Final Model Prediction Confidence')
    plt.grid(True, alpha=0.3)
    plt.savefig('final_prediction_confidence.png')
    plt.show()
    
    # Print execution time
    execution_time = time.time() - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()