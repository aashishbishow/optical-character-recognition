import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to load and preprocess data
def load_and_preprocess_data():
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values (0 to 1)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert labels to one-hot encoding
    y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)

    print(f"Train set: {x_train.shape[0]} samples")
    print(f"Test set: {x_test.shape[0]} samples")

    return (x_train, y_train, y_train_onehot), (x_test, y_test, y_test_onehot)

# Build CRNN model (Convolutional Recurrent Neural Network)
def build_crnn_model(input_shape=(28, 28)):
    # Input layer
    inputs = layers.Input(shape=input_shape)

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
    new_shape = ((input_shape[0] // 4), (input_shape[1] // 4) * 64)
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
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile model
    initial_learning_rate = 0.001

    # Define optimizer with a fixed learning rate (will be adjusted dynamically)
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model


# Data augmentation
def create_data_augmentation():
    return tf.keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomZoom(0.1),
    ])

# Plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Visualize predictions
def visualize_predictions(model, x_test, y_test, num_samples=10):
    # Get random samples
    indices = np.random.choice(range(len(x_test)), num_samples, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        # Get the image and reshape for display
        img = x_test[idx]

        # Get prediction
        pred = model.predict(np.expand_dims(x_test[idx], axis=0))[0]
        pred_label = np.argmax(pred)
        true_label = y_test[idx]

        # Display image
        axes[i].imshow(img, cmap='gray')
        color = 'green' if pred_label == true_label else 'red'
        title = f"True: {true_label}\nPred: {pred_label}\nConf: {pred[pred_label]:.2f}"
        axes[i].set_title(title, color=color)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('prediction_samples.png')
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(model, x_test, y_test):
    # Get predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))

# Visualize CRNN attention
def visualize_crnn_attention(model, x_test, y_test, num_samples=5):
    # Create a model that outputs the activations of the last conv layer
    layer_outputs = [layer.output for layer in model.layers if 'conv2d' in layer.name]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

    # Get random samples
    indices = np.random.choice(range(len(x_test)), num_samples, replace=False)

    for idx in indices:
        img = x_test[idx]
        true_label = y_test[idx]

        # Get prediction
        pred = model.predict(np.expand_dims(img, axis=0))[0]
        pred_label = np.argmax(pred)

        # Get activations
        activations = activation_model.predict(np.expand_dims(img, axis=0))
        last_conv_activation = activations[-1][0]

        # Visualize activation heatmap
        plt.figure(figsize=(12, 4))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Original\nTrue: {true_label}")
        plt.axis('off')

        # Prediction
        plt.subplot(1, 3, 2)
        plt.bar(range(10), pred)
        plt.xticks(range(10))
        plt.xlabel('Digit Class')
        plt.ylabel('Confidence')
        plt.title(f"Prediction: {pred_label}\nConfidence: {pred[pred_label]:.2f}")

        # Activation heatmap
        plt.subplot(1, 3, 3)
        # Average across channels to get activation intensity
        activation_map = np.mean(last_conv_activation, axis=2)
        plt.imshow(activation_map, cmap='jet')
        plt.title('Activation Heatmap')
        plt.colorbar()
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'attention_sample_{idx}.png')
        plt.show()

# Export model artifacts
def export_model(model, model_name="mnist_crnn"):
    # Save model architecture as JSON
    model_json = model.to_json()
    with open(f"{model_name}.json", "w") as json_file:
        json_file.write(model_json)

    # Save model weights
    model.save_weights(f"{model_name}_weights.h5")

    # Save entire model in SavedModel format
    model.save(f"{model_name}_saved", save_format="tf")

    # Save as TFLite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(f"{model_name}.tflite", "wb") as tflite_file:
        tflite_file.write(tflite_model)

    print(f"Model exported as: {model_name}.json, {model_name}_weights.h5, {model_name}_saved/, {model_name}.tflite")

# Main execution function
def main():
    start_time = time.time()

    # Load and preprocess data
    (x_train, y_train, y_train_onehot), (x_test, y_test, y_test_onehot) = load_and_preprocess_data()

    # Create CRNN model
    model = build_crnn_model()
    print(model.summary())

    # Data augmentation
    data_augmentation = create_data_augmentation()

    # Apply data augmentation to training data
    def augment_data(x, y):
        x_aug = np.zeros_like(x)
        for i in range(len(x)):
            x_aug[i] = data_augmentation(x[i].reshape(28, 28, 1)).numpy().reshape(28, 28)
        return x_aug, y

    # Augment a portion of training data
    aug_idx = np.random.choice(range(len(x_train)), size=int(len(x_train)*0.5), replace=False)
    x_train_aug, y_train_onehot_aug = augment_data(x_train[aug_idx], y_train_onehot[aug_idx])

    # Combine original and augmented data
    x_train_combined = np.concatenate([x_train, x_train_aug])
    y_train_onehot_combined = np.concatenate([y_train_onehot, y_train_onehot_aug])

    # Define callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        'best_mnist_crnn_model.h5',
        save_best_only=True,
        monitor='val_accuracy'
    )
    early_stopping_cb = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss",   # Reduce LR when validation loss stops improving
        factor=0.5,           # Reduce LR by 50% (multiplication factor)
        patience=3,           # Wait for 3 epochs before reducing LR
        verbose=1,            # Print LR changes
        min_lr=1e-6           # Minimum possible LR
    )

    # Train the model with validation split
    print("\nTraining CRNN model...")
    history = model.fit(
        x_train_combined, y_train_onehot_combined,
        epochs=25,
        batch_size=128,
        validation_split=0.1,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
        verbose=1,
        shuffle=True
    )

    # Evaluate on test set
    print("\nEvaluating CRNN model...")
    test_results = model.evaluate(x_test, y_test_onehot, verbose=1)
    print(f"Test loss: {test_results[0]:.4f}")
    print(f"Test accuracy: {test_results[1]:.4f}")
    print(f"Test precision: {test_results[2]:.4f}")
    print(f"Test recall: {test_results[3]:.4f}")

    # Plot training history
    plot_training_history(history)

    # Visualize predictions
    visualize_predictions(model, x_test, y_test)

    # Plot confusion matrix
    plot_confusion_matrix(model, x_test, y_test)

    # Visualize CRNN attention
    visualize_crnn_attention(model, x_test, y_test)

    # Export model
    export_model(model)

    # Make a single prediction demonstration
    index = np.random.randint(0, len(x_test))
    image = x_test[index]
    true_label = y_test[index]

    # Show the image
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(f"Sample Test Image (True Label: {true_label})")
    plt.colorbar()
    plt.grid(False)
    plt.savefig('test_sample.png')
    plt.show()

    # Prediction with confidence scores
    prediction = model.predict(np.expand_dims(image, axis=0))[0]
    predicted_label = np.argmax(prediction)

    # Print prediction results
    print(f"\nPrediction for test sample:")
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Confidence: {prediction[predicted_label]:.4f}")

    # Plot confidence for all classes
    plt.figure(figsize=(10, 4))
    plt.bar(range(10), prediction)
    plt.xticks(range(10))
    plt.xlabel('Digit Class')
    plt.ylabel('Prediction Confidence')
    plt.title('Prediction Confidence Across All Classes')
    plt.grid(True, alpha=0.3)
    plt.savefig('prediction_confidence.png')
    plt.show()

    # Print execution time
    execution_time = time.time() - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    # Set up directory for saving artifacts
    os.makedirs("mnist_crnn_results", exist_ok=True)
    os.chdir("mnist_crnn_results")

    # Run main function
    main()