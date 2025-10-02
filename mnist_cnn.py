import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess MNIST dataset
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Normalize pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape data to include channel dimension
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Convert labels to categorical one-hot encoding
y_train_categorical = keras.utils.to_categorical(y_train, 10)
y_test_categorical = keras.utils.to_categorical(y_test, 10)

print(f"Reshaped training data: {X_train.shape}")
print(f"Reshaped testing data: {X_test.shape}")

# Build CNN model
def create_cnn_model():
    model = keras.Sequential([
        # First convolutional layer
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Create and compile model
model = create_cnn_model()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n=== Model Architecture ===")
model.summary()

# Train the model
print("\nTraining CNN model...")
history = model.fit(
    X_train, y_train_categorical,
    batch_size=128,
    epochs=10,
    validation_data=(X_test, y_test_categorical),
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)
print(f"\n=== Final Test Accuracy: {test_accuracy:.4f} ===")

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Visualize predictions on 5 sample images
print("\nVisualizing predictions on 5 sample images...")
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for i in range(5):
    # Randomly select samples
    sample_idx = np.random.randint(0, len(X_test))
    
    # Get prediction and actual label
    predicted_label = y_pred_classes[sample_idx]
    actual_label = y_test[sample_idx]
    
    # Plot the image
    axes[i].imshow(X_test[sample_idx].reshape(28, 28), cmap='gray')
    axes[i].set_title(f'Pred: {predicted_label}, Actual: {actual_label}')
    axes[i].axis('off')
    
    # Color code based on correctness
    color = 'green' if predicted_label == actual_label else 'red'
    axes[i].title.set_color(color)

plt.tight_layout()
plt.show()

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
