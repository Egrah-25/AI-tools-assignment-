# Buggy TensorFlow code (original with errors)
print("=== Debugging Challenge: Fixed Code ===")

# ORIGINAL BUGGY CODE (commented out):
"""
import tensorflow as tf

# Bug 1: Incorrect model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)  # Missing activation function
])

# Bug 2: Incorrect loss function for multi-class classification
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Wrong for 10-class problem
    metrics=['accuracy']
)

# Bug 3: Data shape mismatch
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784)  # Missing normalization
X_test = X_test.reshape(-1, 784)

# Bug 4: No one-hot encoding for labels
model.fit(X_train, y_train, epochs=5)  # y_train not one-hot encoded
"""

# FIXED CODE:
print("Fixed TensorFlow MNIST Classifier:")

import tensorflow as tf
import numpy as np

# Load and preprocess data correctly
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Fix 1: Normalize pixel values
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Fix 2: Reshape data
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Fix 3: One-hot encode labels
y_train_categorical = tf.keras.utils.to_categorical(y_train, 10)
y_test_categorical = tf.keras.utils.to_categorical(y_test, 10)

# Fix 4: Correct model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')  # Correct activation
])

# Fix 5: Correct loss function
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Correct for multi-class
    metrics=['accuracy']
)

print("Model summary:")
model.summary()

# Train the fixed model
print("\nTraining fixed model...")
history = model.fit(
    X_train, y_train_categorical,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)
print(f"\nFixed model test accuracy: {test_accuracy:.4f}")
