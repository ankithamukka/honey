#4 Experiment with different regularization techniques on a neural network

import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# Define paths to your training, validation, and testing data
output_folder_1 = "/home/ankitha-mukka/Train"
output_folder_2 = "/home/ankitha-mukka/Valid"
output_folder_3 = "/home/ankitha-mukka/Test"

# Initialize parameters
learning_rate = 0.001
num_epochs = 20
input_shape = (28, 28, 3)  # Assuming images are resized to 28x28 and have 3 channels

# Function to load data from folders
def load_data(folder_path):
    X = []
    y = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):  # Assuming images are in JPG or PNG format
                img_path = os.path.join(root, file)
                label = 1 if "positive" in root else 0  # Example: folder structure decides the label
                img = cv2.imread(img_path)
                img = cv2.resize(img, (input_shape[0], input_shape[1]))  # Resize image to match input_shape
                X.append(img)
                y.append(label)
    return np.array(X), np.array(y)

# Load data
X_train, y_train = load_data(output_folder_1)
X_val, y_val = load_data(output_folder_2)
X_test, y_test = load_data(output_folder_3)

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# Define the neural network model with L1 regularization and dropout
model_l1_dropout = Sequential([
    Flatten(input_shape=input_shape),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001)),
    Dropout(0.5),  # Dropout layer to prevent overfitting
    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001)),
    Dropout(0.5),  # Dropout layer to prevent overfitting
    Dense(1, activation='sigmoid')
])

# Compile the model with L1 regularization and dropout
model_l1_dropout.compile(optimizer=Adam(learning_rate=learning_rate),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])

# Train the model with L1 regularization and dropout
history_l1_dropout = model_l1_dropout.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val))

# Evaluate on training data with L1 regularization and dropout
train_loss_l1_dropout, train_accuracy_l1_dropout = model_l1_dropout.evaluate(X_train, y_train)
print("\nResults with L1 Regularization and Dropout:")
print(f"Training Accuracy: {train_accuracy_l1_dropout:.4f}")

# Evaluate on validation data with L1 regularization and dropout
val_loss_l1_dropout, val_accuracy_l1_dropout = model_l1_dropout.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy_l1_dropout:.4f}")

# Evaluate on test data with L1 regularization and dropout
test_loss_l1_dropout, test_accuracy_l1_dropout = model_l1_dropout.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy_l1_dropout:.4f}")

# Define the neural network model with L2 regularization and dropout
model_l2_dropout = Sequential([
    Flatten(input_shape=input_shape),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),  # Dropout layer to prevent overfitting
    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),  # Dropout layer to prevent overfitting
    Dense(1, activation='sigmoid')
])

# Compile the model with L2 regularization and dropout
model_l2_dropout.compile(optimizer=Adam(learning_rate=learning_rate),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])

# Train the model with L2 regularization and dropout
history_l2_dropout = model_l2_dropout.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val))

# Evaluate on training data with L2 regularization and dropout
train_loss_l2_dropout, train_accuracy_l2_dropout = model_l2_dropout.evaluate(X_train, y_train)
print("\nResults with L2 Regularization and Dropout:")
print(f"Training Accuracy: {train_accuracy_l2_dropout:.4f}")

# Evaluate on validation data with L2 regularization and dropout
val_loss_l2_dropout, val_accuracy_l2_dropout = model_l2_dropout.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy_l2_dropout:.4f}")

# Evaluate on test data with L2 regularization and dropout
test_loss_l2_dropout, test_accuracy_l2_dropout = model_l2_dropout.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy_l2_dropout:.4f}")