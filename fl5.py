# 5 Compare performance with various optimization algorithms

import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.metrics import accuracy_score

# Define paths to your training, validation, and testing data
output_folder_1 = "/home/ankitha-mukka/Train"
output_folder_2 = "/home/ankitha-mukka/Valid"
output_folder_3 = "/home/ankitha-mukka/Test"

# Initialize parameters
learning_rate = 0.001
num_epochs = 20
input_shape = (28, 28, 3)  # Assuming images are resized to 28x28 and have 3 channels

# Load data
X_train = []
y_train = []
for root, _, files in os.walk(output_folder_1):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):  # Assuming images are in JPG or PNG format
            img_path = os.path.join(root, file)
            label = 1 if "positive" in root else 0  # Example: folder structure decides the label
            img = cv2.imread(img_path)
            img = cv2.resize(img, (input_shape[0], input_shape[1]))  # Resize image to match input_shape
            X_train.append(img)
            y_train.append(label)
X_train = np.array(X_train) / 255.0
y_train = np.array(y_train)

X_val = []
y_val = []
for root, _, files in os.walk(output_folder_2):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):  # Assuming images are in JPG or PNG format
            img_path = os.path.join(root, file)
            label = 1 if "positive" in root else 0  # Example: folder structure decides the label
            img = cv2.imread(img_path)
            img = cv2.resize(img, (input_shape[0], input_shape[1]))  # Resize image to match input_shape
            X_val.append(img)
            y_val.append(label)
X_val = np.array(X_val) / 255.0
y_val = np.array(y_val)

X_test = []
y_test = []
for root, _, files in os.walk(output_folder_3):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):  # Assuming images are in JPG or PNG format
            img_path = os.path.join(root, file)
            label = 1 if "positive" in root else 0  # Example: folder structure decides the label
            img = cv2.imread(img_path)
            img = cv2.resize(img, (input_shape[0], input_shape[1]))  # Resize image to match input_shape
            X_test.append(img)
            y_test.append(label)
X_test = np.array(X_test) / 255.0
y_test = np.array(y_test)

# Define the neural network model
models = {
    'Adam': Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Dropout layer to prevent overfitting
        Dense(64, activation='relu'),
        Dropout(0.5),  # Dropout layer to prevent overfitting
        Dense(1, activation='sigmoid')
    ]),
    'SGD': Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Dropout layer to prevent overfitting
        Dense(64, activation='relu'),
        Dropout(0.5),  # Dropout layer to prevent overfitting
        Dense(1, activation='sigmoid')
    ]),
    'RMSprop': Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Dropout layer to prevent overfitting
        Dense(64, activation='relu'),
        Dropout(0.5),  # Dropout layer to prevent overfitting
        Dense(1, activation='sigmoid')
    ])
}

# Train and evaluate models with different optimizers
results = {}

for optimizer_name, model in models.items():
    print(f"\nTraining with {optimizer_name} optimizer:")
    
    if optimizer_name == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val), verbose=0)
    
    # Evaluate on training data
    train_loss, train_accuracy = model.evaluate(X_train, y_train)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    
    # Evaluate on validation data
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Evaluate on test data
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    results[optimizer_name] = {
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy
    }

# Print results summary
print("\nResults Summary:")
for optimizer_name, result in results.items():
    print(f"Optimizer: {optimizer_name}")
    print(f"Training Accuracy: {result['train_accuracy']:.4f}")
    print(f"Validation Accuracy: {result['val_accuracy']:.4f}")
    print(f"Test Accuracy: {result['test_accuracy']:.4f}")
    print()
