#2 Build and train a simple neural network using a framework like TensorFlow or PyTorch

import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from sklearn.model_selection import train_test_split

# Define paths to your training, validation, and testing data
output_folder_1 = "/home/ankitha-mukka/Train"
output_folder_2 = "/home/ankitha-mukka/Valid"
output_folder_3 = "/home/ankitha-mukka/Test"

# Initialize parameters
learning_rate = 0.01
num_epochs = 10  # Increase the number of epochs for better training
input_shape = (28, 28, 3)

# Function to load data from folders
def load_data(folder_path):
    X = []
    y = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(root, file)
                label = 1 if "positive" in root else 0
                img = cv2.imread(img_path)
                img = cv2.resize(img, (input_shape[0], input_shape[1]))
                X.append(img)
                y.append(label)
    return np.array(X), np.array(y)

X_train, y_train = load_data(output_folder_1)
X_val, y_val = load_data(output_folder_2)
X_test, y_test = load_data(output_folder_3)

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# Define the neural network model
model = Sequential([
    Flatten(input_shape=input_shape),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val))

# Extract training accuracy from history
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

# Print training accuracy for each epoch
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}, Training Accuracy: {training_accuracy[epoch]:.2f}")

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Print final validation accuracy
print(f"Validation Accuracy: {validation_accuracy[-1]:.2f}")
