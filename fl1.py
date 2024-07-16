#1Implement a perceptron from scratch  


import numpy as np
import os
import cv2  # Assuming you have OpenCV installed for image processing

# Define paths to your training, validation, and testing data
output_folder_1 = "/home/ankitha-mukka/Train"
output_folder_2 = "/home/ankitha-mukka/Valid"
output_folder_3 = "/home/ankitha-mukka/Test"

# Initialize perceptron parameters
learning_rate = 0.01
num_epochs = 1
input_shape = (28, 28, 3)  # Assuming images are resized to 28x28 and have 3 channels

# Initialize perceptron weights and bias
num_inputs = np.prod(input_shape)
weights = np.zeros(num_inputs)
bias = 0.0

# Function to load data from folders
X_train = []
y_train = []
for root, dirs, files in os.walk(output_folder_1):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):  # Assuming images are in JPG or PNG format
            img_path = os.path.join(root, file)
            label = 1 if "positive" in root else 0  # Example: folder structure decides the label
            img = cv2.imread(img_path)
            img = cv2.resize(img, (input_shape[0], input_shape[1]))  # Resize image to match input_shape
            X_train.append(img.flatten())
            y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Training the perceptron
for epoch in range(num_epochs):
    for i in range(len(X_train)):
        linear_output = np.dot(weights, X_train[i]) + bias
        prediction = 1 if linear_output >= 0 else 0
        error = y_train[i] - prediction
        weights += learning_rate * error * X_train[i]
        bias += learning_rate * error

    # Calculate training accuracy after each epoch
    correct_train = 0
    for i in range(len(X_train)):
        linear_output = np.dot(weights, X_train[i]) + bias
        prediction = 1 if linear_output >= 0 else 0
        if prediction == y_train[i]:
            correct_train += 1
    training_accuracy = correct_train / len(X_train)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Accuracy: {training_accuracy:.2f}")

# Function to test accuracy on a dataset
def test_accuracy(X, y, weights, bias):
    correct = 0
    for i in range(len(X)):
        linear_output = np.dot(weights, X[i]) + bias
        prediction = 1 if linear_output >= 0 else 0
        if prediction == y[i]:
            correct += 1
    return correct / len(X)

# Load validation data and test accuracy
X_val = []
y_val = []
for root, dirs, files in os.walk(output_folder_2):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):  # Assuming images are in JPG or PNG format
            img_path = os.path.join(root, file)
            label = 1 if "positive" in root else 0  # Example: folder structure decides the label
            img = cv2.imread(img_path)
            img = cv2.resize(img, (input_shape[0], input_shape[1]))  # Resize image to match input_shape
            X_val.append(img.flatten())
            y_val.append(label)

X_val = np.array(X_val)
y_val = np.array(y_val)

validation_accuracy = test_accuracy(X_val, y_val, weights, bias)
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Load testing data and test accuracy
X_test = []
y_test = []
for root, dirs, files in os.walk(output_folder_3):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):  # Assuming images are in JPG or PNG format
            img_path = os.path.join(root, file)
            label = 1 if "positive" in root else 0  # Example: folder structure decides the label
            img = cv2.imread(img_path)
            img = cv2.resize(img, (input_shape[0], input_shape[1]))  # Resize image to match input_shape
            X_test.append(img.flatten())
            y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

test_accuracy = test_accuracy(X_test, y_test, weights, bias)
print(f"Test Accuracy: {test_accuracy:.2f}")
