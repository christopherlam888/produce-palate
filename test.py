# Use this to test the model

import os
import cv2
import numpy as np
import pickle
import json

# Define folder paths
test_folder = "data/test"

# Load the label_dict from the JSON file
with open("label_dict.json", "r") as f:
    label_dict = json.load(f)

# Load test images and labels
test_images = []
test_labels = []
for label_folder in os.listdir(test_folder):
    label_folder_path = os.path.join(test_folder, label_folder)
    label_idx = next(key for key, value in label_dict.items() if value == label_folder)
    for filename in os.listdir(label_folder_path):
        img_path = os.path.join(label_folder_path, filename)
        img = cv2.imread(img_path)
        test_images.append(img)
        test_labels.append(label_idx)

# Preprocess test images
test_data = []
for img in test_images:
    height, width = img.shape[:2]
    crop_size = min(height, width)
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    end_x = start_x + crop_size
    end_y = start_y + crop_size
    img = img[start_y:end_y, start_x:end_x]
    img = cv2.resize(img, (100, 100))
    img = cv2.bilateralFilter(img, 30, 25, 25)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 9
    )
    mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.addWeighted(img, 0.8, mask, 1, 0)
    img = img.flatten()
    test_data.append(img)

test_data = np.array(test_data)
test_labels = np.array(test_labels, dtype=int)

# Load the model
with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

# Use the model for prediction
predictions = best_model.predict(test_data)

# Calculate accuracy
accuracy = np.mean(predictions == test_labels) * 100

print("Total Accuracy: {:.2f}%".format(accuracy))
