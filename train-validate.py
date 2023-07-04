# Use this to train and validate the model

import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import json

# Define folder paths
train_folder = "data/train"
validate_folder = "data/validate"

# Load training images and labels
train_images = []
train_labels = []
label_dict = {}
label_counter = 0
for label_folder in os.listdir(train_folder):
    label_folder_path = os.path.join(train_folder, label_folder)
    label_dict[label_counter] = label_folder
    for filename in os.listdir(label_folder_path):
        img_path = os.path.join(label_folder_path, filename)
        img = cv2.imread(img_path)
        train_images.append(img)
        train_labels.append(label_counter)
    label_counter += 1

# Preprocess training images
train_data = []
for img in train_images:
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
    train_data.append(img)

train_data = np.array(train_data)

# Train KNN model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(train_data, train_labels)

# Define the parameter grid for tuning
# param_grid = {
# "n_estimators": [100, 500, 1000],
# "max_depth": [None, 10, 20],
# "min_samples_split": [2, 5, 10],
# "min_samples_leaf": [1, 2, 4],
# "max_features": ["sqrt", "log2"],
# }

# Create the random forest classifier
# rf_model = RandomForestClassifier()

# Perform grid search to find the best hyperparameters
# grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)
# grid_search.fit(train_data, train_labels)

# Get the best model and its parameters
# best_rf_model = grid_search.best_estimator_
# best_rf_params = grid_search.best_params_

# Print the best parameters
# print("Best Parameters:")
# for param, value in best_rf_params.items():
# print(f"{param}: {value}")

# Train the best model on the entire training data
# rf_model = RandomForestClassifier(**best_rf_params)
# rf_model.fit(train_data, train_labels)

# Train Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    max_features="log2",
    min_samples_leaf=1,
    min_samples_split=5,
)
rf_model.fit(train_data, train_labels)

# Create an SVM model
# svm_model = SVC()

# Define the parameter grid for grid search
# param_grid = {
# "C": [0.1, 1, 10],  # Penalty parameter C
# "kernel": ["linear", "rbf"],  # Kernel type
# "gamma": ["scale", "auto"],  # Kernel coefficient for 'rbf' kernel
# }

# Perform grid search with cross-validation
# grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5)
# grid_search.fit(train_data, train_labels)

# Get the best parameter values and the corresponding accuracy
# best_params = grid_search.best_params_
# best_accuracy = grid_search.best_score_

# Print the best parameters
# print("Best Parameters:")
# for param, value in best_params.items():
# print(f"{param}: {value}")

# Train the SVM model with the best parameters on the entire training data
# svm_model = SVC(**best_params)
# svm_model.fit(train_data, train_labels)

# Train SVM model
svm_model = SVC(kernel="rbf", C=10, gamma="scale")
svm_model.fit(train_data, train_labels)

# Load validate images and labels
validate_images = []
validate_labels = []
for label_folder in os.listdir(validate_folder):
    label_folder_path = os.path.join(validate_folder, label_folder)
    label_idx = next(key for key, value in label_dict.items() if value == label_folder)
    for filename in os.listdir(label_folder_path):
        img_path = os.path.join(label_folder_path, filename)
        img = cv2.imread(img_path)
        validate_images.append(img)
        validate_labels.append(label_idx)

# Preprocess validate images
validate_data = []
for img in validate_images:
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
    validate_data.append(img)

validate_data = np.array(validate_data)

# Use the trained models for prediction
knn_predictions = knn_model.predict(validate_data)
rf_predictions = rf_model.predict(validate_data)
svm_predictions = svm_model.predict(validate_data)

# Calculate accuracy
knn_accuracy = np.mean(knn_predictions == validate_labels) * 100
rf_accuracy = np.mean(rf_predictions == validate_labels) * 100
svm_accuracy = np.mean(svm_predictions == validate_labels) * 100

print("KNN Accuracy: {:.2f}%".format(knn_accuracy))
print("Random Forest Accuracy: {:.2f}%".format(rf_accuracy))
print("SVM Accuracy: {:.2f}%".format(svm_accuracy))

# Print incorrect labels vs correct labels
print("Incorrect Labels vs Correct Labels")
for i in range(len(validate_labels)):
    if rf_predictions[i] != validate_labels[i]:
        print(
            f"Random Forest: {i} {label_dict[rf_predictions[i]]} vs {label_dict[validate_labels[i]]}"
        )
    # if svm_predictions[i] != validate_labels[i]:
    # print(
    # f"SVM: {i} {label_dict[svm_predictions[i]]} vs {label_dict[validate_labels[i]]}"
    # )

# Save the model with the highest accuracy
if knn_accuracy >= max(rf_accuracy, svm_accuracy):
    with open("best_model.pkl", "wb") as f:
        pickle.dump(knn_model, f)
    print("KNN model saved as the best model.")
elif rf_accuracy >= max(knn_accuracy, svm_accuracy):
    with open("best_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    print("Random Forest model saved as the best model.")
else:
    with open("best_model.pkl", "wb") as f:
        pickle.dump(svm_model, f)
    print("SVM model saved as the best model.")

# Save the dict
with open("label_dict.json", "w") as f:
    json.dump(label_dict, f)
