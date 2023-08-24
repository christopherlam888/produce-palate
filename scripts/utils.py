import os
from typing import (
    Final,
    List
)
import cv2
import numpy as np
import pickle
import json
import random


# Define the paths
TEST_DIRECTORY_PATH: Final = os.path.join("data", "test")
VALIDATE_DIRECTORY_PATH: Final = os.path.join("data", "validate")
LABEL_DICT_PATH: Final = os.path.join("scripts", "label_dict.json")
MODEL_PATH: Final = os.path.join("scripts", "best_model.pkl")
IMAGE_PATH: Final = os.path.join("static", "images")


def preprocess_image(img):
    """
    Preprocesses the image using the same method as in test, train and validate
    """
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
    return img


def get_dependencies():
    """
    Gets the dependencies for the program.
    """

    # Load the label_dict from the JSON file
    with open(LABEL_DICT_PATH, "r") as f:
        label_dict = json.load(f)

    # Load the model
    with open(MODEL_PATH, "rb") as f:
        best_model = pickle.load(f)
    
    return label_dict, best_model


def get_predicted_label(filename: str, best_model, label_dict) -> str:
    """
    Gets the predicted label for the image using the best model and the label 
    dictionary.
    """
    image_path = os.path.join(IMAGE_PATH, filename)
    img = cv2.imread(image_path)

    # Preprocess the image
    img = preprocess_image(img)
    test_data = np.array([img])

    # Use the model for prediction
    prediction = best_model.predict(test_data)[0]
    return label_dict[str(prediction)]


def generate_options(predicted_label: str, label_dict: dict, *, num_options=4) -> List[str]:
    """
    Generates the options for the user to choose from.
    """
    output = random.sample(
        [label for label in label_dict.values() if label != predicted_label], 
        num_options
    )

    # Add the predicted label to the list of random labels
    output.append(predicted_label)
    random.shuffle(output)
    return output
