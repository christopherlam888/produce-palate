# This is the main code for Produce Palate

import os
import cv2
import numpy as np
import pickle
import json
import random

# Load the label_dict from the JSON file
with open("label_dict.json", "r") as f:
    label_dict = json.load(f)

# Define folder path
image_folder = "images"

# Load the model
with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

# Get the number of images
num_images = len(os.listdir(image_folder))

# Count the score
score = 0

# Process and predict labels for each image
for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    img = cv2.imread(image_path)

    # Preprocess the image
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
    test_data = np.array([img])

    # Use the model for prediction
    prediction = best_model.predict(test_data)[0]
    predicted_label = label_dict[str(prediction)]

    # Generate four random labels excluding the predicted label
    random_labels = random.sample(
        [label for label in label_dict.values() if label != predicted_label], 4
    )

    # Add the predicted label to the list of random labels
    options = random_labels + [predicted_label]
    random.shuffle(options)

    # Print the options
    print(f"{filename}: Which fruit is this?")
    for i, option in enumerate(options):
        print(f"{i+1}. {option}")

    # Get the user's answer
    answer = int(input("Your answer: "))

    # Check if the answer is correct
    if options[answer - 1] == predicted_label:
        print("Correct!")
        score += 1
    else:
        print("Wrong!")

    # Print the answer
    print(f"It's a {predicted_label}!")

    # Print the score
    print(f"Score: {score}/{num_images}")

    print()

# Done!
print("Congratulations! You're learning about fruits and vegetables!")
