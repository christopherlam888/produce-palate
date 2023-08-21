# This is the main code for Produce Palate
import os
import cv2
import numpy as np
import pickle
import json
import random
from typing import Final
from scripts.utils import (
    IMAGE_PATH,
    get_dependencies,
    get_predicted_label,
)


# Constants
NUM_OPTIONS: Final = 4


def main():
    label_dict, best_model = get_dependencies()

    # Get images and the number of images
    files = os.listdir(IMAGE_PATH)
    num_images = len(files)

    # Count the score
    score = 0

    # Process and predict labels for each image
    for filename in files:
        predicted_label = get_predicted_label(filename, best_model, label_dict)

        # Generate four random labels excluding the predicted label
        random_labels = random.sample(
            [label for label in label_dict.values() if label != predicted_label], 
            NUM_OPTIONS
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


if __name__ == "__main__":
    main()
