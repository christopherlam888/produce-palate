import os
from typing import (
    Final
)
from flask import (
    Flask,
    render_template,
    session,
    request,
)
from scripts.utils import (
    IMAGE_PATH,
    get_dependencies,
    get_predicted_label,
    generate_options,
)
import cv2
import random


# Constants
PORT_NUMBER: Final = 5000
WIDTH_HEIGHT: Final = 300


app = Flask(__name__)
app.secret_key = "secret-key"


def init_session():
    """
    Initializes the session variables. It sets the score to 0, the count to 0 
    and shuffles and resizing the files.
    """

    if "score" not in session:
        session["score"] = 0
    
    if "count" not in session:
        session["count"] = 0
    
    if "files" not in session:
        session["files"] = os.listdir(IMAGE_PATH)
        random.shuffle(session["files"])
        print(f'Files: {session["files"]}')

        # Resize all the images
        for filename in session["files"]:
            img = cv2.imread(f"{IMAGE_PATH}/{filename}", cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (WIDTH_HEIGHT, WIDTH_HEIGHT))
            cv2.imwrite(f"{IMAGE_PATH}/{filename}", img)


def get_next_file():
    """
    Gets the next file in the list of files.
    """
    files = session["files"]
    return files[session["count"] % len(files)]


@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        print(request.form["option"])
        option = request.form["option"]
        predicted_label = session["predicted_label"]

        if option == predicted_label:
            session["score"] += 1

        session["count"] += 1

        return render_template(
            "result.html",
            predicted_label=predicted_label,
            option=option,
            score=session["score"],
            count=session["count"],
        )
    # User navigated to the page so we want to display the image and options
    else:
        init_session()
        
        label_dict, best_model = get_dependencies()
        filename = get_next_file()
        predicted_label = get_predicted_label(filename, best_model, label_dict)
        session["predicted_label"] = predicted_label

        return render_template(
            "options.html",
            options=generate_options(predicted_label, label_dict),
            image_path=f"{IMAGE_PATH}/{filename}"
        )


if __name__ == "__main__":
    app.run(debug=True, port=PORT_NUMBER)
