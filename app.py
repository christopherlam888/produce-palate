import os
from typing import (
    Final
)
from flask import (
    Flask,
    render_template,
    session,
    request,
    redirect,
    url_for,
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
    @effects: Modifies the `session`.
    """
    # The user accessed the play page directly
    if "initialized" not in session:
        return

    # The session has already been initialized
    if session["initialized"]:
        return

    session["score"] = 0
    session["count"] = 0

    session["files"] = os.listdir(IMAGE_PATH)
    random.shuffle(session["files"])

    # Resize all the images to 300x300 if they aren't already
    for filename in session["files"]:
        img = cv2.imread(f"{IMAGE_PATH}/{filename}", cv2.IMREAD_UNCHANGED)
        height, width, _ = img.shape
        if height == width == WIDTH_HEIGHT:
            continue
        img = cv2.resize(img, (WIDTH_HEIGHT, WIDTH_HEIGHT))
        cv2.imwrite(f"{IMAGE_PATH}/{filename}", img)
    
    session["initialized"] = True


def get_file() -> str:
    """
    Gets the next file in the list of files.
    """
    files = session["files"]
    return files[session["count"] % len(files)]


@app.route("/")
def index():
    session["initialized"] = False
    return render_template("index.html")


@app.route("/play", methods=["GET", "POST"])
def play():
    # User accessed the page directly without going through the main page
    if "initialized" not in session:
        return redirect(url_for("index"))
    if request.method == "POST":
        option = request.form["option"]
        predicted_label = session["predicted_label"]

        if option == predicted_label:
            session["score"] += 1

        session["count"] += 1

        session["current_score"] = session["score"]
        session["current_count"] = session["count"]

        return render_template(
            "result.html",
            predicted_label=predicted_label,
            option=option,
            score=session["score"],
            count=session["count"],
            num_files=len(session["files"]),
        )
    elif "count" in session and session["count"] >= len(session["files"]):
        return redirect(url_for("done"))

    # User navigated to the page so we want to display the image and options
    else:
        init_session()
        
        label_dict, best_model = get_dependencies()
        filename = get_file()
        predicted_label = get_predicted_label(filename, best_model, label_dict)
        session["predicted_label"] = predicted_label

        return render_template(
            "options.html",
            options=generate_options(predicted_label, label_dict),
            image_path=f"{IMAGE_PATH}/{filename}"
        )


@app.route("/done")
def done():
    # User accessed the page directly without playing the game
    if "initialized" not in session or "current_score" not in session \
            or "current_count" not in session:
        return redirect(url_for("index"))
    
    score = session["current_score"]
    count = session["current_count"]
    session["initialized"] = False
    init_session()

    return render_template(
        "done.html",
        score=score,
        count=count,
    )


if __name__ == "__main__":
    app.run(debug=True, port=PORT_NUMBER)
