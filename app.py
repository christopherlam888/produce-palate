from flask import (
    Flask,
    render_template,
    session,
    request,
)
from typing import (
    Final
)
from scripts.utils import (
    IMAGE_PATH,
    get_dependencies,
    get_predicted_label,
)

# Constants
PORT_NUMBER: Final = 5000

app = Flask(__name__)
app.secret_key = "secret-key"


@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        print(request.form["option"])
        option = request.form["option"]
        predicted_label = session["predicted_label"]

        if option == predicted_label:
            session["score"] += 1

        return render_template(
            "result.html",
            predicted_label=predicted_label,
            option=option,
            score=session["score"],
            count=session["count"],
        )
    # User navigated to the page so we want to display the image and options
    else:
        if("count" not in session):
            session["count"] = 0
        else:
            session["count"] += 1

        if "score" not in session:
            session["score"] = 0
        
        label_dict, best_model = get_dependencies()
        filename = "banana.jpeg"
        predicted_label = get_predicted_label(filename, best_model, label_dict)
        session["predicted_label"] = predicted_label

        return render_template(
            "options.html",
            options=["apple", "banana", "orange", "pear"],
            image_path=f"{IMAGE_PATH}/{filename}"
        )


if __name__ == "__main__":
    app.run(debug=True, port=PORT_NUMBER)
