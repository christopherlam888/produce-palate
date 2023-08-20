import flask
from typing import (
    Final
)

# Constants
PORT_NUMBER: Final = 5000

app = flask.Flask(__name__)


@app.route("/")
def main():
    return flask.render_template(
        "index.html",
        options=["apple", "banana", "orange", "pear"],
        image_path="static/banana.jpeg"
    )


if __name__ == "__main__":
    app.run(debug=True, port=PORT_NUMBER)
