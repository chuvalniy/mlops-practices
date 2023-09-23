from flask import Flask, jsonify, request

from models.model import generate_and_decode_text

app = Flask(__name__)


@app.route("/")
def generate():
    text = generate_and_decode_text(500)

    return text


if __name__ == "__main__":
    app.run(debug=True)
