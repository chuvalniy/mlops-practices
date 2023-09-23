from flask import Flask, request, jsonify

from models.model import generate_and_decode_text

app = Flask(__name__)


@app.post("/generate")
def generate():
    data = request.json
    try:
        sample = data['text_length']
    except KeyError:
        return jsonify({"error": "No text length sent. You have to use 'text_length' as key to your request."})

    text = generate_and_decode_text(sample)

    return text


if __name__ == "__main__":
    app.run(debug=True)
