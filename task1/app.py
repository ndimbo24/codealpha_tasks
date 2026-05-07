from flask import Flask, render_template, request, jsonify
from deep_translator import GoogleTranslator

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()

    text = data["text"]
    source = data["source"]
    target = data["target"]

    translated = GoogleTranslator(
        source=source,
        target=target
    ).translate(text)

    return jsonify({
        "translated_text": translated
    })

if __name__ == "__main__":
    app.run(debug=True)