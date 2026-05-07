from flask import Flask, request, jsonify, send_from_directory
import time, traceback, os
from nlp_engine import get_matcher

app = Flask(__name__, static_folder="static")

print("\n🔄 Loading message matcher...")
matcher = get_matcher()
print("✅ Ready!\n")

@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return r

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/chat", methods=["POST","OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 204
    t = time.time()
    try:
        data = request.get_json(force=True, silent=True) or {}
        msg  = str(data.get("message","")).strip()
        if not msg:
            return jsonify({"error": "No message provided"}), 400
        result = matcher.match(msg)
        answer = matcher.get_answer(result)
        return jsonify({
            "answer":     answer,
            "category":   result["faq"]["category"] if result["faq"] else None,
            "language":   result["language"],
            "confidence": result["confidence"],
            "score":      result["score"],
            "tokens":     result["preprocessed"],
            "elapsed_ms": round((time.time()-t)*1000, 1),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 http://localhost:{port}")
    app.run(debug=True, port=port, host="0.0.0.0")
