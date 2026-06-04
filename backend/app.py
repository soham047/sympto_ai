"""Flask backend — serves the 5-feature healthcare API + static frontend."""
import os
import sys
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FRONTEND_DIR = os.path.join(ROOT, "frontend")
sys.path.insert(0, ROOT)

from backend.routes.blood import bp as blood_bp
from backend.routes.symptom import bp as symptom_bp
from backend.routes.image import bp as image_bp
from backend.routes.chat import bp as chat_bp
from backend.routes.lifestyle import bp as lifestyle_bp

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

app.register_blueprint(blood_bp, url_prefix="/api/blood")
app.register_blueprint(symptom_bp, url_prefix="/api/symptom")
app.register_blueprint(image_bp, url_prefix="/api/image")
app.register_blueprint(chat_bp, url_prefix="/api/chat")
app.register_blueprint(lifestyle_bp, url_prefix="/api/lifestyle")


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("AI Healthcare Platform -> http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
