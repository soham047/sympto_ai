import os
import io
import json
import base64
import numpy as np
from PIL import Image
from flask import Blueprint, request, jsonify

bp = Blueprint("image", __name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
_models = {}


def _load(kind):
    if kind in _models:
        return _models[kind]
    try:
        import tensorflow as tf
    except ImportError as e:
        raise RuntimeError(
            "TensorFlow is not installed. The image classifier requires it. "
            "On Python 3.11/3.12 run: pip install tensorflow"
        ) from e
    if kind == "xray":
        h5 = os.path.join(MODELS_DIR, "xray_cnn.h5")
        if not os.path.exists(h5):
            raise RuntimeError("xray_cnn.h5 not found — run scripts/train_xray.py first.")
        m = tf.keras.models.load_model(h5)
        labels = ["NORMAL", "PNEUMONIA"]
        gray = True
    elif kind == "skin":
        h5 = os.path.join(MODELS_DIR, "skin_cnn.h5")
        if not os.path.exists(h5):
            raise RuntimeError("skin_cnn.h5 not found — run scripts/train_skin.py first.")
        m = tf.keras.models.load_model(h5)
        with open(os.path.join(MODELS_DIR, "skin_labels.json")) as f:
            labels = json.load(f)
        gray = False
    else:
        raise ValueError(kind)
    _models[kind] = (m, labels, gray)
    return _models[kind]


def _decode_image(file_or_b64):
    if hasattr(file_or_b64, "read"):
        return Image.open(file_or_b64)
    if isinstance(file_or_b64, str):
        if file_or_b64.startswith("data:"):
            file_or_b64 = file_or_b64.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(file_or_b64)))
    raise ValueError("unsupported image input")


def _predict(kind, pil_img):
    model, labels, gray = _load(kind)
    img = pil_img.convert("L" if gray else "RGB").resize((64, 64))
    arr = np.array(img, dtype=np.float32) / 255.0
    if gray:
        arr = arr[..., None]
    arr = arr[None, ...]
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "prediction": labels[idx],
        "confidence": round(float(probs[idx]), 3),
        "probabilities": {l: round(float(p), 3) for l, p in zip(labels, probs)},
    }


@bp.route("/<kind>", methods=["POST"])
def predict(kind):
    if kind not in ("xray", "skin"):
        return jsonify({"error": "kind must be 'xray' or 'skin'"}), 400
    try:
        if "image" in request.files:
            img = _decode_image(request.files["image"])
        else:
            data = request.get_json(force=True)
            img = _decode_image(data["image_b64"])
    except Exception as e:
        return jsonify({"error": f"could not decode image: {e}"}), 400

    try:
        return jsonify(_predict(kind, img))
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
