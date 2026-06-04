import os
import joblib
import pandas as pd
from flask import Blueprint, request, jsonify

bp = Blueprint("symptom", __name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "symptom_model.pkl")
_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


@bp.route("/symptoms", methods=["GET"])
def list_symptoms():
    return jsonify({"symptoms": _load()["symptoms"]})


@bp.route("/predict", methods=["POST"])
def predict():
    b = _load()
    data = request.get_json(force=True)
    selected = set(data.get("symptoms", []))
    values = [1 if s in selected else 0 for s in b["symptoms"]]
    columns = b.get("feature_columns", [f"sx_{s}" for s in b["symptoms"]])
    row = pd.DataFrame([values], columns=columns)
    proba = b["model"].predict_proba(row)[0]
    classes = b["model"].classes_
    top_idx = proba.argsort()[::-1][:3]
    top = []
    for i in top_idx:
        d = classes[i]
        info = b["diseases"][d]
        top.append({
            "disease": d,
            "confidence": round(float(proba[i]), 3),
            "medicines": info["med"],
            "diet": info["diet"],
        })
    return jsonify({"predictions": top})
