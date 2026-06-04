import os
import joblib
import pandas as pd
from flask import Blueprint, request, jsonify

bp = Blueprint("blood", __name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "blood_model.pkl")
_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


# Per-class advice — covers all classes from the new blood_report_dataset
ADVICE = {
    "Normal":
        {"advice": "Your CBC values look within healthy reference ranges. Maintain regular check-ups.",
         "diet": ["Balanced diet", "Stay hydrated", "Adequate protein and iron sources"]},

    "Iron Deficiency Anemia":
        {"advice": "Microcytic, hypochromic picture suggesting iron deficiency. Confirm with iron studies (serum ferritin, TIBC).",
         "diet": ["Red meat, liver, lentils, spinach", "Vitamin C with meals for absorption", "Avoid tea/coffee around meals", "Beetroot, jaggery, dates"]},

    "B12/Folate Deficiency Anemia":
        {"advice": "Macrocytic anemia suggesting B12 or folate deficiency. Common in vegans/vegetarians and older adults. Get B12 + folate levels checked.",
         "diet": ["Dairy, eggs, fortified cereals", "Leafy greens for folate", "B12 supplementation if vegan", "Animal liver if non-vegetarian"]},

    "Aplastic Anemia":
        {"advice": "Pancytopenia (low Hb, low WBC, low platelets) — serious. Immediate haematologist referral required.",
         "diet": ["High-protein, high-calorie meals", "Avoid raw food due to infection risk", "Strict hand hygiene"]},

    "Bacterial Infection":
        {"advice": "Elevated WBC with neutrophilia suggests bacterial infection. Clinical correlation and possible antibiotic therapy.",
         "diet": ["Plenty of fluids", "Citrus and Vitamin C", "Garlic, ginger, turmeric", "Avoid sugar and alcohol"]},

    "Viral Infection":
        {"advice": "Lymphocytosis pattern suggests viral infection. Symptomatic management, rest, hydration.",
         "diet": ["Hydration with ORS / coconut water", "Light meals", "Vitamin D, C, Zinc", "Avoid heavy/oily food"]},

    "Leukopenia":
        {"advice": "Low WBC count — consult a haematologist. Reduces infection resistance.",
         "diet": ["High-protein meals", "Avoid raw / undercooked food", "Strict hand hygiene", "Vitamin B-rich foods"]},

    "Thrombocytopenia":
        {"advice": "Low platelet count. Avoid NSAIDs (aspirin/ibuprofen) and contact sports; see a doctor.",
         "diet": ["Papaya leaf extract (folk remedy)", "Leafy greens", "Citrus fruits", "Avoid alcohol"]},

    "Polycythemia":
        {"advice": "Elevated red cell mass — could indicate smoking-related, sleep-apnea-related, or primary polycythemia. See a clinician.",
         "diet": ["Increase hydration", "Reduce iron-rich foods", "Avoid smoking and dehydration", "Limit red meat"]},

    "Eosinophilia":
        {"advice": "Elevated eosinophil count — commonly seen in allergies, parasitic infections, or atopic conditions.",
         "diet": ["Identify and avoid allergens", "Anti-inflammatory diet", "Omega-3 rich foods", "Avoid food triggers (dust mites, pollen, dairy if intolerant)"]},
}


@bp.route("/features", methods=["GET"])
def features():
    b = _load()
    return jsonify({"features": b["features"], "classes": b["classes"]})


@bp.route("/predict", methods=["POST"])
def predict():
    b = _load()
    data = request.get_json(force=True)
    try:
        vec = [float(data[f]) for f in b["features"]]
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"missing/invalid feature: {e}"}), 400

    row = pd.DataFrame([vec], columns=b["features"])
    pred = b["model"].predict(row)[0]
    proba = b["model"].predict_proba(row)[0]
    classes = b["model"].classes_.tolist()
    probs = {c: round(float(p), 3) for c, p in zip(classes, proba)}
    return jsonify({
        "prediction": pred,
        "probabilities": probs,
        **ADVICE.get(pred, {"advice": "Consult a clinician.", "diet": []}),
    })
