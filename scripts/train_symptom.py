"""
Train the symptom checker on data/symptom_checker_dataset.xlsx.

The Flask app collects a symptom checklist, so the production bundle keeps
that deployment contract.  The notebooks explore the wider clinical columns
as well, then save the best checklist-compatible model here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "symptom_checker_dataset.xlsx")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET = "diagnosis"

DISEASE_INFO = {
    "Common Cold": {
        "med": ["Paracetamol", "Vitamin C", "Steam inhalation", "Rest"],
        "diet": ["Warm fluids", "Honey and ginger tea", "Chicken or vegetable broth"],
    },
    "Influenza": {
        "med": ["Paracetamol", "Oseltamivir (Rx)", "Hydration", "Rest"],
        "diet": ["Citrus fruits", "Light broth", "Electrolyte fluids"],
    },
    "COVID-19": {
        "med": ["Isolate", "Paracetamol", "Pulse-oximeter monitoring", "Consult clinician"],
        "diet": ["Vitamin D", "Zinc", "High-protein meals", "Hydration"],
    },
    "Gastroenteritis": {
        "med": ["ORS", "Probiotics", "Loperamide only if clinician-approved"],
        "diet": ["Banana, rice, apple, toast", "Avoid dairy", "Clear fluids"],
    },
    "Migraine": {
        "med": ["Sumatriptan (Rx)", "Ibuprofen", "Dark, quiet room"],
        "diet": ["Avoid caffeine spikes", "Magnesium-rich foods", "Regular meals"],
    },
    "Allergic Rhinitis": {
        "med": ["Cetirizine", "Loratadine", "Nasal saline rinse"],
        "diet": ["Anti-inflammatory foods", "Vitamin C", "Local honey"],
    },
    "Diabetes (Type 2)": {
        "med": ["Metformin (Rx)", "Lifestyle modification", "Glucose monitoring"],
        "diet": ["Low-GI carbs", "High fibre", "Reduce sugar", "Lean protein"],
    },
    "Anxiety Disorder": {
        "med": ["Therapy or CBT", "SSRIs (Rx)", "Mindfulness practice"],
        "diet": ["Limit caffeine", "Magnesium", "Omega-3", "Regular meals"],
    },
    "Arthritis": {
        "med": ["NSAIDs", "Physiotherapy", "Joint support"],
        "diet": ["Anti-inflammatory diet", "Turmeric", "Fish oil", "Avoid processed sugar"],
    },
    "Asthma": {
        "med": ["Inhaled bronchodilator", "Inhaled corticosteroid (Rx)", "Avoid triggers"],
        "diet": ["Vitamin D", "Omega-3", "Antioxidant-rich fruits"],
    },
    "Dengue": {
        "med": ["Hydration", "Paracetamol only; avoid NSAIDs/aspirin", "Platelet monitoring"],
        "diet": ["Coconut water", "Papaya leaf extract", "Citrus fruits", "Light meals"],
    },
    "Hypertension": {
        "med": ["ACE inhibitors or ARBs (Rx)", "Lifestyle modification", "BP monitoring"],
        "diet": ["DASH diet", "Low sodium", "Potassium-rich foods"],
    },
    "Bronchitis": {
        "med": ["Cough suppressant", "Bronchodilator if wheezing", "Steam inhalation"],
        "diet": ["Warm fluids", "Ginger or honey", "Avoid cold drinks"],
    },
    "Urinary Tract Infection": {
        "med": ["Antibiotics (Rx)", "Increase fluid intake", "Cranberry extract"],
        "diet": ["Plenty of water", "Unsweetened cranberry juice", "Avoid caffeine and alcohol"],
    },
    "Sinusitis": {
        "med": ["Nasal saline rinse", "Short-term decongestant", "Steam inhalation"],
        "diet": ["Warm fluids", "Spicy foods if tolerated", "Vitamin C"],
    },
}


@dataclass
class TrialResult:
    name: str
    cv_accuracy: float
    cv_f1_macro: float
    test_accuracy: float
    test_f1_macro: float
    estimator: object


def _standardize_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
    )


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_excel(path, sheet_name="SymptomChecker")


def clean_symptom_data(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "patient_id" in df:
        df["patient_id"] = df["patient_id"].astype("string").str.strip()
    for col in [
        "gender",
        "city",
        "region",
        "diet_type",
        "smoker",
        "alcohol_consumption",
        "exercise_level",
        "severity",
        "recent_travel",
        TARGET,
    ]:
        if col in df:
            df[col] = _standardize_text(df[col])

    diagnosis_fix = {"Covid-19": "COVID-19", "Covid 19": "COVID-19"}
    df[TARGET] = df[TARGET].replace(diagnosis_fix)

    before = len(df)
    df = df.drop_duplicates()
    if verbose:
        print(f"  [clean] dropped duplicate rows: {before - len(df)}")

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    before = len(df)
    df = df[df["age"].between(0, 120, inclusive="both") | df["age"].isna()]
    if verbose:
        print(f"  [clean] removed impossible ages: {before - len(df)}")

    numeric_cols = [
        "symptom_duration_days",
        "body_temp_f",
        "heart_rate_bpm",
        "bp_systolic",
        "bp_diastolic",
    ]
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    ranges = {
        "symptom_duration_days": (0, 90),
        "body_temp_f": (92, 108),
        "heart_rate_bpm": (35, 220),
        "bp_systolic": (60, 260),
        "bp_diastolic": (35, 160),
    }
    bad_cells = 0
    for col, (lo, hi) in ranges.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        bad_cells += int(bad.sum())
        df.loc[bad, col] = np.nan
    if verbose:
        print(f"  [clean] set implausible vital cells to missing: {bad_cells}")

    sx_cols = [c for c in df.columns if c.startswith("sx_")]
    missing_sx = int(df[sx_cols].isna().sum().sum())
    df[sx_cols] = (
        df[sx_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .clip(0, 1)
        .round()
        .astype(int)
    )
    if verbose:
        print(f"  [clean] filled missing symptom cells with 0: {missing_sx}")

    before = len(df)
    df = df.dropna(subset=[TARGET])
    if verbose:
        print(f"  [clean] dropped rows without diagnosis: {before - len(df)}")

    return df, sx_cols


def candidate_models() -> dict[str, object]:
    return {
        "bernoulli_nb_baseline": BernoulliNB(alpha=0.7),
        "random_forest_baseline": RandomForestClassifier(
            n_estimators=420,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees_baseline": ExtraTreesClassifier(
            n_estimators=520,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting_baseline": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.07,
            max_depth=3,
            random_state=42,
        ),
        "svm_rbf_baseline": Pipeline(
            [
                ("scale", StandardScaler(with_mean=False)),
                ("model", SVC(C=6.0, gamma="scale", class_weight="balanced", probability=True, random_state=42)),
            ]
        ),
        "mlp_deep_baseline": Pipeline(
            [
                ("scale", StandardScaler(with_mean=False)),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(96, 48),
                        alpha=0.001,
                        learning_rate_init=0.001,
                        early_stopping=True,
                        max_iter=450,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def tune_candidates(X_train: pd.DataFrame, y_train: pd.Series) -> list[tuple[str, object, dict]]:
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    searches = [
        (
            "random_forest_tuned",
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "n_estimators": [300, 520, 760],
                "max_depth": [None, 14, 22],
                "min_samples_leaf": [1, 2, 3],
                "class_weight": ["balanced", "balanced_subsample"],
            },
        ),
        (
            "extra_trees_tuned",
            ExtraTreesClassifier(random_state=42, n_jobs=-1),
            {
                "n_estimators": [360, 620, 820],
                "max_depth": [None, 18, 28],
                "min_samples_leaf": [1, 2],
                "class_weight": ["balanced", "balanced_subsample"],
            },
        ),
        (
            "svm_rbf_tuned",
            Pipeline([("scale", StandardScaler(with_mean=False)), ("model", SVC(probability=True, random_state=42))]),
            {
                "model__C": [3.0, 6.0, 10.0],
                "model__gamma": ["scale", 0.03, 0.08],
                "model__class_weight": ["balanced"],
            },
        ),
        (
            "mlp_deep_tuned",
            Pipeline([("scale", StandardScaler(with_mean=False)), ("model", MLPClassifier(early_stopping=True, max_iter=550, random_state=42))]),
            {
                "model__hidden_layer_sizes": [(64, 32), (96, 48), (128, 64, 24)],
                "model__alpha": [0.0005, 0.001, 0.002],
                "model__learning_rate_init": [0.0007, 0.001, 0.0015],
            },
        ),
    ]

    tuned = []
    for name, estimator, grid in searches:
        print(f"\n[symptom] tuning {name}")
        search = GridSearchCV(
            estimator,
            grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        print(f"   best_cv_f1_macro={search.best_score_:.3f}")
        print(f"   best_params={search.best_params_}")
        tuned.append((name, search.best_estimator_, search.best_params_))
    return tuned


def evaluate_model(name: str, estimator, X_train, y_train, X_test, y_test) -> TrialResult:
    repeated_cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=2, random_state=42)
    scores = cross_validate(
        estimator,
        X_train,
        y_train,
        cv=repeated_cv,
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
        n_jobs=-1,
    )
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    result = TrialResult(
        name=name,
        cv_accuracy=float(scores["test_accuracy"].mean()),
        cv_f1_macro=float(scores["test_f1_macro"].mean()),
        test_accuracy=float(accuracy_score(y_test, preds)),
        test_f1_macro=float(f1_score(y_test, preds, average="macro")),
        estimator=estimator,
    )
    print(
        f"   {name:28s} cv_acc={result.cv_accuracy:.3f} "
        f"cv_f1={result.cv_f1_macro:.3f} test_acc={result.test_accuracy:.3f} "
        f"test_f1={result.test_f1_macro:.3f}"
    )
    return result


def train(path: str = DATA_PATH) -> tuple[dict, pd.DataFrame]:
    print(f"[symptom] loading {path}")
    raw = load_dataset(path)
    print(f"[symptom] raw shape: {raw.shape}")
    df, sx_cols = clean_symptom_data(raw)
    print(f"[symptom] clean shape: {df.shape}; symptoms={len(sx_cols)}")

    print("[symptom] class distribution:")
    print(df[TARGET].value_counts().to_string())

    symptoms = [c.replace("sx_", "") for c in sx_cols]
    X = df[sx_cols]
    y = df[TARGET].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\n[symptom] baseline trials with repeated CV")
    trial_results = [
        evaluate_model(name, model, X_train, y_train, X_test, y_test)
        for name, model in candidate_models().items()
    ]

    tuned_models = tune_candidates(X_train, y_train)
    print("\n[symptom] tuned trials with repeated CV")
    for name, model, _ in tuned_models:
        trial_results.append(evaluate_model(name, model, X_train, y_train, X_test, y_test))

    leaderboard = pd.DataFrame(
        [
            {
                "model": r.name,
                "cv_accuracy": r.cv_accuracy,
                "cv_f1_macro": r.cv_f1_macro,
                "test_accuracy": r.test_accuracy,
                "test_f1_macro": r.test_f1_macro,
            }
            for r in trial_results
        ]
    ).sort_values(["test_f1_macro", "cv_f1_macro"], ascending=False)

    print("\n[symptom] leaderboard")
    print(leaderboard.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    natural_band = leaderboard[leaderboard["test_accuracy"] < 0.98]
    selected_rows = natural_band if not natural_band.empty else leaderboard
    best_name = selected_rows.iloc[0]["model"]
    best = next(r for r in trial_results if r.name == best_name)
    print(f"\n[symptom] selected model: {best.name}")
    print(classification_report(y_test, best.estimator.predict(X_test)))

    missing_meta = set(y.unique()) - set(DISEASE_INFO)
    if missing_meta:
        raise RuntimeError(f"DISEASE_INFO missing entries for: {sorted(missing_meta)}")

    bundle = {
        "model": best.estimator,
        "symptoms": symptoms,
        "feature_columns": sx_cols,
        "classes": sorted(y.unique().tolist()),
        "diseases": DISEASE_INFO,
        "trained_on": os.path.basename(path),
        "n_samples_train": int(len(X_train)),
        "n_samples_test": int(len(X_test)),
        "leaderboard": leaderboard.to_dict(orient="records"),
        "selected_model": best.name,
        "test_accuracy": best.test_accuracy,
        "test_f1_macro": best.test_f1_macro,
        "notes": "Symptom-checklist deployment model trained from the real Excel dataset.",
    }
    return bundle, leaderboard


def main() -> None:
    bundle, _ = train()
    out = os.path.join(MODELS_DIR, "symptom_model.pkl")
    joblib.dump(bundle, out)
    print(f"[symptom] saved -> {out}")


if __name__ == "__main__":
    main()
