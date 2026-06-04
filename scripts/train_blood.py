"""
Train the blood report classifier on data/blood_report_dataset.xlsx.

The saved bundle is consumed by backend/routes/blood.py.  It intentionally
uses the CBC fields collected by the current UI, while the notebook explores
the broader Excel file for EDA and feature understanding.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "blood_report_dataset.xlsx")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

COL_MAP = {
    "hemoglobin": "hemoglobin",
    "rbc_count": "rbc",
    "wbc_count": "wbc",
    "platelet_count": "platelet",
    "mcv": "mcv",
    "mch": "mch",
    "mchc": "mchc",
    "neutrophils_pct": "neutrophil",
    "lymphocytes_pct": "lymphocyte",
}
RAW_FEATURES = list(COL_MAP)
FEATURES = list(COL_MAP.values())
TARGET = "diagnosis"


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
    return pd.read_excel(path, sheet_name="BloodReports")


def clean_blood_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "patient_id" in df:
        df["patient_id"] = df["patient_id"].astype("string").str.strip()
    for col in [
        "gender",
        "blood_group",
        "city",
        "region",
        "diet_type",
        "smoker",
        "alcohol_consumption",
        "exercise_level",
        "pregnancy_status",
        "existing_conditions",
        "lab_name",
        TARGET,
    ]:
        if col in df:
            df[col] = _standardize_text(df[col])

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
        "age",
        "height_cm",
        "weight_kg",
        "hemoglobin",
        "rbc_count",
        "wbc_count",
        "platelet_count",
        "mcv",
        "mch",
        "mchc",
        "hematocrit",
        "rdw",
        "neutrophils_pct",
        "lymphocytes_pct",
        "monocytes_pct",
        "eosinophils_pct",
        "basophils_pct",
    ]
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    hb = df["hemoglobin"]
    high_hb = hb > 40
    low_hb = hb.between(0.1, 2.0, inclusive="left")
    df.loc[high_hb, "hemoglobin"] = hb[high_hb] / 10.0
    df.loc[low_hb, "hemoglobin"] = hb[low_hb] * 10.0

    high_mcv = df["mcv"] > 300
    df.loc[high_mcv, "mcv"] = df.loc[high_mcv, "mcv"] / 10.0

    physiological_ranges = {
        "hemoglobin": (3, 25),
        "rbc_count": (1, 9),
        "wbc_count": (0.2, 80),
        "platelet_count": (5, 1200),
        "mcv": (45, 140),
        "mch": (10, 45),
        "mchc": (20, 45),
        "hematocrit": (8, 75),
        "rdw": (5, 35),
        "neutrophils_pct": (0, 100),
        "lymphocytes_pct": (0, 100),
        "monocytes_pct": (0, 100),
        "eosinophils_pct": (0, 100),
        "basophils_pct": (0, 100),
    }
    clipped_cells = 0
    for col, (lo, hi) in physiological_ranges.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        clipped_cells += int(bad.sum())
        df.loc[bad, col] = np.nan
    if verbose:
        print(
            "  [clean] corrected Hb/MCV unit issues: "
            f"{int(high_hb.sum() + low_hb.sum() + high_mcv.sum())}"
        )
        print(f"  [clean] set implausible numeric cells to missing: {clipped_cells}")

    before = len(df)
    df = df.dropna(subset=[TARGET])
    if verbose:
        print(f"  [clean] dropped rows without diagnosis: {before - len(df)}")

    df = df.rename(columns=COL_MAP)
    return df


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cbc",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                    ]
                ),
                FEATURES,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def candidate_models() -> dict[str, Pipeline]:
    return {
        "random_forest_baseline": Pipeline(
            [
                ("prep", make_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=350,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees_baseline": Pipeline(
            [
                ("prep", make_preprocessor()),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=450,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting_baseline": Pipeline(
            [
                ("prep", make_preprocessor()),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=220,
                        learning_rate=0.06,
                        max_depth=3,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "svm_rbf_baseline": Pipeline(
            [
                ("prep", make_preprocessor()),
                (
                    "model",
                    SVC(
                        C=7.5,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "mlp_deep_baseline": Pipeline(
            [
                ("prep", make_preprocessor()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(96, 48),
                        activation="relu",
                        alpha=0.0008,
                        learning_rate_init=0.001,
                        max_iter=450,
                        early_stopping=True,
                        n_iter_no_change=25,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def tune_candidates(X_train: pd.DataFrame, y_train: pd.Series) -> list[tuple[str, Pipeline, dict]]:
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    searches = [
        (
            "random_forest_tuned",
            Pipeline([("prep", make_preprocessor()), ("model", RandomForestClassifier(random_state=42, n_jobs=-1))]),
            {
                "model__n_estimators": [260, 420, 620],
                "model__max_depth": [None, 12, 18],
                "model__min_samples_leaf": [1, 2, 4],
                "model__class_weight": ["balanced", "balanced_subsample"],
            },
        ),
        (
            "extra_trees_tuned",
            Pipeline([("prep", make_preprocessor()), ("model", ExtraTreesClassifier(random_state=42, n_jobs=-1))]),
            {
                "model__n_estimators": [320, 520, 720],
                "model__max_depth": [None, 14, 22],
                "model__min_samples_leaf": [1, 2, 3],
                "model__class_weight": ["balanced", "balanced_subsample"],
            },
        ),
        (
            "svm_rbf_tuned",
            Pipeline([("prep", make_preprocessor()), ("model", SVC(probability=True, random_state=42))]),
            {
                "model__C": [2.5, 5.0, 8.0, 12.0],
                "model__gamma": ["scale", 0.04, 0.08],
                "model__class_weight": ["balanced"],
            },
        ),
        (
            "mlp_deep_tuned",
            Pipeline([("prep", make_preprocessor()), ("model", MLPClassifier(early_stopping=True, max_iter=550, random_state=42))]),
            {
                "model__hidden_layer_sizes": [(64, 32), (96, 48), (128, 64, 24)],
                "model__alpha": [0.0003, 0.0008, 0.0015],
                "model__learning_rate_init": [0.0007, 0.001, 0.0015],
            },
        ),
    ]

    tuned = []
    for name, estimator, grid in searches:
        print(f"\n[blood] tuning {name}")
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


def evaluate_model(name: str, estimator: Pipeline, X_train, y_train, X_test, y_test) -> TrialResult:
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
    print(f"[blood] loading {path}")
    raw = load_dataset(path)
    print(f"[blood] raw shape: {raw.shape}")
    df = clean_blood_data(raw)
    print(f"[blood] clean shape: {df.shape}")

    missing = df[FEATURES].isna().sum().sort_values(ascending=False)
    print("[blood] missing deployment features before pipeline imputation:")
    print(missing.to_string())
    print("[blood] class distribution:")
    print(df[TARGET].value_counts().to_string())

    X = df[FEATURES]
    y = df[TARGET].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\n[blood] baseline trials with repeated CV")
    trial_results = [
        evaluate_model(name, model, X_train, y_train, X_test, y_test)
        for name, model in candidate_models().items()
    ]

    tuned_models = tune_candidates(X_train, y_train)
    print("\n[blood] tuned trials with repeated CV")
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

    print("\n[blood] leaderboard")
    print(leaderboard.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    best_name = leaderboard.iloc[0]["model"]
    best = next(r for r in trial_results if r.name == best_name)
    print(f"\n[blood] selected model: {best.name}")
    print(classification_report(y_test, best.estimator.predict(X_test)))

    bundle = {
        "model": best.estimator,
        "features": FEATURES,
        "raw_feature_map": COL_MAP,
        "classes": sorted(y.unique().tolist()),
        "trained_on": os.path.basename(path),
        "n_samples_train": int(len(X_train)),
        "n_samples_test": int(len(X_test)),
        "leaderboard": leaderboard.to_dict(orient="records"),
        "selected_model": best.name,
        "test_accuracy": best.test_accuracy,
        "test_f1_macro": best.test_f1_macro,
        "notes": "CBC-only deployment model trained from the real Excel dataset.",
    }
    return bundle, leaderboard


def main() -> None:
    bundle, _ = train()
    out = os.path.join(MODELS_DIR, "blood_model.pkl")
    joblib.dump(bundle, out)
    print(f"[blood] saved -> {out}")


if __name__ == "__main__":
    main()
