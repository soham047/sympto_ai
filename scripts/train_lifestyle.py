"""
Lifestyle planner training.

This module builds a realistic tabular training set for calorie and protein
targets, benchmarks a few regressors, tunes the strongest candidate, and saves
the bundle consumed by backend/routes/lifestyle.py.

Why synthetic? A complete lifestyle planner needs labelled targets for calorie
needs, protein targets, schedule constraints, goals, diet pattern, sleep, and
stress. Public datasets usually cover only one slice. Here, public guideline
logic provides guardrails and the model learns the smooth tabular target.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

ACTIVITY = ["sedentary", "light", "moderate", "active", "very_active"]
GOAL = ["lose_fat", "maintain", "gain_muscle", "recompose", "improve_fitness"]
DIET_TYPES = ["balanced", "vegetarian", "vegan", "high_protein", "diabetes_friendly", "heart_friendly"]
EXPERIENCE = ["beginner", "intermediate", "advanced"]
EQUIPMENT = ["none", "bands", "dumbbells", "gym"]
WORK_STYLE = ["desk", "mixed", "standing", "physical"]
STRESS = ["low", "medium", "high"]

NUMERIC_FEATURES = [
    "age",
    "weight_kg",
    "height_cm",
    "bmi",
    "sleep_hours",
    "workout_days",
    "available_minutes",
    "steps_now",
]
CATEGORICAL_FEATURES = [
    "sex",
    "activity",
    "goal",
    "diet_type",
    "experience",
    "equipment",
    "work_style",
    "stress_level",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGETS = ["calories", "protein_g"]


@dataclass
class ModelResult:
    name: str
    mae_calories: float
    mae_protein: float
    r2_calories: float
    r2_protein: float
    estimator: object


def mifflin_st_jeor(age: float, sex: str, weight_kg: float, height_cm: float) -> float:
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    return base + 5 if str(sex).upper().startswith("M") else base - 161


ACTIVITY_MULT = {
    "sedentary": 1.20,
    "light": 1.36,
    "moderate": 1.52,
    "active": 1.70,
    "very_active": 1.88,
}
WORK_STYLE_BONUS = {"desk": 0, "mixed": 90, "standing": 140, "physical": 260}
GOAL_DELTA = {
    "lose_fat": -430,
    "maintain": 0,
    "gain_muscle": 330,
    "recompose": -120,
    "improve_fitness": -40,
}


def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


def _protein_target(weight_kg: float, goal: str, activity: str, diet_type: str) -> float:
    base = {
        "lose_fat": 1.75,
        "maintain": 1.35,
        "gain_muscle": 1.85,
        "recompose": 1.75,
        "improve_fitness": 1.50,
    }[goal]
    if activity in {"active", "very_active"}:
        base += 0.12
    if diet_type == "vegan":
        base += 0.08
    if diet_type == "diabetes_friendly":
        base += 0.05
    return _clip(weight_kg * base, 45, 220)


def synth_dataset(n: int = 9000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n):
        sex = rng.choice(["M", "F"], p=[0.52, 0.48])
        age = int(rng.integers(18, 72))
        height = rng.normal(174, 8) if sex == "M" else rng.normal(162, 7)
        height = _clip(height, 145, 198)

        bmi_center = rng.choice([21.5, 24.0, 27.5, 31.0], p=[0.25, 0.35, 0.25, 0.15])
        bmi = _clip(rng.normal(bmi_center, 2.3), 17.5, 39)
        weight = bmi * (height / 100) ** 2

        activity = rng.choice(ACTIVITY, p=[0.30, 0.27, 0.23, 0.15, 0.05])
        goal = rng.choice(GOAL, p=[0.30, 0.22, 0.18, 0.18, 0.12])
        diet_type = rng.choice(DIET_TYPES, p=[0.36, 0.20, 0.08, 0.14, 0.12, 0.10])
        experience = rng.choice(EXPERIENCE, p=[0.48, 0.36, 0.16])
        equipment = rng.choice(EQUIPMENT, p=[0.28, 0.18, 0.25, 0.29])
        work_style = rng.choice(WORK_STYLE, p=[0.46, 0.30, 0.12, 0.12])
        stress = rng.choice(STRESS, p=[0.28, 0.50, 0.22])

        workout_days = int(rng.integers(2, 7))
        if activity == "sedentary":
            workout_days = min(workout_days, int(rng.integers(2, 5)))
        if activity in {"active", "very_active"}:
            workout_days = max(workout_days, int(rng.integers(4, 7)))

        available_minutes = int(rng.choice([25, 30, 35, 40, 45, 55, 60, 75], p=[.08, .16, .12, .16, .18, .12, .14, .04]))
        sleep = _clip(rng.normal(7.1, 0.9) - (0.35 if stress == "high" else 0), 4.5, 9.5)
        steps_center = {"sedentary": 3600, "light": 5600, "moderate": 7600, "active": 9800, "very_active": 12200}[activity]
        steps_now = int(_clip(rng.normal(steps_center, 1500), 1200, 18000))

        bmr = mifflin_st_jeor(age, sex, weight, height)
        tdee = bmr * ACTIVITY_MULT[activity] + WORK_STYLE_BONUS[work_style]
        tdee += max(0, workout_days - 3) * 28
        tdee += (available_minutes - 40) * 1.8
        tdee += (sleep - 7) * 18
        tdee += {"low": 20, "medium": 0, "high": -45}[stress]

        calories = tdee + GOAL_DELTA[goal] + rng.normal(0, 70)
        if goal == "lose_fat":
            calories = max(calories, bmr * 1.08)
        elif goal == "gain_muscle":
            calories = min(calories, tdee + 520)
        else:
            calories = max(calories, bmr * 1.05)

        protein = _protein_target(weight, goal, activity, diet_type) + rng.normal(0, 5)

        rows.append(
            {
                "age": age,
                "sex": sex,
                "weight_kg": round(weight, 1),
                "height_cm": round(height, 1),
                "bmi": round(weight / ((height / 100) ** 2), 1),
                "activity": activity,
                "goal": goal,
                "diet_type": diet_type,
                "experience": experience,
                "equipment": equipment,
                "work_style": work_style,
                "stress_level": stress,
                "sleep_hours": round(sleep, 1),
                "workout_days": workout_days,
                "available_minutes": available_minutes,
                "steps_now": steps_now,
                "calories": round(_clip(calories, 1200, 4300)),
                "protein_g": round(_clip(protein, 45, 220)),
            }
        )
    return pd.DataFrame(rows)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def candidate_models() -> dict[str, Pipeline]:
    return {
        "ridge_baseline": Pipeline(
            [
                ("pre", make_preprocessor()),
                ("model", MultiOutputRegressor(Ridge(alpha=1.0))),
            ]
        ),
        "random_forest_baseline": Pipeline(
            [
                ("pre", make_preprocessor()),
                (
                    "model",
                    MultiOutputRegressor(
                        RandomForestRegressor(
                            n_estimators=220,
                            min_samples_leaf=3,
                            random_state=42,
                            n_jobs=-1,
                        )
                    ),
                ),
            ]
        ),
        "extra_trees_baseline": Pipeline(
            [
                ("pre", make_preprocessor()),
                (
                    "model",
                    MultiOutputRegressor(
                        ExtraTreesRegressor(
                            n_estimators=260,
                            min_samples_leaf=3,
                            random_state=42,
                            n_jobs=-1,
                        )
                    ),
                ),
            ]
        ),
        "gradient_boosting_baseline": Pipeline(
            [
                ("pre", make_preprocessor()),
                (
                    "model",
                    MultiOutputRegressor(
                        GradientBoostingRegressor(
                            n_estimators=220,
                            learning_rate=0.05,
                            max_depth=3,
                            random_state=42,
                        )
                    ),
                ),
            ]
        ),
    }


def evaluate_model(name: str, estimator, X_train, y_train, X_test, y_test) -> ModelResult:
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        estimator,
        X_train,
        y_train,
        cv=cv,
        scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
        n_jobs=-1,
    )
    estimator.fit(X_train, y_train)
    pred = estimator.predict(X_test)
    mae_by_target = np.mean(np.abs(y_test.to_numpy() - pred), axis=0)
    r2_by_target = [
        r2_score(y_test.iloc[:, 0], pred[:, 0]),
        r2_score(y_test.iloc[:, 1], pred[:, 1]),
    ]
    result = ModelResult(
        name=name,
        mae_calories=float(mae_by_target[0]),
        mae_protein=float(mae_by_target[1]),
        r2_calories=float(r2_by_target[0]),
        r2_protein=float(r2_by_target[1]),
        estimator=estimator,
    )
    print(
        f"   {name:28s} cv_mae={-scores['test_mae'].mean():.1f} "
        f"cv_r2={scores['test_r2'].mean():.3f} "
        f"test_mae_kcal={result.mae_calories:.1f} "
        f"test_mae_protein={result.mae_protein:.1f}"
    )
    return result


def tune_model(X_train, y_train):
    print("\n[lifestyle] tuning gradient_boosting")
    estimator = Pipeline(
        [
            ("pre", make_preprocessor()),
            ("model", MultiOutputRegressor(GradientBoostingRegressor(random_state=42))),
        ]
    )
    grid = {
        "model__estimator__n_estimators": [180, 260, 340],
        "model__estimator__learning_rate": [0.035, 0.05, 0.075],
        "model__estimator__max_depth": [2, 3],
        "model__estimator__min_samples_leaf": [4, 8, 14],
    }
    search = GridSearchCV(
        estimator,
        grid,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=4, shuffle=True, random_state=42),
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"   best_cv_mae={-search.best_score_:.1f}")
    print(f"   best_params={search.best_params_}")
    return "gradient_boosting_tuned", search.best_estimator_, search.best_params_


def train(seed: int = 42) -> tuple[dict, pd.DataFrame]:
    df = synth_dataset(seed=seed)
    print(f"[lifestyle] dataset shape: {df.shape}")
    print("[lifestyle] goal distribution:")
    print(df["goal"].value_counts().to_string())

    X = df[FEATURES]
    y = df[TARGETS]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("\n[lifestyle] baseline trials")
    results = [
        evaluate_model(name, model, X_train, y_train, X_test, y_test)
        for name, model in candidate_models().items()
    ]

    tuned_name, tuned_model, tuned_params = tune_model(X_train, y_train)
    print("\n[lifestyle] tuned trial")
    results.append(evaluate_model(tuned_name, tuned_model, X_train, y_train, X_test, y_test))

    leaderboard = pd.DataFrame(
        [
            {
                "model": r.name,
                "mae_calories": r.mae_calories,
                "mae_protein": r.mae_protein,
                "r2_calories": r.r2_calories,
                "r2_protein": r.r2_protein,
            }
            for r in results
        ]
    ).sort_values(["mae_calories", "mae_protein"])
    print("\n[lifestyle] leaderboard")
    print(leaderboard.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    best_name = leaderboard.iloc[0]["model"]
    best = next(r for r in results if r.name == best_name)
    bundle = {
        "model": best.estimator,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "targets": TARGETS,
        "activity": ACTIVITY,
        "goal": GOAL,
        "diet_types": DIET_TYPES,
        "experience": EXPERIENCE,
        "equipment": EQUIPMENT,
        "work_style": WORK_STYLE,
        "stress": STRESS,
        "selected_model": best.name,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "mae_calories": best.mae_calories,
        "mae_protein": best.mae_protein,
        "r2_calories": best.r2_calories,
        "r2_protein": best.r2_protein,
        "training_notes": (
            "Synthetic lifestyle cohort generated from Mifflin-St Jeor, activity multipliers, "
            "goal deltas, and conservative public-health planning guardrails."
        ),
        "references": [
            "CDC adult activity guidance: 150 minutes moderate activity weekly plus 2 days of muscle-strengthening - https://www.cdc.gov/physical-activity-basics/guidelines/adults.html",
            "Physical Activity Guidelines for Americans: 150-300 minutes moderate activity weekly and muscle-strengthening - https://www.cdc.gov/physical-activity/media/pdfs/Physical_Activity_Guidelines_2nd_edition.pdf",
            "CDC adult sleep guidance: at least 7 hours per 24 hours for adults - https://www.cdc.gov/sleep/about/index.html",
            "Dietary Guidelines for Americans 2020-2025: nutrient-dense foods within calorie limits - https://www.dietaryguidelines.gov/sites/default/files/2020-12/Dietary_Guidelines_for_Americans_2020-2025.pdf",
        ],
    }
    return bundle, leaderboard


def main() -> None:
    bundle, _ = train()
    out = os.path.join(MODELS_DIR, "lifestyle_model.pkl")
    joblib.dump(bundle, out)
    print(f"[lifestyle] saved -> {out}")


if __name__ == "__main__":
    main()
