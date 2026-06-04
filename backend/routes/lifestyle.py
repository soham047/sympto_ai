import math
import os
from datetime import date, timedelta

import joblib
import pandas as pd
from flask import Blueprint, jsonify, request

bp = Blueprint("lifestyle", __name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "lifestyle_model.pkl")
_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


ACTIVITY_LABELS = {
    "sedentary": "Sedentary",
    "light": "Lightly active",
    "moderate": "Moderately active",
    "active": "Active",
    "very_active": "Very active",
}
GOAL_LABELS = {
    "lose_fat": "Lose fat",
    "maintain": "Maintain",
    "gain_muscle": "Gain muscle",
    "recompose": "Body recomposition",
    "improve_fitness": "Improve fitness",
}


def _num(data, key, default=None, lo=None, hi=None):
    raw = data.get(key, default)
    if raw in ("", None):
        raw = default
    val = float(raw)
    if lo is not None and val < lo:
        raise ValueError(f"{key} must be at least {lo}")
    if hi is not None and val > hi:
        raise ValueError(f"{key} must be at most {hi}")
    return val


def _choice(data, key, allowed, default):
    val = str(data.get(key, default)).strip()
    return val if val in allowed else default


def _bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Healthy range"
    if bmi < 30:
        return "Overweight"
    return "Obesity range"


def _mifflin(age, sex, weight, height):
    base = 10 * weight + 6.25 * height - 5 * age
    return base + 5 if sex == "M" else base - 161


def _round_to_25(x):
    return int(round(x / 25) * 25)


def _macro_plan(calories, protein_g, weight, goal, diet_type):
    fat_per_kg = 0.75 if goal in {"lose_fat", "recompose"} else 0.85
    if diet_type == "heart_friendly":
        fat_per_kg = 0.70
    fat_g = max(40, round(weight * fat_per_kg))
    protein_kcal = protein_g * 4
    fat_kcal = fat_g * 9
    carb_g = max(90, round((calories - protein_kcal - fat_kcal) / 4))
    fiber_g = max(25, min(42, round(calories / 1000 * 14)))
    return {
        "protein_g": int(round(protein_g)),
        "carbs_g": int(carb_g),
        "fat_g": int(fat_g),
        "fiber_g": int(fiber_g),
        "split": {
            "protein_pct": round(protein_kcal / calories * 100),
            "carbs_pct": round(carb_g * 4 / calories * 100),
            "fat_pct": round(fat_kcal / calories * 100),
        },
    }


def _habit_score(profile):
    score = 100
    score -= max(0, 7 - profile["sleep_hours"]) * 8
    score -= {"low": 0, "medium": 6, "high": 14}[profile["stress_level"]]
    score -= max(0, 6000 - profile["steps_now"]) / 600
    if profile["bmi"] >= 30 or profile["bmi"] < 18.5:
        score -= 8
    if profile["available_minutes"] < 30:
        score -= 5
    return int(max(42, min(96, round(score))))


def _focus(profile):
    goal = profile["goal"]
    if goal == "lose_fat":
        return {
            "theme": "steady fat loss",
            "primary": "calorie consistency, steps, and strength retention",
            "pace": "aim for roughly 0.25-0.75 kg change per week",
        }
    if goal == "gain_muscle":
        return {
            "theme": "lean muscle gain",
            "primary": "progressive strength work, protein, and sleep",
            "pace": "small surplus; adjust if weight jumps too quickly",
        }
    if goal == "recompose":
        return {
            "theme": "body recomposition",
            "primary": "high protein, lifting quality, and modest calorie control",
            "pace": "track strength and waist more than scale weight",
        }
    if goal == "improve_fitness":
        return {
            "theme": "cardio and strength base",
            "primary": "weekly aerobic minutes, mobility, and repeatable workouts",
            "pace": "increase training volume gradually",
        }
    return {
        "theme": "maintenance and energy",
        "primary": "stable meals, movement, sleep, and stress recovery",
        "pace": "keep weight and energy steady",
    }


def _workout_library(profile):
    equipment = profile["equipment"]
    experience = profile["experience"]
    minutes = int(profile["available_minutes"])
    base_sets = 2 if experience == "beginner" else 3 if experience == "intermediate" else 4
    rest = "60-90 sec" if experience != "advanced" else "90-120 sec"

    if equipment == "gym":
        strength_a = [
            f"Squat or leg press - {base_sets} x 6-10",
            f"Bench press or chest press - {base_sets} x 6-10",
            f"Lat pulldown or row - {base_sets} x 8-12",
            f"Romanian deadlift - {base_sets} x 8-10",
            f"Plank - {base_sets} x 30-45 sec",
        ]
        strength_b = [
            f"Deadlift or hip thrust - {base_sets} x 5-8",
            f"Overhead press - {base_sets} x 6-10",
            f"Seated row - {base_sets} x 8-12",
            f"Split squat - {base_sets} x 8 each side",
            f"Farmer carry - {base_sets} x 30 m",
        ]
    elif equipment == "dumbbells":
        strength_a = [
            f"Dumbbell goblet squat - {base_sets} x 8-12",
            f"Dumbbell floor press - {base_sets} x 8-12",
            f"One-arm dumbbell row - {base_sets} x 10 each side",
            f"Dumbbell Romanian deadlift - {base_sets} x 8-12",
            f"Dead bug - {base_sets} x 8 each side",
        ]
        strength_b = [
            f"Reverse lunge - {base_sets} x 8 each side",
            f"Dumbbell overhead press - {base_sets} x 8-12",
            f"Hip bridge - {base_sets} x 12-15",
            f"Renegade row or supported row - {base_sets} x 8 each side",
            f"Side plank - {base_sets} x 25-40 sec",
        ]
    elif equipment == "bands":
        strength_a = [
            f"Band squat - {base_sets} x 12",
            f"Band chest press - {base_sets} x 10-14",
            f"Band row - {base_sets} x 12-15",
            f"Band good morning - {base_sets} x 12",
            f"Bird dog - {base_sets} x 8 each side",
        ]
        strength_b = [
            f"Step-up - {base_sets} x 10 each side",
            f"Band overhead press - {base_sets} x 10-12",
            f"Band pulldown - {base_sets} x 12",
            f"Glute bridge - {base_sets} x 12-15",
            f"Pallof press - {base_sets} x 10 each side",
        ]
    else:
        strength_a = [
            f"Bodyweight squat - {base_sets} x 12",
            f"Incline push-up - {base_sets} x 8-12",
            f"Backpack row or towel row - {base_sets} x 10-12",
            f"Hip hinge good morning - {base_sets} x 12",
            f"Plank - {base_sets} x 25-40 sec",
        ]
        strength_b = [
            f"Reverse lunge - {base_sets} x 8 each side",
            f"Push-up variation - {base_sets} x 6-12",
            f"Glute bridge - {base_sets} x 12-15",
            f"Wall sit - {base_sets} x 30-45 sec",
            f"Mountain climber - {base_sets} x 20",
        ]

    cardio_easy = f"Zone-2 walk/cycle for {max(20, minutes - 10)} min"
    cardio_intervals = "6 rounds: 1 min brisk + 2 min easy" if minutes < 40 else "8 rounds: 1 min brisk + 2 min easy"
    mobility = ["Hip flexor stretch", "Thoracic rotations", "Hamstring floss", "Ankle rocks", "Box breathing 3 min"]
    return {
        "strength_a": {"title": "Strength A", "items": strength_a, "rest": rest},
        "strength_b": {"title": "Strength B", "items": strength_b, "rest": rest},
        "cardio_easy": {"title": "Easy Cardio", "items": [cardio_easy, "Keep effort conversational", "Finish with 5 min easy cooldown"]},
        "cardio_intervals": {"title": "Intervals", "items": [cardio_intervals, "Warm up 6-8 min", "Stop if dizzy or chest pain occurs"]},
        "mobility": {"title": "Mobility + Recovery", "items": mobility},
    }


def _weekly_schedule(profile):
    lib = _workout_library(profile)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    templates = {
        2: ["strength_a", "mobility", "cardio_easy", "mobility", "strength_b", "walk", "recovery"],
        3: ["strength_a", "mobility", "cardio_easy", "strength_b", "mobility", "cardio_easy", "recovery"],
        4: ["strength_a", "cardio_easy", "strength_b", "mobility", "strength_a", "cardio_intervals", "recovery"],
        5: ["strength_a", "cardio_easy", "strength_b", "mobility", "strength_a", "cardio_intervals", "strength_b"],
        6: ["strength_a", "cardio_easy", "strength_b", "mobility", "strength_a", "cardio_intervals", "strength_b"],
    }
    plan_keys = templates.get(int(profile["workout_days"]), templates[4])
    schedule = []
    for day, key in zip(days, plan_keys):
        if key == "walk":
            block = {"title": "Steps Focus", "items": [f"Walk until you reach {profile['step_target']:,} steps", "Add 5 min mobility if stiff"]}
        elif key == "recovery":
            block = {"title": "Recovery", "items": ["Gentle walk", "Light stretching", "Prepare meals for tomorrow"]}
        else:
            block = lib[key]
        schedule.append({"day": day, **block})
    return schedule


def _monthly_schedule(profile):
    today = date.today()
    monday = today - timedelta(days=today.weekday())
    focus = _focus(profile)
    return [
        {
            "week": i + 1,
            "dates": f"{(monday + timedelta(days=i*7)).strftime('%d %b')} - {(monday + timedelta(days=i*7+6)).strftime('%d %b')}",
            "focus": [
                "Baseline week: learn portions, log workouts, keep sessions easy.",
                "Build week: add one set to main strength moves or 5 min cardio.",
                "Push week: keep nutrition steady and make the hardest workout slightly harder.",
                "Deload and review: reduce volume 20%, check weight, waist, energy, and sleep.",
            ][i],
            "checkpoint": [
                "Take starting weight, waist, resting pulse, and step average.",
                "Review hunger, soreness, and schedule friction.",
                "Check whether performance is improving without sleep falling.",
                "Adjust calories by 100-150 only if the trend is clearly off.",
            ][i],
            "theme": focus["theme"],
        }
        for i in range(4)
    ]


MEAL_BANK = {
    "balanced": {
        "breakfast": ["oats with milk, banana, nuts", "eggs with whole-grain toast and fruit"],
        "lunch": ["rice or roti bowl with dal/chicken, salad, curd", "whole-grain wrap with paneer/chicken and vegetables"],
        "snack": ["fruit plus yogurt", "roasted chana or nuts"],
        "dinner": ["grilled protein, vegetables, and potatoes", "dal, sabzi, roti, and salad"],
    },
    "vegetarian": {
        "breakfast": ["paneer bhurji with toast", "Greek yogurt, fruit, and seeds"],
        "lunch": ["dal, rice, salad, and curd", "rajma/chole bowl with vegetables"],
        "snack": ["sprouts chaat", "lassi or yogurt with fruit"],
        "dinner": ["paneer/tofu stir-fry with roti", "khichdi with vegetables and curd"],
    },
    "vegan": {
        "breakfast": ["soy milk oats with peanut butter", "tofu scramble with toast"],
        "lunch": ["lentil bowl with rice and vegetables", "chickpea salad wrap"],
        "snack": ["fruit and peanuts", "hummus with carrots"],
        "dinner": ["tofu curry with rice", "bean chili with salad"],
    },
    "high_protein": {
        "breakfast": ["eggs or tofu with toast", "protein yogurt bowl"],
        "lunch": ["lean chicken/paneer bowl with rice and salad", "tuna/chickpea wrap with vegetables"],
        "snack": ["protein shake or milk", "cottage cheese/paneer cubes"],
        "dinner": ["fish/chicken/tofu with vegetables", "dal plus grilled paneer/chicken"],
    },
    "diabetes_friendly": {
        "breakfast": ["besan chilla with curd", "eggs/tofu with vegetables"],
        "lunch": ["dal, salad, curd, and measured roti/rice", "lean protein bowl with non-starchy vegetables"],
        "snack": ["nuts and fruit portion", "sprouts or roasted chana"],
        "dinner": ["protein, vegetables, and small whole-grain serving", "soup plus paneer/tofu salad"],
    },
    "heart_friendly": {
        "breakfast": ["oats with fruit and seeds", "vegetable poha with curd"],
        "lunch": ["dal, brown rice/roti, vegetables, salad", "fish/tofu bowl with greens"],
        "snack": ["fruit and unsalted nuts", "curd with flaxseed"],
        "dinner": ["grilled protein, vegetables, and legumes", "vegetable soup, dal, and roti"],
    },
}


def _meal_plan(profile, macros):
    diet = profile["diet_type"]
    bank = MEAL_BANK.get(diet, MEAL_BANK["balanced"])
    calories = profile["calories"]
    meals = int(profile["meals_per_day"])
    distribution = {
        3: [("Breakfast", 0.28), ("Lunch", 0.38), ("Dinner", 0.34)],
        4: [("Breakfast", 0.25), ("Lunch", 0.34), ("Snack", 0.13), ("Dinner", 0.28)],
        5: [("Breakfast", 0.22), ("Lunch", 0.30), ("Snack 1", 0.10), ("Dinner", 0.28), ("Snack 2", 0.10)],
    }.get(meals, [("Breakfast", 0.25), ("Lunch", 0.34), ("Snack", 0.13), ("Dinner", 0.28)])

    lookup = {"Breakfast": "breakfast", "Lunch": "lunch", "Dinner": "dinner", "Snack": "snack", "Snack 1": "snack", "Snack 2": "snack"}
    out = []
    for idx, (name, pct) in enumerate(distribution):
        options = bank[lookup[name]]
        out.append(
            {
                "name": name,
                "target_kcal": _round_to_25(calories * pct),
                "idea": options[idx % len(options)],
                "plate_rule": "protein palm + fiber carbs + 2 fists vegetables" if name != "Snack" else "protein or fiber first",
            }
        )
    return out


def _grocery_list(profile):
    diet = profile["diet_type"]
    proteins = {
        "vegan": ["tofu/tempeh", "lentils", "chickpeas", "soy milk", "beans"],
        "vegetarian": ["paneer", "curd/yogurt", "dal", "eggs if acceptable", "sprouts"],
        "high_protein": ["eggs", "chicken/fish or tofu", "Greek yogurt", "paneer", "dal"],
    }.get(diet, ["eggs/tofu", "dal", "curd/yogurt", "chicken/fish or paneer", "beans"])
    return {
        "proteins": proteins,
        "carbs": ["oats", "rice or millet", "whole-wheat roti/bread", "potatoes", "fruit"],
        "vegetables": ["leafy greens", "carrots", "cucumber", "beans", "seasonal vegetables"],
        "fats": ["nuts", "seeds", "olive/mustard oil", "avocado if budget allows"],
        "extras": ["ORS/electrolytes for hot days", "spices", "lemon", "unsweetened tea/coffee"],
    }


def _habits(profile):
    habits = [
        f"Walk {profile['step_target']:,} steps on most days; start from current average, not perfection.",
        f"Train {profile['workout_days']} days/week for about {profile['available_minutes']} minutes.",
        f"Hit about {profile['protein_g']} g protein and {profile['fiber_g']} g fiber daily.",
        f"Drink around {profile['water_l']} L water; add more on hot or sweaty days.",
    ]
    if profile["sleep_hours"] < 7:
        habits.append("Move bedtime 15 minutes earlier for one week before changing anything else.")
    if profile["stress_level"] == "high":
        habits.append("Add a 5-minute breathing block after work or before sleep.")
    return habits


def _safety(profile):
    flags = []
    if profile["age"] < 18:
        flags.append("Under 18: use this only with parent/clinician guidance; growth needs can change calorie targets.")
    if profile["bmi"] < 18.5 and profile["goal"] == "lose_fat":
        flags.append("Fat-loss goal is not recommended at this BMI without clinician supervision.")
    if profile["bmi"] >= 35:
        flags.append("BMI is in a higher-risk range; consider medical review before aggressive dieting or intense training.")
    if profile["conditions"]:
        flags.append("Because you listed medical conditions, keep intensity conservative and confirm major diet/training changes with a clinician.")
    if profile["injury"]:
        flags.append("Modify or skip painful movements; pain that changes gait or sleep deserves professional assessment.")
    if profile["sleep_hours"] < 6:
        flags.append("Sleep is currently low; prioritize recovery before adding hard intervals.")
    return flags


def _parse_profile(data, bundle):
    age = _num(data, "age", 25, 13, 90)
    sex = _choice(data, "sex", ["M", "F"], "M")
    weight = _num(data, "weight", data.get("weight_kg", 70), 30, 220)
    height = _num(data, "height", data.get("height_cm", 170), 120, 220)
    bmi = weight / ((height / 100) ** 2)

    goal = _choice(data, "goal", bundle["goal"], "maintain")
    activity = _choice(data, "activity", bundle["activity"], "light")
    diet_type = _choice(data, "diet_type", bundle["diet_types"], "balanced")
    experience = _choice(data, "experience", bundle["experience"], "beginner")
    equipment = _choice(data, "equipment", bundle["equipment"], "none")
    work_style = _choice(data, "work_style", bundle["work_style"], "desk")
    stress_level = _choice(data, "stress_level", bundle["stress"], "medium")

    workout_days = int(_num(data, "workout_days", 4, 2, 6))
    available_minutes = int(_num(data, "available_minutes", 40, 20, 90))
    sleep_hours = _num(data, "sleep_hours", 7, 4, 10)
    steps_now = int(_num(data, "steps_now", 5500, 500, 30000))
    meals_per_day = int(_num(data, "meals_per_day", 4, 3, 5))
    budget = _choice(data, "budget", ["low", "medium", "high"], "medium")

    conditions = str(data.get("conditions", "") or "").strip()
    injury = str(data.get("injury", "") or "").strip()
    cuisine = str(data.get("cuisine", "flexible") or "flexible").strip()[:40]

    return {
        "age": age,
        "sex": sex,
        "weight_kg": weight,
        "height_cm": height,
        "bmi": bmi,
        "activity": activity,
        "goal": goal,
        "diet_type": diet_type,
        "experience": experience,
        "equipment": equipment,
        "work_style": work_style,
        "stress_level": stress_level,
        "sleep_hours": sleep_hours,
        "workout_days": workout_days,
        "available_minutes": available_minutes,
        "steps_now": steps_now,
        "meals_per_day": meals_per_day,
        "budget": budget,
        "conditions": conditions,
        "injury": injury,
        "cuisine": cuisine,
    }


@bp.route("/options", methods=["GET"])
def options():
    b = _load()
    return jsonify(
        {
            "activity": b["activity"],
            "goal": b["goal"],
            "diet_types": b["diet_types"],
            "experience": b["experience"],
            "equipment": b["equipment"],
            "work_style": b["work_style"],
            "stress": b["stress"],
        }
    )


@bp.route("/predict", methods=["POST"])
def predict():
    b = _load()
    data = request.get_json(force=True)
    try:
        profile = _parse_profile(data, b)
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"missing/invalid field: {e}"}), 400

    model_row = pd.DataFrame([{k: profile[k] for k in b["features"]}])
    prediction = b["model"].predict(model_row)[0]
    calories = _round_to_25(float(prediction[0]))
    bmr = _mifflin(profile["age"], profile["sex"], profile["weight_kg"], profile["height_cm"])
    calories = int(max(calories, math.ceil(bmr * 1.05)))
    protein_g = int(round(float(prediction[1])))

    step_target = int(min(12000, max(profile["steps_now"] + 1500, 7000 if profile["activity"] == "sedentary" else 8500)))
    water_l = round(min(4.5, max(1.8, profile["weight_kg"] * 0.033 + profile["workout_days"] * 0.08)), 1)

    profile.update(
        {
            "calories": calories,
            "protein_g": protein_g,
            "step_target": step_target,
            "water_l": water_l,
        }
    )
    macros = _macro_plan(calories, protein_g, profile["weight_kg"], profile["goal"], profile["diet_type"])
    profile.update(macros)
    focus = _focus(profile)

    response = {
        "summary": {
            "goal": GOAL_LABELS[profile["goal"]],
            "activity": ACTIVITY_LABELS[profile["activity"]],
            "focus": focus,
            "readiness_score": _habit_score(profile),
            "model": b.get("selected_model"),
            "model_mae_calories": round(float(b.get("mae_calories", 0)), 1),
            "model_mae_protein": round(float(b.get("mae_protein", 0)), 1),
        },
        "body": {
            "bmi": round(profile["bmi"], 1),
            "bmi_category": _bmi_category(profile["bmi"]),
            "bmr": round(bmr),
            "calories_per_day": calories,
            "estimated_weekly_change": focus["pace"],
        },
        "nutrition": {
            "calories_per_day": calories,
            "macros": macros,
            "water_l": water_l,
            "meal_plan": _meal_plan(profile, macros),
            "grocery_list": _grocery_list(profile),
            "notes": [
                "Use the target as a starting estimate; adjust after 2 weeks of trend data.",
                "Keep most meals simple enough to repeat on busy days.",
                f"Cuisine preference noted: {profile['cuisine']}. Swap foods within the same protein/carb/fat role.",
            ],
        },
        "training": {
            "weekly_schedule": _weekly_schedule(profile),
            "progression": [
                "Week 1: leave 2-3 reps in reserve on strength sets.",
                "Week 2: add one set to two main movements if recovery is fine.",
                "Week 3: add a small load, slower tempo, or 5 cardio minutes.",
                "Week 4: deload 20% and review measurements.",
            ],
            "step_target": step_target,
        },
        "monthly_plan": _monthly_schedule(profile),
        "daily_routine": [
            {"time": "Morning", "task": "Water, light exposure, 5-10 min walk or mobility."},
            {"time": "Work block", "task": "Stand or walk 3-5 min each hour; keep a protein-forward lunch."},
            {"time": "Training window", "task": f"{profile['available_minutes']} min planned session or step target fallback."},
            {"time": "Evening", "task": "Prepare one meal component and reduce screens before bed."},
        ],
        "habits": _habits(profile),
        "safety_flags": _safety(profile),
        "references": b.get("references", []),
    }
    return jsonify(response)
