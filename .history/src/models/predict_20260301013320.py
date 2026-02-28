import joblib
import pandas as pd

from src.recommendation.stress_advice import get_stress_advice

MODEL_PATH = "models/stress_classifier.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_columns.pkl"


LABEL_MAP = {
    0: "LOW",
    1: "MODERATE",
    2: "HIGH"
}


def predict_stress(input_features):

    # Load artifacts
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    # Create dataframe in correct order
    df = pd.DataFrame([input_features])
    df = df[feature_columns]

    # Scale
    X_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=feature_columns
    )
    # Prediction
    pred = model.predict(X_scaled)[0]

    # ---------- TOP 5 FACTORS ----------
    top_factors = []

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        feat_imp = sorted(
            zip(feature_columns, importances),
            key=lambda x: x[1],
            reverse=True
        )

        top_factors = feat_imp[:5]

    # ---------- ADVICE ----------
    advice = get_stress_advice(input_features, pred)

    return {
        "stress_level": LABEL_MAP[pred],
        "top_factors": top_factors,
        "advice": advice
    }


# ===========================
# TEST RUN (example input)
# ===========================
if __name__ == "__main__":

    sample_input = {
        "sleep_hours": 5.2,
        "sleep_quality": 2,
        "workload_hours": 9,
        "assignment_pressure": 4,
        "physical_activity": 10,
        "screen_time": 9,
        "social_interaction": 1,
        "caffeine_intake": 3,
        "eye_ratio": 0.22,
        "mouth_ratio": 0.30,
        "angry": 0.20,
        "fear": 0.10,
        "sad": 0.35,
        "neutral": 0.20,
        "happy": 0.10,
        "surprise": 0.05
    }

    result = predict_stress(sample_input)

    print("\n===== PREDICTION RESULT =====")
    print("Stress Level:", result["stress_level"])

    print("\nTop 5 Factors:")
    for f, v in result["top_factors"]:
        print(f"{f} -> {v:.4f}")

    print("\nAdvice:")
    for a in result["advice"]:
        print("-", a)