import joblib
import pandas as pd
import shap

from src.recommendation.stress_advice import get_stress_advice
from src.llm.llama_reasoner import generate_ai_report

MODEL_PATH = "models/stress_classifier.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_columns.pkl"

LABEL_MAP = {
    0: "LOW",
    1: "MODERATE",
    2: "HIGH"
}

# ---------- LOAD MODEL ONCE ----------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)

explainer = shap.TreeExplainer(model)


def predict_stress(input_features):

    # ---------- DATAFRAME ----------
    df = pd.DataFrame([input_features])
    df = df[feature_columns]

    # ---------- SCALE ----------
    X_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=feature_columns
    )

    # ---------- PREDICTION ----------
    pred = model.predict(X_scaled)[0]
    stress_label = LABEL_MAP[pred]

    # ---------- SHAP EXPLANATION ----------
    shap_values = explainer.shap_values(X_scaled)

    if isinstance(shap_values, list):
        class_shap = shap_values[pred][0]
    else:
        class_shap = shap_values[0, pred]

    contributions = list(zip(feature_columns, class_shap))
    contributions = sorted(
        contributions,
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_factors = contributions[:5]

    # ---------- ADVICE ----------
    advice = get_stress_advice(input_features, pred)

    # ---------- LLM REPORT ----------
    shap_summary = [
        f"{feature}: {value:.4f}"
        for feature, value in top_factors
    ]

    try:
        ai_report = generate_ai_report(
            stress_label,
            shap_summary,
            input_features,
            {
                "eye_ratio": input_features["eye_ratio"],
                "neutral": input_features["neutral"],
                "happy": input_features["happy"],
                "sad": input_features["sad"]
            }
        )
    except Exception as e:
        ai_report = "AI reasoning engine unavailable. Please ensure Ollama is running."

    return {
        "stress_level": stress_label,
        "top_factors": top_factors,
        "advice": advice,
        "ai_report": ai_report
    }


# ---------- TEST ----------
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

    print("\nTop Factors (SHAP):")
    for f, v in result["top_factors"]:
        print(f"{f}: {v:.4f}")

    print("\nAI REPORT:\n")
    print(result["ai_report"])