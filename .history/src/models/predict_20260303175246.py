import joblib
import pandas as pd
import shap

from src.llm.llama_reasoner import generate_ai_report

MODEL_PATH = "models/stress_classifier.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_columns.pkl"

LABEL_MAP = {
    0: "LOW",
    1: "MODERATE",
    2: "HIGH"
}

# Load once
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)

explainer = shap.TreeExplainer(model)


def predict_stress(input_features, name, age):

    df = pd.DataFrame([input_features])
    df = df[feature_columns]

    X_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=feature_columns
    )

    pred = model.predict(X_scaled)[0]
    stress_label = LABEL_MAP[pred]

    # SHAP explanation
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

    shap_summary = [
        f"{feature}: {value:.4f}"
        for feature, value in top_factors
    ]

    try:
        ai_report = generate_ai_report(
            name,
            age,
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
    except Exception:
        ai_report = "AI reasoning engine unavailable. Please ensure Ollama is running."

    return {
        "stress_level": stress_label,
        "top_factors": top_factors,
        "ai_report": ai_report
    }