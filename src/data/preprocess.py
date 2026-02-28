import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

RAW_PATH = "data/raw/synthetic_stress_dataset.csv"

PROCESSED_DIR = "data/processed"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_columns.pkl"


def preprocess():

    # Load data
    df = pd.read_csv(RAW_PATH)

    # Remove leakage feature
    if "stress_score_internal" in df.columns:
        df = df.drop(columns=["stress_score_internal"])

    # Separate target
    X = df.drop(columns=["stress_level"])
    y = df["stress_level"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Save folders
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Save scaled full dataset temporarily
    full_df = X_scaled.copy()
    full_df["stress_level"] = y

    full_df.to_csv(f"{PROCESSED_DIR}/scaled_full.csv", index=False)

    # Save scaler + feature order
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(list(X.columns), FEATURES_PATH)

    print("Preprocessing completed.")
    print("Features:", len(X.columns))


if __name__ == "__main__":
    preprocess()