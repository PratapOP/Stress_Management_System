import joblib
import pandas as pd

MODEL_PATH = "models/stress_classifier.pkl"
TRAIN_PATH = "data/processed/train.csv"


def show_feature_importance():

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(TRAIN_PATH)
    X = df.drop(columns=["stress_level"])

    # Works for Random Forest
    if hasattr(model, "feature_importances_"):

        importance = model.feature_importances_

        feat_df = pd.DataFrame({
            "feature": X.columns,
            "importance": importance
        })

        feat_df = feat_df.sort_values(
            by="importance",
            ascending=False
        )

        print("\n===== FEATURE IMPORTANCE =====")
        print(feat_df)

    else:
        print("Model does not support feature importance.")


if __name__ == "__main__":
    show_feature_importance()