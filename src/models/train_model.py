import pandas as pd
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, f1_score

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

MODEL_SAVE_PATH = "models/stress_classifier.pkl"


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=["stress_level"])
    y_train = train_df["stress_level"]

    X_test = test_df.drop(columns=["stress_level"])
    y_test = test_df["stress_level"]

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
        "svm": SVC(kernel="rbf")
    }

    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def evaluate_models(models, X_test, y_test):

    best_model = None
    best_f1 = -1

    print("\n===== MODEL RESULTS =====")

    for name, model in models.items():

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        print(f"{name} -> Accuracy: {acc:.4f} | F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    return best_model


def main():

    X_train, X_test, y_train, y_test = load_data()

    models = train_models(X_train, y_train)

    best_model = models["random_forest"]
    print("\nUsing Random Forest for explainability.")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_SAVE_PATH)

    print("\nBest model saved:", MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()