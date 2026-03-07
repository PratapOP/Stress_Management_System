import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# ==========================
# Load Dataset
# ==========================

data = pd.read_csv("data/processed/synthetic_stress_dataset.csv")

X = data.drop("stress_level", axis=1)
y = data["stress_level"]


# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==========================
# Define Models
# ==========================

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}


# ==========================
# Evaluate Models
# ==========================

print("\n===== MODEL PERFORMANCE =====\n")

results = []

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    results.append((name, accuracy, f1))

    print(f"{name} -> Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}\n")


# ==========================
# Display Table
# ==========================

print("===== FINAL RESULTS TABLE =====")

for model, acc, f1 in results:
    print(f"{model} | Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")