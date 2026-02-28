import pandas as pd
import numpy as np

DATA_PATH = r"C:\Users\mksme\Desktop\PROJECTS I am Working On\New folder (2)\data\raw\synthetic_stress_dataset.csv"

EMOTIONS = ["angry", "fear", "sad", "neutral", "happy", "surprise"]


def validate_dataset():

    df = pd.read_csv(DATA_PATH)

    print("\n===== BASIC INFO =====")
    print("Shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())

    # -------------------------
    # LABEL DISTRIBUTION
    # -------------------------
    print("\n===== STRESS DISTRIBUTION =====")
    print(df["stress_level"].value_counts())

    # -------------------------
    # RANGE CHECKS
    # -------------------------
    print("\n===== RANGE CHECKS =====")

    checks = {
        "sleep_hours": (3, 9),
        "workload_hours": (1, 12),
        "physical_activity": (0, 120),
        "screen_time": (1, 12),
        "social_interaction": (0, 5),
        "eye_ratio": (0.18, 0.35),
        "mouth_ratio": (0.20, 0.60)
    }

    for col, (low, high) in checks.items():
        min_val = df[col].min()
        max_val = df[col].max()

        print(f"{col}: min={min_val:.3f}, max={max_val:.3f}")

        if min_val < low or max_val > high:
            print("WARNING:", col, "out of expected range!")

    # -------------------------
    # EMOTION SUM CHECK
    # -------------------------
    print("\n===== EMOTION NORMALIZATION =====")

    emotion_sum = df[EMOTIONS].sum(axis=1)
    max_error = np.abs(emotion_sum - 1).max()

    print("Max normalization error:", max_error)

    if max_error > 0.01:
        print("WARNING: Emotion probabilities not normalized!")

    # -------------------------
    # INTERNAL FEATURE CHECK
    # -------------------------
    print("\n===== INTERNAL FEATURES =====")
    if "stress_score_internal" in df.columns:
        print("Found stress_score_internal (GOOD)")
    else:
        print("WARNING: stress_score_internal missing!")

    print("\nValidation completed successfully.")


if __name__ == "__main__":
    validate_dataset()