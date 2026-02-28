import os
import numpy as np
import pandas as pd

# ==========================
# CONFIG
# ==========================
TOTAL_SAMPLES = 5000
OUTPUT_PATH = "data/raw/synthetic_stress_dataset.csv"
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

CLASS_COUNTS = {
    0: 1667,   # Low
    1: 1666,   # Moderate
    2: 1667    # High
}

EMOTIONS = ["angry", "fear", "sad", "neutral", "happy", "surprise"]


# ==========================
# HELPERS
# ==========================

def clip(val, low, high):
    return np.clip(val, low, high)


def normalize_emotions(arr):
    arr = np.maximum(arr, 0.001)
    return arr / np.sum(arr)


def generate_emotions(stress_level):
    """
    Emotion bias based on stress level
    """
    if stress_level == 0:  # Low
        base = np.array([0.08, 0.07, 0.08, 0.30, 0.40, 0.07])

    elif stress_level == 1:  # Moderate
        base = np.array([0.12, 0.10, 0.15, 0.35, 0.18, 0.10])

    else:  # High
        base = np.array([0.25, 0.15, 0.30, 0.20, 0.05, 0.05])

    noise = np.random.normal(0, 0.03, size=6)
    return normalize_emotions(base + noise)


def generate_sample(stress_level):

    # ---------- ROUTINE FEATURES ----------
    if stress_level == 0:  # LOW
        sleep_hours = np.random.normal(7.8, 0.6)
        sleep_quality = np.random.randint(4, 6)
        workload_hours = np.random.normal(3.5, 1.2)
        assignment_pressure = np.random.randint(1, 3)
        physical_activity = np.random.normal(60, 20)
        screen_time = np.random.normal(4, 1.5)
        social_interaction = np.random.normal(3, 1)
        caffeine_intake = np.random.randint(0, 2)

        eye_ratio = np.random.normal(0.31, 0.02)
        mouth_ratio = np.random.normal(0.35, 0.05)

    elif stress_level == 1:  # MODERATE
        sleep_hours = np.random.normal(6.2, 0.7)
        sleep_quality = np.random.randint(2, 5)
        workload_hours = np.random.normal(6.5, 1.5)
        assignment_pressure = np.random.randint(2, 4)
        physical_activity = np.random.normal(35, 15)
        screen_time = np.random.normal(6.5, 1.5)
        social_interaction = np.random.normal(2, 1)
        caffeine_intake = np.random.randint(1, 4)

        eye_ratio = np.random.normal(0.27, 0.02)
        mouth_ratio = np.random.normal(0.33, 0.06)

    else:  # HIGH
        sleep_hours = np.random.normal(4.8, 0.8)
        sleep_quality = np.random.randint(1, 3)
        workload_hours = np.random.normal(9.5, 1.2)
        assignment_pressure = np.random.randint(4, 6)
        physical_activity = np.random.normal(10, 8)
        screen_time = np.random.normal(9, 1.5)
        social_interaction = np.random.normal(0.8, 0.6)
        caffeine_intake = np.random.randint(3, 6)

        eye_ratio = np.random.normal(0.23, 0.02)
        mouth_ratio = np.random.normal(0.30, 0.07)

    # ---------- CLIPPING ----------
    sleep_hours = clip(sleep_hours, 3, 9)
    workload_hours = clip(workload_hours, 1, 12)
    physical_activity = clip(physical_activity, 0, 120)
    screen_time = clip(screen_time, 1, 12)
    social_interaction = clip(social_interaction, 0, 5)

    eye_ratio = clip(eye_ratio, 0.18, 0.35)
    mouth_ratio = clip(mouth_ratio, 0.20, 0.60)

    # ---------- EMOTIONS ----------
    emotions = generate_emotions(stress_level)

    # ---------- INTERNAL STRESS SCORE (hidden realism) ----------
    stress_score = (
        0.35 * workload_hours +
        0.30 * assignment_pressure +
        0.20 * screen_time -
        0.30 * sleep_hours -
        0.20 * physical_activity / 30 -
        0.15 * social_interaction +
        0.25 * emotions[2] +   # sad
        0.20 * emotions[0] -   # angry
        0.20 * emotions[4]     # happy
    )

    stress_score += np.random.normal(0, 2.0)

    sample = {
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "workload_hours": workload_hours,
        "assignment_pressure": assignment_pressure,
        "physical_activity": physical_activity,
        "screen_time": screen_time,
        "social_interaction": social_interaction,
        "caffeine_intake": caffeine_intake,
        "eye_ratio": eye_ratio,
        "mouth_ratio": mouth_ratio,
        "stress_score_internal": stress_score,  # REMOVE later before training
        "stress_level": stress_level
    }

    for i, emo in enumerate(EMOTIONS):
        sample[emo] = emotions[i]

    return sample


# ==========================
# DATASET GENERATION
# ==========================

def generate_dataset():
    rows = []

    for stress_level, count in CLASS_COUNTS.items():
        for _ in range(count):
            rows.append(generate_sample(stress_level))

    df = pd.DataFrame(rows)

    # Shuffle
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Dataset saved:", OUTPUT_PATH)
    print(df["stress_level"].value_counts())


if __name__ == "__main__":
    generate_dataset()