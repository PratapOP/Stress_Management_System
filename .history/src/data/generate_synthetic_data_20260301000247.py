import os
import numpy as np
import pandas as pd

np.random.seed(42)

OUTPUT_PATH = "data/raw/synthetic_stress_dataset.csv"
N_SAMPLES = 5000

EMOTIONS = ["angry","fear","sad","neutral","happy","surprise"]


def normalize(arr):
    arr = np.maximum(arr, 0.001)
    return arr / arr.sum()


rows = []

for _ in range(N_SAMPLES):

    # -------- RANDOM HUMAN FEATURES --------
    sleep_hours = np.random.uniform(3, 9)
    sleep_quality = np.random.randint(1, 6)

    workload_hours = np.random.uniform(1, 12)
    assignment_pressure = np.random.randint(1, 6)

    physical_activity = np.random.uniform(0, 120)
    screen_time = np.random.uniform(1, 12)
    social_interaction = np.random.uniform(0, 5)

    caffeine_intake = np.random.randint(0, 6)

    eye_ratio = np.clip(
    0.35 - (workload_hours / 40) + np.random.normal(0, 0.02),
    0.18,
    0.35
)
    mouth_ratio = np.random.uniform(0.2, 0.6)

    # emotions random but realistic
    emo_base = np.random.rand(6)

    # subtle stress correlation
    emo_base[2] += workload_hours / 12   # sad increases with workload
    emo_base[4] += sleep_hours / 9       # happy increases with sleep
    emotions = normalize(emo_base)

    # -------- STRESS SCORE (HIDDEN) --------
    stress_score = (
        0.35 * workload_hours
        + 0.30 * assignment_pressure
        + 0.25 * screen_time
        - 0.35 * sleep_hours
        - 0.25 * (physical_activity / 30)
        - 0.20 * social_interaction
        + 0.30 * emotions[2]   # sad
        + 0.20 * emotions[0]   # angry
        - 0.25 * emotions[4]   # happy
    )

    # BIGGER HUMAN VARIATION
    stress_score += np.random.normal(0, 1.2)

    # -------- LABEL CREATION --------
    if stress_score < 2:
        stress_level = 0
    elif stress_score < 5:
        stress_level = 1
    else:
        stress_level = 2

    row = {
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
        "stress_level": stress_level
    }

    for i,e in enumerate(EMOTIONS):
        row[e] = emotions[i]

    rows.append(row)


df = pd.DataFrame(rows)

# balance classes roughly
df = df.groupby("stress_level").sample(
    n=min(df["stress_level"].value_counts()),
    random_state=42
)

df = df.sample(frac=1).reset_index(drop=True)

os.makedirs("data/raw", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(df["stress_level"].value_counts())