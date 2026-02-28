import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/scaled_full.csv"

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"


def split_data():

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["stress_level"])
    y = df["stress_level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_df = X_train.copy()
    train_df["stress_level"] = y_train

    test_df = X_test.copy()
    test_df["stress_level"] = y_test

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)


if __name__ == "__main__":
    split_data()