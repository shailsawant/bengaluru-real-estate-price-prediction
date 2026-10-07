from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PREPARED_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prepared_housing_data.csv"
)


TRAIN_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_data.csv"
)


TEST_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_data.csv"
)


df = pd.read_csv(PREPARED_DATA_FILE)


train_data, test_data = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["area_type_clean"],
)


train_data.to_csv(
    TRAIN_DATA_FILE,
    index=False
)


test_data.to_csv(
    TEST_DATA_FILE,
    index=False
)


print("\nDATA SPLIT SUMMARY")

print(
    "Complete prepared data:",
    len(df)
)

print(
    "Training data:",
    len(train_data)
)

print(
    "Test data:",
    len(test_data)
)


print("\nTRAINING AREA-TYPE PERCENTAGES")

print(
    train_data["area_type_clean"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


print("\nTEST AREA-TYPE PERCENTAGES")

print(
    test_data["area_type_clean"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)