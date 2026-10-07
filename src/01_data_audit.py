from pathlib import Path

import pandas as pd


# __file__ is this Python file. parents[1] moves up from src/ to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "Bengaluru_House_Data.csv"


# Load the raw CSV without changing it.
df = pd.read_csv(DATA_FILE)


print("\n1. DATASET SIZE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n2. COLUMN NAMES")
print(df.columns.tolist())

print("\n3. FIRST FIVE ROWS")
print(df.head())

print("\n4. DATA TYPES AND NON-NULL COUNTS")
df.info()

print("\n5. MISSING VALUES")
missing_report = pd.DataFrame(
    {
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
    }
).sort_values("missing_count", ascending=False)
print(missing_report)

print("\n6. EXACT DUPLICATE ROWS")
print(df.duplicated().sum())

print("\n7. UNIQUE VALUES IN EACH COLUMN")
print(df.nunique(dropna=False).sort_values(ascending=False))

print("\n8. NUMERIC COLUMN SUMMARY")
print(df.describe().T)

print("\n9. TEXT COLUMN SUMMARY")
# print(df.describe(include="object").T)
print(df.describe(include="str").T)

print("\n10. SAMPLE SIZE VALUES")
print(df["size"].value_counts(dropna=False).head(20))

print("\n11. SAMPLE TOTAL_SQFT VALUES")
print(df["total_sqft"].drop_duplicates().head(30).tolist())

print("\n12. LOCATION FREQUENCY")
print(df["location"].value_counts(dropna=False).head(20))

