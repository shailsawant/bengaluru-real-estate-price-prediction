from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "Bengaluru_House_Data.csv"

df = pd.read_csv(DATA_FILE)

sqft_as_number = pd.to_numeric(
    df["total_sqft"], 
    errors="coerce"
)

problem_sqft = df.loc[
    sqft_as_number.isna() & df["total_sqft"].notna(),
    ["total_sqft"]
].copy()

print("\nTOTAL ROWS")
print(len(df))

print("\nDIRECTLY CONVERTIBLE VALUES")
print(sqft_as_number.notna().sum())

print("\nVALUES REQUIRING SPECIAL CLEANING")
print(len(problem_sqft))

print("\nUNIQUE PROBLEMATIC VALUES")
print(problem_sqft["total_sqft"].nunique())

print("\nPROBLEMATIC VALUES AND COUNTS")
print(
    problem_sqft["total_sqft"]
    .value_counts()
    .to_string()
)

# This selects the problematic total_sqft column.
# .str.strip() removes spaces from the beginning and end:
# " 2100 - 2850 " → "2100 - 2850"
# It does not remove spaces inside the value.
problem_values = (
    problem_sqft["total_sqft"].str.strip()
)

is_range = problem_values.str.fullmatch(
    r"\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?"
)

range_values = problem_values[is_range]

unit_values = problem_values[~is_range]

units = (
    unit_values
    .str.extract(
        r"^[\d.]+\s*(.+)$",
        expand=False
    )
    .str.strip()
)

print("\nNUMBER OF RANGE VALUES")
print(is_range.sum())

print("\nNUMBER OF VALUES WITH UNITS")
print((~is_range).sum())

print("\nMEASUREMENT UNITS FOUND")
print(units.value_counts())
