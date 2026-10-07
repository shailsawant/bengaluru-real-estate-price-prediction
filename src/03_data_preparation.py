from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "raw" / "Bengaluru_House_Data.csv"

PROCESSED_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prepared_housing_data.csv"
)


REMOVED_RECORDS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "removed_records_initial.csv"
)

df = pd.read_csv(DATA_FILE)

UNIT_TO_SQFT = {
    "Sq. Yards": 9,
    "Sq. Meter": 10.7639,
    "Acres": 43560,
    "Cents": 435.6,
    "Perch": 272.25,
    "Grounds": 2400,
    "Guntha": 1089,
    "Ares": 1076.39,
    "Bigha": 27225,
    "Kottah": 720,
    "Marla": 272.25,
    "Kanal": 5445,
    "Hectares": 107639.104,
    "Square Feet": 1,
}

def convert_to_sqft(value: str) -> float:
    """
    Convert a total_sqft value to square feet.
    If the value is a range, return the average of the two numbers.
    If the value has a unit, convert it to square feet.
    If the value cannot be converted, return np.nan.
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    # Check for range values
    if "-" in value:
        try:
            low, high = map(float, value.split("-"))
            return (low + high) / 2
        except ValueError:
            return np.nan

    # Check for unit values
    for unit, factor in UNIT_TO_SQFT.items():
        if value.endswith(unit):
            try:
                number = float(value.replace(unit, "").strip())
                return number * factor
            except ValueError:
                return np.nan

    # Try to convert directly to float
    try:
        return float(value)
    except ValueError:
        return np.nan

df["total_sqft_clean"] = (
    df["total_sqft"]
    .apply(convert_to_sqft)
    .round(2)
)

size_parts = df["size"].str.extract(
    r"^\s*(\d+)\s*(.+?)\s*$"
)

size_parts.columns = [
    "bedrooms",
    "layout_type"
]

df["bedrooms"] = pd.to_numeric(
    size_parts["bedrooms"],
    errors="coerce"
)

df["sqft_per_bedroom"] = (
    df["total_sqft_clean"]
    / df["bedrooms"]
)

print("\n20 LOWEST AREAS PER BEDROOM")

print(
    df.nsmallest(
        20,
        "sqft_per_bedroom"
    )[
        [
            "area_type",
            "location",
            "size",
            "bedrooms",
            "total_sqft",
            "total_sqft_clean",
            "sqft_per_bedroom",
            "bath",
            "price",
        ]
    ].to_string(index=False)
)

print("\nAREA-PER-BEDROOM GROUPS")

print(
    "Below 100 sqft per bedroom:",
    (df["sqft_per_bedroom"] < 100).sum()
)

print(
    "Below 200 sqft per bedroom:",
    (df["sqft_per_bedroom"] < 200).sum()
)

print(
    "Below 300 sqft per bedroom:",
    (df["sqft_per_bedroom"] < 300).sum()
)


print("\nAREA PER BEDROOM BY AREA TYPE")
area_type_summary = (
    df.groupby("area_type")
    ["sqft_per_bedroom"]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
)

print(area_type_summary.to_string())
print("\nLOW AREA-PER-BEDROOM COUNTS BY AREA TYPE")
for area_type in df["area_type"].unique():

    area_type_rows = df[
        df["area_type"] == area_type
    ]

    print("\n", area_type)

    print(
        "Total records:",
        len(area_type_rows)
    )

    print(
        "Below 100 sqft per bedroom:",
        (
            area_type_rows["sqft_per_bedroom"]
            < 100
        ).sum()
    )

    print(
        "Below 200 sqft per bedroom:",
        (
            area_type_rows["sqft_per_bedroom"]
            < 200
        ).sum()
    )

    print(
        "Below 300 sqft per bedroom:",
        (
            area_type_rows["sqft_per_bedroom"]
            < 300
        ).sum()
    )


df["bathroom_difference"] = (
    df["bath"] - df["bedrooms"]
)

df["bathrooms_per_bedroom"] = (
    df["bath"] / df["bedrooms"]
)

print("\nMISSING BATHROOM VALUES")

print(
    df.loc[
        df["bath"].isna(),
        [
            "area_type",
            "location",
            "size",
            "bedrooms",
            "total_sqft_clean",
            "bath",
            "price",
        ]
    ].to_string(index=False)
)


print("\nBATHROOM SUMMARY")

print(df["bath"].describe())


print("\nBATHROOM DIFFERENCE SUMMARY")

print(df["bathroom_difference"].describe())


print("\nBATHROOM BUSINESS-RULE COUNTS")

print(
    "Bathrooms greater than bedrooms:",
    (df["bath"] > df["bedrooms"]).sum()
)

print(
    "Bathrooms more than bedrooms + 1:",
    (
        df["bath"]
        > df["bedrooms"] + 1
    ).sum()
)

print(
    "Bathrooms more than bedrooms + 2:",
    (
        df["bath"]
        > df["bedrooms"] + 2
    ).sum()
)


print("\nLARGEST BATHROOM DIFFERENCES")

print(
    df.nlargest(
        25,
        "bathroom_difference"
    )[
        [
            "area_type",
            "location",
            "size",
            "bedrooms",
            "total_sqft_clean",
            "bath",
            "bathroom_difference",
            "bathrooms_per_bedroom",
            "price",
        ]
    ].to_string(index=False)
)


df["price_per_sqft"] = (
    df["price"] * 100000
    / df["total_sqft_clean"]
)


print("\nPRICE PER SQFT SUMMARY")

print(
    df["price_per_sqft"]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
)


print("\n20 LOWEST PRICE-PER-SQFT RECORDS")

print(
    df.nsmallest(
        20,
        "price_per_sqft"
    )[
        [
            "area_type",
            "location",
            "size",
            "total_sqft",
            "total_sqft_clean",
            "bath",
            "price",
            "price_per_sqft",
        ]
    ].to_string(index=False)
)


print("\n20 HIGHEST PRICE-PER-SQFT RECORDS")

print(
    df.nlargest(
        20,
        "price_per_sqft"
    )[
        [
            "area_type",
            "location",
            "size",
            "total_sqft",
            "total_sqft_clean",
            "bath",
            "price",
            "price_per_sqft",
        ]
    ].to_string(index=False)
)


print("\nPRICE-PER-SQFT GROUP COUNTS")

print(
    "Below ₹100 per sqft:",
    (df["price_per_sqft"] < 100).sum()
)

print(
    "Below ₹1,000 per sqft:",
    (df["price_per_sqft"] < 1000).sum()
)

print(
    "Above ₹50,000 per sqft:",
    (df["price_per_sqft"] > 50000).sum()
)

print(
    "Above ₹100,000 per sqft:",
    (df["price_per_sqft"] > 100000).sum()
)

print("\nLOCATION COUNT BEFORE CLEANING")

print(
    df["location"]
    .nunique(dropna=True)
)


df["location_clean"] = (
    df["location"]
    .str.strip()
    .str.replace(
        r"\s+",
        " ",
        regex=True
    )
)


print("\nLOCATION COUNT AFTER CLEANING")

print(
    df["location_clean"]
    .nunique(dropna=True)
)


location_variations = (
    df.groupby("location_clean")
    ["location"]
    .agg(
        record_count="size",
        original_variations="nunique",
        original_values=lambda values: sorted(
            values.dropna().unique().tolist()
        )
    )
)


location_variations = location_variations[
    location_variations["original_variations"] > 1
]


print("\nLOCATIONS MERGED BY CLEANING")

print(
    location_variations
    .sort_values(
        "original_variations",
        ascending=False
    )
    .to_string()
)


print("\nMISSING CLEANED LOCATIONS")

print(
    df.loc[
        df["location_clean"].isna(),
        [
            "area_type",
            "location",
            "size",
            "total_sqft_clean",
            "bath",
            "price",
        ]
    ]
)


print("\nTOP 20 CLEANED LOCATIONS")

print(
    df["location_clean"]
    .value_counts(dropna=False)
    .head(20)
)

for value in df.loc[
    df["location_clean"] == "Anekal",
    "location"
].unique():
    print(repr(value))

print("\nAVAILABILITY VALUES")

print(
    df["availability"]
    .value_counts(dropna=False)
    .to_string()
)


df["availability_clean"] = (
    df["availability"]
    .str.strip()
    .str.replace(
        r"\s+",
        " ",
        regex=True
    )
)


ready_values = [
    "Ready To Move",
    "Immediate Possession",
]


df["availability_status"] = np.where(
    df["availability_clean"].isin(ready_values),
    "Ready",
    "Under Construction"
)


print("\nAVAILABILITY STATUS COUNTS")

print(
    df["availability_status"]
    .value_counts(dropna=False)
)


print("\nORIGINAL VALUES BY STATUS")

print(
    df.groupby("availability_status")
    ["availability_clean"]
    .unique()
)

print("\nSOCIETY AUDIT")

print(
    "Missing society values:",
    df["society"].isna().sum()
)

print(
    "Unique known societies:",
    df["society"].nunique()
)

print("\nTOP 20 SOCIETIES")

print(
    df["society"]
    .value_counts(dropna=False)
    .head(20)
)


print("\nBALCONY AUDIT")

print(
    df["balcony"]
    .value_counts(dropna=False)
    .sort_index()
)


print("\nMISSING BALCONY BY AREA TYPE")

print(
    df.assign(
        balcony_missing=df["balcony"].isna()
    )
    .groupby("area_type")
    ["balcony_missing"]
    .agg(
        total_records="size",
        missing_balcony="sum",
        missing_percent="mean"
    )
    .assign(
        missing_percent=lambda table:
        table["missing_percent"] * 100
    )
)


print("\nMISSING BALCONY BY BEDROOM COUNT")

print(
    df.assign(
        balcony_missing=df["balcony"].isna()
    )
    .groupby("bedrooms")
    ["balcony_missing"]
    .agg(
        total_records="size",
        missing_balcony="sum",
        missing_percent="mean"
    )
    .assign(
        missing_percent=lambda table:
        table["missing_percent"] * 100
    )
)

quality_flags = pd.DataFrame(
    index=df.index
)


quality_flags["missing_location"] = (
    df["location_clean"].isna()
)


quality_flags["missing_bedrooms"] = (
    df["bedrooms"].isna()
)


quality_flags["area_below_100"] = (
    df["total_sqft_clean"] < 100
)


quality_flags["area_above_50000"] = (
    df["total_sqft_clean"] > 50000
)


quality_flags["area_per_bedroom_below_100"] = (
    df["sqft_per_bedroom"] < 100
)


quality_flags["bedrooms_above_10"] = (
    df["bedrooms"] > 10
)


quality_flags["bathrooms_above_bedrooms_plus_2"] = (
    df["bath"] > df["bedrooms"] + 2
)


quality_flags["price_per_sqft_below_100"] = (
    df["price_per_sqft"] < 100
)


quality_flags["price_per_sqft_above_100000"] = (
    df["price_per_sqft"] > 100000
)


print("\nQUALITY FLAG COUNTS")

print(
    quality_flags
    .sum()
    .sort_values(ascending=False)
)


df["quality_flag_count"] = (
    quality_flags.sum(axis=1)
)


print("\nROWS WITH AT LEAST ONE FLAG")

print(
    (df["quality_flag_count"] > 0).sum()
)


print("\nNUMBER OF FLAGS PER ROW")

print(
    df["quality_flag_count"]
    .value_counts()
    .sort_index()
)


flagged_rows = df.loc[
    df["quality_flag_count"] > 0,
    [
        "area_type",
        "location_clean",
        "size",
        "bedrooms",
        "total_sqft",
        "total_sqft_clean",
        "sqft_per_bedroom",
        "bath",
        "price",
        "price_per_sqft",
        "quality_flag_count",
    ]
]


print("\nROWS WITH THREE OR MORE FLAGS")

print(
    flagged_rows.loc[
        flagged_rows["quality_flag_count"] >= 3
    ]
    .sort_values(
        "quality_flag_count",
        ascending=False
    )
    .to_string(index=False)
)

df["layout_type"] = (
    size_parts["layout_type"]
    .str.upper()
    .str.strip()
)

print("\nBEDROOM EXTRACTION EXAMPLES")
print(
    df[
        [
            "size",
            "bedrooms",
            "layout_type"
        ]
    ]
    .drop_duplicates()
    .sort_values("bedrooms")
    .to_string(index=False)
)

print("\nFAILED BEDROOM EXTRACTIONS")
print(
    df.loc[
        df["bedrooms"].isna(),
        ["size"]
    ]
)

print("\nBEDROOM SUMMARY")

print(df["bedrooms"].describe())


print("\nLAYOUT TYPES")

print(
    df["layout_type"]
    .value_counts(dropna=False)
)

columns_to_show = [
    "area_type",
    "location",
    "size",
    "total_sqft",
    "total_sqft_clean",
    "bath",
    "price",
]

smallest_areas = df.nsmallest(
    20,
    "total_sqft_clean"
)

largest_areas = df.nlargest(
    20,
    "total_sqft_clean"
)

print("\n20 SMALLEST PROPERTIES")

print(
    smallest_areas[
        columns_to_show
    ].to_string(index=False)
)


print("\n20 LARGEST PROPERTIES")

print(
    largest_areas[
        columns_to_show
    ].to_string(index=False)
)

print("\nAREA SIZE GROUPS")

print(
    "Below 100 sqft:",
    (df["total_sqft_clean"] < 100).sum()
)

print(
    "Between 100 and 300 sqft:",
    (
        (df["total_sqft_clean"] >= 100)
        & (df["total_sqft_clean"] < 300)
    ).sum()
)

print(
    "Above 10,000 sqft:",
    (df["total_sqft_clean"] > 10000).sum()
)

print(
    "Above 50,000 sqft:",
    (df["total_sqft_clean"] > 50000).sum()
)

print("\nCONVERSION EXAMPLES")
example_values = [
    "1056",
    "2100 - 2850",
    "34.46Sq. Meter",
    "1100Sq. Yards",
    "5.31Acres",
    "24Guntha",
    "3Cents",
    "4125Perch",
    "1Grounds",
]

for value in example_values:
    print(f"{value} -> {convert_to_sqft(value)}")


print("\nFAILED CONVERSIONS")
print(df["total_sqft_clean"].isna().sum())

print("\nFAILED RECORDS")
print(
    df.loc[
        df["total_sqft_clean"].isna(),
        ["total_sqft"]
    ]
)

print("\nCLEANED AREA SUMMARY")
print(df["total_sqft_clean"].describe())


df["area_type_clean"] = (
    df["area_type"]
    .str.strip()
    .str.replace(
        r"\s+",
        " ",
        regex=True
    )
)


hard_removal_flags = [
    "missing_location",
    "missing_bedrooms",
    "area_below_100",
]


df["removal_reason"] = ""


for flag_name in hard_removal_flags:

    rows_with_flag = quality_flags[flag_name]

    df.loc[
        rows_with_flag,
        "removal_reason"
    ] += flag_name + "; "


hard_removal_mask = (
    quality_flags[
        hard_removal_flags
    ]
    .any(axis=1)
)


removed_records = df.loc[
    hard_removal_mask
].copy()


prepared_data = df.loc[
    ~hard_removal_mask,
    [
        "area_type_clean",
        "availability_status",
        "location_clean",
        "layout_type",
        "total_sqft_clean",
        "bedrooms",
        "bath",
        "balcony",
        "price",
    ]
].copy()


PROCESSED_DATA_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


REMOVED_RECORDS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


prepared_data.to_csv(
    PROCESSED_DATA_FILE,
    index=False
)


removed_records.to_csv(
    REMOVED_RECORDS_FILE,
    index=False
)


print("\nINITIAL PREPARATION SUMMARY")

print(
    "Original records:",
    len(df)
)

print(
    "Removed records:",
    len(removed_records)
)

print(
    "Prepared records:",
    len(prepared_data)
)


print("\nREMOVAL REASONS")

print(
    removed_records["removal_reason"]
    .value_counts()
)


print("\nPREPARED DATA MISSING VALUES")

print(
    prepared_data.isna().sum()
)


print("\nPREPARED DATA FILE")

print(PROCESSED_DATA_FILE)