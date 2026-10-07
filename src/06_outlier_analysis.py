from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------
# 1. Build project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAINING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_data.csv"
)

FIGURES_DIRECTORY = PROJECT_ROOT / "reports" / "figures"

FIGURES_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# 2. Configure pandas display
# ---------------------------------------------------------

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.2f}".format)


# ---------------------------------------------------------
# 3. Load training data only
# ---------------------------------------------------------

train_data = pd.read_csv(TRAINING_DATA_PATH)

print("\nTRAINING RECORDS")
print(len(train_data))


# ---------------------------------------------------------
# 4. Calculate price per square foot
#
# Price is stored in lakh rupees.
# 1 lakh = 100,000 rupees.
# ---------------------------------------------------------

train_data["price_per_sqft"] = (
    train_data["price"] * 100_000
) / train_data["total_sqft_clean"]


# ---------------------------------------------------------
# 5. Price-per-square-foot summary by area type
# ---------------------------------------------------------

price_per_sqft_summary = (
    train_data.groupby("area_type_clean")["price_per_sqft"]
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
    .round(2)
)

print("\nPRICE PER SQFT BY AREA TYPE")
print(price_per_sqft_summary)


# ---------------------------------------------------------
# 6. Find a display limit for the chart
#
# We use the 99th percentile only to make the chart readable.
# No records are removed from the training data here.
# ---------------------------------------------------------

price_per_sqft_99th = train_data["price_per_sqft"].quantile(0.99)

chart_data = train_data[
    train_data["price_per_sqft"] <= price_per_sqft_99th
].copy()

records_outside_chart = (
    train_data["price_per_sqft"] > price_per_sqft_99th
).sum()

print("\nPRICE-PER-SQFT CHART LIMIT")
print(f"99th percentile: {price_per_sqft_99th:.2f}")
print(f"Records displayed: {len(chart_data)}")
print(f"Records outside chart: {records_outside_chart}")


# ---------------------------------------------------------
# 7. Create boxplot by area type
# ---------------------------------------------------------

area_type_order = [
    "Super built-up Area",
    "Plot Area",
    "Built-up Area",
    "Carpet Area",
]

plt.figure(figsize=(14, 8))

sns.boxplot(
    data=chart_data,
    x="area_type_clean",
    y="price_per_sqft",
    order=area_type_order,
    showfliers=False,
)

plt.title(
    "Training Data: Price per Square Foot by Area Type",
    fontsize=16,
)

plt.xlabel("Area Type", fontsize=12)
plt.ylabel("Price per Square Foot (₹)", fontsize=12)

plt.xticks(rotation=15)
plt.tight_layout()

figure_path = (
    FIGURES_DIRECTORY
    / "04_price_per_sqft_by_area_type.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\nFIGURE SAVED TO")
print(figure_path)


# ---------------------------------------------------------
# 8. Create comparable property groups
#
# Properties are compared only when they have:
# - the same location
# - the same area type
# ---------------------------------------------------------

group_columns = [
    "location_clean",
    "area_type_clean",
]

train_data["comparison_group_size"] = (
    train_data.groupby(group_columns)["price_per_sqft"]
    .transform("size")
)


# ---------------------------------------------------------
# 9. Calculate Q1 and Q3 inside each comparison group
#
# Q1 = 25th percentile
# Q3 = 75th percentile
# IQR = Q3 - Q1
# ---------------------------------------------------------

train_data["group_q1"] = (
    train_data.groupby(group_columns)["price_per_sqft"]
    .transform(lambda values: values.quantile(0.25))
)

train_data["group_q3"] = (
    train_data.groupby(group_columns)["price_per_sqft"]
    .transform(lambda values: values.quantile(0.75))
)

train_data["group_iqr"] = (
    train_data["group_q3"]
    - train_data["group_q1"]
)


# ---------------------------------------------------------
# 10. Calculate the IQR outlier limits
# ---------------------------------------------------------

train_data["group_lower_limit"] = (
    train_data["group_q1"]
    - 1.5 * train_data["group_iqr"]
).clip(lower=0)

train_data["group_upper_limit"] = (
    train_data["group_q3"]
    + 1.5 * train_data["group_iqr"]
)


# ---------------------------------------------------------
# 11. Use only groups containing at least 10 properties
#
# Quartiles from tiny groups are unreliable.
# ---------------------------------------------------------

eligible_group = (
    train_data["comparison_group_size"] >= 10
)


# ---------------------------------------------------------
# 12. Flag unusual price-per-square-foot records
#
# These are audit flags only.
# We are not deleting these records.
# ---------------------------------------------------------

train_data["unusual_price_per_sqft"] = (
    eligible_group
    & (
        (
            train_data["price_per_sqft"]
            < train_data["group_lower_limit"]
        )
        |
        (
            train_data["price_per_sqft"]
            > train_data["group_upper_limit"]
        )
    )
)


# ---------------------------------------------------------
# 13. Print the outlier audit summary
# ---------------------------------------------------------

eligible_record_count = eligible_group.sum()

unusual_record_count = (
    train_data["unusual_price_per_sqft"].sum()
)

print("\nLOCATION AND AREA-TYPE OUTLIER AUDIT")
print(
    "Records in groups with at least 10 properties:",
    eligible_record_count,
)
print(
    "Unusual price-per-sqft records:",
    unusual_record_count,
)

if eligible_record_count > 0:
    unusual_percentage = (
        unusual_record_count
        / eligible_record_count
        * 100
    )

    print(
        "Percentage of eligible records flagged:",
        f"{unusual_percentage:.2f}%",
    )


# ---------------------------------------------------------
# 14. Show flagged counts by area type
# ---------------------------------------------------------

print("\nFLAGGED RECORDS BY AREA TYPE")

flagged_by_area_type = (
    train_data.loc[
        train_data["unusual_price_per_sqft"],
        "area_type_clean",
    ]
    .value_counts()
)

print(flagged_by_area_type)


# ---------------------------------------------------------
# 15. Columns used when inspecting flagged properties
# ---------------------------------------------------------

columns_to_display = [
    "location_clean",
    "area_type_clean",
    "total_sqft_clean",
    "bedrooms",
    "bath",
    "price",
    "price_per_sqft",
    "comparison_group_size",
    "group_lower_limit",
    "group_upper_limit",
]


# ---------------------------------------------------------
# 16. Display the most expensive flagged properties
# ---------------------------------------------------------

print("\n20 HIGHEST FLAGGED PRICE-PER-SQFT RECORDS")

highest_flagged_records = (
    train_data.loc[
        train_data["unusual_price_per_sqft"],
        columns_to_display,
    ]
    .sort_values(
        "price_per_sqft",
        ascending=False,
    )
    .head(20)
)

print(highest_flagged_records.to_string(index=False))


# ---------------------------------------------------------
# 17. Display the least expensive flagged properties
# ---------------------------------------------------------

print("\n20 LOWEST FLAGGED PRICE-PER-SQFT RECORDS")

lowest_flagged_records = (
    train_data.loc[
        train_data["unusual_price_per_sqft"],
        columns_to_display,
    ]
    .sort_values(
        "price_per_sqft",
        ascending=True,
    )
    .head(20)
)

print(lowest_flagged_records.to_string(index=False))