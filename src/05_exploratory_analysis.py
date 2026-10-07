from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]


TRAIN_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_data.csv"
)


FIGURES_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)


FIGURE_FILE = (
    FIGURES_DIRECTORY
    / "01_numeric_distributions.png"
)


FIGURES_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


train_df = pd.read_csv(
    TRAIN_DATA_FILE
)


area_99th_percentile = (
    train_df["total_sqft_clean"]
    .quantile(0.99)
)


price_99th_percentile = (
    train_df["price"]
    .quantile(0.99)
)


normal_area_data = train_df[
    train_df["total_sqft_clean"]
    <= area_99th_percentile
]


normal_price_data = train_df[
    train_df["price"]
    <= price_99th_percentile
]


print("\nVISUALISATION LIMITS")

print(
    "Area 99th percentile:",
    area_99th_percentile
)

print(
    "Price 99th percentile:",
    price_99th_percentile
)


print("\nROWS ABOVE VISUALISATION LIMITS")

print(
    "Area:",
    (
        train_df["total_sqft_clean"]
        > area_99th_percentile
    ).sum()
)

print(
    "Price:",
    (
        train_df["price"]
        > price_99th_percentile
    ).sum()
)


sns.set_theme(
    style="whitegrid"
)


figure, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)


sns.histplot(
    data=train_df,
    x="total_sqft_clean",
    bins=50,
    ax=axes[0, 0],
)

axes[0, 0].set_title(
    "Complete Area Distribution"
)

axes[0, 0].set_xlabel(
    "Total Area in Square Feet"
)


sns.histplot(
    data=normal_area_data,
    x="total_sqft_clean",
    bins=50,
    ax=axes[0, 1],
)

axes[0, 1].set_title(
    "Area Distribution Up to 99th Percentile"
)

axes[0, 1].set_xlabel(
    "Total Area in Square Feet"
)


sns.histplot(
    data=train_df,
    x="price",
    bins=50,
    ax=axes[1, 0],
)

axes[1, 0].set_title(
    "Complete Price Distribution"
)

axes[1, 0].set_xlabel(
    "Price in Lakh Rupees"
)


sns.histplot(
    data=normal_price_data,
    x="price",
    bins=50,
    ax=axes[1, 1],
)

axes[1, 1].set_title(
    "Price Distribution Up to 99th Percentile"
)

axes[1, 1].set_xlabel(
    "Price in Lakh Rupees"
)


figure.suptitle(
    "Training Data: Area and Price Distributions",
    fontsize=16
)


plt.tight_layout()


plt.savefig(
    FIGURE_FILE,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


print("\nFIGURE SAVED TO")

print(FIGURE_FILE)

relationship_data = train_df[
    (
        train_df["total_sqft_clean"]
        <= area_99th_percentile
    )
    &
    (
        train_df["price"]
        <= price_99th_percentile
    )
].copy()


complete_correlation = (
    train_df[
        [
            "total_sqft_clean",
            "price"
        ]
    ]
    .corr()
    .loc[
        "total_sqft_clean",
        "price"
    ]
)


zoomed_correlation = (
    relationship_data[
        [
            "total_sqft_clean",
            "price"
        ]
    ]
    .corr()
    .loc[
        "total_sqft_clean",
        "price"
    ]
)


print("\nAREA AND PRICE CORRELATION")

print(
    "Complete training data:",
    round(complete_correlation, 4)
)

print(
    "Up to 99th percentiles:",
    round(zoomed_correlation, 4)
)


print("\nSCATTER-PLOT RECORDS")

print(
    "Training records:",
    len(train_df)
)

print(
    "Records displayed:",
    len(relationship_data)
)

print(
    "Records outside chart limits:",
    len(train_df) - len(relationship_data)
)


SCATTER_FIGURE_FILE = (
    FIGURES_DIRECTORY
    / "02_area_vs_price.png"
)


plt.figure(
    figsize=(12, 8)
)


sns.scatterplot(
    data=relationship_data,
    x="total_sqft_clean",
    y="price",
    hue="area_type_clean",
    alpha=0.45,
    s=35,
)


plt.title(
    "Training Data: Property Area Versus Price"
)


plt.xlabel(
    "Total Area in Square Feet"
)


plt.ylabel(
    "Price in Lakh Rupees"
)


plt.legend(
    title="Area Type"
)


plt.tight_layout()


plt.savefig(
    SCATTER_FIGURE_FILE,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


print("\nSCATTER FIGURE SAVED TO")

print(SCATTER_FIGURE_FILE)


location_counts = (
    train_df["location_clean"]
    .value_counts()
)


print("\nLOCATION FREQUENCY SUMMARY")

print(
    "Unique training locations:",
    len(location_counts)
)


frequency_thresholds = [
    1,
    2,
    5,
    10,
    20,
]


for threshold in frequency_thresholds:

    rare_locations = location_counts[
        location_counts <= threshold
    ].index

    affected_records = (
        train_df["location_clean"]
        .isin(rare_locations)
        .sum()
    )

    frequent_location_count = (
        location_counts > threshold
    ).sum()

    print(
        f"\nThreshold: {threshold}"
    )

    print(
        "Locations grouped as Other:",
        len(rare_locations)
    )

    print(
        "Training records affected:",
        affected_records
    )

    print(
        "Final location categories:",
        frequent_location_count + 1
    )


top_locations = (
    location_counts
    .head(20)
    .sort_values()
)


LOCATION_FIGURE_FILE = (
    FIGURES_DIRECTORY
    / "03_top_locations.png"
)


plt.figure(
    figsize=(12, 8)
)


sns.barplot(
    x=top_locations.values,
    y=top_locations.index,
    orient="h",
)


plt.title(
    "Top 20 Locations in Training Data"
)


plt.xlabel(
    "Number of Properties"
)


plt.ylabel(
    "Location"
)


plt.tight_layout()


plt.savefig(
    LOCATION_FIGURE_FILE,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


print("\nLOCATION FIGURE SAVED TO")

print(LOCATION_FIGURE_FILE)