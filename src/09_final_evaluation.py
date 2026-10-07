from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# 1. Create project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAINING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_data.csv"
)

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_data.csv"
)

REPORTS_DIRECTORY = PROJECT_ROOT / "reports"
FIGURES_DIRECTORY = REPORTS_DIRECTORY / "figures"
MODELS_DIRECTORY = PROJECT_ROOT / "models"

FIGURES_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# 2. Load training and test data
# ---------------------------------------------------------

train_data = pd.read_csv(TRAINING_DATA_PATH)
test_data = pd.read_csv(TEST_DATA_PATH)

print("\nDATASET SIZES")
print(f"Training records: {len(train_data)}")
print(f"Test records: {len(test_data)}")


# ---------------------------------------------------------
# 3. Define model columns
# ---------------------------------------------------------

numeric_features = [
    "total_sqft_clean",
    "bedrooms",
    "bath",
    "balcony",
]

categorical_features = [
    "area_type_clean",
    "availability_status",
    "location_clean",
    "layout_type",
]

feature_columns = (
    numeric_features
    + categorical_features
)

target_column = "price"


# ---------------------------------------------------------
# 4. Separate inputs and targets
# ---------------------------------------------------------

X_train = train_data[feature_columns].copy()
y_train = train_data[target_column].copy()

X_test = test_data[feature_columns].copy()
y_test = test_data[target_column].copy()

print("\nMODEL DATA SHAPES")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")


# ---------------------------------------------------------
# 5. Create numeric preprocessing
# ---------------------------------------------------------

numeric_preprocessor = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


# ---------------------------------------------------------
# 6. Create categorical preprocessing
# ---------------------------------------------------------

categorical_preprocessor = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent",
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=10,
            ),
        ),
    ]
)


# ---------------------------------------------------------
# 7. Combine preprocessing
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_preprocessor,
            numeric_features,
        ),
        (
            "categorical",
            categorical_preprocessor,
            categorical_features,
        ),
    ]
)


# ---------------------------------------------------------
# 8. Create the tuned Random Forest pipeline
# ---------------------------------------------------------

final_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=400,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=1,
                max_features=1.0,
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# ---------------------------------------------------------
# 9. Train using all training records
# ---------------------------------------------------------

print("\nTRAINING FINAL MODEL")

final_model.fit(
    X_train,
    y_train,
)

print("Final model training completed.")


# ---------------------------------------------------------
# 10. Predict the untouched test data
# ---------------------------------------------------------

test_predictions = final_model.predict(X_test)


# ---------------------------------------------------------
# 11. Calculate final test measurements
# ---------------------------------------------------------

test_mae = mean_absolute_error(
    y_test,
    test_predictions,
)

test_median_ae = median_absolute_error(
    y_test,
    test_predictions,
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions,
    )
)

test_r2 = r2_score(
    y_test,
    test_predictions,
)


print("\nFINAL TEST RESULTS")
print(f"MAE: {test_mae:.2f} lakh")
print(
    f"Median absolute error: "
    f"{test_median_ae:.2f} lakh"
)
print(f"RMSE: {test_rmse:.2f} lakh")
print(f"R²: {test_r2:.4f}")


# ---------------------------------------------------------
# 12. Compare with the training median baseline
#
# This baseline always predicts the median training price.
# ---------------------------------------------------------

training_median_price = y_train.median()

baseline_predictions = np.full(
    shape=len(y_test),
    fill_value=training_median_price,
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions,
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions,
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_predictions,
)

comparison = pd.DataFrame(
    [
        {
            "model": "Median baseline",
            "mae": baseline_mae,
            "rmse": baseline_rmse,
            "r2": baseline_r2,
        },
        {
            "model": "Tuned Random Forest",
            "mae": test_mae,
            "rmse": test_rmse,
            "r2": test_r2,
        },
    ]
)

print("\nFINAL MODEL COMPARISON")
print(
    comparison.round(2).to_string(
        index=False,
    )
)


# ---------------------------------------------------------
# 13. Calculate percentage errors
# ---------------------------------------------------------

absolute_percentage_error = (
    np.abs(y_test - test_predictions)
    / y_test
    * 100
)

median_percentage_error = np.median(
    absolute_percentage_error
)

within_10_percent = (
    absolute_percentage_error <= 10
).mean() * 100

within_20_percent = (
    absolute_percentage_error <= 20
).mean() * 100

within_30_percent = (
    absolute_percentage_error <= 30
).mean() * 100

print("\nPERCENTAGE-ERROR SUMMARY")
print(
    f"Median absolute percentage error: "
    f"{median_percentage_error:.2f}%"
)
print(
    f"Predictions within 10%: "
    f"{within_10_percent:.2f}%"
)
print(
    f"Predictions within 20%: "
    f"{within_20_percent:.2f}%"
)
print(
    f"Predictions within 30%: "
    f"{within_30_percent:.2f}%"
)


# ---------------------------------------------------------
# 14. Build a test-results DataFrame
# ---------------------------------------------------------

test_results = X_test.copy()

test_results["actual_price"] = y_test.to_numpy()
test_results["predicted_price"] = test_predictions

test_results["absolute_error"] = np.abs(
    test_results["actual_price"]
    - test_results["predicted_price"]
)

test_results["percentage_error"] = (
    test_results["absolute_error"]
    / test_results["actual_price"]
    * 100
)


# ---------------------------------------------------------
# 15. Evaluate each area type
# ---------------------------------------------------------

def calculate_group_metrics(group):
    actual = group["actual_price"]
    predicted = group["predicted_price"]

    return pd.Series(
        {
            "records": len(group),
            "mae": mean_absolute_error(
                actual,
                predicted,
            ),
            "rmse": np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            ),
            "r2": (
                r2_score(actual, predicted)
                if len(group) >= 2
                else np.nan
            ),
        }
    )


area_type_results = (
    test_results.groupby("area_type_clean")
    .apply(
        calculate_group_metrics,
        include_groups=False,
    )
    .round(2)
)

print("\nTEST RESULTS BY AREA TYPE")
print(area_type_results)


# ---------------------------------------------------------
# 16. Display the largest test errors
# ---------------------------------------------------------

error_columns = [
    "location_clean",
    "area_type_clean",
    "total_sqft_clean",
    "bedrooms",
    "bath",
    "actual_price",
    "predicted_price",
    "absolute_error",
    "percentage_error",
]

largest_errors = (
    test_results[error_columns]
    .sort_values(
        "absolute_error",
        ascending=False,
    )
    .head(20)
)

print("\n20 LARGEST TEST ERRORS")
print(
    largest_errors.round(2).to_string(
        index=False,
    )
)


# ---------------------------------------------------------
# 17. Save all test predictions
# ---------------------------------------------------------

predictions_path = (
    REPORTS_DIRECTORY
    / "final_test_predictions.csv"
)

test_results.to_csv(
    predictions_path,
    index=False,
)

print("\nTEST PREDICTIONS SAVED TO")
print(predictions_path)


# ---------------------------------------------------------
# 18. Prepare an actual-versus-predicted chart
#
# The 99th percentile is used only for displaying the chart.
# It does not change the test measurements.
# ---------------------------------------------------------

chart_limit = max(
    y_test.quantile(0.99),
    np.quantile(test_predictions, 0.99),
)

chart_data = test_results[
    (
        test_results["actual_price"]
        <= chart_limit
    )
    & (
        test_results["predicted_price"]
        <= chart_limit
    )
].copy()

records_outside_chart = (
    len(test_results)
    - len(chart_data)
)

print("\nACTUAL-VERSUS-PREDICTED CHART")
print(f"Chart limit: {chart_limit:.2f} lakh")
print(f"Records displayed: {len(chart_data)}")
print(
    f"Records outside chart: "
    f"{records_outside_chart}"
)


# ---------------------------------------------------------
# 19. Create the prediction chart
# ---------------------------------------------------------

plt.figure(figsize=(10, 8))

sns.scatterplot(
    data=chart_data,
    x="actual_price",
    y="predicted_price",
    hue="area_type_clean",
    alpha=0.65,
)

plt.plot(
    [0, chart_limit],
    [0, chart_limit],
    color="black",
    linestyle="--",
    label="Perfect prediction",
)

plt.xlim(0, chart_limit)
plt.ylim(0, chart_limit)

plt.title(
    "Test Data: Actual Price vs Predicted Price",
    fontsize=15,
)

plt.xlabel("Actual Price (Lakh ₹)")
plt.ylabel("Predicted Price (Lakh ₹)")

plt.legend(
    title="Area Type",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

plt.tight_layout()

figure_path = (
    FIGURES_DIRECTORY
    / "05_actual_vs_predicted.png"
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
# 20. Save the complete trained pipeline
#
# This saves preprocessing and the Random Forest together.
# ---------------------------------------------------------

model_path = (
    MODELS_DIRECTORY
    / "bengaluru_price_model.joblib"
)

joblib.dump(
    final_model,
    model_path,
)

print("\nFINAL MODEL SAVED TO")
print(model_path)