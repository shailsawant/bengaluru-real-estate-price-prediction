from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# 1. Create the training-data path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAINING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_data.csv"
)

# Use the exact filename that worked in 07_model_training.py.


# ---------------------------------------------------------
# 2. Load training data
# ---------------------------------------------------------

train_data = pd.read_csv(TRAINING_DATA_PATH)

print("\nTRAINING RECORDS")
print(len(train_data))


# ---------------------------------------------------------
# 3. Define model features
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

target_column = "price"


# ---------------------------------------------------------
# 4. Create X and y
# ---------------------------------------------------------

X = train_data[
    numeric_features + categorical_features
].copy()

y = train_data[target_column].copy()

print("\nMODEL INPUT SHAPE")
print(X.shape)

print("\nTARGET SHAPE")
print(y.shape)


# ---------------------------------------------------------
# 5. Prepare numeric columns
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
# 6. Prepare categorical columns
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
# 8. Create the Random Forest pipeline
#
# n_jobs=1 is intentional here.
# RandomizedSearchCV will handle parallel processing.
# ---------------------------------------------------------

random_forest_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            RandomForestRegressor(
                random_state=42,
                n_jobs=1,
            ),
        ),
    ]
)


# ---------------------------------------------------------
# 9. Define parameter combinations
#
# RandomizedSearchCV will select 12 random combinations
# from these possible values.
# ---------------------------------------------------------

parameter_options = {
    "model__n_estimators": [
        150,
        200,
        300,
        400,
    ],
    "model__max_depth": [
        None,
        10,
        20,
        30,
    ],
    "model__min_samples_split": [
        2,
        5,
        10,
    ],
    "model__min_samples_leaf": [
        1,
        2,
        4,
    ],
    "model__max_features": [
        "sqrt",
        0.7,
        1.0,
    ],
    "model__bootstrap": [
        True,
        False,
    ],
}


# ---------------------------------------------------------
# 10. Configure three-fold tuning
#
# Three folds keep the tuning time reasonable.
# We will validate the winning configuration with five folds.
# ---------------------------------------------------------

tuning_cross_validation = KFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)


# ---------------------------------------------------------
# 11. Define evaluation measurements
#
# RandomizedSearchCV selects the model with the best MAE.
# It also records RMSE and R².
# ---------------------------------------------------------

scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


# ---------------------------------------------------------
# 12. Configure RandomizedSearchCV
#
# 12 combinations × 3 folds = 36 model fits.
# ---------------------------------------------------------

random_search = RandomizedSearchCV(
    estimator=random_forest_pipeline,
    param_distributions=parameter_options,
    n_iter=12,
    scoring=scoring,
    refit="mae",
    cv=tuning_cross_validation,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    return_train_score=False,
    error_score="raise",
)


# ---------------------------------------------------------
# 13. Run parameter tuning
#
# This uses training data only.
# ---------------------------------------------------------

print("\nSTARTING RANDOM FOREST TUNING")
print("Testing 12 parameter combinations using 3 folds.")
print("Total model fits: 36")

random_search.fit(X, y)


# ---------------------------------------------------------
# 14. Print the best parameter combination
# ---------------------------------------------------------

print("\nBEST RANDOM FOREST PARAMETERS")

for parameter_name, parameter_value in random_search.best_params_.items():
    clean_name = parameter_name.replace("model__", "")
    print(f"{clean_name}: {parameter_value}")


# ---------------------------------------------------------
# 15. Print the tuning score
#
# Scikit-learn stores MAE as a negative number.
# We multiply by -1 to display a normal positive error.
# ---------------------------------------------------------

best_tuning_mae = -random_search.best_score_

print("\nBEST THREE-FOLD TUNING RESULT")
print(f"Average MAE: {best_tuning_mae:.2f} lakh")


# ---------------------------------------------------------
# 16. Display the five best tested combinations
# ---------------------------------------------------------

tuning_results = pd.DataFrame(
    random_search.cv_results_
)

tuning_results["mean_mae"] = (
    -tuning_results["mean_test_mae"]
)

tuning_results["mean_rmse"] = (
    -tuning_results["mean_test_rmse"]
)

top_results = (
    tuning_results[
        [
            "rank_test_mae",
            "mean_mae",
            "mean_rmse",
            "mean_test_r2",
            "params",
        ]
    ]
    .sort_values("rank_test_mae")
    .head(5)
    .copy()
)

top_results = top_results.rename(
    columns={
        "rank_test_mae": "rank",
        "mean_test_r2": "mean_r2",
    }
)

print("\nTOP FIVE PARAMETER COMBINATIONS")
print(
    top_results.to_string(
        index=False,
    )
)


# ---------------------------------------------------------
# 17. Get the best model
#
# RandomizedSearchCV automatically refits the winning model
# using all training records.
# ---------------------------------------------------------

best_random_forest = random_search.best_estimator_


# ---------------------------------------------------------
# 18. Validate the winning configuration with five folds
#
# This makes the result comparable with script 07.
# ---------------------------------------------------------

final_cross_validation = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

final_scores = cross_validate(
    estimator=best_random_forest,
    X=X,
    y=y,
    cv=final_cross_validation,
    scoring=scoring,
    n_jobs=-1,
)

final_mae_scores = -final_scores["test_mae"]
final_rmse_scores = -final_scores["test_rmse"]
final_r2_scores = final_scores["test_r2"]


# ---------------------------------------------------------
# 19. Print five-fold validation results
# ---------------------------------------------------------

print("\nTUNED RANDOM FOREST: FIVE-FOLD VALIDATION")

print("MAE scores:")
print(np.round(final_mae_scores, 2))

print("RMSE scores:")
print(np.round(final_rmse_scores, 2))

print("R² scores:")
print(np.round(final_r2_scores, 4))

print(
    f"Average MAE: "
    f"{final_mae_scores.mean():.2f} lakh"
)

print(
    f"Average RMSE: "
    f"{final_rmse_scores.mean():.2f} lakh"
)

print(
    f"Average R²: "
    f"{final_r2_scores.mean():.4f}"
)


# ---------------------------------------------------------
# 20. Compare tuned and untuned Random Forest
# ---------------------------------------------------------

untuned_mae = 33.53
untuned_rmse = 93.33
untuned_r2 = 0.5852

comparison = pd.DataFrame(
    [
        {
            "model": "Untuned Random Forest",
            "mae": untuned_mae,
            "rmse": untuned_rmse,
            "r2": untuned_r2,
        },
        {
            "model": "Tuned Random Forest",
            "mae": final_mae_scores.mean(),
            "rmse": final_rmse_scores.mean(),
            "r2": final_r2_scores.mean(),
        },
    ]
)

print("\nTUNING COMPARISON")
print(
    comparison.round(2).to_string(
        index=False,
    )
)

print("\nTEST DATA STATUS")
print("The test dataset has not been loaded or evaluated.")