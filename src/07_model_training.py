from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


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

# Use the exact filename that worked in 06_outlier_analysis.py.


# ---------------------------------------------------------
# 2. Load training data
# ---------------------------------------------------------

train_data = pd.read_csv(TRAINING_DATA_PATH)

print("\nTRAINING RECORDS")
print(len(train_data))

print("\nTRAINING COLUMNS")
print(train_data.columns.tolist())


# ---------------------------------------------------------
# 3. Define input and target columns
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
#
# X contains the information used for prediction.
# y contains the price that the model must predict.
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
# 5. Numeric preprocessing
#
# Missing numeric values are replaced with the median.
# Missing-value indicators are also created.
# Values are then scaled for Ridge Regression.
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
# 6. Categorical preprocessing
#
# Missing text values are replaced with the most common value.
#
# OneHotEncoder converts text categories into numeric columns.
#
# min_frequency=10 groups rare categories automatically.
# handle_unknown allows new locations during prediction.
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
# 7. Combine numeric and categorical preprocessing
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
# 8. Create a dummy baseline
#
# DummyRegressor always predicts the median training price.
# A real model must perform better than this.
# ---------------------------------------------------------

dummy_model = DummyRegressor(
    strategy="median",
)


# ---------------------------------------------------------
# 9. Create the Ridge Regression pipeline
#
# The pipeline first prepares the data and then trains Ridge.
# ---------------------------------------------------------

ridge_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            Ridge(alpha=1.0),
        ),
    ]
)


random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


gradient_boosting_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            GradientBoostingRegressor(
                loss="huber",
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=5,
                random_state=42,
            ),
        ),
    ]
)


# ---------------------------------------------------------
# 10. Configure five-fold cross-validation
#
# The training data is divided into five parts.
# Four parts train the model and one part validates it.
# This process repeats five times.
# ---------------------------------------------------------

cross_validation = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# ---------------------------------------------------------
# 11. Define evaluation measurements
# ---------------------------------------------------------

scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


# ---------------------------------------------------------
# 12. Function for evaluating a model
# ---------------------------------------------------------

def evaluate_model(model_name, model, X, y):
    results = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=cross_validation,
        scoring=scoring,
        n_jobs=-1,
    )

    mae_scores = -results["test_mae"]
    rmse_scores = -results["test_rmse"]
    r2_scores = results["test_r2"]

    print(f"\n{model_name}")
    print("-" * len(model_name))

    print("MAE scores:")
    print(np.round(mae_scores, 2))

    print("RMSE scores:")
    print(np.round(rmse_scores, 2))

    print("R² scores:")
    print(np.round(r2_scores, 4))

    print(f"Average MAE: {mae_scores.mean():.2f} lakh")
    print(f"Average RMSE: {rmse_scores.mean():.2f} lakh")
    print(f"Average R²: {r2_scores.mean():.4f}")

    return {
        "model": model_name,
        "mae": mae_scores.mean(),
        "rmse": rmse_scores.mean(),
        "r2": r2_scores.mean(),
    }


# ---------------------------------------------------------
# 13. Evaluate the dummy baseline
# ---------------------------------------------------------

dummy_result = evaluate_model(
    model_name="Dummy Regressor",
    model=dummy_model,
    X=X,
    y=y,
)


# ---------------------------------------------------------
# 14. Evaluate Ridge Regression
# ---------------------------------------------------------

ridge_result = evaluate_model(
    model_name="Ridge Regression",
    model=ridge_model,
    X=X,
    y=y,
)


random_forest_result = evaluate_model(
    model_name="Random Forest",
    model=random_forest_model,
    X=X,
    y=y,
)


gradient_boosting_result = evaluate_model(
    model_name="Gradient Boosting",
    model=gradient_boosting_model,
    X=X,
    y=y,
)

# ---------------------------------------------------------
# 15. Compare model results
# ---------------------------------------------------------

comparison = pd.DataFrame(
    [
        dummy_result,
        ridge_result,
        random_forest_result,
        gradient_boosting_result
    ]
).sort_values(
    by="mae",
    ascending=True,
)

print("\nMODEL COMPARISON")
print(comparison.round(2).to_string(index=False))


# ---------------------------------------------------------
# 16. Train Ridge on all training records
#
# We fit it now, but we still do not evaluate test data.
# ---------------------------------------------------------

# ridge_model.fit(X, y)

# print("\nRIDGE MODEL TRAINED")
# print("The model was fitted using all training records.")
# print("The test data has not been used.")