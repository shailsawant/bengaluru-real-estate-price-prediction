from pathlib import Path

import joblib
import pandas as pd


# ---------------------------------------------------------
# 1. Locate the saved model
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "bengaluru_price_model.joblib"
)


# ---------------------------------------------------------
# 2. Load the complete trained pipeline
#
# The file contains:
# - missing-value handling
# - numeric scaling
# - one-hot encoding
# - rare-category handling
# - trained Random Forest
# ---------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file was not found: {MODEL_PATH}\n"
        "Run 09_final_evaluation.py first."
    )

model = joblib.load(MODEL_PATH)

print("\nMODEL LOADED FROM")
print(MODEL_PATH)


# ---------------------------------------------------------
# 3. Define the expected input columns
# ---------------------------------------------------------

FEATURE_COLUMNS = [
    "total_sqft_clean",
    "bedrooms",
    "bath",
    "balcony",
    "area_type_clean",
    "availability_status",
    "location_clean",
    "layout_type",
]


# ---------------------------------------------------------
# 4. Create a prediction function
# ---------------------------------------------------------

def predict_property_price(
    total_sqft,
    bedrooms,
    bathrooms,
    balconies,
    area_type,
    availability,
    location,
    layout_type,
):
    """
    Predict the price of one Bengaluru property.

    The predicted price is returned in lakh rupees.
    """

    # Basic validation

    if total_sqft <= 0:
        raise ValueError(
            "Total square feet must be greater than zero."
        )

    if bedrooms <= 0:
        raise ValueError(
            "Bedrooms must be greater than zero."
        )

    if bathrooms is not None and bathrooms <= 0:
        raise ValueError(
            "Bathrooms must be greater than zero."
        )

    if balconies is not None and balconies < 0:
        raise ValueError(
            "Balconies cannot be negative."
        )

    # Create one row with the same columns used during training

    new_property = pd.DataFrame(
        [
            {
                "total_sqft_clean": total_sqft,
                "bedrooms": bedrooms,
                "bath": bathrooms,
                "balcony": balconies,
                "area_type_clean": area_type,
                "availability_status": availability,
                "location_clean": location.strip(),
                "layout_type": layout_type,
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    # The model returns an array because it can predict
    # multiple properties at once.

    prediction = model.predict(new_property)[0]

    return prediction, new_property


# ---------------------------------------------------------
# 5. Enter a property for prediction
#
# Change these values when testing another property.
# ---------------------------------------------------------

property_details = {
    "total_sqft": 1200,
    "bedrooms": 2,
    "bathrooms": 2,
    "balconies": 1,
    "area_type": "Super built-up Area",
    "availability": "Ready",
    "location": "Whitefield",
    "layout_type": "BHK",
}


# ---------------------------------------------------------
# 6. Predict the property price
# ---------------------------------------------------------

predicted_price_lakh, property_input = (
    predict_property_price(
        **property_details
    )
)

predicted_price_crore = (
    predicted_price_lakh / 100
)


# ---------------------------------------------------------
# 7. Display the input
# ---------------------------------------------------------

print("\nPROPERTY DETAILS")
print(
    property_input.to_string(
        index=False,
    )
)


# ---------------------------------------------------------
# 8. Display the prediction
# ---------------------------------------------------------

print("\nPREDICTED PROPERTY PRICE")
print(
    f"₹{predicted_price_lakh:,.2f} lakh"
)

print(
    f"₹{predicted_price_crore:,.2f} crore"
)


# ---------------------------------------------------------
# 9. Display an accuracy reminder
# ---------------------------------------------------------

print("\nMODEL ACCURACY CONTEXT")
print(
    "Final test MAE: ₹32.60 lakh"
)

print(
    "The prediction is an estimate, not a property valuation."
)

if property_details["area_type"] == "Plot Area":
    print(
        "Warning: Plot Area predictions had a test "
        "MAE of ₹83.39 lakh and are less reliable."
    )