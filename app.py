from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# 1. Configure the web page
#
# This must be the first Streamlit command.
# ---------------------------------------------------------

st.set_page_config(
    page_title="Bengaluru Property Price Predictor",
    page_icon="🏠",
    layout="centered",
)


# ---------------------------------------------------------
# 2. Locate the saved model
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "bengaluru_price_model.joblib"
)


# ---------------------------------------------------------
# 3. Load the model
#
# cache_resource prevents Streamlit from loading the model
# again every time the user changes an input.
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model was not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

except Exception as error:
    st.error(
        f"The model could not be loaded: {error}"
    )
    st.stop()


# ---------------------------------------------------------
# 4. Extract known locations from the trained encoder
#
# This avoids manually writing hundreds of locations.
# Index 2 is location_clean because the categorical feature
# order during training was:
#
# 0: area_type_clean
# 1: availability_status
# 2: location_clean
# 3: layout_type
# ---------------------------------------------------------

def get_frequent_locations(trained_model):
    try:
        preprocessor = trained_model.named_steps[
            "preprocessor"
        ]

        categorical_pipeline = (
            preprocessor.named_transformers_[
                "categorical"
            ]
        )

        encoder = (
            categorical_pipeline.named_steps[
                "encoder"
            ]
        )

        all_locations = encoder.categories_[2]

        infrequent_locations = (
            encoder.infrequent_categories_[2]
        )

        if infrequent_locations is None:
            frequent_locations = all_locations

        else:
            infrequent_set = set(
                infrequent_locations
            )

            frequent_locations = [
                location
                for location in all_locations
                if location not in infrequent_set
            ]

        return sorted(
            str(location)
            for location in frequent_locations
        )

    except Exception:
        return []


locations = get_frequent_locations(model)


# ---------------------------------------------------------
# 5. Display application heading
# ---------------------------------------------------------

st.title("Bengaluru Property Price Predictor")

st.write(
    "Enter the property details to estimate its price."
)

st.caption(
    "The model was trained using 10,636 Bengaluru "
    "property records and evaluated on 2,660 unseen records."
)


# ---------------------------------------------------------
# 6. Create the input form
#
# Code inside the form runs when the user presses the
# Predict Price button.
# ---------------------------------------------------------

with st.form("property_form"):

    st.subheader("Property Details")

    selected_location = st.selectbox(
        label="Location",
        options=[
            "Select a location"
        ]
        + locations
        + [
            "Other / location not listed"
        ],
    )

    other_location = ""

    if selected_location == "Other / location not listed":
        other_location = st.text_input(
            label="Enter the location",
            placeholder="Example: New Bengaluru Layout",
        )

    total_sqft = st.number_input(
        label="Total area in square feet",
        min_value=100.0,
        max_value=1000000.0,
        value=1200.0,
        step=50.0,
    )

    first_column, second_column = st.columns(2)

    with first_column:
        bedrooms = st.number_input(
            label="Bedrooms",
            min_value=1,
            max_value=50,
            value=2,
            step=1,
        )

        bathrooms = st.number_input(
            label="Bathrooms",
            min_value=1,
            max_value=50,
            value=2,
            step=1,
        )

    with second_column:
        balconies = st.number_input(
            label="Balconies",
            min_value=0,
            max_value=3,
            value=1,
            step=1,
        )

        layout_type = st.selectbox(
            label="Layout type",
            options=[
                "BHK",
                "BEDROOM",
                "RK",
            ],
        )

    area_type = st.selectbox(
        label="Area type",
        options=[
            "Super built-up Area",
            "Built-up Area",
            "Plot Area",
            "Carpet Area",
        ],
    )

    availability = st.selectbox(
        label="Availability",
        options=[
            "Ready",
            "Under Construction",
        ],
    )

    predict_button = st.form_submit_button(
        label="Predict Price",
        type="primary",
        use_container_width=True,
    )


# ---------------------------------------------------------
# 7. Validate the form and make a prediction
# ---------------------------------------------------------

if predict_button:

    if selected_location == "Select a location":
        st.error("Select a location.")

    elif (
        selected_location
        == "Other / location not listed"
        and not other_location.strip()
    ):
        st.error("Enter the property location.")

    else:
        if selected_location == "Other / location not listed":
            final_location = other_location.strip()

        else:
            final_location = selected_location

        new_property = pd.DataFrame(
            [
                {
                    "total_sqft_clean": total_sqft,
                    "bedrooms": bedrooms,
                    "bath": bathrooms,
                    "balcony": balconies,
                    "area_type_clean": area_type,
                    "availability_status": availability,
                    "location_clean": final_location,
                    "layout_type": layout_type,
                }
            ]
        )

        try:
            predicted_price_lakh = model.predict(
                new_property
            )[0]

            predicted_price_crore = (
                predicted_price_lakh / 100
            )

            st.success("Prediction completed")

            st.metric(
                label="Estimated Property Price",
                value=(
                    f"₹{predicted_price_lakh:,.2f} lakh"
                ),
            )

            if predicted_price_lakh >= 100:
                st.write(
                    f"Approximately "
                    f"**₹{predicted_price_crore:,.2f} crore**"
                )

            st.write("Property used for prediction:")

            display_data = pd.DataFrame(
                {
                    "Property detail": [
                        "Location",
                        "Total area",
                        "Bedrooms",
                        "Bathrooms",
                        "Balconies",
                        "Area type",
                        "Availability",
                        "Layout type",
                    ],
                    "Value": [
                        final_location,
                        f"{total_sqft:,.0f} sqft",
                        bedrooms,
                        bathrooms,
                        balconies,
                        area_type,
                        availability,
                        layout_type,
                    ],
                }
            )

            st.dataframe(
                display_data,
                hide_index=True,
                use_container_width=True,
            )

            # ---------------------------------------------
            # Display area-type-specific accuracy context
            # ---------------------------------------------

            area_type_mae = {
                "Super built-up Area": 21.26,
                "Built-up Area": 32.21,
                "Plot Area": 83.39,
                "Carpet Area": 18.70,
            }

            relevant_mae = area_type_mae[
                area_type
            ]

            st.info(
                f"On the test dataset, the average error "
                f"for {area_type} properties was "
                f"₹{relevant_mae:,.2f} lakh."
            )

            if area_type == "Plot Area":
                st.warning(
                    "Plot Area predictions are less reliable. "
                    "Land size and constructed area are mixed "
                    "in the source dataset."
                )

            if (
                selected_location
                == "Other / location not listed"
            ):
                st.warning(
                    "This location was not frequent in the "
                    "training data. The prediction is less "
                    "location-specific."
                )

        except Exception as error:
            st.error(
                f"Prediction failed: {error}"
            )


# ---------------------------------------------------------
# 8. Display model information
# ---------------------------------------------------------

st.divider()

st.subheader("Model Information")

st.write(
    """
    - Model: Tuned Random Forest Regressor
    - Final test MAE: ₹32.60 lakh
    - Final test R²: 0.6483
    - Predictions within 20% of actual price: 56.39%
    """
)

st.caption(
    "This application is an educational ML prototype. "
    "The result is an estimate and not a professional "
    "property valuation."
)