import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

FEATURE_NAMES = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order",
]

BINARY_FEATURES = {
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "fraud_model.json"
)

model = XGBClassifier()

model.load_model(MODEL_PATH)

def build_input(
    distance_from_home: float,
    distance_from_last_transaction: float,
    ratio_to_median_purchase_price: float,
    repeat_retailer: int,
    used_chip: int,
    used_pin_number: int,
    online_order: int,
) -> dict:

    return {
        "distance_from_home": float(
            distance_from_home
        ),

        "distance_from_last_transaction": float(
            distance_from_last_transaction
        ),

        "ratio_to_median_purchase_price": float(
            ratio_to_median_purchase_price
        ),

        "repeat_retailer": float(
            repeat_retailer
        ),

        "used_chip": float(
            used_chip
        ),

        "used_pin_number": float(
            used_pin_number
        ),

        "online_order": float(
            online_order
        ),
    }

def validate_input(data: dict):

    errors = []

    missing = [
        feature
        for feature in FEATURE_NAMES
        if feature not in data
    ]

    if missing:

        errors.append(
            f"Missing features: {missing}"
        )

    for feature, value in data.items():

        # Ignore unknown fields
        if feature not in FEATURE_NAMES:
            continue

        # Check numeric
        if not isinstance(value, (int, float)):

            errors.append(
                f"'{feature}' must be numeric."
            )

            continue

        if np.isnan(value) or np.isinf(value):

            errors.append(
                f"'{feature}' cannot be NaN or Inf."
            )

            continue

        if feature in BINARY_FEATURES:

            if value not in (0, 1, 0.0, 1.0):

                errors.append(
                    f"'{feature}' must be 0 or 1. "
                    f"Received: {value}"
                )

        else:

            if value < 0:

                errors.append(
                    f"'{feature}' must be >= 0. "
                    f"Received: {value}"
                )

    if errors:

        raise ValueError(
            "Input validation failed:\n"
            + "\n".join(
                f"• {error}"
                for error in errors
            )
        )

def get_confidence(fraud_probability: float) -> str:

    if (
        fraud_probability >= 0.80
        or fraud_probability <= 0.20
    ):
        return "High"

    elif (
        fraud_probability >= 0.60
        or fraud_probability <= 0.40
    ):
        return "Medium"

    else:
        return "Low"

def predict(input_data: dict) -> dict:

    validate_input(input_data)

    X = pd.DataFrame(
        [input_data],
        columns=FEATURE_NAMES
    )

    probabilities = model.predict_proba(X)[0]

    legit_probability = float(
        probabilities[0]
    )

    fraud_probability = float(
        probabilities[1]
    )

    label = int(
        np.argmax(probabilities)
    )

    label_text = (
        "Fraud"
        if label == 1
        else "Legit"
    )

    confidence = get_confidence(
        fraud_probability
    )

    return {
        "label": label,

        "label_text": label_text,

        "fraud_probability": round(
            fraud_probability,
            4
        ),

        "legit_probability": round(
            legit_probability,
            4
        ),

        "confidence": confidence,

        "input_used": input_data,
    }