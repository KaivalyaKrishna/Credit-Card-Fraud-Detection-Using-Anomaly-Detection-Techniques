import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "fraud_model.json"
)

model = XGBClassifier()

model.load_model(MODEL_PATH)

print("XGBoost model loaded ✅\n")


# ============================================================
# TEST CASES
# ============================================================

test_cases = [
    {
        "label": "Should be LEGIT — nearby, normal price, repeat retailer",
        "data": {
            "distance_from_home": 2.5,
            "distance_from_last_transaction": 0.5,
            "ratio_to_median_purchase_price": 0.9,
            "repeat_retailer": 1.0,
            "used_chip": 1.0,
            "used_pin_number": 1.0,
            "online_order": 0.0,
        },
    },

    {
        "label": "Should be FRAUD — far away, inflated price, no chip/pin, online",
        "data": {
            "distance_from_home": 500.0,
            "distance_from_last_transaction": 300.0,
            "ratio_to_median_purchase_price": 8.5,
            "repeat_retailer": 0.0,
            "used_chip": 0.0,
            "used_pin_number": 0.0,
            "online_order": 1.0,
        },
    },

    {
        "label": "Edge case — very close, slightly above median, chip used",
        "data": {
            "distance_from_home": 0.5,
            "distance_from_last_transaction": 0.1,
            "ratio_to_median_purchase_price": 1.2,
            "repeat_retailer": 1.0,
            "used_chip": 1.0,
            "used_pin_number": 0.0,
            "online_order": 0.0,
        },
    },

    {
        "label": "Bad input test — binary field = 5",
        "data": {
            "distance_from_home": 10.0,
            "distance_from_last_transaction": 1.0,
            "ratio_to_median_purchase_price": 1.0,
            "repeat_retailer": 5.0,
            "used_chip": 0.0,
            "used_pin_number": 0.0,
            "online_order": 1.0,
        },
    },
]


# ============================================================
# VALIDATION
# ============================================================

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


def validate_input(data):

    for feature in FEATURE_NAMES:

        if feature not in data:
            raise ValueError(
                f"Missing feature: {feature}"
            )

    for feature in BINARY_FEATURES:

        if data[feature] not in (0, 1, 0.0, 1.0):
            raise ValueError(
                f"{feature} must be 0 or 1. "
                f"Received: {data[feature]}"
            )


# ============================================================
# RUN TESTS
# ============================================================

SEP = "─" * 60

for i, test_case in enumerate(test_cases, 1):

    print(SEP)
    print(f"Test {i}: {test_case['label']}")

    try:

        data = test_case["data"]

        validate_input(data)

        X = pd.DataFrame(
            [data],
            columns=FEATURE_NAMES
        )

        print("  Running prediction...")

        probabilities = model.predict_proba(X)[0]

        prediction = int(np.argmax(probabilities))

        fraud_probability = float(probabilities[1])
        legit_probability = float(probabilities[0])

        label = (
            "Fraud"
            if prediction == 1
            else "Legit"
        )

        confidence = max(
            fraud_probability,
            legit_probability
        )

        print(f"  Prediction     : {label}")
        print(f"  Label          : {prediction}")
        print(f"  Fraud prob     : {fraud_probability:.4f}")
        print(f"  Legit prob     : {legit_probability:.4f}")
        print(f"  Confidence     : {confidence:.4f}")

    except ValueError as e:

        print(f"  ✅ Caught expected error: {e}")

    except Exception as e:

        print(
            f"  ❌ Unexpected error: "
            f"{type(e).__name__}: {e}"
        )


print(SEP)
print("\nAll tests complete ✅")