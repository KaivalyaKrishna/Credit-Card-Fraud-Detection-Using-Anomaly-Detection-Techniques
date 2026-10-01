import cloudpickle
import numpy as np
import pandas as pd

PKL_PATH = "fraud_detector.pkl"

FEATURE_NAMES = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order",
]

BINARY_FEATURES = {"repeat_retailer", "used_chip", "used_pin_number", "online_order"}


# ── Load model (cached so it only loads once) ─────────────────────────────────
_detector = None

def load_detector():
    global _detector
    if _detector is None:
        with open(PKL_PATH, "rb") as f:
            _detector = cloudpickle.load(f)
    return _detector


# ── Build input dict from raw form values ─────────────────────────────────────
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
        "distance_from_home": float(distance_from_home),
        "distance_from_last_transaction": float(distance_from_last_transaction),
        "ratio_to_median_purchase_price": float(ratio_to_median_purchase_price),
        "repeat_retailer": float(repeat_retailer),
        "used_chip": float(used_chip),
        "used_pin_number": float(used_pin_number),
        "online_order": float(online_order),
    }


# ── Run prediction, return structured result ──────────────────────────────────
def predict(input_data: dict) -> dict:
    detector = load_detector()
    result = detector.predict(input_data)
    return {
        "label": result.label,
        "label_text": result.label_text,
        "fraud_probability": result.fraud_probability,
        "legit_probability": result.legit_probability,
        "confidence": result.confidence,
        "input_used": result.input_used,
    }
