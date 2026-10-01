import cloudpickle

PKL_PATH = "fraud_detector.pkl"

# ── Load model ────────────────────────────────────────────────────────────────
with open(PKL_PATH, "rb") as f:
    detector = cloudpickle.load(f)

print("Model loaded ✅\n")

# ── Test cases ────────────────────────────────────────────────────────────────
test_cases = [
    {
        "label": "Should be LEGIT  — nearby, normal price, repeat retailer",
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
        "label": "Should be FRAUD  — far away, inflated price, no chip/pin, online",
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
        "label": "Edge case     — very close, slightly above median, chip used",
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
        "label": "Bad input test — should raise ValueError (binary field = 5)",
        "data": {
            "distance_from_home": 10.0,
            "distance_from_last_transaction": 1.0,
            "ratio_to_median_purchase_price": 1.0,
            "repeat_retailer": 5.0,   # ← invalid
            "used_chip": 0.0,
            "used_pin_number": 0.0,
            "online_order": 1.0,
        },
    },
]

# ── Run tests ─────────────────────────────────────────────────────────────────
SEP = "─" * 60

for i, tc in enumerate(test_cases, 1):
    print(SEP)
    print(f"Test {i}: {tc['label']}")
    try:
        result = detector.predict(tc["data"])
        print(f"  Prediction  : {result.label_text}  (label={result.label})")
        print(f"  Fraud prob  : {result.fraud_probability:.4f}")
        print(f"  Legit prob  : {result.legit_probability:.4f}")
        print(f"  Confidence  : {result.confidence}")
    except ValueError as e:
        print(f"  ✅ Caught expected error: {e}")

print(SEP)
print("\nAll tests complete ✅")
