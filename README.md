# Credit-Card-Fraud-Detection-Using-Anomaly-Detection-Techniques

# 🔍 Credit Card Fraud Detector

A Streamlit web app that uses an XGBoost model to predict whether a credit card transaction is fraudulent.

---

## 📁 Project Structure

```
fraud-detector/
├── app.py                  # Streamlit frontend
├── utils.py                # Input building, model loading, prediction
├── test.py                 # Standalone test script for the .pkl
├── fraud_detector.pkl      # Trained model (generated from Kaggle notebook)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Setup

**1. Clone / download the project**

**2. Place your `fraud_detector.pkl` in the root folder**

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## 🧪 Run Tests

```bash
python test.py
```

Tests 4 hardcoded cases:
- Clear legit transaction
- Clear fraud transaction
- Edge case
- Invalid input (validation check)

---

## 🧠 Features Used by the Model

| Feature | Type | Description |
|---|---|---|
| `distance_from_home` | Float | Distance (km) from cardholder's home |
| `distance_from_last_transaction` | Float | Distance (km) from last transaction |
| `ratio_to_median_purchase_price` | Float | Amount ÷ cardholder's median spend |
| `repeat_retailer` | Binary (0/1) | Shopped here before? |
| `used_chip` | Binary (0/1) | Card chip used? |
| `used_pin_number` | Binary (0/1) | PIN entered? |
| `online_order` | Binary (0/1) | Online transaction? |

---

## 📦 Model Details

- **Algorithm:** XGBoost Classifier
- **Class imbalance handling:** `scale_pos_weight`
- **Serialization:** `cloudpickle` (safe across environments)
- **Training data:** 1,000,000 transactions (~8.7% fraud rate)