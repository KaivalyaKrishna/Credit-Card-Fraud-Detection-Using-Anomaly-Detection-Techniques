# 🔍 Credit Card Fraud Detector

A Streamlit web app that uses an XGBoost model to predict whether a credit card transaction is fraudulent.

---

## 📁 Project Structure

```
fraud-detector/
├── app.py                  # Streamlit frontend
├── utils.py                # Input building, validation, model loading, prediction
├── test.py                 # Standalone test script for the model
├── fraud_model.json        # Trained XGBoost model (exported from the notebook)
├── credit-card.ipynb       # Training and evaluation notebook (Kaggle)
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Setup

Developed with Python 3.13.

**1. Clone / download the project**

**2. Make sure `fraud_model.json` is in the root folder** (it is included in the repository)

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

Loads `fraud_model.json` and runs 4 hardcoded cases, printing the prediction and probabilities for each:
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

- **Algorithm:** XGBoost Classifier (300 trees, max depth 6, learning rate 0.1, subsample 0.8, colsample_bytree 0.8)
- **Class imbalance handling:** `scale_pos_weight` (≈ 10.44)
- **Serialization:** XGBoost native JSON (`fraud_model.json`), portable across Python versions
- **XGBoost version:** 3.2.0
- **Training data:** 1,000,000 transactions (~8.7% fraud rate), 80/20 stratified train/test split
- **Dataset file:** `card_transdata.csv` (see the notebook for the Kaggle source)

### Test-set results (200,000 transactions)

| Metric | Value |
|---|---|
| Precision | 0.9768 |
| Recall | 0.9989 |
| F1 score | 0.9877 |
| Accuracy | 0.9978 |

Confusion matrix: 182,105 legit correctly identified, 414 legit flagged as fraud, 20 frauds missed, 17,461 frauds caught.

---

## 🧾 How Predictions Work

- **Validation:** all seven features are required. Binary features must be 0 or 1, and the three numeric features must be 0 or greater.
- **Label:** the class with the higher predicted probability (fraud or legit).
- **Confidence:** based on the fraud probability: **High** if it is ≥ 0.80 or ≤ 0.20, **Medium** if ≥ 0.60 or ≤ 0.40, otherwise **Low**.

---

## ⚠️ Limitations

- The dataset has sharp cut-offs (for example, the fraud rate jumps once the purchase ratio passes about 4.0), so the near-perfect metrics reflect this dataset and should not be read as real-world performance.
- Because of `scale_pos_weight`, the fraud probability is a model score, not a calibrated chance of fraud.
- Legacy `.pkl` model files from earlier iterations are git-ignored and are not used by the app.