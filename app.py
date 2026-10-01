import streamlit as st
from utils import build_input, predict

st.set_page_config(
    page_title="Fraud Detector",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 Credit Card Fraud Detector")
st.markdown("Fill in the transaction details below and click **Predict**.")
st.divider()

with st.form("prediction_form"):
    st.subheader("Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        distance_from_home = st.number_input(
            "Distance from Home (km)",
            min_value=0.0, value=10.0, step=0.1,
            help="How far was the transaction from the cardholder's home?"
        )
        distance_from_last_transaction = st.number_input(
            "Distance from Last Transaction (km)",
            min_value=0.0, value=1.0, step=0.1,
            help="How far from the previous transaction location?"
        )
        ratio_to_median_purchase_price = st.number_input(
            "Ratio to Median Purchase Price",
            min_value=0.0, value=1.0, step=0.01,
            help="Transaction amount ÷ cardholder's median purchase price. >1 means above average."
        )

    with col2:
        repeat_retailer = st.selectbox(
            "Repeat Retailer?",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Has the cardholder shopped at this retailer before?"
        )
        used_chip = st.selectbox(
            "Used Chip?",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Was the card chip used for this transaction?"
        )
        used_pin_number = st.selectbox(
            "Used PIN?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Was a PIN entered for this transaction?"
        )
        online_order = st.selectbox(
            "Online Order?",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Was this an online transaction?"
        )

    submitted = st.form_submit_button("🔎 Predict", use_container_width=True)

if submitted:
    try:
        input_data = build_input(
            distance_from_home=distance_from_home,
            distance_from_last_transaction=distance_from_last_transaction,
            ratio_to_median_purchase_price=ratio_to_median_purchase_price,
            repeat_retailer=repeat_retailer,
            used_chip=used_chip,
            used_pin_number=used_pin_number,
            online_order=online_order,
        )

        result = predict(input_data)

        st.divider()
        st.subheader("Prediction Result")

        is_fraud = result["label"] == 1

        if is_fraud:
            st.error(f"🚨 **FRAUD DETECTED** — Confidence: {result['confidence']}")
        else:
            st.success(f"✅ **Legitimate Transaction** — Confidence: {result['confidence']}")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Prediction", result["label_text"])
        col_b.metric("Fraud Probability", f"{result['fraud_probability'] * 100:.2f}%")
        col_c.metric("Legit Probability", f"{result['legit_probability'] * 100:.2f}%")

        st.markdown("#### Fraud Probability")
        st.progress(result["fraud_probability"])

        with st.expander("📋 Raw Input Sent to Model"):
            st.json(result["input_used"])

    except ValueError as e:
        st.error(f"⚠️ Validation Error:\n\n{e}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")