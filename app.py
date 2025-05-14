import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load the model
model = joblib.load("fraud_model_smote.pkl")  # Make sure this file is in your project folder

def add_features(df):
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['high_value'] = df['amount'] > 10000
    df['high_risk_location'] = df['location'].isin(['Caracas', 'Kabul', 'Mogadishu', 'Pyongyang'])
    df['odd_hour'] = df['hour'].isin(range(0, 5))
    return df

st.set_page_config(page_title="FraudFilter Lite", layout="wide")
st.title("🔍 FraudFilter Lite")
st.markdown("""
Welcome to **FraudFilter Lite** — a lightweight AI tool that helps you flag suspicious financial transactions.

**How to Use:**
- Upload a CSV file with columns: `transaction_id`, `amount`, `location`, `time`
- The app will process your data and flag transactions that may be fraudulent
- Download the flagged transactions for your records
""")

uploaded_file = st.file_uploader("📁 Upload your transaction CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = add_features(df)
    features = ['amount', 'hour', 'day_of_week', 'high_value', 'high_risk_location', 'odd_hour']
    df['fraud_prediction'] = model.predict(df[features])

    st.subheader("📊 Full Dataset with Predictions")
    st.dataframe(df)

    flagged = df[df['fraud_prediction'] == 1]
    st.subheader(f"🚨 Flagged Transactions: {len(flagged)}")
    st.dataframe(flagged)

    st.download_button("⬇️ Download Flagged Report", data=flagged.to_csv(index=False), file_name="flagged_report.csv")

else:
    st.info("Upload a CSV file to begin fraud analysis.")
