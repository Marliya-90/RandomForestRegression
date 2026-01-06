import streamlit as st
import pandas as pd
import os

st.title("🚖 Taxi Trip Price Prediction")

DATA_PATH = "taxi_trip_pricingR.csv"

st.write("Files in repo:", os.listdir())

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    st.success("Taxi dataset loaded successfully ✅")
else:
    st.error("Taxi CSV not found in GitHub repo ❌")
    st.stop()

st.dataframe(df.head())
