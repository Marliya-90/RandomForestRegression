# taxi_frontend.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

st.title("🚖 Taxi Fare Prediction (RandomForest Regressor)")

# 1️⃣ Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Dataset Preview")
    st.dataframe(df.head())

    # 2️⃣ Select target column
    target_column = st.selectbox("Select the target column", df.columns)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Drop rows with missing target
    df = df.dropna(subset=[target_column])
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # 3️⃣ Encode categorical columns
    le_dict = {}
