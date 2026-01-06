import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Sales Prediction using Random Forest", layout="wide")
st.title("📊 Sales Prediction using Random Forest")

# -----------------------------
# CSV LOAD (DEPLOY SAFE)
# -----------------------------
DATA_PATH = "taxi_trip_pricingR.csv"
   # 👉 change ONLY if your csv name is different

def load_data():
    if os.path.exists(DATA_PATH):
        st.success("✅ CSV loaded from GitHub repo")
        return pd.read_csv(DATA_PATH)
    else:
        st.warning("⚠️ CSV not found in repo. Please upload file.")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        else:
            st.stop()

df = load_data()

# -----------------------------
# DATA PREVIEW
# -----------------------------
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

st.write("Shape:", df.shape)
st.write("Columns:", df.columns.tolist())

# -----------------------------
# FEATURE & TARGET SELECTION
# -----------------------------
st.subheader("🎯 Feature Selection")

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

target = st.selectbox("Select Target Column", numeric_cols)
features = st.multiselect(
    "Select Feature Columns",
    [col for col in numeric_cols if col != target]
)

if len(features) == 0:
    st.warning("Please select at least one feature")
    st.stop()

X = df[features]
y = df[target]

# -----------------------------
# MODEL TRAINING
# -----------------------------
if st.button("🚀 Train Random Forest Model"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    st.success("✅ Model trained successfully!")
    st.metric("R² Score", round(r2, 3))
    st.metric("RMSE", round(rmse, 3))

# -----------------------------
# FILE DEBUG (ONLY FOR DEPLOY)
# -----------------------------
with st.expander("📂 Debug: Files in server"):
    st.write(os.listdir())
